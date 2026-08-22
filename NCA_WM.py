from typing import Literal, Optional, Union
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

class NCA_WM(eqx.Module):
    """
    A JAX implementation of a 'Neural Adaptive Cellular Engine', a model capable of learning emergent behaviors for game-like simulations
    
    Args:
        `actions`: Number of action channels that each cell perceives
        `vis_channels`: Number of one-hot visible channels that each cell can update and read from, or "RGB" for 3 continuous channels
        `hid_channels`: Number of hidden channels that each cell can update and read from
        `hidden_neurons`: Number of hidden neurons of the update net second layer (default: 128)
        `padding_mode`: Padding mode for the perceive function: 'reflect', 'circular', 'replicate', 'random', constant (float value), or default 'zeros'
        `kernel`: Matrix that defines the neighborhood pattern for cell interactions (default: Von Neumann 3x3)
        `downscale_factor`: Factor to step the model through a smaller grid and upscale back (1, 2, 4, ...), reduces VRAM and accuracy (default: 1)
        `embedding_dim`: Dimension to compress visual channels and use index map data loading instead of one-hot (default: None - no embedding)
        `global_context`: Whether to give information about the global visible state of every cell (mean) as input to every cell (default: False)
        `dtype`: Data type of the weights of the neural networks (default: jnp.float32)
    """

    actions: int = eqx.field(static=True)
    vis_channels: int = eqx.field(static=True)
    is_continuous: bool = eqx.field(static=True)
    hid_channels: int = eqx.field(static=True)
    padding_mode: str | float = eqx.field(static=True)
    downscale_factor: int = eqx.field(static=True)
    embedding_dim: int | None = eqx.field(static=True)
    global_context: bool = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)
    kernel_h: int = eqx.field(static=True)
    kernel_w: int = eqx.field(static=True)
    
    kernel: jnp.ndarray # not static, but excluded as a training parameter
    
    net: eqx.nn.Sequential
    vis_embedding_encoder: eqx.nn.Embedding | None
    vis_embedding_decoder: eqx.nn.Conv2d | None
    
    def __init__(self,
            actions: int,
            vis_channels: Union[int, Literal["RGB"]],
            hid_channels: int,
            *,
            hidden_neurons: int = 128,
            padding_mode: Union[float, int,
                Literal['zeros'],
                Literal['reflect'],
                Literal['circular'], Literal['wrap'],
                Literal['replicate'], Literal['edge'],
                Literal['random']
            ] = 'zeros',
            kernel: Optional[list[list[int]]] = None,
            downscale_factor: int = 1,
            embedding_dim: Optional[int] = None,
            global_context: bool = False, # WIP
            dtype: jnp.dtype = jnp.float32,

            key: jax.Array
        ):

        # validate parameters
        assert actions == 0 or actions > 1, f"actions must be 0 or > 1, got {actions}"

        is_continuous = False
        if vis_channels == "RGB":
            is_continuous = True
        else:
            assert vis_channels > 0, f"vis_channels must be > 0 or \"RGB\", got {vis_channels}"
        
        assert hid_channels >= 0, f"hid_channels must be >= 0, got {hid_channels}"
        
        paddings = {'zeros', 'reflect', 'circular', 'wrap', 'replicate', 'edge', 'random'}
        if isinstance(padding_mode, str):
            assert padding_mode in paddings, f"padding_mode must be one of: {', '.join(paddings)}"
        else:
            padding_mode = float(padding_mode) # to float if it's an int
        
        assert downscale_factor >= 1, f"downscale_factor must be >= 1, got {downscale_factor}"
        assert embedding_dim is None or embedding_dim >= 1, f"embedding_dim must be None or >= 1, got {embedding_dim}"
        if is_continuous:
            assert embedding_dim is None, f"embedding_dim must be None if vis_channels are continuous (RGB)"
        assert hidden_neurons >= 1, f"hidden_neurons must be >= 1, got {hidden_neurons}"

        # set parameters

        self.actions = actions

        self.vis_channels = 3 if vis_channels == "RGB" else vis_channels
        self.is_continuous = is_continuous # whether vis_channels is one-hot discrete or continuous (e.g: 3 RGB channels)
        self.hid_channels = hid_channels

        self.padding_mode = padding_mode
        self.downscale_factor = downscale_factor
        self.embedding_dim = embedding_dim
        self.global_context = global_context
        self.dtype = dtype

        self.channels = self.vis_repr_dim + hid_channels

        keys = jax.random.split(key, 5) # to randomize neural networks' weights

        _actions = self.actions # copy

        # increase inputs exponentially if downscaling is applied
        if self.downscale_factor > 1:
            factor = self.downscale_factor ** 2

            self.channels *= factor
            _actions *= factor # note that self.actions remains the same! This is only used in input_dim calc

        if kernel is None:
            # Von Neumann 3x3 as default
            kernel = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]
        
        self.kernel_size = sum(sum(row) for row in kernel) # total number of 1s in the kernel

        # what each cell receives as input:
        self.input_dim = (self.channels * self.kernel_size) + _actions

        # layers of the update net
        conv1 = eqx.nn.Conv2d(self.input_dim, hidden_neurons, kernel_size=1, key=keys[0])
        # convhalf = eqx.nn.Conv2d(hidden_neurons, hidden_neurons, kernel_size=1, key=keys[4])
        conv2 = eqx.nn.Conv2d(hidden_neurons, self.channels, kernel_size=1, key=keys[1]) # output visual and hidden channels
        
        # init second layer to zero for stability (the used key is wasted)
        conv2 = eqx.tree_at(lambda c: c.weight, conv2, jnp.zeros(conv2.weight.shape))
        conv2 = eqx.tree_at(lambda c: c.bias, conv2, jnp.zeros(conv2.bias.shape))

        # update net
        self.net = eqx.nn.Sequential([
            conv1,
            eqx.nn.Lambda(jax.nn.silu),
            # convhalf,
            # eqx.nn.Lambda(jax.nn.silu),
            conv2
        ])

        # embedding encoder and decoder
        self.vis_embedding_encoder = None
        self.vis_embedding_decoder = None

        if self.embedding_dim is not None:
            # input is index-map instead of one-hot
            self.vis_embedding_encoder = eqx.nn.Embedding(self.vis_channels, self.embedding_dim, key=keys[2])
            # output is visual channels
            self.vis_embedding_decoder = eqx.nn.Conv2d(self.embedding_dim, self.vis_channels, kernel_size=1, key=keys[3])

        # setup kernel (what each cell perceives)
        self.kernel_h = len(kernel) # length of outer array (columns)
        self.kernel_w = len(kernel[0]) # length of inner array (rows)

        # isolate each '1' in the kernel into separate filters for convolution
        kernels = np.zeros((self.kernel_size, self.kernel_h, self.kernel_w), dtype=np.float32) # create with CPU
        idx = 0

        for dy in range(self.kernel_h):
            for dx in range(self.kernel_w):
                # set 1 where kernel has 1
                if kernel[dy][dx] == 1:
                    kernels[idx, dy, dx] = 1.0
                    idx += 1

        self.kernel = jnp.array(kernels[:, np.newaxis, :, :], dtype=self.dtype) # move to GPU

        # change floating point precision if isn't default float32
        if self.dtype != jnp.float32:
            def cast(x):
                return x.astype(self.dtype) if eqx.is_inexact_array(x) else x
            
            # apply cast to trainable objects
            self.net = jax.tree_util.tree_map(cast, self.net)
            
            if self.vis_embedding_encoder is not None:
                self.vis_embedding_encoder = jax.tree_util.tree_map(cast, self.vis_embedding_encoder)
                self.vis_embedding_decoder = jax.tree_util.tree_map(cast, self.vis_embedding_decoder)

    def perceive(self, x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        # x shape is (C, H, W)
        x = x.astype(self.dtype) # make sure dtypes match

        # prepare full kernel for convolution
        # this calculation is static and could be done once outside of perceive
        # but thanks to XLA, this is compiled right away and does not slow down
        full_kernel = jnp.tile(self.kernel, (self.channels, 1, 1, 1))
        
        # calculate padding (no dilation)
        pad_h = (self.kernel_h // 2)
        pad_w = (self.kernel_w // 2)
        pad_spec = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))

        # pad with zeros
        if self.padding_mode == 'zeros':
            img_input = jnp.pad(x, pad_spec, mode='constant', constant_values=0.0)
        
        # pad with reflection of the inner cells
        elif self.padding_mode == 'reflect':
            img_input = jnp.pad(x, pad_spec, mode='reflect')

        # pad with replication of the border cells
        elif self.padding_mode in ("replicate", "edge"):
            img_input = jnp.pad(x, pad_spec, mode='edge')

        # wrap around (left-right, top-bottom)
        elif self.padding_mode in ("circular", "wrap"):
            img_input = jnp.pad(x, pad_spec, mode='wrap')
        
        # pad with random noise 0-1
        elif self.padding_mode == 'random':
            # temporary constant padding
            img_input = jnp.pad(x, pad_spec, mode='constant', constant_values=0.0)

            # create noise between -1.0 and 1.0 - rather than between 0.0 and 1.0
            # the mean will be 0.0 and make the model more stable during autoregressivity
            noise = jax.random.uniform(key, img_input.shape, dtype=self.dtype, minval=-1.0, maxval=1.0)
            
            # 1 where state is, 0 where padding is
            mask = jnp.pad(jnp.ones_like(x), pad_spec, mode='constant', constant_values=0.0)
            
            # Replace the 0.0 padding with the noise
            img_input = jnp.where(mask == 0.0, noise, img_input)
        
        # pad with a custom constant
        else:
            img_input = jnp.pad(x, pad_spec, mode='constant', constant_values=float(self.padding_mode))

        # specify shapes
        dims = jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2, 3), # input NCHW
            rhs_spec=(0, 1, 2, 3), # kernel OIHW
            out_spec=(0, 1, 2, 3) # output NCHW
        )
        
        res = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(img_input, axis=0), # add batch dimension (1CHW)
            rhs=full_kernel,
            window_strides=(1, 1), # move by 1 pixel at a time
            padding='VALID', # padding was applied manually
            lhs_dilation=(1, 1),
            rhs_dilation=(1, 1), # (dilation)
            dimension_numbers=dims,
            feature_group_count=self.channels
        )

        return jnp.squeeze(res, axis=0) # remove batch dimension (no more required)
    
    @staticmethod
    def pixel_unshuffle(x: jnp.ndarray, factor: int) -> jnp.ndarray:
        if factor == 1:
            return x
        
        # x is (C, H, W)
        C, H, W = x.shape
        H //= factor
        W //= factor

        x = x.reshape(C, H, factor, W, factor)
        x = x.transpose(0, 2, 4, 1, 3)
        # (C*(factor^2), H//factor, W//factor)
        return x.reshape(C * (factor ** 2), H, W)

    @staticmethod
    def pixel_shuffle(x: jnp.ndarray, factor: int) -> jnp.ndarray:
        if factor == 1:
            return x
        
        # x is (C, H, W)
        C, H, W = x.shape
        C //= (factor ** 2)
        
        x = x.reshape(C, factor, factor, H, W)
        x = x.transpose(0, 3, 1, 4, 2)
        # (C//(factor^2), H*factor, W*factor)
        return x.reshape(C, H * factor, W * factor)
    
    @property
    def trainable_mask(self):
        mask = jax.tree_util.tree_map(eqx.is_inexact_array, self)
        return eqx.tree_at(
            lambda m: (m.kernel), # excluded from training
            mask,
            replace=(False)
        )
    
    # representation of visual channels dimension (embedding_dim if valid, otherwise vis_channels)
    @property
    def vis_repr_dim(self):
        return self.embedding_dim if self.embedding_dim is not None else self.vis_channels
    
    # converts index-map to model input
    def encode_vis(self, vis: jnp.ndarray) -> jnp.ndarray:
        # vis is (H, W) uint8 (or similar) or (vis_channels, H, W) one-hot float32
        if self.vis_embedding_encoder is not None:
            # encode - transpose (H, W, embedding_dim) to (embedding_dim, H, W)
            return self.vis_embedding_encoder.weight[vis].transpose(2, 0, 1)
        
        return vis
    
    # converts model output to one-hot
    def decode_vis(self, vis: jnp.ndarray) -> jnp.ndarray:
        if self.vis_embedding_decoder is not None:
            return self.vis_embedding_decoder(vis)
        
        return vis

    # forward, updates cells once
    def __call__(self, state: jnp.ndarray, action_map: jnp.ndarray | None, key: jax.Array) -> jnp.ndarray:
        # apply perception layer and padding
        inp = self.perceive(state, key)
        
        # append action map if provided
        if action_map is not None:
            inp = jnp.concatenate([inp, action_map], axis=0)
        
        # apply update net to every cell at once
        dx = self.net(inp.astype(self.dtype)).astype(state.dtype) # cast to self.dtype and back to float32

        return state + dx

    # applies forward sequentially 'substeps' amount of times
    def step(self, state: jnp.ndarray, action_map: jnp.ndarray | None, substeps: int, key: jax.Array) -> jnp.ndarray:
        # downscale if factor > 1
        state = NCA_WM.pixel_unshuffle(state, self.downscale_factor)
    
        # downscale action map
        action = None
        if action_map is not None:
            action = NCA_WM.pixel_unshuffle(action_map, self.downscale_factor)

        # forward steps - a python for loop is being used rather than jax's fori_loop
        # or scan because "substeps" is usually a small static number, the overhead is not worth it
        for _ in range(substeps):
            # randomize key each forward
            key, step_key = jax.random.split(key)
            
            state = self(state, action, step_key)
            
        # upscale back to original resolution
        return NCA_WM.pixel_shuffle(state, self.downscale_factor)

def save_model_and_optstate(model: NCA_WM, path: str, opt_state: optax.OptState):
    trainable_model = eqx.filter(model, model.trainable_mask) # only save trainable parameters! exclude kernel
    eqx.tree_serialise_leaves(path, (trainable_model, opt_state))

def load_model_and_optstate(path: str, skeleton: NCA_WM, optimizer: optax.GradientTransformation) -> tuple[NCA_WM, optax.OptState]:
    # skeleton must be created with the same parameters as the saved model

    # get model weights and opt state from skeleton
    base_params = eqx.filter(skeleton, skeleton.trainable_mask)
    base_opt_state = optimizer.init(base_params)

    # deserialize
    loaded_params, loaded_opt_state = eqx.tree_deserialise_leaves(path, (base_params, base_opt_state))

    # return combined new model and opt state
    return eqx.combine(loaded_params, skeleton), loaded_opt_state

# only load model and discard opt_state
def load_model(path: str, skeleton: NCA_WM) -> NCA_WM:
    base_params = eqx.filter(skeleton, skeleton.trainable_mask)
    
    loaded_params, _ = eqx.tree_deserialise_leaves(path, (base_params, None))
    
    return eqx.combine(loaded_params, skeleton)