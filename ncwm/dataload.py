from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
from glob import glob
from PIL import Image

# default behavior of data preprocessing (convert to one-hot or keep index map if embedding)
def preprocess_data(
        file: str,
        vis_channels: int,
        use_embedding: bool,
        is_continuous: bool,
        color_map: Optional[list[tuple[int, int, int]]],
        verbose: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

    raw_states = None
    raw_actions = None
    raw_infos = None

    data = np.load(file) # load to RAM

    if file.endswith(".npy"):
        # simple s = s' model: only states, no actions/infos
        raw_states = data
    else:
        # the whole loading process uses numpy (CPU), whether data is moved in VRAM right away or later on is decided externally
        raw_states = np.array(data['states']) # BCHW - C must be 1 if index map
        raw_actions = np.array(data['actions']) if 'actions' in data else None # B - where B is raw_states's B - 1 (since last state has no action)
        raw_infos = np.array(data['infos']) if 'infos' in data else None # extra info if any (e.g: about the state for hidden channels loss calc)

    # if states are integers (one-hot or index map)
    is_int = np.issubdtype(raw_states.dtype, np.integer)

    # if index map, squeeze C dim
    if is_int and raw_states.ndim == 4 and raw_states.shape[1] == 1:
        raw_states = np.squeeze(raw_states, axis=1) # B1HW to BHW

    # if RGB (use_embedding always false)
    if is_continuous:
        # if vis_channels is "RGB" but COLOR_MAP is defined, assume data is one-hot and convert to RGB with a warning:
        if color_map is not None:

            # NOTE: vis_channels is usually len(COLOR_MAP), but just in case, they are two different args here
            # but in this specific case, vis_channels is always 3 and len(COLOR_MAP) can be anything

            # if onehot BCHW convert to index map BHW (easier to map colors)
            if raw_states.ndim == 4 and raw_states.shape[1] == len(color_map):
                raw_states = np.argmax(raw_states, axis=1)
            
            cmap = np.array(color_map, dtype=np.float32) / 255.0 # normalized
            rgb_states = cmap[raw_states]
            
            # BHWC to BCHW, where C is always 3
            raw_states = np.transpose(rgb_states, (0, 3, 1, 2))

            # this shouldn't really be a warning, as it could be intentional (e.g: to test if the model can
            # learn with the same data being RGB rather than one-hot)
            if verbose:
                print("Note: COLOR_MAP is defined but vis_channels are continuous (RGB), data will be converted to RGB")

        # if uint8 just normalize
        elif np.issubdtype(raw_states.dtype, np.integer):
            raw_states = raw_states.astype(np.float32) / 255.0
        else:
            raw_states = raw_states.astype(np.float32)

        return raw_states, raw_actions, raw_infos
    
    if use_embedding:
        if not is_int:
            if verbose:
                print("Warning: embedding is enabled but data is one-hot. It is recommended to have data as index map or to disable embedding")
            
            raw_states = np.argmax(raw_states, axis=1).astype(np.uint8) # one-hot to index map (using uint8 in case of more than 255 classes)
        
        return raw_states, raw_actions, raw_infos
    
    # if one-hot or index map without embedding
    if is_int:
        # convert to one-hot if it's not already, always float32
        raw_states = np.eye(vis_channels, dtype=jnp.float32)[raw_states]

        if raw_states.ndim == 5: # if B1HWC -> BHWC
            raw_states = np.squeeze(raw_states, axis=1)

        # one-hot transposes to BHWC, so convert it back to BCHW
        raw_states = np.transpose(raw_states, (0, 3, 1, 2))
    
    # BCHW, B, B(any)
    return raw_states, raw_actions, raw_infos

# default behavior of data preprocessing (convert image to one-hot)
def preprocess_image(file: str, color_map: list[tuple[int, int, int]] | None, use_embedding: bool) -> jax.Array:
    img = Image.open(file).convert("RGBA") # if it has transparency, turn it to solid black
    img_array = np.array(img) # HW4, as uint8
    
    h, w, _ = img_array.shape
    rgb = img_array[..., :3] # HW3
    alpha = img_array[..., 3]

    if color_map is None:
        # TODO implement, this method should have model as arg0 maybe
        raise NotImplementedError("Not implemented: COLOR_MAP is None, intent is expected to be to load a RGB image for inference purposes (will be implemented), if not, whenever COLOR_MAP is None but model's 'vis_channels' isn't \"RGB\", inference is simply not possible, so make sure to either set COLOR_MAP appropriately or use a model with RGB output")

    rgb[alpha < 255] = [0, 0, 0] # set every transparent pixel to black
    
    # create an index map of -1s
    state_idx = np.full((h, w), -1, dtype=np.uint8)

    # TODO revise (might not be performant)
    
    for cls_idx, color in enumerate(color_map):
        # find where the pixel matches the colormap color
        mask = np.all(rgb == color, axis=-1)
        state_idx[mask] = cls_idx

    wrong_idx = np.argmax(state_idx == -1) # first occurrence of -1

    # if any pixel remains -1, it means it wasn't found in the COLOR_MAP
    if state_idx.ravel()[wrong_idx] == -1:

        # find the first wrong pixel to show in the error
        y, x = np.unravel_index(wrong_idx, state_idx.shape)

        raise ValueError(f"Color {tuple(rgb[y, x])} mismatch in {file} at pixel (x{x}, y{y}), not defined in COLOR_MAP of config (Note that pixels with any transparency are set to black)")

    # move to VRAM always! (preprocess_image is only meant for inference)
    state_idx = jnp.array(state_idx)

    if use_embedding: # index map
        return state_idx # HW uint8
    
    # convert to one-hot
    one_hot = jax.nn.one_hot(state_idx, len(color_map)) # HWC
    
    # transpose from HWC to CHW and convert to float32
    return jnp.transpose(one_hot, (2, 0, 1)).astype(jnp.float32)

# get first state from a .npz, .npy, or .png file
def load_first(
        data_path: str,
        vis_channels: int,
        color_map: Optional[list[tuple[int, int, int]]],
        use_embedding: bool,
        is_continuous: bool,
        index: int = 0
    ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:

    # if it's a glob with multiple files, take the first one
    file = sorted(glob(data_path))[0]

    # data_path can be either a .npz file, .npy file, or a .png file
    if file.endswith(".npz") or file.endswith(".npy"):

        s, a, i = preprocess_data(
            file=file,
            vis_channels=vis_channels,
            use_embedding=use_embedding,
            is_continuous=is_continuous,
            color_map=color_map
        )

        return (
            jnp.array(s[index]),
            jnp.array(a[index]) if a is not None else None,
            jnp.array(i[index]) if i is not None else None
        ) # all to VRAM

    elif file.endswith(".png"): # TODO allow other formats (preferably lossless only)

        # instantly to VRAM
        return preprocess_image(file, color_map, use_embedding), None, None # handles color_map None internally

    else:
        raise NotImplementedError(f"Unsupported file format: {file}. Only .npz, .npy, and .png (partially) are supported")