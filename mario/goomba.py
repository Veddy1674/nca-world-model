import jax
import jax.numpy as jnp
import optax
import numpy as np
import cv2
from NACE import NACE

# training
STEPS = 10000
BATCH_SIZE = 64
LOG_SEGMENTS = 100
LOAD_MODEL = None#"mario/model.eqx"
SAVE_MODEL = "mario/goomba_model.eqx"
LOSS_GRAPH = "mario/goomba_loss_graph.png"

# data
DATA_GLOB = "mario/data/goomba_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "VRAM"

# visualizer
LOAD_MODEL_INF = "mario/goomba_model.eqx"
LOAD_DATA_INF = "mario/data/goomba_*.npz"
WIN_SIZE = None # auto-calculated
KEY_MAP = {
    # 'd': 0,
    # 'a': 1,
}
DEFAULT_ACTION = None
FPS = None

# model
SUBSTEPS = 2
TRUNCATED_BPTT = 1

COLOR_MAP = [
    [33, 33, 33], # background/non-solid
    [17, 17, 17], # solid
    [142, 107, 55], # goomba
]

_class_weights = jnp.array([0.8, 0.8, 1.0])

def make_model(key):
    return NACE(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=2, # for this simple environment, cells only read/write colors
        hidden_neurons=128,
        padding_mode="zeros",
        key=key,
    )

def make_optimizer():
    # scheduler = optax.warmup_cosine_decay_schedule(
    #     init_value=1e-4,
    #     peak_value=1e-2,
    #     warmup_steps=100,
    #     decay_steps=900,
    # )
    scheduler = optax.exponential_decay(1e-3, LOG_SEGMENTS, decay_rate=0.99)
    return optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    ), scheduler

def init_hidden(state: jnp.ndarray, info: jnp.ndarray, hid_channels: int, GRID_H: int, GRID_W: int) -> jnp.ndarray:

    if state.ndim == 3: # INFERENCE (unbatched)
        
        hid = jnp.zeros((hid_channels, GRID_H, GRID_W), dtype=jnp.float32)

        goomba_mask = state[2:3, :, :] 
        
        info_expanded = info[:, None, None]
        
        return jnp.where(goomba_mask == 1, info_expanded, hid)

    else: # TRAINING (batched)    
        # info shape: (B, 2) - where info[:, 0] is offsetX and info[:, 1] is direction
        B = state.shape[0] # same as info B

        hid = jnp.zeros((B, hid_channels, GRID_H, GRID_W), dtype=jnp.float32)

        goomba_mask = state[:, 2:3, :, :] # grab state to see where there is a goomba
        
        # get info and expand
        offset_expanded = info[:, 0:1, None, None]
        dir_expanded = info[:, 1:2, None, None]
        
        info_expanded = jnp.concatenate([offset_expanded, dir_expanded], axis=1)
        
        # zeroed where there is no goomba, offsetX and direction where there is goomba
        return jnp.where(goomba_mask == 1, info_expanded, hid)

def loss_calc(preds: jnp.ndarray, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    preds_vis = jnp.moveaxis(preds[:, :len(COLOR_MAP)], 1, -1) # BHWC
    targets_vis = jnp.moveaxis(targets, 1, -1) # BHWC
    
    celoss = optax.softmax_cross_entropy(preds_vis, targets_vis)

    class_idx = jnp.argmax(targets_vis, axis=-1) # argmax of C
    weights = _class_weights[class_idx]

    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)

    preds_hid = jnp.moveaxis(preds[:, len(COLOR_MAP):], 1, -1)

    goomba_mask = targets_vis[:, :, :, 2:3]
    
    info_target_expanded = infos[:, None, None, :]
    
    hidden_squared_error = (preds_hid - info_target_expanded) ** 2
    
    masked_hidden_error = hidden_squared_error * goomba_mask
    
    hidden_loss = jnp.sum(masked_hidden_error) / jnp.sum(goomba_mask) * 2

    return celoss + hidden_loss

# custom inference logic:
# def state_convert(model: NACE, state: jax.Array):
#     # get visible channels from state and move to RAM
#     if state.ndim == 3: # if CHW (one-hot)
#         visible = np.array(jnp.argmax(state[:model.vis_channels], axis=0), dtype=np.uint8)
#     else: # if HW (index map)
#         visible = np.array(state, dtype=np.uint8)

#     # (15, 15, 16, 16, 3) - get sprites for each position
#     img_blocks = _sprites[visible]
    
#     # (15, 16, 15, 16, 3) - swap dimensions
#     img_blocks = np.transpose(img_blocks, (0, 2, 1, 3, 4))
    
#     # create empty HWC BGR image, upscale to 240x240
#     img = img_blocks.reshape(15 * 16, 15 * 16, 3) # (uint8)

#     img = img[:, :, ::-1] # to BGR
        
#     # upscale
#     return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)