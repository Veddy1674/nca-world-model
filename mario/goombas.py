import jax
import jax.numpy as jnp
import optax
import numpy as np
import cv2
from NCA_WM import NCA_WM

# training
STEPS = 15000
BATCH_SIZE = 128
LOG_SEGMENTS = 100
LOAD_MODEL = None
SAVE_MODEL = "mario/goombas_model.eqx"
LOSS_GRAPH = "mario/goombas_loss_graph.png"
TRUNCATED_BPTT = 2
HIDDEN_NOISE_STD = 0.001

# data
DATA_GLOB = "mario/data/goombas_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "VRAM"

# visualizer
LOAD_MODEL_INF = "mario/goombas_model.eqx"
LOAD_DATA_INF = "mario/data/goombas_*.npz"
WIN_SIZE = None # auto-calculated
KEY_MAP = {
    # 'd': 0,
    # 'a': 1,
}
DEFAULT_ACTION = None
FPS = None

# model
SUBSTEPS = 2

COLOR_MAP = [
    [10, 10, 10], # background/non-solid
    [142, 107, 55], # goomba
]

_class_weights = jnp.array([0.1, 1.0])

def make_model(key):
    return NCA_WM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=2,
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
    scheduler = optax.exponential_decay(7e-4, LOG_SEGMENTS, decay_rate=0.99)
    return optax.chain(
        optax.clip(0.5),
        optax.adam(learning_rate=scheduler),
    ), scheduler

def init_hidden(state: jnp.ndarray, info: jnp.ndarray, hid_channels: int, GRID_H: int, GRID_W: int) -> jnp.ndarray:
    # NOTA: hid_channels deve essere impostato a 2 nel config per ospitare [sin, cos]

    if state.ndim == 3: # INFERENCE (unbatched: C, H, W)
        hid = jnp.zeros((hid_channels, GRID_H, GRID_W), dtype=jnp.float32)
        goomba_mask = state[1:2, :, :] # Canale del Goomba
        
        # info shape: (2,) -> diventa (2, 1, 1) per il broadcast
        info_expanded = info[:, None, None]
        return jnp.where(goomba_mask == 1, info_expanded, hid)

    else: # TRAINING (batched: B, C, H, W)
        B = state.shape[0]
        hid = jnp.zeros((B, hid_channels, GRID_H, GRID_W), dtype=jnp.float32)
        goomba_mask = state[:, 1:2, :, :] # Canale del Goomba (B, 1, H, W)
        
        # info shape: (B, 2) -> diventa (B, 2, 1, 1) per il broadcast sui pixel
        info_expanded = info[:, :, None, None]
        return jnp.where(goomba_mask == 1, info_expanded, hid)

def loss_calc(preds: jnp.ndarray, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    B, _, H, W = targets.shape
    num_hid_channels = preds.shape[1] - len(COLOR_MAP)

    # --- 1. VISUAL LOSS ---
    preds_vis = jnp.moveaxis(preds[:, :len(COLOR_MAP)], 1, -1)
    preds_hid = jnp.moveaxis(preds[:, len(COLOR_MAP):], 1, -1) 

    targets_vis = jnp.moveaxis(targets, 1, -1)
    
    celoss = optax.softmax_cross_entropy(preds_vis, targets_vis)
    class_idx = jnp.argmax(targets_vis, axis=-1)
    weights = _class_weights[class_idx]
    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)

    # --- 2. HIDDEN CHANNEL LOSS (The Clock) ---
    goomba_mask_bhwc = targets_vis[:, :, :, 1:2] 
    info_target_expanded = infos[:, None, None, :]
    
    hidden_squared_error = (preds_hid - info_target_expanded) ** 2
    masked_hidden_error = hidden_squared_error * goomba_mask_bhwc
    
    # FIX: Do NOT divide by total_pixels! The Goomba is only 1 pixel.
    hidden_loss = jnp.sum(masked_hidden_error) / (B * num_hid_channels)

    # --- 3. SILENCE PENALTY (Keep the background clean) ---
    # Without this, the Goomba steps into garbage when it moves!
    not_goomba_mask = 1.0 - goomba_mask_bhwc
    silence_penalty = jnp.mean((preds_hid * not_goomba_mask) ** 2)

    # Weight the hidden loss heavily so it respects the clock as much as the visual data!
    return celoss + (hidden_loss * 1.6) + (silence_penalty * 1.0)

_sprites = np.load("mario/env/color_map.npz")["sprites"] # (36, 16, 16, 3)

MAGIC = 32

def get_offset_from_circular(sin_cos):
    sin, cos = sin_cos[0], sin_cos[1]
    angle = np.arctan2(sin, cos)

    if angle < 0:
        angle += 2 * np.pi
        
    return int(round((angle / (2 * np.pi)) * MAGIC)) % MAGIC

# custom inference logic:
def state_convert(model: NCA_WM, state: jax.Array, hid: jax.Array):
    # move to RAM
    visible = np.array(state, dtype=np.uint8)
    hidden = np.array(hid, dtype=np.float32)

    # find goomba coordinate (always exactly one)
    goomba_y, goomba_x = np.argwhere(visible[1] == 1)[0]
    
    # extract [sin, cos] and calculate pixel offset
    goomba_hidden = hidden[:, goomba_y, goomba_x]
    offset = int(get_offset_from_circular(goomba_hidden))
    print(f"Goomba counter (offset): {offset}")

    # grid with background everywhere (index 16)
    sprite_ids = np.full((15, 16), 16, dtype=np.uint8)

    # (15, 16, 16, 16, 3) - get sprites for each position
    img_blocks = _sprites[sprite_ids]
    
    # (15, 16, 16, 16, 3) - swap dimensions
    img_blocks = np.transpose(img_blocks, (0, 2, 1, 3, 4))
    
    # reshape to background canvas HWC (240x256)
    img = img_blocks.reshape(15 * 16, 16 * 16, 3)

    # get the standalone goomba sprite (index 30)
    goomba_sprite = _sprites[30]

    # calculate dynamic horizontal pixel position on the canvas
    y_start = goomba_y * 16
    x_start = (goomba_x * 16) + offset
    
    # draw goomba directly onto background canvas with pixel-level offset
    img[y_start : y_start + 16, x_start : x_start + 16] = goomba_sprite

    img = img[:, :, ::-1] # to BGR
        
    # upscale
    return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)