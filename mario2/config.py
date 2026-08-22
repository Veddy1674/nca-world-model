import jax
import jax.numpy as jnp
import optax
from NCA_WM import NCA_WM

# training
STEPS = 4000
BATCH_SIZE = 24
LOG_SEGMENTS = 100
LOAD_MODEL = None
SAVE_MODEL = "mario2/model.eqx"
LOSS_GRAPH = "mario2/loss_graph.png"
TRUNCATED_BPTT = 4
HIDDEN_NOISE_STD = 0.0008

# data
DATA_GLOB = "mario2/data/example_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "VRAM"

# visualizer
LOAD_MODEL_INF = "mario2/model.eqx"
LOAD_DATA_INF = "mario2/data/example_*.npz"
DATA_IDX_INF = 0
WIN_SIZE = None # auto-calculated
KEY_MAP = {}
DEFAULT_ACTION = None
FPS = None

# model
SUBSTEPS = 2

COLOR_MAP = [
    [240, 240, 240], # player
    [33, 33, 33], # background
]

_class_weights = jnp.array([63.0, 1.0])

def make_model(key):
    return NCA_WM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=2,
        hidden_neurons=128,
        padding_mode='zeros',
        key=key,
    )

def make_optimizer():
    # return optax.adamw(learning_rate=1e-2, weight_decay=1e-4), None
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-3)
    ), None

def init_hidden(state: jnp.ndarray, info: jnp.ndarray, hid_channels: int, GRID_H: int, GRID_W: int) -> jnp.ndarray:
    if state.ndim == 3: # INFERENCE (unbatched: C, H, W)

        hid = jnp.zeros((hid_channels, GRID_H, GRID_W), dtype=jnp.float32)
        goomba_mask = state[0:1, :, :]
        
        info_expanded = info[:, None, None]
        return jnp.where(goomba_mask == 1, info_expanded, hid)

    else: # TRAINING (batched: B, C, H, W)

        B = state.shape[0]
        hid = jnp.zeros((B, hid_channels, GRID_H, GRID_W), dtype=jnp.float32)
        goomba_mask = state[:, 0:1, :, :]
        
        info_expanded = info[:, :, None, None]
        return jnp.where(goomba_mask == 1, info_expanded, hid)

def loss_calc(vis_preds: jnp.ndarray, hid_preds: jnp.ndarray, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    # Visible loss (same as yours)
    vis_preds = jnp.moveaxis(vis_preds, 1, -1)
    vis_targets = jnp.moveaxis(targets, 1, -1)
    celoss = optax.softmax_cross_entropy(vis_preds, vis_targets)
    class_idx = jnp.argmax(vis_targets, axis=-1)
    weights = _class_weights[class_idx]
    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)
    
    # We know the true next hidden state should be the `infos` at the new Goomba position
    target_goomba_mask = targets[:, 0:1, :, :] # 1 where Goomba is in the target
    info_expanded = infos[:, :, None, None]
    
    # Where Goomba is, it should be the subpixel offset. Everywhere else, 0.
    true_hid = jnp.where(target_goomba_mask == 1, info_expanded, 0.0)
    
    # MSE loss on hidden channels
    hid_loss = jnp.mean((hid_preds - true_hid)**2)

    return celoss + hid_loss