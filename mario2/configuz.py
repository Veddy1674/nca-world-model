import jax
import jax.numpy as jnp
import optax
import optuna
from NCA_WM import NCA_WM

# training
STEPS = 4000
BATCH_SIZE = 64
LOG_SEGMENTS = 100
LOAD_MODEL = None#"mario2/model3.eqx"
SAVE_MODEL = "mario2/model3.eqx"
LOSS_GRAPH = "mario2/loss_graph.png"
TRUNCATED_BPTT = 4

# data
DATA_GLOB = "mario2/data/goombas_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "VRAM"

# visualizer
LOAD_MODEL_INF = "mario2/model3.eqx"
LOAD_DATA_INF = "mario2/data/goombas_*.npz"
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

# _class_weights = jnp.array([1.0, 0.0])
_class_weights = jnp.array([63.0, 1.0])

def make_model(key):
    return NCA_WM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=1,
        hidden_neurons=64,
        padding_mode='zeros',
        dtype=jnp.bfloat16,
        key=key,
    )

def make_optimizer() -> tuple[optax.GradientTransformation, optax.Schedule | None]:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    ), None

def init_hidden(state: jnp.ndarray, info: jnp.ndarray, hid_channels: int, GRID_H: int, GRID_W: int) -> jnp.ndarray:
    if state.ndim == 3: # INFERENCE (unbatched: C, H, W)

        hid = jnp.zeros((hid_channels, GRID_H, GRID_W), dtype=jnp.bfloat16)
        mask = state[0:1, :, :] # where player is
        
        info_expanded = info[:, None, None].astype(jnp.bfloat16)
        return jnp.where(mask == 1, info_expanded, hid)

    else: # TRAINING (batched: B, C, H, W)

        B = state.shape[0]
        hid = jnp.zeros((B, hid_channels, GRID_H, GRID_W), dtype=jnp.bfloat16)
        mask = state[:, 0:1, :, :]
        
        info_expanded = info[:, :, None, None].astype(jnp.bfloat16)
        return jnp.where(mask == 1, info_expanded, hid)

# def add_noise(model: NCA_WM, vis: jnp.ndarray, hid: jnp.ndarray, key) -> tuple[jnp.ndarray, jnp.ndarray]:
#     vis += jax.random.normal(key, vis.shape, dtype=model.dtype) * 0.05
#     hid *= 0.9
#     return vis, hid

def loss_calc(vis_preds: jnp.ndarray, hid_preds: jnp.ndarray, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    
    hid_loss = jnp.sum(hid_preds**2)

    # 1. VISIBLE LOSS (With Class Weights)
    vis_preds_hwc = jnp.moveaxis(vis_preds, 1, -1)
    vis_targets_hwc = jnp.moveaxis(targets, 1, -1)

    # Calculate raw cross entropy per pixel
    raw_ce = optax.softmax_cross_entropy(vis_preds_hwc, vis_targets_hwc)
    
    # Apply weights: Targets is [B, C, H, W], so argmax gets the class index
    class_indices = jnp.argmax(targets, axis=1) # [B, H, W]
    weights = _class_weights[class_indices]     # [B, H, W]
    
    celoss = jnp.mean(raw_ce * weights)

    # 2. HIDDEN CHANNELS LOSS (The Offset)
    # FIX: == 1.0 to select the player, not the background!
    player_mask = (targets[:, 0:1] == 1.0) # shape: (B, 1, H, W)
    
    # Mask out background hidden channels
    hid_preds_masked = hid_preds * player_mask 
    
    # Sum hidden predictions where player is
    hid_sum = jnp.sum(hid_preds_masked, axis=(2, 3)) # shape: (B, hid_channels)
    
    # Count how many player pixels exist (should be 1, but this protects shapes)
    player_count = jnp.sum(player_mask, axis=(2, 3)) # shape: (B, 1)
    
    # Add epsilon (1e-6) to prevent division by zero!
    hid_preds_mean = hid_sum / (player_count + 1e-6) # shape: (B, hid_channels)
    
    # MSE Loss between predicted offset and true info offset
    mse_loss = jnp.mean((hid_preds_mean - infos) ** 2)

    background_mask = (targets[:, 0:1] == 0.0)
    hid_background = hid_preds * background_mask
    bg_loss = jnp.mean(hid_background ** 2)

    return celoss + mse_loss + hid_loss * 0.1# + mse_loss + bg_loss * 0.1

# MSE_LOSS_WEIGHT = 5.0

# def hyperparam_sweep(key: jax.Array, trial: optuna.Trial) -> tuple[NCA_WM, optax.GradientTransformation]:
#     global MSE_LOSS_WEIGHT

#     MSE_LOSS_WEIGHT = trial.suggest_float("MSE_LOSS_WEIGHT", 0.5, 10.0)

#     return make_model(key), make_optimizer()[0]