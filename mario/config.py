import jax
import jax.numpy as jnp
import optax
import optuna
import numpy as np
import cv2
from NCA_WM import NCA_WM

# training
STEPS = 500
BATCH_SIZE = 128
LOG_SEGMENTS = 100 # in such a simple environment, logging means CPU overhead
LOAD_MODEL = "mario/model.eqx"
SAVE_MODEL = "mario/model.eqx"
LOSS_GRAPH = "mario/loss_graph.png"
TRUNCATED_BPTT = 1

# data
DATA_GLOB = "mario/data/example_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "VRAM"

# visualizer
LOAD_MODEL_INF = "mario/model.eqx"
LOAD_DATA_INF = "mario/data/example_*.npz"
DATA_IDX_INF = 0
WIN_SIZE = None # auto-calculated
KEY_MAP = {}
DEFAULT_ACTION = None
FPS = None

_kernel = [
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1],
]

# model
SUBSTEPS = 4

_colors_and_sprites = np.load("mario/env/color_map.npz")
_colors = _colors_and_sprites['colors'] # (36, 3)
_sprites = _colors_and_sprites['sprites'] # (36, 16, 16, 3)

COLOR_MAP = _colors.tolist()

_class_weights = jnp.ones(len(COLOR_MAP))
# set weights for specific classes: SKY and GROUND
_class_weights = _class_weights.at[([16, 33],)].set([0.8, 0.4])

def make_model(key):
    return NCA_WM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=0,
        hidden_neurons=64,
        padding_mode='zeros',
        dtype=jnp.bfloat16,
        kernel=_kernel,
        key=key,
    )

def make_optimizer() -> tuple[optax.GradientTransformation, optax.Schedule | None]:
    scheduler = optax.exponential_decay(init_value=0.01, transition_steps=100, decay_rate=0.95)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler, weight_decay=1e-4)
    ), scheduler

def loss_calc(vis_preds: jnp.ndarray, hid_preds: jnp.ndarray, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    # BCHW to BHWC
    vis_preds = jnp.moveaxis(vis_preds, 1, -1)
    targets = jnp.moveaxis(targets, 1, -1)
    
    celoss = optax.softmax_cross_entropy(vis_preds, targets)
    class_idx = jnp.argmax(targets, axis=-1)

    weights = _class_weights[class_idx]

    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)

    return celoss

# def hyperparam_sweep(key: jax.Array, trial: optuna.Trial) -> tuple[NCA_WM, optax.GradientTransformation]:
#     return NCA_WM(
#         # neural network
#         actions=len(KEY_MAP),
#         vis_channels=len(COLOR_MAP),
#         hid_channels=0,
#         hidden_neurons=128,
#         padding_mode='zeros',
#         dtype=jnp.bfloat16,
#         # embedding_dim=16,
#         kernel=krnl,
#         key=key,
#     ), make_optimizer()[0]

# custom inference logic:
def state_convert(model: NCA_WM, state: jax.Array, hid: jax.Array):
    # get visible channels from state and move to RAM
    visible = np.array(jnp.argmax(state, axis=0), dtype=np.uint8)

    img_blocks = _sprites[visible] # (36, 15, 16, 16, 16, 3)
    
    img_blocks = np.transpose(img_blocks, (0, 2, 1, 3, 4))
    
    # create empty HWC BGR image, upscale to 256x240
    img = img_blocks.reshape(15*16, 16*16, 3) # (uint8)

    img = img[:, :, ::-1] # to BGR
        
    # upscale
    return cv2.resize(img, WIN_SIZE, interpolation=cv2.INTER_NEAREST)