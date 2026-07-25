import jax
import jax.numpy as jnp
import optax
import optuna
from NACE import NACE

# training
STEPS = 4000
BATCH_SIZE = 4
LOG_SEGMENTS = 100
LOAD_MODEL = None
SAVE_MODEL = "mario3/model.eqx"
LOSS_GRAPH = "mario3/loss_graph.png"
TRUNCATED_BPTT = 1
HIDDEN_NOISE_STD = 0#0.1

# data
DATA_GLOB = "mario3/data/e_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "RAM"

# visualizer
LOAD_MODEL_INF = "mario3/model.eqx"
LOAD_DATA_INF = "mario3/data/e_0.npz"
DATA_IDX_INF = 10
WIN_SIZE = None # auto-calculated
KEY_MAP = {
    'q': 0,
    'w': 1,
    'e': 2,
    'r': 3,
    't': 4,
    'y': 5,
    'u': 6,
    'i': 7,
    'o': 8
}
DEFAULT_ACTION = None
FPS = None

# model
SUBSTEPS = 2

COLOR_MAP = [
    [0, 0, 0],
    [0, 168, 0],
    [60, 188, 252],
    [92, 148, 252],
    [128, 208, 16],
    [200, 76, 12],
    [252, 152, 56],
    [252, 188, 176],
    [252, 252, 252],
]

def make_model(key):
    return NACE(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=0,
        hidden_neurons=256,
        padding_mode='zeros',
        embedding_dim=6,
        downscale_factor=4,
        dtype=jnp.bfloat16,
        key=key,
    )

def make_optimizer() -> tuple[optax.GradientTransformation, optax.Schedule | None]:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
    ), None

def loss_calc(vis_preds: jnp.ndarray, hid_preds: jnp.ndarray, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    # visible loss
    vis_preds = jnp.moveaxis(vis_preds, 1, -1)
    
    return optax.softmax_cross_entropy_with_integer_labels(vis_preds, targets).mean()

# MSE_LOSS_WEIGHT = 5.0

# def hyperparam_sweep(key: jax.Array, trial: optuna.Trial) -> tuple[NACE, optax.GradientTransformation]:
#     global MSE_LOSS_WEIGHT

#     MSE_LOSS_WEIGHT = trial.suggest_float("MSE_LOSS_WEIGHT", 0.5, 10.0)

#     return make_model(key), make_optimizer()[0]