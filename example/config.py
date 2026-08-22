import jax
import jax.numpy as jnp
import optax
import optuna
from NCA_WM import NCA_WM

# Note that for showcases purposes, this model is underparameterized and
# undertrained to grant fair performance in little training time
# The model can learn just as fine with even less parameters (e.g: lower hidden_neurons) but more training steps
# Equally, the model could learn in less training time with more parameters and slightly worse performance

# training
STEPS = 500
BATCH_SIZE = 64
LOG_SEGMENTS = 100 # in such a simple environment, logging means CPU overhead
LOAD_MODEL = None
SAVE_MODEL = "example/model.eqx"
LOSS_GRAPH = "example/loss_graph.png"
TRUNCATED_BPTT = 1

# data
DATA_GLOB = "example/data/example_*.npz"
DATA_LIMIT = None # all files
LOADING_MODE = "VRAM"

# visualizer
LOAD_MODEL_INF = "example/model.eqx"
LOAD_DATA_INF = "example/data/example_0000.npz"
DATA_IDX_INF = 0
WIN_SIZE = None # auto-calculated
KEY_MAP = {
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3
}
DEFAULT_ACTION = None
FPS = None

# model
SUBSTEPS = 2

COLOR_MAP = [
    [240, 240, 240], # player
    [33, 33, 33] # background
]

_class_weights = jnp.array([63.0, 1.0]) # background is 63 pixels, player is 1 pixel

def make_model(key: jax.Array) -> NCA_WM:
    return NCA_WM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels="RGB",#len(COLOR_MAP),
        hid_channels=0, # for this simple environment, cells only read/write colors
        hidden_neurons=24,
        padding_mode='zeros',
        dtype=jnp.bfloat16,
        key=key,
    )

def make_optimizer() -> tuple[optax.GradientTransformation, optax.Schedule | None]:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.05, weight_decay=1e-4)
    ), None

def loss_calc(vis_preds: jnp.ndarray, hid_preds: None, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    # BCHW to BHWC
    vis_preds = jnp.moveaxis(vis_preds, 1, -1) # visible predictions
    vis_targets = jnp.moveaxis(targets, 1, -1) # visible targets
    
    celoss = optax.softmax_cross_entropy(vis_preds, vis_targets)

    class_idx = jnp.argmax(vis_targets, axis=-1) # argmax of C
    weights = _class_weights[class_idx]

    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)

    return celoss

# example way of using this function, to test what learning rate is the best to use
def hyperparam_sweep(key: jax.Array, trial: optuna.Trial) -> tuple[NCA_WM, optax.GradientTransformation]:
    lrinit = trial.suggest_float("lr_init_value", 9e-2, 5e-1)

    return make_model(key), optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lrinit, weight_decay=1e-4)
    )