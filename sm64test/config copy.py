import jax
import jax.numpy as jnp
import optax
import optuna
from NACE import NACE

# Note that for showcases purposes, this model is underparameterized and
# undertrained to grant fair performance in little training time
# The model can learn just as fine with even less parameters (e.g: lower hidden_neurons) but more training steps
# Equally, the model could learn in less training time with more parameters and slightly worse performance

# training
STEPS = 5000
BATCH_SIZE = 4
LOG_SEGMENTS = 100 # in such a simple environment, logging means CPU overhead
LOAD_MODEL = None
SAVE_MODEL = "sm64test/model1.eqx"
LOSS_GRAPH = "sm64test/loss_graph.png"
TRUNCATED_BPTT = 1

# data
DATA_GLOB = "sm64test/data/*.npz"
DATA_LIMIT = 40 # all files
LOADING_MODE = "RAM"

# visualizer
LOAD_MODEL_INF = "sm64test/model1.eqx"
LOAD_DATA_INF = "sm64test/data/*.npz"
DATA_IDX_INF = 0
WIN_SIZE = None # auto-calculated
KEY_MAP = {
    'a': 0,
    'd': 1,
}
DEFAULT_ACTION = None
FPS = None

# model
SUBSTEPS = 4

COLOR_MAP = None

def make_model(key: jax.Array) -> NACE:
    return NACE(
        # neural network
        actions=len(KEY_MAP),
        vis_channels="RGB",#len(COLOR_MAP),
        hid_channels=0, # for this simple environment, cells only read/write colors
        hidden_neurons=128,
        padding_mode='zeros',
        dtype=jnp.bfloat16,
        key=key,
    )

def make_optimizer() -> tuple[optax.GradientTransformation, optax.Schedule | None]:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, weight_decay=1e-4)
    ), None

def loss_calc(vis_preds: jnp.ndarray, hid_preds: None, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    print(vis_preds.shape, targets.shape)
    return jnp.mean((vis_preds - targets) ** 2)

# example way of using this function, to test what learning rate is the best to use
def hyperparam_sweep(key: jax.Array, trial: optuna.Trial) -> tuple[NACE, optax.GradientTransformation]:
    lrinit = trial.suggest_float("lr_init_value", 9e-2, 5e-1)

    return make_model(key), optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lrinit, weight_decay=1e-4)
    )