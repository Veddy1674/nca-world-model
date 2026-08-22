import jax
import jax.numpy as jnp
import optax
from NCA_WM import NCA_WM

# training
STEPS = 8000
BATCH_SIZE = 512
LOG_SEGMENTS = 100
LOAD_MODEL = None#"snake/model.eqx"
SAVE_MODEL = "snake/model.eqx"
LOSS_GRAPH = "snake/loss_graph.png"

# data
DATA_GLOB = "snake/data/*.npz"

# visualizer
LOAD_MODEL_INF = "snake/model.eqx"
LOAD_DATA_INF = "snake/data/example_28.npz"
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
SUBSTEPS = 4
TRUNCATED_BPTT = 1

COLOR_MAP = [
    [17, 119, 17], # snake head
    [17, 163, 17], # snake body
    [201, 31, 19], # apple
    [17, 17, 17], # background
]

_class_weights = jnp.array([1.0, 1.0, 0.0, 0.02])

_kernel = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

def make_model(key):
    return NCA_WM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=0, # for this simple environment, cells only read/write colors
        hidden_neurons=128,
        padding_mode='zeros',
        kernel=_kernel,
        downscale_factor=1,
        key=key,
    )

def make_optimizer():
    scheduler = optax.exponential_decay(3e-3, STEPS, 0.3)
    # optim = optax.adamw(learning_rate=scheduler, weight_decay=1e-4)
    optim = optax.adam(learning_rate=scheduler)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optim
    ), scheduler

def _apple_count_loss(preds: jnp.ndarray) -> jnp.ndarray:
    # preds: B C H W (raw logits, channel-first, same preds as loss_calc gets)
    probs = jax.nn.softmax(preds[:, :len(COLOR_MAP)], axis=1)
    apple_probs = probs[:, 2]  # apple channel
    apple_count = jnp.sum(apple_probs, axis=(1, 2))  # per-sample expected count
    return jnp.mean((apple_count - 1.0) ** 2)

def loss_calc(preds: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    apple_loss = _apple_count_loss(preds)

    # BCHW to BHWC
    preds = jnp.moveaxis(preds[:, :len(COLOR_MAP)], 1, -1)
    targets = jnp.moveaxis(targets, 1, -1)
    
    celoss = optax.softmax_cross_entropy(preds, targets)

    class_idx = jnp.argmax(targets, axis=-1) # argmax of C
    weights = _class_weights[class_idx]

    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)

    return celoss + apple_loss * 0.4