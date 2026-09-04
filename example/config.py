from typing import Optional
import jax
import jax.numpy as jnp
import optax
import optuna
from ncwm import NCWM

# Note that for showcases purposes, this model is underparameterized and undertrained
# to grant fair performance in little training time
# The model can learn just as fine with even less parameters (e.g: lower hidden_neurons) but more training steps
# Equally, the model could learn in less training time with more parameters and worse performance

# training-related
STEPS = 500
BATCH_SIZE = 64
LOG_SEGMENTS = 100 # how many times to log during training (100 means every 5 steps in this case)
LOAD_MODEL = None # path to a previously saved model to continue training
SAVE_MODEL = "example/model.eqx" # where to save the model
LOSS_GRAPH = "example/loss_graph.png" # where to save the loss graph
TRUNCATED_BPTT = 1 # not necessary for this example (as the environment is Markovian)

# data-related
DATA_GLOB = "example/data/*.npz" # glob pattern of data files used for training
DATA_LIMIT = None # None means all files will be included, look at ncwm/base_config.py for more info
LOADING_MODE = "VRAM" # "VRAM" or "RAM" or "DISK", if device is CPU, "VRAM" will be interpreted as "RAM"

# for inference
LOAD_MODEL_INF = "example/model.eqx" # model to load
LOAD_DATA_INF = "example/data/*.npz" # which data file to load as the first state, can be .npz, .npy or .png
DATA_IDX_INF = 0 # which index of the data file to load as the first state (if applicable)
WIN_SIZE = None # if None it will be calculated automatically based on the loaded data shape (height and width)
KEY_MAP = { # key to action mapping, actions are what the model expects as external input (converted to one-hot during model step)
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3
}
DEFAULT_ACTION = None # default action (useful when FPS is set)
FPS = None # if None, the simulation will wait for inputs after every step

# model-related
SUBSTEPS = 2 # how many forward passes per step
# higher values of SUBSTEPS allows the information to propagate further away between pixels
# more SUBSTEPS is quite similar (as in learning capacity) to adding more layers to the model

# COLOR_MAP to represent states as images (order matters)
COLOR_MAP = [
    [33, 33, 33], # background (dark gray)
    [240, 240, 240] # player (white)
]

# as the background is 63 pixels and player is 1 pixel, player should be weighted 63 times
# this avoids the loss to drop instantly even though the model hasn't learned how the player moves
_class_weights = jnp.array([1.0, 63.0]) # (order matters)

# creating the actual model:
def make_model(key: jax.Array) -> NCWM:
    return NCWM(
        # neural network
        actions=len(KEY_MAP),
        vis_channels=len(COLOR_MAP),
        hid_channels=0, # as the environment is Markovian, the model can learn without additional info or hidden states required
        hidden_neurons=24, # number of hidden neurons in the first layer of the CNN
        padding_mode='zeros', # what cells see at the edge of the grid (with zeros they have a clear information that they are at the edge)
        dtype=jnp.bfloat16, # data type (float32 by default)
        key=key, # random key (leave as is)
    )

# optimizer (for training):
def make_optimizer() -> tuple[optax.GradientTransformation, Optional[optax.Schedule]]:
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=0.09, weight_decay=1e-4)
    ), None

# loss function (for training):
def loss_calc(vis_preds: jnp.ndarray, hid_preds: None, targets: jnp.ndarray, actions: jnp.ndarray, infos: jnp.ndarray) -> jnp.ndarray:
    # BCHW to BHWC
    vis_preds = jnp.moveaxis(vis_preds, 1, -1) # visible predictions
    vis_targets = jnp.moveaxis(targets, 1, -1) # visible targets
    
    celoss = optax.softmax_cross_entropy(vis_preds, vis_targets)

    class_idx = jnp.argmax(vis_targets, axis=-1) # argmax of C
    weights = _class_weights[class_idx]

    # making the loss weighted
    celoss = jnp.sum(celoss * weights) / jnp.sum(weights)

    return celoss

# example way of using this function, to test what learning rate is the best to use
def hyperparam_sweep(key: jax.Array, trial: optuna.Trial) -> tuple[NCWM, optax.GradientTransformation]:
    lrinit = trial.suggest_float("lr_init_value", 0.001, 0.1)

    return make_model(key), optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lrinit, weight_decay=1e-4)
    )