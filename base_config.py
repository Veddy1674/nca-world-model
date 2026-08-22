import jax
import equinox as eqx
import optax
import optuna
from pydantic import BaseModel, Field
from typing import Callable, Literal, Any
from cv2.typing import MatLike
from NCA_WM import NCA_WM
import numpy as np
import importlib.util
import os
import sys

# all the constants and functions required in a configuration
class NCA_WM_Config(BaseModel):
    # training parameters
    STEPS: int = Field(gt=0)
    BATCH_SIZE: int = Field(gt=0)
    LOG_SEGMENTS: int = Field(ge=-1) # how many logging prints in the entire training 0 or -1 to disable logging
    LOAD_MODEL: str | None # path to load the model and optimizer state at the beginning (None to train from scratch)
    SAVE_MODEL: str | None # path to save the model and optimizer state (None to skip saving)
    LOSS_GRAPH: str | None # path to save a matplotlib loss graph at the end
    TRUNCATED_BPTT: int = Field(gt=0) # how long hidden channels persist for before reset (zeroed or noise applied), makes batched training sequential

    # data loading parameters (training only)
    DATA_GLOB: str # glob path to load .npz data files
    DATA_LIMIT: int | tuple[int | None, int | None] | None # limit data to include, where None is "all", e.g: (None, 200) includes the first 200 files (both ends inclusive)
    LOADING_MODE: Literal["RAM", "VRAM", "DISK"] # VRAM = all data to GPU right away, RAM = all data in RAM, given to GPU each train chunk/step, DISK = load data to RAM and GPU progressively

    # inference parameters
    LOAD_MODEL_INF: str # path to load the model for inference
    LOAD_DATA_INF: str # path to load the first state for inference (.npz data file or .png to start from, if glob matches multiple files, the first one is used)
    DATA_IDX_INF: int # index of the data file (LOAD_DATA_INF) to load for inference
    WIN_SIZE: tuple[int, int] | None # width, height - window size for inference, if None it is auto-calculated from the data shape
    KEY_MAP: dict # key mapping for inference (length should be equal to model actions) TODO dont allow R or specials
    DEFAULT_ACTION: int | None # action to take when no key is pressed, if None wait for input (must be defined if FPS is valid)
    FPS: int | None # if 0 or None wait for input, otherwise wait for 1000 // FPS milliseconds and step with DEFAULT_ACTION

    # model/shared parameters
    SUBSTEPS: int = Field(gt=0) # how many forwards for each step, preferably even
    COLOR_MAP: list[tuple[int, int, int]] | None # list of RGB colors the data states have (length should be equal to model visual channels), or None if RGB
    
    make_model: Callable
    make_optimizer: Callable[[], tuple[optax.GradientTransformation, optax.Schedule]]
    init_hidden: Callable[[jax.Array, jax.Array | None, int, int, int], jax.Array] | None = None # initialize hidden channels each train sequence to certain values (by default zeroed)
    loss_calc: Callable[[jax.Array, jax.Array | None, jax.Array, jax.Array | None, jax.Array | None], jax.Array]
    add_noise: Callable[[NCA_WM, jax.Array, jax.Array, Any], jax.Array] | None = None

    hyperparam_sweep: Callable[[jax.Array, optuna.Trial], tuple[NCA_WM, optax.GradientTransformation]] | None = None

    state_convert: Callable[[NCA_WM, jax.Array, jax.Array | None], MatLike] | None = None # optional function to convert model prediction to cv2 render format with a custom logic

    model_config = {"arbitrary_types_allowed": True}

# called from scripts that require a configuration file
def load_configuration() -> NCA_WM_Config:
    # arg0 is configPath, if null ask via input()
    if len(sys.argv) < 2:
        try:
            config_path = input("Config Path: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(1)
    else:
        config_path = sys.argv[1]
    
    # config exists check
    if not os.path.exists(config_path):
        print(f"File not found: {config_path}")
        sys.exit(1)
    
    # load module and execute
    spec = importlib.util.spec_from_file_location(os.path.basename(config_path), config_path)
    if spec is None or spec.loader is None:
        print(f"Invalid module: {config_path}")
        sys.exit(1)
    
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    
    # ignore __name__, __file__ etc.
    config_vars = {
        k: v for k, v in vars(cfg_module).items() if not k.startswith('__')
    }
    
    try:
        # just to validate, unused
        valid_config = NCA_WM_Config(**config_vars)
    except Exception as e:
        print(f"Validation of config {config_path} failed: {str(e)}")
        sys.exit(1)

    # define non-defined attributes or functions (e.g: add_noise, if it wasn't defined it becomes add_noise = None)
    for key, value in valid_config.model_dump(exclude_none=False).items():
        if not hasattr(cfg_module, key):
            setattr(cfg_module, key, value)
    
    for key in valid_config.__dict__:
        if not hasattr(cfg_module, key):
            setattr(cfg_module, key, getattr(valid_config, key))
    
    # return module instead of the config itself, so that the script's variables can be modified from other scripts...
    # it is treated as if its type is NCA_WM_Config
    return cfg_module # type: ignore

# print info about inputs, parameters and such
def print_model_info(model: NCA_WM, details: bool = False):
    inputs_details = ""
    if details:
        # what input_dim is made out of
        vis = model.vis_channels
        hid = model.hid_channels
        acts = model.actions
        krnl = model.kernel_size
        inputs_details = f" (krnl: {krnl} x (vis: {vis} + hid: {hid}) + acts: {acts})"

    print(f"Model inputs: {model.input_dim}{inputs_details}") # TODO projection channels
    
    params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, model.trainable_mask)))
    print(f"Model parameters: {params:,}")