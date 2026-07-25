import jax
import jax.numpy as jnp
import numpy as np
import cv2

from NACE import NACE, load_model
from inference import compile_model_inference
from dataload import load_first

from base_config import * # intellisense for configs

# convert CHW jax array to cv2 matlike to render
# hidden channels could be used in custom implementations
def state_convert(state: jax.Array, hid: jax.Array | None):
    if cfg.state_convert is not None:
        return cfg.state_convert(model, state, hid)
    
    if model.is_continuous: # RGB
        # convert state 3HW float32 in HWC uint8
        img = np.array(jnp.moveaxis(state, 0, -1) * 255.0, dtype=np.uint8)

        img = img[:, :, ::-1] # to BGR
        
        return cv2.resize(img, cfg.WIN_SIZE, interpolation=cv2.INTER_NEAREST)

    # get visible channels from state and move to RAM
    if state.ndim == 3: # if CHW (one-hot)
        visible = np.array(jnp.argmax(state[:model.vis_channels], axis=0), dtype=np.uint8)
    else: # if HW (index map)
        visible = np.array(state, dtype=np.uint8)

    # create empty HWC BGR image
    img = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

    assert cfg.COLOR_MAP is not None, "This assert should never fail - COLOR_MAP was set to None due to is_continuous being True"
    
    # paint each class with its color from COLOR_MAP
    for class_idx, color in enumerate(cfg.COLOR_MAP):
        img[visible == class_idx] = color[::-1] # to BGR
        
    # upscale
    return cv2.resize(img, cfg.WIN_SIZE, interpolation=cv2.INTER_NEAREST)

def reset_hidden():
    if cfg.init_hidden is not None:
        # pass the first state and info (if any) to initialize hidden channels
        return cfg.init_hidden(FIRST_STATE, FIRST_INFO, model.hid_channels, GRID_H, GRID_W)
    elif model.hid_channels > 0:
        return jnp.zeros((model.hid_channels, GRID_H, GRID_W), dtype=model.dtype)
    else:
        return None

def main(model: NACE):
    # to silence pyright
    assert cfg.FPS is not None
    
    FPS = 1000 // cfg.FPS

    curr_state = FIRST_STATE
    curr_hid = reset_hidden()
    curr_image = state_convert(curr_state, curr_hid)

    while True:
        # render
        cv2.imshow(WIN_NAME, curr_image)

        keyPress = cv2.waitKey(FPS) & 0xFF

        action_onehot: np.ndarray | jax.Array | None = None # defined here to allow custom one-hot

        try: # getWindowProperty can raise cv2.error
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1 or keyPress == 27: # if closing window or ESC (exit)
                break
        except cv2.error:
            break

        if keyPress == ord('q'): # Q (print info about hidden channels)
            if curr_hid is None:
                print("\nNo hidden channels")
                continue
            
            # curr_hid has shape (num_hid, H, W)
            
            print("\nHidden channels:")
            print(f"  Mean: {curr_hid.mean()}")
            print(f"  Std: {curr_hid.std()}")
            print(f"  Max: {curr_hid.max()}")
            print(f"  Min: {curr_hid.min()}")
            continue
        
        elif keyPress == ord('y'): # Y (visualize numbers on each pixel/cell representing their hidden channels)
            if curr_hid is None:
                print("\nNo hidden channels")
                continue
            
            # curr_hid has shape (num_hid, H, W)

            # all mean values of every cell's mean hidden channels
            mean_hid = np.array(jnp.mean(curr_hid, axis=0)) # (GRID_H, GRID_W)

            # size of each cell when upscaled
            cell_h = cfg.WIN_SIZE[1] // GRID_H # type: ignore
            cell_w = cfg.WIN_SIZE[0] // GRID_W # type: ignore

            # constants to draw text (and calc text size)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 1

            for y in range(GRID_H):
                for x in range(GRID_W):
                    mean = float(mean_hid[y, x]) # mean of hidden channels for this cell
                    text = f"{mean:.1f}"
                    
                    # formula to center text horizontally and vertically
                    (textW, textH), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    textX = (x * cell_w + cell_w // 2) - textW // 2
                    textY = (y * cell_h + cell_h // 2) + textH // 2

                    # text is appended to curr_image, so it will disappear when the next frame is displayed
                    
                    # black
                    cv2.putText(
                        img=curr_image,
                        text=text,
                        org=(textX, textY), 
                        fontFace=font,
                        fontScale=font_scale,
                        color=(0, 0, 0),
                        thickness=thickness * 3
                    )

                    # white
                    cv2.putText(
                        img=curr_image,
                        text=text,
                        org=(textX, textY), 
                        fontFace=font,
                        fontScale=font_scale,
                        color=(255, 255, 255),
                        thickness=thickness
                    )
            
        elif keyPress == ord('t'): # T (allow custom one-hot action)

            if model.actions == 0: # actions can be 0 or >= 2, not 1
                print("\nModel takes no actions as input")
                continue

            print(f"\nInsert an one-hot action of {model.actions} values (e.g: 1 0 0 0.5)")
            
            try:
                inp = input(">> ") # block rendering
                values = [float(x) for x in inp.split()]
                
                if len(values) == model.actions:
                    action_onehot = np.array(values, dtype=np.float32)
                    print(f"Action: {action_onehot}")
                else:
                    print(f"Error: You must insert exactly {model.actions} values.")
                
            except ValueError:
                print("Error: Invalid format. Only numbers and spaces allowed.")

        elif keyPress == ord('r'): # R (reset)
            curr_state = FIRST_STATE
            curr_hid = reset_hidden()
            curr_image = state_convert(curr_state, curr_hid)
            continue

        # if no actions, any key triggers the prediction
        if action_onehot is None and model.actions > 0: # default index to onehot logic
        
            # check if pressed key is mapped to an action or set to invalid
            action_idx: int = CV2_KEY_MAP.get(keyPress, -1)

            if action_idx == -1:
                # action was invalid and FPS is blocking, ignore (found a key not in key map)
                if FPS == 0:
                    continue
                else:
                    # FPS is defined, no key was pressed, so use default
                    action_idx = cfg.DEFAULT_ACTION # type: ignore
        
            # to one-hot
            if model.actions > 0:
                action_onehot = np.zeros(model.actions, dtype=np.float32)
                action_onehot[action_idx] = 1.0
        
        # to jax array
        if action_onehot is not None:
            action_onehot = jnp.array(action_onehot, dtype=model.dtype)

        # forward state and return next state (autoregressive)
        global key
        curr_state, curr_hid, key = model_inference(curr_state, curr_hid, action_onehot, 0.0, key)

        # update image
        curr_image = state_convert(curr_state, curr_hid)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # allow config loading as first argument
    cfg = load_configuration()

    # init model and optimizer
    model: NACE = cfg.make_model(jax.random.key(0))

    # print using device name
    print(f"Using {jax.devices()[0].device_kind.upper()}")

    # load model, ignore optimizer and opt_state
    model = load_model(cfg.LOAD_MODEL_INF, model)

    print_model_info(model)
    
    # load first file
    FIRST_STATE, _, FIRST_INFO = load_first(
        data_path=cfg.LOAD_DATA_INF,
        vis_channels=model.vis_channels,
        color_map=cfg.COLOR_MAP,
        use_embedding=model.embedding_dim is not None,
        is_continuous=model.is_continuous,
        index=cfg.DATA_IDX_INF
    ) # CHW

    if cfg.COLOR_MAP is not None and model.is_continuous:
        cfg.COLOR_MAP = None # setting COLOR_MAP to None just to "make it clear" that it isn't being used
        # (other than to load data, in fact, it MUST be set to None AFTER loading data)
        print("Note: COLOR_MAP is defined but vis_channels are continuous (RGB), it is being ignored")

    # setup renderer
    WIN_NAME = "NACE - Inference with opencv-python"
    cv2.namedWindow(WIN_NAME)

    CV2_KEY_MAP = {ord(k): v for k, v in cfg.KEY_MAP.items()} # ord(key string)s

    # get grid dims from first loaded file, first state
    GRID_H, GRID_W = FIRST_STATE.shape[-2:] # BCHW (one-hot) or BHW (index-map)

    key = jax.random.key(1)
    np.random.seed(1)

    model_inference, key = compile_model_inference(model, GRID_H, GRID_W, FIRST_STATE, cfg.SUBSTEPS, key)

    # adjust win size if none
    if cfg.WIN_SIZE is None:
        # base 832x832 - because 832 is easily divided by 2 and powers of 2
        scale = 832 / max(GRID_H, GRID_W)

        WIN_H = int(GRID_H * scale)
        WIN_W = int(GRID_W * scale)

        # e.g: 64x32 = 832x416 (WH)
        cfg.WIN_SIZE = (WIN_W, WIN_H)
    
    # adjust fps if none or 0
    if cfg.FPS is None or cfg.FPS <= 0:
        cfg.FPS = 1001 # 1000 // 1001 = 0, which makes cv2.waitKey blocking
    
    print(f"Window initialized - {cfg.WIN_SIZE[0]}x{cfg.WIN_SIZE[1]}")
    
    try:
        main(model)
    except KeyboardInterrupt:
        pass