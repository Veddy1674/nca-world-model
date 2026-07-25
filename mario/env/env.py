from PIL import Image
import numpy as np
import random

SMALL_MAP = "mario/env/map211x15.png"
COLOR_SPRITE_MAP = np.load("mario/env/color_map.npz")

COLOR_MAP = COLOR_SPRITE_MAP['colors']

class MarioEnv():
    def __init__(self):
        # 211x15 or so
        self.map_img = np.array(Image.open(SMALL_MAP).convert("RGB"), dtype=np.uint8)
        self.maxCamX = self.map_img.shape[1] - 16
        self.reset()
    
    def reset(self, camX: int = 0):
        # index map into 0-255
        H, W, _ = self.map_img.shape
        pixels = self.map_img.reshape(-1, 3)

        indices = np.where((pixels[:, None] == COLOR_MAP).all(axis=2))[1]
        self.full_map = indices.reshape(H, W)

        self.camX = camX
        self.state = self.full_map[:, self.camX:self.camX+16]

        return self.state, self._norm_camX()
    
    def _norm_camX(self):
        return self.camX / self.maxCamX
    
    def step(self, action: int):
        # action 0 = go right, action 1 = go left
        if action == 0:
            if self.camX < self.maxCamX:
                self.camX += 1
        elif action == 1:
            if self.camX > 0:
                self.camX -= 1
        
        # Update state with new camX
        self.state = self.full_map[:, self.camX:self.camX+16]
        return self.state, self._norm_camX()

# env = MarioEnv()
# state, camX = env.reset()

# for i in range(200):
#     state, camX = env.step(0)

# print("CamX:",camX)
# Image.fromarray(COLOR_MAP[state]).show()
# exit()

def rand_action(episode_idx: int) -> int:
    return 0
    # if episode_idx < 3000:
    #     # 70% right, 30% left
    #     return 0
    #     # return 1 if random.random() < 0.7 else 0
    # elif episode_idx < 300:
    #     # 30% right, 70% left
    #     return 0 if random.random() < 0.3 else 1
    # else:
    #     # 50% left, 50% right
    #     return 0 if random.random() < 0.5 else 1

def get_initial_camX(episode_idx: int, maxCamX: int) -> int:
    return 0
    # if episode_idx < 3000: # first
    #     return 0
    # # if episode_idx < 300:
    # #     # More on the left (0 to 1/3 of maxCamX)
    # #     return random.randint(0, max(1, maxCamX // 4))
    # elif episode_idx < 300:
    #     # More on the right (2/3 to maxCamX)
    #     return random.randint(max(1, 2 * maxCamX // 4), maxCamX)
    # else:
    #     # More centered (1/3 to 2/3 of maxCamX)
    #     return random.randint(maxCamX // 3, 2 * maxCamX // 3)

def make_dataset(env, STEPS, episode_idx: int):
    states = np.empty((STEPS, 1, 15, 16), dtype=np.uint8)
    actions = np.empty(STEPS-1, dtype=np.uint8)
    infos = np.empty(STEPS, dtype=np.float32)
    
    initial_camX = get_initial_camX(episode_idx, env.maxCamX)
    state, camX = env.reset(camX=initial_camX)
    
    for i in range(STEPS-1):
        action = rand_action(episode_idx)
        
        states[i] = state[None, :, :]
        actions[i] = action
        infos[i] = camX
        
        state, camX = env.step(action)
    
    states[STEPS-1] = state
    infos[STEPS-1] = camX
    return states, actions, infos

if __name__ == '__main__':
    from tqdm import tqdm
    
    env = MarioEnv()
    
    for i in tqdm(range(1), desc="Making datasets"):
        states, actions, infos = make_dataset(env, 200, i)
        
        if i == 0:
            print("States shape:", states.shape)
            print("Actions shape:", actions.shape)
            print("Infos shape:", infos.shape)
        
        np.savez_compressed(f"mario/data/example_{i}.npz", states=states, actions=actions, infos=infos)