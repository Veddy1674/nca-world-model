import numpy as np

COLOR_SPRITE_MAP = np.array([
    [10, 10, 10], # background/non-solid
    [142, 107, 55], # goomba
], np.uint8)

MAGIC = 32

class GoombaEnv():
    def __init__(self):

        self.reset()
    
    def reset(self):
        # index map 16x15 of background
        self.state = np.full((15, 16), 0, dtype=np.uint8)

        self.goombaPos = (12, 2)
        self.goombaOffsetX = 0 # increases by 1
        self.state[self.goombaPos] = 1  # goomba
        
        return self.state, self.goombaOffsetX
    
    def set_goomba_pos(self, pos):
        self.state[self.goombaPos] = 0 # background
        self.goombaPos = pos
        self.state[self.goombaPos] = 1 # goomba
    
    def step(self, action: int):
        self.goombaOffsetX += 1

        if self.goombaOffsetX >= MAGIC:
            self.goombaOffsetX = 0
            self.set_goomba_pos((self.goombaPos[0], self.goombaPos[1] + 1))

        return self.state, self.goombaOffsetX
    
    @staticmethod
    def get_circular_offset(goombaOffsetX):
        angle = (goombaOffsetX / MAGIC) * 2 * np.pi
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

# from PIL import Image

# env = GoombaEnv()

# state, goombaOffsetX = env.reset()

# for i in range(48):
#     state, goombaOffsetX = env.step(0)

#     print("Offset:", goombaOffsetX, "Circular:", env.get_circular_offset(goombaOffsetX).tolist())

# Image.fromarray(COLOR_SPRITE_MAP[state]).show()
# exit()

def rand_action():
    return 0
    # return np.random.randint(0, 2)

def make_dataset(env: GoombaEnv, STEPS):
    states = np.empty((STEPS, 1, 15, 16), dtype=np.uint8)
    actions = np.empty(STEPS-1, dtype=np.uint8)
    infos = np.empty((STEPS, 2), dtype=np.float32)
    
    state, offset = env.reset()
    offset = env.get_circular_offset(offset)
    
    for i in range(STEPS-1):
        action = rand_action()
        
        states[i] = state[None, :, :]
        actions[i] = action
        infos[i] = offset
        
        state, offset = env.step(action)
        offset = env.get_circular_offset(offset)
    
    states[STEPS-1] = state
    infos[STEPS-1] = offset
    return states, actions, infos

if __name__ == '__main__':
    from tqdm import tqdm
    
    env = GoombaEnv()
    
    for i in tqdm(range(1), desc="Making datasets"):
        states, actions, infos = make_dataset(env, 12*MAGIC)
        
        if i == 0:
            print("States shape:", states.shape)
            print("Actions shape:", actions.shape)
            print("Infos shape:", infos.shape)
        
        np.savez_compressed(f"mario/data/goombas_{i}.npz", states=states, actions=actions, infos=infos)