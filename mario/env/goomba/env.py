import numpy as np

COLOR_SPRITE_MAP = np.array([
    [33, 33, 33], # background/non-solid
    [17, 17, 17], # solid
    [142, 107, 55], # goomba
], np.uint8)

class GoombaEnv():
    def __init__(self):

        self.reset()
    
    def reset(self):
        # index map 16x15 of background
        self.state = np.full((15, 16), 0, dtype=np.uint8)

        # solid areas
        self.state[-2:, 0:] = 1
        self.state[-4:, -2:] = 1
        self.state[1:3, -3:] = 1
        self.state[-3, 1] = 1
        self.state[7, 3] = 1
        self.state[7, 7:12] = 1
        self.state[4, 9] = 1

        self.goombaPos = (12, 12)
        self.goombaOffsetX = 0.0 # increases by 1/32
        self.goombaDir = 1 # 1 or -1
        self.state[self.goombaPos] = 2  # goomba
        
        return self.state, self.goombaOffsetX, self.goombaDir
    
    def set_goomba_pos(self, pos):
        self.state[self.goombaPos] = 0 # background
        self.goombaPos = pos
        self.state[self.goombaPos] = 2 # goomba
    
    def step(self, action: int):
        self.goombaOffsetX += 1/32

        if self.goombaOffsetX >= 1.0:
            self.goombaOffsetX = 0.0
            self.set_goomba_pos((self.goombaPos[0], self.goombaPos[1] + self.goombaDir))

            # if at the new position a wall is found at the direction it is currently moving, invert direction
            if self.state[self.goombaPos[0], self.goombaPos[1] + self.goombaDir] == 1:
                self.goombaDir = -self.goombaDir # 1 or -1
                self.goombaOffsetX += 1/32 # another step to instantly move after turn
        
        return self.state, self.goombaOffsetX, self.goombaDir

from PIL import Image

env = GoombaEnv()

# state, goombaOffsetX, goombaDir = env.reset()

# for i in range(8900):
#     _, goombaOffsetX, goombaDir = env.step(0)

#     # print("Offset:", goombaOffsetX, "Direction:", goombaDir)


# # for _ in range(128+34):
# #     state, goombaOffsetX, goombaDir = env.step(0)

# # print("Offset:", goombaOffsetX, "Direction:", goombaDir)

# Image.fromarray(COLOR_SPRITE_MAP[state]).show()
# exit()

# OTHER ONE
# c = np.load("mario/env/color_map.npz")
# sprites = c["sprites"]

# id_map = {0: 16, 1: 33, 2: 34}
# canvas = np.zeros((240, 256, 3), dtype=np.uint8)

# # 1. Creiamo la mappa di sfondo (mappiamo gli 1 sul terreno e tutto il resto sul cielo)
# bg_mask = np.where(state == 1, id_map[1], id_map[0])

# # 2. Ricostruiamo lo sfondo 256x240 prendendo i blocchi degli sprite in un colpo solo
# # Cambiamo la forma per allineare i blocchi, poi trasponiamo e rimodelliamo a 240x256x3
# canvas = sprites[bg_mask].transpose(0, 2, 1, 3, 4).reshape(240, 256, 3)

# # 3. Troviamo la posizione del Goomba (2) nella griglia usando NumPy
# goomba_indices = np.argwhere(state == 2)

# if goomba_indices.size > 0:
#     r, c_idx = goomba_indices[0]
    
#     y = r * 16
#     x = c_idx * 16 + (int(goombaOffsetX * 16) * goombaDir)
    
#     if 0 <= x <= 240:
#         canvas[y:y+16, x:x+16] = sprites[id_map[2]]

# final_image = Image.fromarray(canvas)
# final_image.show()

def rand_action():
    return 0
    # return np.random.randint(0, 2)

def make_dataset(env, STEPS):
    states = np.empty((STEPS, 1, 15, 16), dtype=np.uint8)
    actions = np.empty(STEPS-1, dtype=np.uint8)
    infos = np.empty((STEPS, 2), dtype=np.float32)
    
    state, offset, direction = env.reset()
    
    for i in range(STEPS-1):
        action = rand_action()
        
        states[i] = state[None, :, :]
        actions[i] = action
        infos[i] = [offset, direction]
        
        state, offset, direction = env.step(action)
    
    states[STEPS-1] = state
    infos[STEPS-1] = [offset, direction]
    return states, actions, infos

if __name__ == '__main__':
    from tqdm import tqdm
    
    env = GoombaEnv()
    
    for i in tqdm(range(1), desc="Making datasets"):
        states, actions, infos = make_dataset(env, 80000)
        
        if i == 0:
            print("States shape:", states.shape)
            print("Actions shape:", actions.shape)
            print("Infos shape:", infos.shape)
        
        np.savez_compressed(f"mario/data/goomba_{i}.npz", states=states, actions=actions, infos=infos)