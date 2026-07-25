import numpy as np

IDX_PLAYER = 0
IDX_BACKGROUND = 1

COUNTER = 2

class ExampleEnv():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.state = np.full((8, 8), IDX_BACKGROUND, dtype=np.uint8) # HW index map (fill with background)

        # spawn player randomly
        y = np.random.randint(1, 7)
        x = np.random.randint(1, 7)
        self.playerPos = (y, x)
        self.playerOffset = (0, 0)

        self.state[self.playerPos] = IDX_PLAYER # set player

        return self.state, self._ai_offset()
    
    def _ai_offset(self):
        y = self.playerOffset[0]# / COUNTER
        x = self.playerOffset[1]# / COUNTER

        return np.array([y, x], dtype=np.float32)
    
    def step(self, action: int):
        y, x = self.playerPos
        off_y, off_x = self.playerOffset

        # apply sub-pixel action
        if action == 0: off_y -= 1
        elif action == 1: off_y += 1
        elif action == 2: off_x -= 1
        elif action == 3: off_x += 1

        # resolve Y axis crossing
        if off_y <= -COUNTER:
            if y > 0:
                y -= 1
                off_y = 0
            else:
                off_y = -COUNTER + 1
        elif off_y >= COUNTER:
            if y < 7:
                y += 1
                off_y = 0
            else:
                off_y = COUNTER - 1

        # resolve X axis crossing
        if off_x <= -COUNTER:
            if x > 0:
                x -= 1
                off_x = 0
            else:
                off_x = -COUNTER + 1
        elif off_x >= COUNTER:
            if x < 7:
                x += 1
                off_x = 0
            else:
                off_x = COUNTER - 1

        # update board
        self.state[self.playerPos] = IDX_BACKGROUND # set old to background
        self.playerPos = (y, x)
        self.state[self.playerPos] = IDX_PLAYER # set new to player
        self.playerOffset = (off_y, off_x)

        return self.state, self._ai_offset()

if __name__ == '__main__':
    from tqdm import tqdm

    env = ExampleEnv()

    EPISODES = 10
    STEPS = 4000
    OUT = "mario2/data/example_{:04d}.npz"

    def make_episode(steps: int):
        states = np.empty((steps, 8, 8), dtype=np.uint8) # BHW
        actions = np.empty((steps-1), dtype=np.uint8) # B
        infos = np.empty((steps, 2), dtype=np.float32) # B2

        state, info = env.reset()

        for i in range(steps-1):
            action = np.random.randint(0, 4)

            states[i] = state
            actions[i] = action
            infos[i] = info

            state, info = env.step(action)
        
        states[-1] = state
        infos[-1] = info

        return states, actions, infos
    
    for i in tqdm(range(EPISODES), "Making episodes"):
        states, actions, infos = make_episode(STEPS)

        states = np.eye(2, dtype=np.float32)[states]
        states = np.moveaxis(states, -1, 1)

        # save
        np.savez_compressed(OUT.format(i), states=states, actions=actions, infos=infos)

        # if first iteration, print shape
        if i == 0:
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
            print(f"Infos shape: {infos.shape}")