import numpy as np

IDX_PLAYER = 0
IDX_BACKGROUND = 1

class ExampleEnv():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.state = np.full((8, 8), IDX_BACKGROUND, dtype=np.uint8) # HW index map (fill with background)

        # spawn player randomly
        y = np.random.randint(1, 7)
        x = np.random.randint(1, 7)
        self.playerPos = (y, x)

        self.state[self.playerPos] = IDX_PLAYER # set player

        return self.state
    
    def _move_player(self, pos: tuple[int, int]):
        # just in case
        if pos[0] < 0 or pos[0] > 7 or pos[1] < 0 or pos[1] > 7:
            return
        
        self.state[self.playerPos] = IDX_BACKGROUND # set old to background
        self.playerPos = pos
        self.state[self.playerPos] = IDX_PLAYER # set new to player
    
    def step(self, action: int):
        y, x = self.playerPos

        if action == 0:
            # move up
            self._move_player((y - 1, x))
        elif action == 1:
            # move down
            self._move_player((y + 1, x))
        elif action == 2:
            # move left
            self._move_player((y, x - 1))
        elif action == 3:
            # move right
            self._move_player((y, x + 1))

        return self.state

if __name__ == '__main__':
    from tqdm import tqdm

    env = ExampleEnv()

    EPISODES = 10
    STEPS = 400
    OUT = "example/data/example_{:04d}.npz"

    def make_episode(steps: int):
        states = np.empty((steps, 8, 8), dtype=np.uint8) # BHW
        actions = np.empty((steps-1), dtype=np.uint8) # B

        state = env.reset()

        for i in range(steps-1):
            action = np.random.randint(0, 4)

            states[i] = state
            actions[i] = action

            state = env.step(action)
        
        states[-1] = state

        return states, actions
    
    for i in tqdm(range(EPISODES), "Making episodes"):
        states, actions = make_episode(STEPS)

        # save
        np.savez_compressed(OUT.format(i), states=states, actions=actions)

        # if first iteration, print shape
        if i == 0:
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
