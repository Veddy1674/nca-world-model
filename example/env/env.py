import numpy as np

IDX_BACKGROUND = 0
IDX_PLAYER = 1

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
