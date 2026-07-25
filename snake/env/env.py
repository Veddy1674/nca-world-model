import numpy as np
import random

COLOR_MAP = np.array([
    [17, 119, 17],   # snake head
    [17, 163, 17],   # snake body
    [201, 31, 19],   # apple
    [17, 17, 17],    # background
], dtype=np.uint8)

HEAD, BODY, APPLE, BG = 0, 1, 2, 3


class SnakeEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.full((self.size, self.size), BG, dtype=np.uint8)
        head = (random.randrange(self.size), random.randrange(self.size))
        self.snake = [head]
        self.state[head] = HEAD
        self._spawn_apple()
        return self.state

    def _spawn_apple(self):
        empty = [(r, c) for r in range(self.size) for c in range(self.size)
                  if self.state[r, c] == BG]
        if not empty:
            raise Exception("Apple couldn't spawn anywhere in the map")
        
        self.apple = random.choice(empty)
        self.state[self.apple] = APPLE

    def step(self, action: int):
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        if not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size):
            return self.state
        if new_head in self.snake:
            return self.state

        ate_apple = new_head == self.apple
        self.snake.insert(0, new_head)

        if not ate_apple:
            tail = self.snake.pop()
            self.state[tail] = BG

        self.state[new_head] = HEAD
        if len(self.snake) > 1:
            self.state[self.snake[1]] = BODY

        if ate_apple:
            self._spawn_apple()

        return self.state


def best_action(head: tuple, apple: tuple):
    dr = apple[0] - head[0]
    dc = apple[1] - head[1]
    candidates = []
    if dr < 0:
        candidates.append(0)  # up
    if dr > 0:
        candidates.append(1)  # down
    if dc < 0:
        candidates.append(2)  # left
    if dc > 0:
        candidates.append(3)  # right
    return random.choice(candidates) if candidates else random.randrange(4)

def rand_action(step):
    if step < 100:
        return random.randrange(4)

    if step < 500:
        # do best 30% of the times, 70% do random
        if random.random() < 0.3:
            return best_action(env.snake[0], env.apple)
        else:
            return random.randrange(4)

    # do best 55% of the times, 45% do random
    if random.random() < 0.55:
        return best_action(env.snake[0], env.apple)
    else:
        return random.randrange(4)

def make_dataset(env, STEPS):
    states = np.empty((STEPS, 1, 8, 8), dtype=np.uint8)
    actions = np.empty(STEPS-1, dtype=np.uint8)

    state = env.reset()

    for i in range(STEPS-1):
        action = rand_action(i)

        states[i] = state[None, :, :]
        actions[i] = action

        state = env.step(action)

    states[STEPS-1] = state
    return states, actions

if __name__ == '__main__':
    from tqdm import tqdm

    env = SnakeEnv()
    # state = env.reset()
    # for _ in range(100):
    #     state = env.step(rand_action())

    # from PIL import Image
    # Image.fromarray(COLOR_MAP[state]).show()
    # exit()

    for i in tqdm(range(100), desc="Making datasets"):
        states, actions = make_dataset(env, 1_000)

        if i == 0:
            print("States shape:", states.shape)
            print("Actions shape:", actions.shape)

        np.savez_compressed(f"snake/data/example_{i}.npz", states=states, actions=actions)

