import numpy as np
from env import ExampleEnv

if __name__ == '__main__':
    from tqdm import tqdm

    env = ExampleEnv()

    # to keep the dataset small, only 1 episode (file) is generated
    EPISODES = 1
    STEPS = 10_000
    OUT = "example/data/example_{:03d}.npz"

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
