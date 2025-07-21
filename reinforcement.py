import gym
import numpy as np
import pandas as pd
from first import grid_based_pack, get_unique_rotations
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import random


df = pd.read_parquet("flight_ICN_to_BUD.parquet")

boxes = []
for idx, row in df.iterrows():
    box_id = (
        row['mstdocnum'], row['docowridr'], row['dupnum'],
        row['seqnum'], row['ratlinsernum'], row['dimsernum']
    ) 
    length = float(row['pcslen']) 
    width = float(row['pcswid'])
    height = float(row['pcshgt']) 
    numpcs = int(row['dim_numpcs'])
    weight = float(row['dim_wgt'])

    boxes.append({
        'box_id': box_id,
        'dimensions': (length, width, height),
        'number' : numpcs, 
        'weight': weight

    })      


#Class for packing environment

class BoxPackingEnv(gym.Env):
    def __init__(self, boxes, container_dims=(92, 60.4, 64), grid_step=1):
        super(BoxPackingEnv, self).__init__()
        self.boxes = boxes
        self.container_dims = container_dims
        self.grid_step = grid_step

        # Observation: Grid flattened + current box dims
        grid_cells = int((container_dims[0] * container_dims[1] * container_dims[2]) / (grid_step**3))
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_cells + 3,), dtype=np.float32)

        # Action: [x, y, z, rotation_index]
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Your existing reset code here...
        self.placed_boxes = []
        self.current_box_index = 0
        self.rejected_rotations = []
        self.grid = np.zeros((int(self.container_dims[2]),
                            int(self.container_dims[1]),
                            int(self.container_dims[0])), dtype=int)
        obs = self._get_state()
        info = {}  # Add any useful info here if needed
        return obs, info


    def step(self, action):
        if self.current_box_index >= len(self.boxes):
            return self._get_state(), 0.0, True, {}

        box = self.boxes[self.current_box_index]
        rotations = get_unique_rotations(box['dimensions'])

        # Decode normalized action
        pos = tuple(int(action[i] * (self.grid.shape[2 - i] - 1)) for i in range(3))
        rot_index = min(int(action[3] * len(rotations)), len(rotations) - 1)
        chosen_rotation = rotations[rot_index]

        # Build box to try
        trial_box = {
            'box_id': box['box_id'],
            'dimensions': chosen_rotation,
            'position': pos,
            'number': 1,
            'weight': box.get('weight', 1),
            'colour': box.get('colour', (1, 0, 0))
        }

        # Try placing with grid_based_pack
        trial_list = self.placed_boxes + [trial_box]
        placed_boxes = grid_based_pack(trial_list, self.container_dims, self.grid_step)

        if len(placed_boxes) > len(self.placed_boxes):
            reward = 1.0
            self.placed_boxes = placed_boxes
        else:
            reward = -1.0

        self.current_box_index += 1
        done = self.current_box_index >= len(self.boxes)
        return self._get_state(), reward, done, {}

    def _get_state(self):
        flat_grid = self.grid.flatten() / (np.max(self.grid) if np.max(self.grid) > 0 else 1)
        if self.current_box_index < len(self.boxes):
            dims = np.array(self.boxes[self.current_box_index]['dimensions'])
        else:
            dims = np.zeros(3)
        return np.concatenate([flat_grid, dims.astype(np.float32)])






# Assuming boxes and dims are defined
env = BoxPackingEnv(boxes, container_dims=(92, 60.4, 64), grid_step=1)

check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# To test:
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
