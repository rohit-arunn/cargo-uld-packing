"""
rl_packing_example.py

Simplified end-to-end RL example for 3D grid packing using PPO (stable-baselines3). 

Key ideas:
- State: flattened occupancy grid + remaining box features + frontier occupancy (one-hot)
- Action: a single Discrete integer that encodes (box_index, frontier_index, rotation_index)
- Reward: volume placed (voxels) for valid placement, -0.5 penalty for invalid action
- Episode: ends when no boxes remain or max steps reached 
"""

import numpy as np
import random
from copy import deepcopy
import gym
from gym import spaces
import plotly.graph_objects as go
import plotly.io as pio
from plotly_main import create_ld1, create_ld1_edges
pio.renderers.default = "browser"

# If you want to run training:
# pip install stable-baselines3[extra]  (install before running)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------- Helper utilities ----------

def get_unique_rotations(dim):
    """Return integer rotations (dx,dy,dz) (grid units). For simplicity we use 3 unique rotations."""
    x,y,z = dim
    rotations = {(x,y,z),(y,x,z),(z,y,x)}  # not exhaustive but illustrative
    # remove duplicates and convert to list of tuples
    return list(rotations)

def can_place(grid, x, y, z, dx, dy, dz):
    """Check if block (dx,dy,dz) fits in grid at (x,y,z) without overlapping."""
    Lz, Ly, Lx = grid.shape
    if x < 0 or y < 0 or z < 0: 
        return False
    if x + dx > Lx or y + dy > Ly or z + dz > Lz:
        return False
    # check overlap
    sub = grid[z:z+dz, y:y+dy, x:x+dx]
    return np.all(sub == 0)

def place_box(grid, x, y, z, dx, dy, dz, box_marker=1):
    """Place the box into the grid by setting voxels to box_marker. Returns volume placed."""
    grid[z:z+dz, y:y+dy, x:x+dx] = box_marker
    return dx * dy * dz

def generate_frontier_from_grid(grid, max_frontier=20):
    """Simple heuristic: return up to max_frontier candidate (x,y,z) positions: lowest empty cells scanning x,y."""
    Lz, Ly, Lx = grid.shape
    frontier = []    
    for x in range(Lx):
        for y in range(Ly):
            # find first z that's zero
            for z in range(Lz):
                if grid[z,y,x] == 0:
                    frontier.append((x,y,z))
                    break
            if len(frontier) >= max_frontier:
                return frontier
    return frontier

# ---------- A simple gym.Env for packing ----------

class PackingEnv(gym.Env):
    """
    Simplified packing environment.
    Observation: concatenated vector of:
       - flattened occupancy grid (Lz*Ly*Lx, 0/1)
       - remaining boxes features: for up to N boxes, dims flattened (N*3) and mask (N)
       - frontier one-hot mask of length max_frontier (1 if candidate exists)
    Action (Discrete): index in [0 .. num_boxes * max_frontier * num_rotations - 1]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, boxes, container_dims=(6,6,6), grid_step=1, max_frontier=20, max_boxes=None):
        super().__init__()
        # boxes: list of dicts with 'dimensions' (in grid units), 'number' (integer)
        self.raw_boxes = deepcopy(boxes)
        self.container_dims = container_dims  # (Lx, Ly, Lz)
        # store grid shape as (Lz, Ly, Lx) for indexing convenience
        Lx, Ly, Lz = container_dims
        self.grid_shape = (Lz, Ly, Lx)

        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)

        self.grid_step = grid_step
        self.max_frontier = max_frontier
        self.max_boxes = max_boxes if max_boxes is not None else len(boxes)
        self.viewer = None
        self.placed_boxes = []

        # prepare the expanded box list (account for 'number' field) 
        self.box_list = []
        for b in self.raw_boxes:
            count = b.get('number', 1)
            for _ in range(count):
                self.box_list.append({'dimensions': b['dimensions'], 'weight': b.get('weight',1)})
        self.num_boxes_total = len(self.box_list)

        # limit number of boxes considered (pad if necessary)
        self.N = min(self.max_boxes, self.num_boxes_total)

        # For simplicity: we only allow up to R rotations per box (we compute per-box)
        # but the action space will use a fixed R (max_rotations)
        self.max_rotations = 3

        # Observation vector size calculation
        grid_size = np.prod(self.grid_shape)
        # box features: N * 3 dims + N mask
        box_feat_size = self.N * 3 + self.N
        frontier_size = self.max_frontier * 3  # we'll encode frontier coordinates flattened (or -1 if unused)

        obs_size = grid_size + box_feat_size + frontier_size

        # Use a continuous Box observation (float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Action space: choose box_index (0..N-1) * frontier_idx (0..max_frontier-1) * rotation_idx (0..max_rotations-1)
        self.action_space = spaces.Discrete(self.N * self.max_frontier * self.max_rotations)

        # state tracking
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_shape, dtype=np.uint8)
        # remaining box indices list (we will refer to indices in original box_list)
        self.remaining = list(range(self.N))  # if N < available boxes, only first N
        random.shuffle(self.remaining)

        # step counters
        self.steps = 0
        self.max_steps = max(200, self.N * 5)

        # initial frontier
        self.frontier = generate_frontier_from_grid(self.grid, self.max_frontier)

        # build observation
        return self._get_obs()

    def _get_obs(self):
        # flattened grid
        grid_flat = self.grid.flatten().astype(np.float32)  # 0/1

        # box features: for first N boxes we provide dims (normalized) and mask (1=available, 0=used)
        dims = np.zeros((self.N,3), dtype=np.float32)
        mask = np.zeros((self.N,), dtype=np.float32)
        for i in range(self.N):
            if i < len(self.remaining):
                # map index i_in_list -> actual box id in box_list
                bidx = self.remaining[i]
                d = self.box_list[bidx]['dimensions']
                dims[i,:] = np.array(d, dtype=np.float32) / float(max(self.container_dims))  # normalize dims
                mask[i] = 1.0
            else:
                dims[i,:] = 0.0
                mask[i] = 0.0
        dims_flat = dims.flatten()
        box_mask = mask

        # frontier positions: we return up to max_frontier coordinates normalized; if fewer, pad with -1
        frontier_coords = np.full((self.max_frontier, 3), -1.0, dtype=np.float32)
        for i, (x,y,z) in enumerate(self.frontier[:self.max_frontier]):
            # normalize coords by container dims
            frontier_coords[i,0] = x / float(self.container_dims[0])
            frontier_coords[i,1] = y / float(self.container_dims[1])
            frontier_coords[i,2] = z / float(self.container_dims[2])
        frontier_flat = frontier_coords.flatten()

        obs = np.concatenate([grid_flat, dims_flat, box_mask, frontier_flat]).astype(np.float32)
        return obs

    def step(self, action):
        """
        Decode action into (box_slot, frontier_index, rot_idx)
        - box_slot: integer index into the *visible* remaining slots (0..N-1)
        - then map to actual box_list index
        """
        self.steps += 1
        done = False
        info = {}

        slot = action // (self.max_frontier * self.max_rotations)
        rem = action % (self.max_frontier * self.max_rotations)
        frontier_idx = rem // self.max_rotations
        rot_idx = rem % self.max_rotations

        reward = 0.0

        # Check slot validity: slot must refer to available slot (slot < len(remaining))
        if slot >= len(self.remaining):
            # invalid action: choose random small negative
            reward = -0.5
            if self.steps >= self.max_steps:
                done = True
            return self._get_obs(), reward, done, info

        # map to actual box id
        b_idx = self.remaining[slot]
        box = self.box_list[b_idx]
        dims = box['dimensions']
        rotations = get_unique_rotations(dims)
        # clamp rot_idx
        rot_idx = min(rot_idx, len(rotations)-1)
        dx,dy,dz = rotations[rot_idx]

        # check frontier idx valid
        if frontier_idx >= len(self.frontier):
            # invalid frontier -> small penalty
            reward = -0.5
            if self.steps >= self.max_steps:
                done = True
            return self._get_obs(), reward, done, info

        x,y,z = self.frontier[frontier_idx]

        # Attempt to place
        if can_place(self.grid, x, y, z, dx, dy, dz):
            vol = place_box(self.grid, x, y, z, dx, dy, dz, box_marker=1)
            reward = float(vol) / float(np.prod(self.grid_shape))  # normalized placed volume 
            # remove this box from remaining
            self.remaining.pop(slot)
            # update frontier
            self.frontier = generate_frontier_from_grid(self.grid, self.max_frontier)
            # If no remaining boxes or grid full, end episode
            if len(self.remaining) == 0:
                done = True
            if self.steps >= self.max_steps:
                done = True
        else:
            # invalid placement (overlap or out of bounds)
            reward = -0.5

        # small shaping: encourage filling bottom layers by extra reward for low z placements
        reward += 0.01 * (1.0 - (z / float(self.container_dims[2] + 1)))  # slight bias for low z

        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, info

    def render(self, mode="human"):
        

        fig = go.Figure()

        # --- Add the LD1 container ---
        ld1_mesh, ld1_edges = create_ld1(color="lightblue", opacity=0.2, name="LD1 Container")
        fig.add_trace(ld1_mesh)
        for edge in ld1_edges:
            fig.add_trace(edge)

        # --- Add placed boxes ---
        for box in getattr(self, "placed_boxes", []):
            x, y, z = box["position"]
            dx, dy, dz = box["dimensions"]

            fig.add_trace(go.Mesh3d(
                x=[x, x+dx, x+dx, x, x, x+dx, x+dx, x],
                y=[y, y, y+dy, y+dy, y, y, y+dy, y+dy],
                z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
                i=[0, 0, 0, 4, 4, 4, 2, 1, 5, 6, 7, 7],
                j=[1, 2, 3, 5, 6, 7, 6, 5, 6, 7, 6, 5],
                k=[2, 3, 0, 6, 7, 4, 7, 6, 7, 4, 5, 4],
                color="orange",
                opacity=0.6
            ))

        # Layout
        fig.update_layout(
            scene=dict(
                xaxis_title='Length',
                yaxis_title='Width',
                zaxis_title='Height',
                aspectmode="data"
            ),
            title="ULD Packing (LD1 with Boxes)"
        )

        if mode == "human":
            return fig


        # Update existing figure instead of opening a new one
        #self.viewer.show(renderer="browser")
# ---------- Example usage: train PPO ----------

def build_example_boxes():
    # Simple set of boxes (dimensions in grid units) AI is a hype content is similar bro, MS in AI
    # number field is handled by making repeated entries earlier
    return [
        {'dimensions': (2,2,1), 'number': 3},
        {'dimensions': (3,2,1), 'number': 2},
        {'dimensions': (1,1,2), 'number': 4},
        {'dimensions': (2,1,1), 'number': 2},
    ]

def main_train():
    boxes = build_example_boxes()
    env = DummyVecEnv([lambda: PackingEnv(boxes, container_dims=(6,6,6), max_frontier=20, max_boxes=10)])
    model = PPO('MlpPolicy', env, verbose=1, n_steps=128, batch_size=64, learning_rate=3e-4)
    # Train for a modest number of timesteps for demo; increase for real training
    model.learn(total_timesteps=25_000)
    model.save("ppo_packing_demo")

    # Evaluate a few episodes
    eval_env = PackingEnv(boxes, container_dims=(6,6,6), max_frontier=20, max_boxes=10)
    obs = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
    return env

if __name__ == "__main__":
    env = main_train()
    real_env = env.envs[0]   # unwrap DummyVecEnv to get your PackingEnv
    real_env.render()
    real_env.viewer.show(renderer="browser")

