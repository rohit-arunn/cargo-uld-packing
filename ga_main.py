import numpy as np
import pandas as pd
import random
from multiprocessing import Pool
import random
import random
from packing_main import GridBasedPacking

GRID_STEP = 1 

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


def evaluate_fitness_wrapper(args):
    self, chromosome = args
    return self.evaluate_fitness(chromosome)
   

class GeneticAlgorithm:
    def __init__(self, boxes, uld_checker, container_dims, grid_step=1,
                 pop_size=4, generations=2, mutation_rate=0.3, processes=8):
        self.boxes = boxes
        self.uld_checker = uld_checker
        self.container_dims = container_dims
        self.GRID_STEP = grid_step
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.processes = processes
        #self.crossover = crossover   

    # -------------------- 
    # FITNESS HELPERS
    # --------------------
    def calculate_weight_penalty(self, placed_boxes):
        penalty = 0
        for box in placed_boxes:
            x, y, z = box['position']
            dx, dy, dz = box['dimensions']
            weight = box['weight']

            for other in placed_boxes:
                if other == box:
                    continue
                ox, oy, oz = other['position']
                odx, ody, odz = other['dimensions']
                oweight = other['weight']

                same_xy = (
                    ox < x + dx and ox + odx > x and
                    oy < y + dy and oy + ody > y
                )
                touching_z = abs(z - (oz + oz)) <= self.GRID_STEP or \
                             (abs((oz + odz) - z) <= self.GRID_STEP)

                if same_xy and touching_z:
                    if weight > oweight:
                        penalty += (weight - oweight)
                    break
        return penalty

    def grid_fill_ratio(self, grid):
        lz, ly, lx = grid.shape

        for z in reversed(range(lz)):
            if np.any(grid[z]):
                highest_layer = z + 1
                break
        else:
            return 0.0

        used_grid = grid[:highest_layer]
        filled_spaces = np.count_nonzero(used_grid)
        total_spaces = used_grid.size

        return filled_spaces / total_spaces

    
    #presently I have commented out the weight penalty, so the volume optimization is the only factor considered. 



    def generate_initial_population(self):
        population = []
        for _ in range(self.pop_size):
            chromo = []
            for box in self.boxes:
                chromo.append({
                    'box_id': box['box_id'],
                    'dimensions': box['dimensions'],
                    'number': box.get('number', 1),
                    'weight': box['weight']
                })
            population.append(chromo)
        return population

    def evaluate_fitness(self, chromosome):
        boxes_for_fitness = []
        for src in chromosome:
            boxes_for_fitness.append({
                'box_id': src['box_id'],
                'dimensions': src['dimensions'],
                'number': src.get('number', 1),
                'weight': src.get('weight'),
                'colour': src.get('colour', (
                    random.random(),
                    random.random(),
                    random.random()
                ))
            })

        thepacking = GridBasedPacking(boxes=boxes_for_fitness, uld_checker=self.uld_checker, grid_step=self.GRID_STEP)

        placed, grid = thepacking.grid_based_pack(boxes_for_fitness, self.container_dims, self.GRID_STEP)

        if not placed:
            return 0.0, None

        weight_penalty = self.calculate_weight_penalty(placed)
        volume_reward = self.grid_fill_ratio(grid)

        fitness = volume_reward         # - 0.005 * weight_penalty
        return fitness, placed

    def crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = parent1[a:b]

        existing_ids = {box['box_id'] for box in child if box is not None}
        fill_boxes = [box for box in parent2 if box['box_id'] not in existing_ids]

        fill_idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill_boxes[fill_idx]
                fill_idx += 1

        return child

    def mutate(self, chromo):
        num_swaps = max(1, int(self.mutation_rate * len(chromo)))
        for _ in range(num_swaps):
            i, j = random.sample(range(len(chromo)), 2)
            chromo[i], chromo[j] = chromo[j], chromo[i]
        return chromo

    def select(self, pop, fitnesses, k=3):
        contenders = random.sample(list(zip(pop, fitnesses)), k)
        contenders.sort(key=lambda t: t[1], reverse=True)
        return contenders[0][0]

    # --------------------
    # MAIN LOOP
    # -------------------- 
    def run(self):
        pop = self.generate_initial_population()

        fitnesses_main = []
        placements_main = []
        generations_plot = []
        fitness_history = []

        print("Type of self.evaluate_fitness:", type(self.evaluate_fitness))


        for gen in range(self.generations):
            with Pool(processes=self.processes) as pool:
                results = pool.map(self.evaluate_fitness, pop) 

            fitnesses, placements = map(list, zip(*results)) 

            fitnesses_main += fitnesses
            placements_main += placements

            print(f'GEN {gen:03d}  best = {max(fitnesses):.3f}')
            generations_plot.append(gen)          
            fitness_history.append(max(fitnesses))

            next_pop = []
            while len(next_pop) < self.pop_size:
                p1, p2 = self.select(pop, fitnesses), self.select(pop, fitnesses)
                child = self.crossover(p1, p2)
                child = self.mutate(child)      
                next_pop.append(child)

            pop = next_pop

        idx = fitnesses_main.index(max(fitnesses_main))
        print("Best fitness - ", fitnesses_main[idx])
        print("Best fitness generation - ", idx)
        return placements_main[idx], generations_plot, fitness_history
    

# ga = GeneticAlgorithm(
#     boxes=boxes, 
#     container_dims=(92, 60.4, 64), 
#     grid_step=1,
#     pop_size=10,
#     generations=50,
#     mutation_rate=0.3,
#     processes=8
# )   