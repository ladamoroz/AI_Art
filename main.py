import numpy as np
import copy
import random
import cv2
from PIL import Image
import multiprocessing
from skimage.metrics import structural_similarity as ssim
import time

# constants and global variables
HEX_SIZE = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
SEGMENT_HEIGHT = 8
SEGMENT_WIDTH = 8
POPULATION_SIZE = 8
ROW = SEGMENT_WIDTH // HEX_SIZE
CHROMOSOME_SIZE = (SEGMENT_WIDTH // HEX_SIZE) * (SEGMENT_HEIGHT // HEX_SIZE) + 1
MUTATION_PROBABILITY = 0.001
SEGMENT_COLORS = []
COORDS = []
NUM_SEG = IMAGE_HEIGHT * IMAGE_WIDTH // (SEGMENT_HEIGHT * SEGMENT_WIDTH)
SEG_ROW = IMAGE_WIDTH // SEGMENT_WIDTH
FITNESS = 0.15
NUM_ITERATIONS = 500

# produce final image from chromosome contains colors for every hexagon on the picture
def final_image(chromosome):
    image = np.full((IMAGE_WIDTH, IMAGE_HEIGHT, 3), fill_value=255, dtype=np.uint8)
    image_coords = []
    for i in range(0, NUM_SEG):
        for j in range(0, CHROMOSOME_SIZE):
            temp = copy.deepcopy(COORDS[j])
            for coord in temp:
                coord[0] += SEGMENT_WIDTH * (i % SEG_ROW)
                coord[1] += SEGMENT_HEIGHT * (i // SEG_ROW)
            image_coords.append(temp)

    for i in range(len(chromosome)):
        t = copy.deepcopy(image_coords[i])
        t = t.reshape((-1, 1, 2))
        cv2.fillPoly(image, [t], chromosome[i])
    return image


# compare structural similarity of solution produced by the algorithm and the goal
def compute_fitness(segment, solution):
    image = create_image(solution)
    return ssim(image, segment, multichannel=True)


# produce image from given chromosome within segment
def create_image(chromosome):
    image = np.full((SEGMENT_WIDTH, SEGMENT_HEIGHT, 3), fill_value=255, dtype=np.uint8)
    for i in range(len(chromosome)):
        t = copy.deepcopy(COORDS[i])
        t = t.reshape((-1, 1, 2))
        cv2.fillPoly(image, [t], chromosome[i])
    return image


# extract all colors from goal
def getcolors(segment):
    global SEGMENT_COLORS
    img = Image.fromarray(segment).convert("RGB")
    temp = Image.Image.getcolors(img, maxcolors=100000)
    colors = []
    for i in temp:
        colors.append(i[1])
    SEGMENT_COLORS = copy.deepcopy(colors)


# creating of chromosome
# chromosome is colors for each hexagon on the segment
def create_chromosome():
    chromosome = []
    init_coords = np.array([[0, 1], [1, 0], [2, 1], [2, 2], [1, 3], [0, 2]])
    COORDS.append(init_coords)
    temp = copy.deepcopy(init_coords)
    for i in range(0, CHROMOSOME_SIZE):
        temp = copy.deepcopy(init_coords)
        for coord in temp:
            coord[0] += HEX_SIZE * (i % ROW)
            coord[1] += HEX_SIZE * (i // ROW)
        c = random.choice(SEGMENT_COLORS)
        COORDS.append(temp)
        chromosome.append(c)
    return chromosome


# producing POPULATION_SIZE number of chromosomes to construct population
def create_population():
    new_population = []
    for i in range(POPULATION_SIZE):
        new_population.append(create_chromosome())
    return new_population


# reproduction within population
def reproduce(population):

    # producing 2 children from 2 parents with use of single-point crossover
    def crossover(parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        point = np.random.randint(0, 3)
        child1[:point] = parent2[:point]
        child1[:point] = parent1[:point]

        return child1, child2

    # create new population with addition of children from each two chromosomes
    new_population = []
    for i in range(POPULATION_SIZE):
        for j in range(POPULATION_SIZE):
            if i != j:
                child1, child2 = crossover(population[i], population[j])
                new_population.append(child1)
                new_population.append(child2)
            else:
                new_population.append(population[i])

    return new_population


# mutate some chromosomes in the population
def mutate(population):

    # mutation of chromosome by changing color in the chromosome
    def mutation(gene):
        c = random.choice(SEGMENT_COLORS)
        gene = c
        return gene

    for i in range(POPULATION_SIZE):
        for j in range(CHROMOSOME_SIZE):
            if np.random.uniform(low=0.0) <= MUTATION_PROBABILITY:
                population[i][j] = mutation(population[i][j])
    return population


# sort segments by its fitness
def best(segment, population):
    fitness = []
    for i in range(len(population)):
        fitness.append([population[i], compute_fitness(segment, population[i])])

    population = sorted(fitness, key=lambda chromosome: chromosome[1], reverse=True)
    return population


# producing segment with best fitness for the image
def main(segment):
    getcolors(segment)
    population = create_population()
    fit = 0.0
    iterations = 0
    while fit < FITNESS and iterations < NUM_ITERATIONS:
        iterations += 1
        new_population = best(segment, population)[:POPULATION_SIZE]
        best_chromosome = new_population[0][0]
        fit = new_population[0][1]
        best_image = create_image(best_chromosome)
        temp_pop = []
        for i in range(POPULATION_SIZE):
            temp_pop.append(new_population[i][0])
        population = reproduce(mutate(temp_pop))
    return best_chromosome


# creating image from segments with best fitness
def create_art():
    GOAL = cv2.imread("input.jpg")
    GOAL = cv2.resize(GOAL, (IMAGE_HEIGHT, IMAGE_WIDTH))

    x1 = 0
    y1 = 0
    x2 = SEGMENT_WIDTH
    y2 = SEGMENT_HEIGHT
    image = []
    for i in range(0, NUM_SEG):
        segment = GOAL[y1:y2, x1:x2]
        image += main(segment)
        print(i)
        if x2 == IMAGE_WIDTH:
            x1 = 0
            x2 = SEGMENT_WIDTH
            y1 += SEGMENT_HEIGHT
            y2 += SEGMENT_HEIGHT
        else:
            x1 += SEGMENT_WIDTH
            x2 += SEGMENT_WIDTH
    image = final_image(image)
    cv2.imwrite("out.jpg", image)


start_time = time.time()
create_art()
print("--- %s seconds ---" % (time.time() - start_time))
