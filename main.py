import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import random as rand

# CONSTANTS
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
MUTATION = 1
GENERATION_SIZE = 100
ITERATIONS_NUMBER = 1000000000
SAVE_ITERATION = 50

input_image = mpimg.imread('input.png')
plt.imshow(input_image)

style_image = mpimg.imread('input.png')
plt.imshow(style_image)


def create_image_randomized():
    return np.random.rand(IMAGE_WIDTH, IMAGE_HEIGHT, 3)


def create_image():
    return np.ones((IMAGE_WIDTH, IMAGE_HEIGHT, 3), np.float)


def get_fitness(image):
    return np.sum((image - style_image) ** 2)


#def add_mutation(image):
#    for i in range(MUTATION):
#        cv.circle(image,
#                  (rand.randint(0, IMAGE_WIDTH),
#                   rand.randint(0, IMAGE_HEIGHT)),
#                  1,
#                  (rand.uniform(0, 1), rand.randint(0, 1), rand.randint(0, 1)),
#                  2)


def add_mutation(image):
    for i in range(MUTATION):
        p_1 = np.random.randint(0, IMAGE_WIDTH, 2)
        p_2 = np.random.randint(0, IMAGE_WIDTH, 2)
        cv.line(image,
                (p_1[0], p_1[1]),
                (p_2[0], p_2[1]),
                np.random.uniform(0, 1, 3))


def cross_images(img_a, img_b):
    result_image = (img_a + img_b) / 2
    result_image = np.clip(a=result_image, a_min=0, a_max=1)
    add_mutation(result_image)
    return result_image


generation = []
cash_image = None
try:
    cash_image = mpimg.imread('output.png')
    cash_image = cash_image[:, :, :3]
except FileNotFoundError:
    cash_image = create_image()

for i in range(GENERATION_SIZE):
    generation.append([0, cash_image])

for i in range(ITERATIONS_NUMBER):
    for g_i in range(GENERATION_SIZE):
        generation[g_i][0] = get_fitness(generation[g_i][1])

    generation.sort(key=lambda x: x[0])

    new_generation = list()

    new_generation.append(generation[0])
    new_generation.append(generation[1])

    for j in range(GENERATION_SIZE - 2):
        new_generation.append([0, cross_images(generation[0][1], generation[1][1])])

    generation = new_generation

    if i % SAVE_ITERATION == 0:
        plt.imsave('output.png', generation[0][1])
        print('Iteration: ' + str(i) + '. Fitness: ' + str(generation[0][0]))

plt.imsave('output.png', generation[0][1])
