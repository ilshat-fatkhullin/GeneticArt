import re
import sys
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from skimage.measure import compare_mse


# READING INPUTS

input_image = Image.open('./inputs/input.jpg').convert('RGBA')

word_array = []

for line in open('words.txt', 'r').readlines():
    for w in line.split():
        filtered_word = re.sub('[!@#$;:,.\n]', '', w)
        if len(filtered_word) > 0:
            word_array.append(filtered_word)

fonts = [ImageFont.truetype("./fonts/Pacifico-Regular.ttf", 20)]

# CONSTANTS

IMAGE_WIDTH = input_image.size[0]
IMAGE_HEIGHT = input_image.size[1]
TEXT_MUTATION_RATE = 1

LINE_MUTATION_RATE = 1
LINE_LENGTH = 50
LINE_WIDTH = 1

CIRCLE_MUTATION_RATE = 1
CIRCLE_RADIUS = 10

POPULATION_SIZE = 25
NUMBER_OF_ITERATIONS = 3000
SAVING_ITERATION_INDEX = 50

RANDOM_CACHE_SIZE = 1000


def create_image():
    return Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0, 255))


# FITNESS FUNCTIONS

def rms_difference(image_1, image_2):
    difference = ImageChops.difference(image_1, image_2)
    histogram = difference.histogram()
    squares = (value * ((index % 256) ** 2) for index, value in enumerate(histogram))
    sum_of_squares = np.sum(squares)
    rms = sum_of_squares / float(image_1.size[0] * image_1.size[1])
    return rms


def ms_error(image_1, image_2):
    return compare_mse(np.asarray(image_1), np.asarray(image_2))


def get_fitness(image):
    # return ms_error(image, input_image)
    return rms_difference(image, input_image)


# RANDOM OPTIMIZERS

def get_random(random_array, random_counter, min_value, max_value, shape):
    if random_array is None:
        random_array = np.random.randint(min_value, max_value, shape)
    result = random_array[random_counter]
    random_counter += 1
    if random_counter == RANDOM_CACHE_SIZE:
        random_counter = 0
        random_array = np.random.randint(min_value, max_value, shape)
    return result, random_array, random_counter


random_xs_counter = 0
random_xs = None


def get_random_x():
    global random_xs, random_xs_counter
    result, random_xs, random_xs_counter = get_random(random_xs, random_xs_counter, 0, IMAGE_WIDTH, RANDOM_CACHE_SIZE)
    return result


random_ys_counter = 0
random_ys = None


def get_random_y():
    global random_ys, random_ys_counter
    result, random_ys, random_ys_counter = get_random(random_ys, random_ys_counter, 0, IMAGE_HEIGHT, RANDOM_CACHE_SIZE)
    return result


random_color_counter = 0
random_colors = None


def get_random_color():
    global random_colors, random_color_counter
    result, random_colors, random_color_counter = get_random(random_colors, random_color_counter, 0, 256,
                                                             (RANDOM_CACHE_SIZE, 3))
    return result


random_font_size_counter = 0
random_font_sizes = None


def get_random_font():
    global random_font_sizes, random_font_size_counter
    result, random_font_sizes, random_font_size_counter = get_random(random_font_sizes, random_font_size_counter, 0,
                                                                     len(fonts),
                                                                     RANDOM_CACHE_SIZE)
    return fonts[result]


random_bit_counter = 0
random_bits = None


def get_random_bit():
    global random_bits, random_bit_counter
    result, random_bits, random_bit_counter = get_random(random_bits, random_bit_counter, 0, 2,
                                                         RANDOM_CACHE_SIZE)
    return result


random_word_index_counter = 0
random_word_indexes = None


def get_random_word():
    global random_word_indexes, random_word_index_counter
    result, random_word_indexes, random_word_index_counter = get_random(random_word_indexes, random_word_index_counter,
                                                                        0, len(word_array),
                                                                        RANDOM_CACHE_SIZE)
    return word_array[result]


# MUTATION FUNCTIONS

def add_text_mutation(image, mutation):
    for i in range(mutation):
        word = get_random_word()
        font = get_random_font()
        size = font.getsize(word)
        text_image = Image.new('RGBA', (size[0], size[0]))
        draw = ImageDraw.Draw(text_image)
        color = get_random_color()
        draw.text((0, 0), word, font=font,
                  fill=(color[0], color[1], color[2], 255))
        if get_random_bit() == 0:
            text_image = text_image.rotate(90)
        image.paste(text_image, (get_random_x(), get_random_y()), text_image)
    return image


def add_line_mutation(image, mutation):
    draw = ImageDraw.Draw(image)
    for i in range(mutation):
        color = np.random.randint(0, 256, 3)
        draw.line([(get_random_x(), get_random_y()), (get_random_x(), get_random_y())],
                  (color[0], color[1], color[2], 255), LINE_WIDTH)
    return image


def add_circle_mutation(image, mutation):
    draw = ImageDraw.Draw(image)
    for i in range(mutation):
        color = np.random.randint(0, 256, 3)
        color = (color[0], color[1], color[2], 255)
        x = get_random_x()
        y = get_random_y()
        draw.ellipse([(x, y), (x + CIRCLE_RADIUS, y + CIRCLE_RADIUS)], fill=color, outline=color)
    return image


def cross_images(img_a, img_b):
    return Image.blend(img_a, img_b, 0.5)


# LEARNING FUNCTION

def learn(min_fitness, mutation_function, mutation):
    population = list()

    try:
        cash_image = Image.open('./outputs/output.png').convert('RGBA')
    except FileNotFoundError:
        cash_image = create_image()

    if get_fitness(cash_image) < min_fitness:
        return

    for i in range(POPULATION_SIZE):
        population.append(deepcopy(cash_image))

    best = None
    pre_best = None
    for i in range(NUMBER_OF_ITERATIONS):
        best_fitness = sys.float_info.max

        for individual in population:
            current_fitness = get_fitness(individual)

            if individual is best:
                continue

            if best is None or get_fitness(individual) <= best_fitness:
                pre_best = best
                best = individual
                best_fitness = current_fitness

        new_population = list()
        new_population.append(best)
        new_population.append(pre_best)

        population.remove(best)
        if pre_best in population:
            population.remove(pre_best)

        cross = cross_images(best, pre_best)

        for individual in population:
            new_population.append(mutation_function(deepcopy(cross), mutation))

        population = new_population

        if i % SAVING_ITERATION_INDEX == 0:
            best.save('./outputs/output.png')
            print('Iteration: ' + str(i) + '. Fitness: ' + str(best_fitness))

            if best_fitness < min_fitness:
                return

    best.save('./outputs/output.png')


learn(1000, add_text_mutation, TEXT_MUTATION_RATE)
# learn(1000, add_line_mutation, LINE_MUTATION_RATE)
# learn(1000, add_circle_mutation, CIRCLE_MUTATION_RATE)
