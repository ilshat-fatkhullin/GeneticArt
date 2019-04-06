import re
import sys
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

input_image = Image.open('./inputs/input.jpg').convert('RGBA')

# CONSTANTS
IMAGE_WIDTH = input_image.size[0]
IMAGE_HEIGHT = input_image.size[1]
TEXT_MUTATION = 1

LINE_MUTATION = 1
LINE_LENGTH = 50
LINE_WIDTH = 1

ELLIPSE_MUTATION = 1
CIRCLE_RADIUS = 10

GENERATION_SIZE = 25
ITERATIONS_NUMBER = 1000000000
SAVE_ITERATION = 50

RANDOM_CACHE_SIZE = 1000


def create_image():
    return Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))


# Calculate the root-mean-square difference between two images
def rms_difference(im1, im2):
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(h))
    sum_of_squares = np.sum(sq)
    rms = sum_of_squares / float(im1.size[0] * im1.size[1])
    return rms


def get_fitness(image):
    return rms_difference(image, input_image)


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
        draw.line([(get_random_x(), get_random_y()), (get_random_x(), get_random_y())], (color[0], color[1], color[2], 255), LINE_WIDTH)
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


def learn_until_fitness(min_fitness, mutation_function, mutation):
    generation = list()

    try:
        cash_image = Image.open('./outputs/output.png').convert('RGBA')
    except FileNotFoundError:
        cash_image = create_image()

    if get_fitness(cash_image) < min_fitness:
        return

    for i in range(GENERATION_SIZE):
        generation.append(deepcopy(cash_image))

    best = None
    pre_best = None
    for i in range(ITERATIONS_NUMBER):
        best_fitness = sys.float_info.max
        for individual in generation:
            current_fitness = get_fitness(individual)

            if individual is best:
                continue

            if best is None or get_fitness(individual) <= best_fitness:
                pre_best = best
                best = individual
                best_fitness = current_fitness

        new_generation = list()
        new_generation.append(best)
        new_generation.append(pre_best)

        generation.remove(best)
        if pre_best in generation:
            generation.remove(pre_best)

        cross = cross_images(best, pre_best)

        for individual in generation:
            new_generation.append(mutation_function(deepcopy(cross), mutation))

        generation = new_generation

        if i % SAVE_ITERATION == 0:
            best.save('./outputs/output.png')
            print('Iteration: ' + str(i) + '. Fitness: ' + str(best_fitness) + ' Size of generation ' + str(len(generation)))
            if best_fitness < min_fitness:
                return
            if i == 6000:
                return

    best.save('./outputs/output.png')


word_array = []

for line in open('words.txt', 'r').readlines():
    for w in line.split():
        filtered_word = re.sub('[!@#$;:,.\n]', '', w)
        if len(filtered_word) > 0:
            word_array.append(filtered_word)

fonts = [ImageFont.truetype("./fonts/Pacifico-Regular.ttf", 20)]

learn_until_fitness(3000, add_text_mutation, TEXT_MUTATION)
