import re
import numpy as np
import random as rand
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops


class Word:
    pos_x = 0.
    pos_y = 0.
    color_r = 0.
    color_g = 0.
    color_b = 0.
    word = ' '

    def __init__(self, pos_x, pos_y, color_r, color_g, color_b, word):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.word = word

    @staticmethod
    def clamp(x, a, b):
        if x < a:
            return a
        elif x > b:
            return b
        return x

    def get_cross(self, other):
        return Word(
            (self.pos_x + other.pos_x) // 2,
            (self.pos_y + other.pos_y) // 2,
            (self.color_r + other.color_r) // 2,
            (self.color_g + other.color_g) // 2,
            (self.color_b + other.color_b) // 2,
            self.word)

    def add_mutation(self):
        self.pos_x = Word.clamp(self.pos_x + rand.randint(-POSITION_MUTATION, POSITION_MUTATION), 0, IMAGE_WIDTH)
        self.pos_y = Word.clamp(self.pos_y + rand.randint(-POSITION_MUTATION, POSITION_MUTATION), 0, IMAGE_HEIGHT)
        self.color_r = Word.clamp(self.color_r + rand.randint(-COLOR_MUTATION, COLOR_MUTATION), 0, 255)
        self.color_g = Word.clamp(self.color_g + rand.randint(-COLOR_MUTATION, COLOR_MUTATION), 0, 255)
        self.color_b = Word.clamp(self.color_b + rand.randint(-COLOR_MUTATION, COLOR_MUTATION), 0, 255)


class Individual:
    words = []

    @classmethod
    def from_words(cls, words):
        individual = cls()
        individual.words = words
        return individual

    @classmethod
    def from_str_array(cls, array):
        individual = cls()
        for i in range(WORDS_NUMBER):
            individual.words.append(Word(
                rand.randint(0, IMAGE_WIDTH),
                rand.randint(0, IMAGE_HEIGHT),
                0,
                0,
                0,
                array[i]))
        return individual

    def get_image(self):
        image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT))
        draw = ImageDraw.Draw(image)
        for w in self.words:
            draw.text((w.pos_x, w.pos_y), w.word, font=font, fill=(w.color_r, w.color_g, w.color_b))
        return image

    def get_cross(self, other):
        words = []
        for i in range(WORDS_NUMBER):
            words.append(self.words[i].get_cross(other.words[i]))
        result = Individual.from_words(words)
        result.add_mutation()
        return result

    def add_mutation(self):
        for w in self.words:
            w.add_mutation()

    @staticmethod
    def rms_difference(im1, im2):
        diff = ImageChops.difference(im1, im2)
        h = diff.histogram()
        sq = (value * ((idx % 256) ** 2) for idx, value in enumerate(h))
        sum_of_squares = sum(sq)
        rms = sum_of_squares / float(im1.size[0] * im1.size[1])
        return rms

    def get_fitness(self):
        return Individual.rms_difference(self.get_image(), input_image)


str_array = []

for line in open('words.txt', 'r').readlines():
    for word in line.split():
        filtered_word = re.sub('[!@#$,.\n]', '', word)
        if len(filtered_word) > 0:
            str_array.append(filtered_word)

WORDS_NUMBER = len(str_array)

# CONSTANTS
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

POSITION_MUTATION = 10
COLOR_MUTATION = 5

GENERATION_SIZE = 5
ITERATIONS_NUMBER = 1000000000

SAVE_ITERATION = 50

input_image = Image.open('input.png').convert('RGB')
font = ImageFont.truetype("Roboto-Regular.ttf", 15)

generation = []

for i in range(GENERATION_SIZE):
    generation.append([0., Individual.from_str_array(str_array)])

for i in range(ITERATIONS_NUMBER):
    for g_i in range(GENERATION_SIZE):
        generation[g_i][0] = generation[g_i][1].get_fitness()

    generation.sort(key=lambda x: x[0])

    new_generation = list()

    new_generation.append([generation[0][0], generation[0][1]])
    new_generation.append([generation[1][0], generation[1][1]])

    for j in range(GENERATION_SIZE - 2):
        new_generation.append([0, generation[j][1].get_cross(generation[0][1])])

    generation = new_generation

    if i % SAVE_ITERATION == 0:
        generation[0][1].get_image().save('output.png')
        generation[0][1].get_image().save('output.png')
        print('Iteration: ' + str(i) + '. Fitness: ' + str(generation[0][0]))

generation[0][1].get_image().save('output.png')
