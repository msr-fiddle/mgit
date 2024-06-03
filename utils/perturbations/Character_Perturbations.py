"""
Authors: Milad Moradi
NLP-perturbation: https://github.com/mmoradi-iut/NLP-perturbation
Paper: https://arxiv.org/ftp/arxiv/papers/2108/2108.12237.pdf
 
Common misspelled data: https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines 
"""

import nltk
from random import seed
from random import randint
import csv
import os


def return_random_number(begin, end):
    return randint(begin, end)


def random_changing_type():
    random_num = randint(1, 2)
    if random_num == 1:
        return "FirstChar"
    else:
        return "AllChars"


def swap_characters(input_word, position, adjacent):
    temp_word = ""
    if adjacent == "left":
        if position == 1:
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif position == len(input_word) - 1:
            temp_word = input_word[0 : position - 1]
            temp_word += input_word[position]
            temp_word += input_word[position - 1]
        elif position > 1 and position < len(input_word) - 1:
            temp_word = input_word[0 : position - 1]
            temp_word += input_word[position]
            temp_word += input_word[position - 1]
            temp_word += input_word[position + 1 :]

    elif adjacent == "right":
        if position == 0:
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif position == len(input_word) - 2:
            temp_word = input_word[0:position]
            temp_word += input_word[position + 1]
            temp_word += input_word[position]
        elif position > 0 and position < len(input_word) - 2:
            temp_word = input_word[0:position]
            temp_word += input_word[position + 1]
            temp_word += input_word[position]
            temp_word += input_word[position + 2 :]

    return temp_word


def return_adjacent_char(input_char):

    if input_char == "a":
        return "s"

    elif input_char == "b":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "v"
        else:
            return "n"

    elif input_char == "c":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "x"
        else:
            return "v"

    elif input_char == "d":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "s"
        else:
            return "f"

    elif input_char == "e":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "w"
        else:
            return "r"

    elif input_char == "f":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "d"
        else:
            return "g"

    elif input_char == "g":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "f"
        else:
            return "h"

    elif input_char == "h":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "g"
        else:
            return "j"

    elif input_char == "i":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "u"
        else:
            return "o"

    elif input_char == "j":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "h"
        else:
            return "k"

    elif input_char == "k":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "j"
        else:
            return "l"

    elif input_char == "l":
        return "k"

    elif input_char == "m":
        return "n"

    elif input_char == "n":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "b"
        else:
            return "m"

    elif input_char == "o":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "i"
        else:
            return "p"

    elif input_char == "p":
        return "o"

    elif input_char == "q":
        return "w"

    elif input_char == "r":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "e"
        else:
            return "t"

    elif input_char == "s":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "a"
        else:
            return "d"

    elif input_char == "t":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "r"
        else:
            return "y"

    elif input_char == "u":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "y"
        else:
            return "i"

    elif input_char == "v":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "c"
        else:
            return "b"

    elif input_char == "w":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "q"
        else:
            return "e"

    elif input_char == "x":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "z"
        else:
            return "c"

    elif input_char == "y":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "t"
        else:
            return "u"

    elif input_char == "z":
        return "x"
    # ---------------------------------------------
    elif input_char == "A":
        return "S"

    elif input_char == "B":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "V"
        else:
            return "N"

    elif input_char == "C":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "X"
        else:
            return "V"

    elif input_char == "D":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "S"
        else:
            return "F"

    elif input_char == "E":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "W"
        else:
            return "R"

    elif input_char == "F":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "D"
        else:
            return "G"

    elif input_char == "G":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "F"
        else:
            return "H"

    elif input_char == "H":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "G"
        else:
            return "J"

    elif input_char == "I":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "U"
        else:
            return "O"

    elif input_char == "J":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "H"
        else:
            return "K"

    elif input_char == "K":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "J"
        else:
            return "L"

    elif input_char == "L":
        return "K"

    elif input_char == "M":
        return "N"

    elif input_char == "N":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "B"
        else:
            return "M"

    elif input_char == "O":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "I"
        else:
            return "P"

    elif input_char == "P":
        return "O"

    elif input_char == "Q":
        return "W"

    elif input_char == "R":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "E"
        else:
            return "T"

    elif input_char == "S":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "A"
        else:
            return "D"

    elif input_char == "T":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "R"
        else:
            return "Y"

    elif input_char == "U":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "Y"
        else:
            return "I"

    elif input_char == "V":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "C"
        else:
            return "B"

    elif input_char == "W":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "Q"
        else:
            return "E"

    elif input_char == "X":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "Z"
        else:
            return "C"

    elif input_char == "Y":
        which_adjacent = return_random_number(1, 2)
        if which_adjacent == 1:
            return "T"
        else:
            return "U"

    elif input_char == "Z":
        return "X"

    else:
        return "*"


def perturb_char_Deletion(sample_text, PPS=1):
    if PPS <= 0:
        return [sample_text]
    cur_PPS = PPS - 1

    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    if True in [len(sample_tokenized[idx]) > 2 for idx in range(len(sample_tokenized))]:
        while random_word_selected != True:
            random_word_index = return_random_number(0, len(sample_tokenized) - 1)
            if len(sample_tokenized[random_word_index]) > 2:
                random_word_selected = True
    else:
        return [sample_text]
    # print('Selected random word:', sample_tokenized[random_word_index])

    # --------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    random_char_index = return_random_number(1, len(selected_word) - 2)
    # print('Random position:', random_char_index)
    # print('Character to delete:', selected_word[random_char_index])

    # --------------------------- delete the character

    temp_word = selected_word[:random_char_index]
    temp_word += selected_word[random_char_index + 1 :]

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    # print('After deletion:', perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "
    is_sample_perturbed = True

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_char_Deletion(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


def perturb_char_Insertion(sample_text, PPS=1):
    if PPS <= 0:
        return [sample_text]
    cur_PPS = PPS - 1
    is_sample_perturbed = False
    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    if True in [len(sample_tokenized[idx]) > 2 for idx in range(len(sample_tokenized))]:
        while random_word_selected != True:
            random_word_index = return_random_number(0, len(sample_tokenized) - 1)
            if len(sample_tokenized[random_word_index]) > 2:
                random_word_selected = True
    else:
        return [sample_text]
    # print('Selected random word:', sample_tokenized[random_word_index])

    # --------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    random_char_index = return_random_number(1, len(selected_word) - 2)
    # print('Random position:', random_char_index)

    # --------------------------- select a random character

    random_char_code = return_random_number(97, 122)
    # print('Random character:', chr(random_char_code))

    temp_word = selected_word[:random_char_index]
    temp_word += chr(random_char_code)
    temp_word += selected_word[random_char_index:]

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    # print('After insertion:', perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "
    is_sample_perturbed = True

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_char_Insertion(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


def perturb_char_LetterCaseChanging(sample_text, PPS=1):
    if PPS <= 0:
        return [sample_text]
    cur_PPS = PPS - 1
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    if True in [len(sample_tokenized[idx]) > 2 for idx in range(len(sample_tokenized))]:
        while random_word_selected != True:
            random_word_index = return_random_number(0, len(sample_tokenized) - 1)
            if len(sample_tokenized[random_word_index]) > 2:
                random_word_selected = True
    else:
        return [sample_text]

    # print('Selected random word:', sample_tokenized[random_word_index])

    # --------------------------- select the type of letter case changing

    selected_word = sample_tokenized[random_word_index]

    temp_word = ""

    change_type = random_changing_type()

    # --------------------------- change the letter case

    if change_type == "FirstChar":
        # print('Letter case changing: First character')
        if ord(selected_word[0]) >= 97 and ord(selected_word[0]) <= 122:
            temp_word = chr(ord(selected_word[0]) - 32)
            temp_word += selected_word[1:]
            is_sample_perturbed = True
        elif ord(selected_word[0]) >= 65 and ord(selected_word[0]) <= 90:
            temp_word = chr(ord(selected_word[0]) + 32)
            temp_word += selected_word[1:]
            is_sample_perturbed = True
        else:
            temp_word = selected_word

    elif change_type == "AllChars":
        # print('Letter case changing: All characters')
        for i in range(0, len(selected_word)):
            if ord(selected_word[i]) >= 97 and ord(selected_word[i]) <= 122:
                temp_word += chr(ord(selected_word[i]) - 32)
                is_sample_perturbed = True
            elif ord(selected_word[i]) >= 65 and ord(selected_word[i]) <= 90:
                temp_word += chr(ord(selected_word[i]) + 32)
                is_sample_perturbed = True
            else:
                temp_word += selected_word[i]

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    # print('After letter case changing:', perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_char_LetterCaseChanging(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


MISSPELLED_FILE_PATH = os.path.join(os.path.dirname(__file__), "MisspelledWords.tsv")


def perturb_char_MisspelledWords(sample_text, PPS=1):
    max_perturb = PPS
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)

    # --------------------------- search in the misspelled words corpus

    # corpus_address = (
    #    "/mnt3/dmendoza/mgit/utils/perturbations/MisspelledWords.tsv"
    # )

    corpus_address = MISSPELLED_FILE_PATH

    perturbed_sample = ""
    word_replaced = False

    possible_misspelling = []

    with open(corpus_address) as corpus_file:
        corpus_data = csv.reader(corpus_file, delimiter="\t")

        # ----- find all possible misspellings
        for entry in corpus_data:

            misspelling_position = sample_text.find(entry[0])

            if misspelling_position > -1:
                # print('Can be replaced:', entry[0], '----- Misspelling:', entry[1])
                possible_misspelling.append(entry)

        if len(possible_misspelling) > 0:

            # ----- create a list of unique words in the sample that can be replaced by a misspelling
            unique_words = []
            for i in range(0, len(possible_misspelling)):
                temp_row = possible_misspelling[i]
                if temp_row[0] not in unique_words:
                    unique_words.append(temp_row[0])

            # print('Unique words that can be replaced with a misspelling:', unique_words, '\n')
            num_unique_words = len(unique_words)

            # ----- randomly choose a misspelling and perturb the sample
            num_replacements = 0
            already_replaced = []

            perturbed_sample = sample_text

            while (
                num_replacements < max_perturb and num_replacements < num_unique_words
            ):

                random_index = return_random_number(0, len(possible_misspelling) - 1)
                temp_row = possible_misspelling[random_index]

                if temp_row[0] not in already_replaced:
                    # print(temp_row[0], 'is replaced with', temp_row[1])
                    misspelling_position = perturbed_sample.find(temp_row[0])

                    temp_text = perturbed_sample[0:misspelling_position]
                    temp_text += temp_row[1]
                    temp_text += perturbed_sample[
                        misspelling_position + len(temp_row[0]) :
                    ]

                    perturbed_sample = temp_text

                    already_replaced.append(temp_row[0])
                    word_replaced = True

                    num_replacements += 1

    if word_replaced == False:
        # print('No misspelled word was replaced')
        perturbed_sample = sample_text

    if word_replaced == True:
        is_sample_perturbed = True

    # print('Perturbed sample:', perturbed_sample)

    if is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


def perturb_char_Repetition(sample_text, PPS=1):
    if PPS <= 0:
        return [sample_text]
    cur_PPS = PPS - 1
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    if True in [len(sample_tokenized[idx]) > 2 for idx in range(len(sample_tokenized))]:
        while random_word_selected != True:
            random_word_index = return_random_number(0, len(sample_tokenized) - 1)
            if len(sample_tokenized[random_word_index]) > 2:
                random_word_selected = True
    else:
        return [sample_text]

    # print('Selected random word:', sample_tokenized[random_word_index])

    # --------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    random_char_index = return_random_number(1, len(selected_word) - 2)
    # print('Random position:', random_char_index)
    # print('Character to repeat:', selected_word[random_char_index])

    # --------------------------- repeat the character

    temp_word = selected_word[:random_char_index]
    temp_word += selected_word[random_char_index] + selected_word[random_char_index]
    temp_word += selected_word[random_char_index + 1 :]

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    # print('After repetition:', perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "
    is_sample_perturbed = True

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_char_Repetition(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


def perturb_char_Replacement(sample_text, PPS=1):
    if PPS <= 0:
        return [sample_text]
    cur_PPS = PPS - 1
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    if True in [len(sample_tokenized[idx]) > 2 for idx in range(len(sample_tokenized))]:
        while random_word_selected != True:
            random_word_index = return_random_number(0, len(sample_tokenized) - 1)
            if len(sample_tokenized[random_word_index]) > 2:
                random_word_selected = True
    else:
        return [sample_text]

    # print('Selected random word:', sample_tokenized[random_word_index])

    # --------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    char_is_letter = False
    tries_number = 0

    while char_is_letter != True and tries_number <= 20:
        random_char_index = return_random_number(1, len(selected_word) - 2)
        tries_number += 1
        if (
            ord(selected_word[random_char_index]) >= 97
            and ord(selected_word[random_char_index]) <= 122
        ) or (
            ord(selected_word[random_char_index]) >= 65
            and ord(selected_word[random_char_index]) <= 90
        ):
            char_is_letter = True
            is_sample_perturbed = True

    # print('Random position:', random_char_index)
    # print('Character to replace:', selected_word[random_char_index])

    # --------------------------- replace the character

    char_to_replace = selected_word[random_char_index]

    adjacent_char = return_adjacent_char(char_to_replace)

    # print('Adjacent character:', adjacent_char)

    temp_word = selected_word[:random_char_index]
    temp_word += adjacent_char
    temp_word += selected_word[random_char_index + 1 :]

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    # print('After replacement:', perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_char_Replacement(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


def perturb_char_Swapping(sample_text, PPS=1):
    if PPS <= 0:
        return [sample_text]
    cur_PPS = PPS - 1
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    if True in [len(sample_tokenized[idx]) > 2 for idx in range(len(sample_tokenized))]:
        while random_word_selected != True:
            random_word_index = return_random_number(0, len(sample_tokenized) - 1)
            if len(sample_tokenized[random_word_index]) > 2:
                random_word_selected = True
    else:
        return [sample_text]

    # print('Selected random word:', sample_tokenized[random_word_index])

    # --------------------------- select a random position

    selected_word = sample_tokenized[random_word_index]

    random_char_index = return_random_number(0, len(selected_word) - 1)
    # print('Random position:', random_char_index)
    # print('Char in random position:', selected_word[random_char_index])

    # --------------------------- select an adjacent for swapping

    adjacent_for_swapping = ""

    if random_char_index == 0:
        adjacent_for_swapping = "right"
    elif random_char_index == len(selected_word) - 1:
        adjacent_for_swapping = "left"
    else:
        adjacent = return_random_number(1, 2)
        if adjacent == 1:
            adjacent_for_swapping = "left"
        else:
            adjacent_for_swapping = "right"

    # print('Adjacent for swapping:', adjacent_for_swapping)

    # --------------------------- swap the character and the adjacent

    temp_word = swap_characters(selected_word, random_char_index, adjacent_for_swapping)

    perturbed_word = ""
    for i in range(0, len(temp_word)):
        perturbed_word += temp_word[i]

    # print('After swapping:', perturbed_word)

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += perturbed_word + " "
    is_sample_perturbed = True

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_char_Swapping(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]
