"""
Authors: Milad Moradi
NLP-perturbation: https://github.com/mmoradi-iut/NLP-perturbation
Paper: https://arxiv.org/ftp/arxiv/papers/2108/2108.12237.pdf
"""

import nltk
from random import seed
from random import randint
from nltk.stem.wordnet import WordNetLemmatizer
import mlconjug3
import numpy as np


def return_random_number(begin, end):
    # return randint(begin, end)
    return np.random.randint(begin, end + 1)


def change_ordering(input_length, input_side, input_changes):
    ordering = []

    if input_side == 1:
        for i in range(0, input_length):
            if i < input_changes:

                candidates = []
                for j in range(0, input_changes):
                    if j != i and j not in ordering:
                        candidates.append(j)

                if len(candidates) > 0:
                    random_index = return_random_number(0, len(candidates) - 1)
                    ordering.append(candidates[random_index])
                else:
                    ordering.append(i)
            else:
                ordering.append(i)

    elif input_side == 2:
        for i in range(0, input_length):
            if i < input_length - input_changes:
                ordering.append(i)

            else:
                candidates = []
                for j in range(input_length - input_changes, input_length):
                    if j != i and j not in ordering:
                        candidates.append(j)

                if len(candidates) > 0:
                    random_index = return_random_number(0, len(candidates) - 1)
                    ordering.append(candidates[random_index])
                else:
                    ordering.append(i)

    return ordering


def is_third_person(input_pos_tag):
    subject = ""
    for i in range(0, len(input_pos_tag)):
        token = input_pos_tag[i]
        if subject == "":
            if token[0].lower() in ("it", "this", "that", "he", "she"):
                subject = "third person"
            elif token[1] in ("NNP"):
                subject = "third person"
            elif token[0].lower() in (
                "i",
                "we",
                "you",
                "they",
                "she",
                "these",
                "those",
            ):
                subject = "not third person"
            elif token[0].lower() in ("NNPS"):
                subject = "not third person"
    if subject == "third person":
        return "third person"
    elif subject == "not third person":
        return "not third person"
    else:
        return "none"


# NOTE: PPS > 1 same result as PPS = 1
def perturb_word_Ordering(sample_text, PPS=1):
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)

    random_word_index = 0
    random_word_selected = False

    perturbed_sample = ""

    if len(sample_tokenized) > 3:
        # print('Sample can be perturbed.')

        last_token = ""
        if sample_tokenized[len(sample_tokenized) - 1] in (
            ".",
            "?",
            "!",
            ";",
            ",",
        ):
            last_token = sample_tokenized[len(sample_tokenized) - 1]
            sample_tokenized = sample_tokenized[0 : len(sample_tokenized) - 1]

        ordering_side = return_random_number(1, 2)

        # if (ordering_side == 1): #----- change word ordering in the beginning
        #    print('Change ordering side: Beginning')
        # elif (ordering_side == 2): #----- change word ordering in the end
        #    print('Change ordering side: End')

        num_changed_words = return_random_number(2, len(sample_tokenized) - 1)
        # print('Number of words for changing the order:', num_changed_words)

        new_word_order = change_ordering(
            len(sample_tokenized), ordering_side, num_changed_words
        )

        # print('New word order:', new_word_order)

        for i in range(0, len(new_word_order)):
            temp_index = new_word_order[i]
            perturbed_sample += sample_tokenized[temp_index] + " "
        perturbed_sample += last_token

        is_sample_perturbed = True

    else:
        perturbed_sample = sample_text

    # print('Perturbed sample:', perturbed_sample)

    if is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


def perturb_word_Repetition(sample_text, PPS=1):
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
            if len(sample_tokenized[random_word_index]) > 1:
                random_word_selected = True
    else:
        return [sample_text]

    # print('Selected random word:', sample_tokenized[random_word_index])

    selected_word = sample_tokenized[random_word_index]

    # --------------------------- reconstruct the perturbed sample

    perturbed_sample = ""

    for i in range(0, random_word_index):

        perturbed_sample += sample_tokenized[i] + " "

    perturbed_sample += selected_word + " " + selected_word + " "
    is_sample_perturbed = True

    for i in range(random_word_index + 1, len(sample_tokenized)):
        perturbed_sample += sample_tokenized[i] + " "

    # print('Perturbed sample:', perturbed_sample)

    if (is_sample_perturbed == True) and cur_PPS > 0:
        return perturb_word_Repetition(perturbed_sample, PPS=cur_PPS)
    elif is_sample_perturbed == True:
        return [perturbed_sample]
    else:
        return [sample_text]


# NOTE: PPS > 1 same result as PPS = 1
def perturb_word_SingularPluralVerb(sample_text, PPS=1):
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)
    sample_pos_tag = nltk.pos_tag(sample_tokenized)

    # print(sample_pos_tag)

    Perturbed_sample = ""

    remove_negation = False

    for i in range(0, len(sample_pos_tag)):
        token = sample_pos_tag[i]
        # print(token[0], token[1])
        if remove_negation == False:
            if (
                token[0] == "has"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][1] == "VBN"
            ):  # ----- third person singular present perfect
                verb = "have"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "have"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][1] == "VBN"
            ):  # ----- present perfect
                verb = "has"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "does"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, third person present simple
                verb = "do not"
                remove_negation = True
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "do"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, present simple
                verb = "does not"
                remove_negation = True
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "has"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, third person present perfect
                remove_negation = True
                Perturbed_sample += "have not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "have"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, present perfect
                remove_negation = True
                Perturbed_sample += "has not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "is"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "are not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "are"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "is not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "was"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "were not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "were"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "was not" + " "
                is_sample_perturbed = True

            elif token[0] == "does":  # ----- negative, third person present simple
                verb = "do"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif token[0] == "do":  # ----- negative, present simple
                verb = "does"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif (
                token[0] == "is"
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "are" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "are"
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "is" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "was"
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "were" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "were"
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "was" + " "
                is_sample_perturbed = True

            elif token[1] == "VBZ":  # ----- third person singular present
                verb = token[0]
                length = len(verb)
                if verb == "has":
                    verb = "have"
                elif verb[length - 3 :] == "oes":
                    verb = verb[: length - 2]
                elif verb[length - 4 :] == "ches":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "ses":
                    verb = verb[: length - 2]
                elif verb[length - 4 :] == "shes":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "xes":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "zes":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "ies":
                    verb = verb[: length - 3] + "y"
                else:
                    verb = verb[: length - 1]
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            elif token[1] == "VBP":  # ----- basic form present
                verb = token[0]
                length = len(verb)
                if verb == "have":
                    verb = "has"
                elif verb == "go":
                    verb = "goes"
                elif verb[length - 2 :] == "ch":
                    verb = verb + "es"
                elif verb[length - 1 :] == "s":
                    verb = verb + "es"
                elif verb[length - 2 :] == "sh":
                    verb = verb + "es"
                elif verb[length - 1 :] == "x":
                    verb = verb + "es"
                elif verb[length - 1 :] == "z":
                    verb = verb + "es"
                elif verb[length - 1 :] == "y":
                    verb = verb[: length - 1] + "ies"
                else:
                    verb = verb + "s"
                Perturbed_sample += verb + " "
                is_sample_perturbed = True

            else:
                Perturbed_sample += token[0] + " "

        elif remove_negation == True:
            if token[0] in (
                "not",
                "n't",
            ):  # ----- removing not after do or does
                Perturbed_sample += ""
                remove_negation = False

    # print('Perturbed sample:', Perturbed_sample)

    if is_sample_perturbed == True:
        return [Perturbed_sample]
    else:
        return [sample_text]


# NOTE: PPS > 1 same result as PPS = 1
def perturb_word_VerbTense(sample_text, PPS=1):
    is_sample_perturbed = False

    sample_tokenized = nltk.word_tokenize(sample_text)
    sample_pos_tag = nltk.pos_tag(sample_tokenized)

    # print(sample_pos_tag)

    Perturbed_sample = ""

    remove_negation = False
    can_change_basic_form = True

    for i in range(0, len(sample_pos_tag)):
        token = sample_pos_tag[i]
        # print(token[0], token[1])
        if remove_negation == False and can_change_basic_form == True:

            if (
                token[0] == "does"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, third person present simple
                remove_negation = True
                Perturbed_sample += "did not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "do"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, present simple
                remove_negation = True
                Perturbed_sample += "did not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "did"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, past simple
                if is_third_person(sample_pos_tag) == "third person":
                    remove_negation = True
                    can_change_basic_form = False
                    Perturbed_sample += "does not" + " "
                    is_sample_perturbed = True
                elif is_third_person(sample_pos_tag) == "not third person":
                    remove_negation = True
                    Perturbed_sample += "do not" + " "
                    is_sample_perturbed = True

            elif (
                token[0] in ("is", "am")
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "was not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "are"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "were not" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "was"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                if is_third_person(sample_pos_tag) == "third person":
                    remove_negation = True
                    Perturbed_sample += "is not" + " "
                    is_sample_perturbed = True
                elif is_third_person(sample_pos_tag) == "not third person":
                    remove_negation = True
                    Perturbed_sample += "am not" + " "
                    is_sample_perturbed = True

            elif (
                token[0] == "were"
                and i + 1 < len(sample_pos_tag)
                and sample_pos_tag[i + 1][0] in ("not", "n't")
            ):  # ----- negative, to be present and past, continuous present and past
                remove_negation = True
                Perturbed_sample += "are not" + " "
                is_sample_perturbed = True

            elif token[0] in (
                "is",
                "am",
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "was" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "are"
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "were" + " "
                is_sample_perturbed = True

            elif (
                token[0] == "was"
            ):  # ----- to be present and past, continuous present and past
                if is_third_person(sample_pos_tag) == "third person":
                    Perturbed_sample += "is" + " "
                    is_sample_perturbed = True
                elif is_third_person(sample_pos_tag) == "not third person":
                    Perturbed_sample += "am" + " "
                    is_sample_perturbed = True

            elif (
                token[0] == "were"
            ):  # ----- to be present and past, continuous present and past
                Perturbed_sample += "are" + " "
                is_sample_perturbed = True

            elif token[1] == "VBZ":  # ----- third person singular present
                verb = token[0]
                length = len(verb)
                if verb == "has":
                    verb = "have"
                elif verb[length - 3 :] == "oes":
                    verb = verb[: length - 2]
                elif verb[length - 4 :] == "ches":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "ses":
                    verb = verb[: length - 2]
                elif verb[length - 4 :] == "shes":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "xes":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "zes":
                    verb = verb[: length - 2]
                elif verb[length - 3 :] == "ies":
                    verb = verb[: length - 3] + "y"
                else:
                    verb = verb[: length - 1]

                past_tense = ""

                default_conjugator = mlconjug3.Conjugator(language="en")
                past_verb = default_conjugator.conjugate(verb)
                all_conjugates = past_verb.iterate()

                for j in range(0, len(all_conjugates)):
                    if all_conjugates[j][1] == "indicative past tense":
                        past_tense = all_conjugates[j][3]

                Perturbed_sample += past_tense + " "
                is_sample_perturbed = True

            elif token[1] == "VBP":  # ----- basic form present
                verb = token[0]

                past_tense = ""

                default_conjugator = mlconjug3.Conjugator(language="en")
                past_verb = default_conjugator.conjugate(verb)
                all_conjugates = past_verb.iterate()

                for j in range(0, len(all_conjugates)):
                    if all_conjugates[j][1] == "indicative past tense":
                        past_tense = all_conjugates[j][3]

                Perturbed_sample += past_tense + " "
                is_sample_perturbed = True

            elif token[1] == "VBD":  # ----- past
                if is_third_person(sample_pos_tag) == "third person":
                    verb = token[0]
                    verb = WordNetLemmatizer().lemmatize(verb, "v")

                    length = len(verb)
                    if verb == "have":
                        verb = "has"
                    elif verb == "go":
                        verb = "goes"
                    elif verb == "do":
                        verb = "does"
                    elif verb[length - 2 :] == "ch":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "s":
                        verb = verb + "es"
                    elif verb[length - 2 :] == "sh":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "x":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "z":
                        verb = verb + "es"
                    elif verb[length - 1 :] == "y":
                        verb = verb[: length - 1] + "ies"
                    else:
                        verb = verb + "s"

                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

                elif is_third_person(sample_pos_tag) == "not third person":
                    verb = token[0]
                    verb = WordNetLemmatizer().lemmatize(verb, "v")

                    Perturbed_sample += verb + " "
                    is_sample_perturbed = True

            else:
                Perturbed_sample += token[0] + " "

        elif remove_negation == True:
            if token[0] in (
                "not",
                "n't",
            ):  # ----- removing not after do or does
                Perturbed_sample += ""
                remove_negation = False

        elif can_change_basic_form == False:
            if token[1] == "VB":  # ----- do not change basic form
                verb = token[0]
                Perturbed_sample += verb + " "
                can_change_basic_form = True

    # print('Perturbed sample:', Perturbed_sample)

    if is_sample_perturbed == True:
        return [Perturbed_sample]
    else:
        return [sample_text]
