#from lineage.graph import *
import numpy as np
import sys
from utils.perturbations import Character_Perturbations
from utils.perturbations import Word_Perturbations
#from checklist.perturb import Perturb

# perturb_args = {'perturb_fn': Word_Perturbations.perturb_word_Ordering, 'PPS':1}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_LetterCaseChanging, 'PPS':4}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_Deletion, 'PPS':4}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_Replacement, 'PPS':4}
# perturb_args = {'perturb_fn': Perturb.add_typos, 'typos':4}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_Swapping, 'PPS':4}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_MisspelledWords, 'PPS':4}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_Insertion, 'PPS':4}
# perturb_args = {'perturb_fn': Character_Perturbations.perturb_char_Repetition, 'PPS':4}
# perturb_keys = [perturb_sentence_key]


def perturb_word_order(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Word_Perturbations.perturb_word_Ordering,
        "PPS": 1,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


def perturb_char_lettercase(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_LetterCaseChanging,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


def perturb_char_delete(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_Deletion,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


def perturb_char_replace(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_Replacement,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


# def perturb_addtypos(dataset, feature_keys, **kwargs):
#    perturb_args = {"perturb_fn": Perturb.add_typos, "typos": 4}
#    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
#    return perturb_dataset


def perturb_char_swap(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_Swapping,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


def perturb_char_misspelledword(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_MisspelledWords,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


def perturb_char_insert(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_Insertion,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


def perturb_char_repetition(dataset, feature_keys, **kwargs):
    perturb_args = {
        "perturb_fn": Character_Perturbations.perturb_char_Repetition,
        "PPS": 4,
    }
    perturb_dataset = custom_perturb_wrapper(dataset, perturb_args, feature_keys)
    return perturb_dataset


"""eventually the following will become functions within mgit:"""


def custom_perturb_wrapper(cur_dataset, perturb_args, perturb_keys):
    custom_perturb_function = lambda examples: inner_custom_perturb_function(
        examples, perturb_args, perturb_keys
    )
    perturb_dataset = cur_dataset.map(
        custom_perturb_function, num_proc=16, batched=True
    )
    return perturb_dataset

def perturb_helper(in_str,perturb_fn,**kwargs):
    res = perturb_fn(in_str)
    return res

def inner_custom_perturb_function(examples, perturb_args, perturb_keys):
    #from checklist.perturb import Perturb
    import itertools

    new_examples = dict([(k, []) for k in examples])
    new_examples.update(dict([("org_" + k, []) for k in examples if k in perturb_keys]))
    for i in range(len(examples[perturb_keys[0]])):
        p_lists = []
        found_valid_perturb = False
        for p_key in perturb_keys:
            pdata = examples[p_key][i]
            if len(pdata) > 1:
                #expanded = Perturb.perturb(
                #    [pdata], **perturb_args, keep_original=False
                #).data
                expanded = perturb_helper(pdata,**perturb_args)
                cur_perturb_list = (
                    [entry for entry in expanded if entry != pdata]
                )
            else:
                cur_perturb_list = []
            if len(cur_perturb_list) > 0:
                p_lists.append(cur_perturb_list)
                found_valid_perturb = True
            else:
                p_lists.append([pdata])
        if found_valid_perturb:
            for entry in itertools.product(*p_lists):
                for p_it in range(len(entry)):
                    new_examples[perturb_keys[p_it]].append(entry[p_it])
                for k in examples:
                    if k not in perturb_keys:
                        new_examples[k].append(examples[k][i])
                    else:
                        new_examples["org_" + k].append(examples[k][i])
        else:
            for entry in itertools.product(*p_lists):
                for k in examples:
                    if k not in perturb_keys:
                        new_examples[k].append(examples[k][i])
                    else:
                        new_examples[k].append(examples[k][i])
                        new_examples["org_" + k].append(examples[k][i])
    return new_examples
