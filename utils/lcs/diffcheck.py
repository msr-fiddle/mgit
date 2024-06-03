import inspect
import re
import torch
from .fx import *
from deepdiff import DeepDiff
from termcolor import colored

__all__ = [
    "find_submodules",
    "module_diff",
    "strict_equal",
    "partial_equal",
    "diff_lcs",
    "lcs_one",
]


def strict_equal(x, y):
    return x == y


def partial_equal(x, y):
    assert isinstance(x, ModuleInfo)
    assert isinstance(y, ModuleInfo)

    if x.anchor == y.anchor == start or x.anchor == y.anchor == end:
        return True

    if x.class_type == y.class_type and x.args == y.args:
        return True
    else:
        return False


def lcs_one(x, y, equal=partial_equal):
    """
    Dynamic Programming implementation of the Longest Common Subsequence problem
    :param equal: callable equality specification
    :param x: list of submodules
    :param y: list of submodules
    :return: list of common submodules
    """

    # find the length of the strings
    m = len(x)
    n = len(y)

    # declaring the array for storing the dp values
    l = [[0] * (n + 1) for _ in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                l[i][j] = 0
            elif equal(x[i - 1], y[j - 1]):
                l[i][j] = l[i - 1][j - 1] + 1
            else:
                l[i][j] = max(l[i - 1][j], l[i][j - 1])

    # l[m][n] contains the length of LCS of x[0..m-1] & y[0..n-1]

    len_lcs = l[m][n]

    if len_lcs == 0:
        return []

    lcs_index = [(0, 0)] * len_lcs

    i = m
    j = n

    while i > 0 and j > 0:

        if equal(x[i - 1], y[j - 1]):
            lcs_index[len_lcs - 1] = (i - 1, j - 1)
            i -= 1
            j -= 1
            len_lcs -= 1

        elif l[i - 1][j] > l[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs_index


def find_submodules(module, input_names=None):
    """
    Find all children modules in execution order
    :param input_names: list of input names
    :param module:the root module
    :return:list of ModuleInfo
    """
    if hasattr(module, 'device'):
        device = module.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if input_names is None:
        input_names = ["input_ids", "attention_mask"]
    module = module.to('cpu')
    input_names = input_names
    sig = inspect.signature(module.forward)
    concrete_args = {
        p.name: None for p in sig.parameters.values() if p.name not in input_names
    }
    tracer = ModulePathTracer()
    tracer.trace(module, concrete_args)
    module = module.to(device)
    return [ModuleInfo(anchor=start)] + tracer.submodule_info + [ModuleInfo(anchor=end)]


def module_diff(
    module_list1,
    module_list2,
    strict_equal=strict_equal,
    partial_equal=partial_equal,
):
    """ """
    strict_eq_index = lcs_one(module_list1, module_list2, strict_equal)
    partial_eq_index = {}
    last_eq_index = (0, 0)
    for index in strict_eq_index[1:]:
        if index[0] > last_eq_index[0] + 1 and index[1] > last_eq_index[1] + 1:
            neq_index1 = [i for i in range(last_eq_index[0] + 1, index[0])]
            neq_index2 = [i for i in range(last_eq_index[1] + 1, index[1])]
            module_list1_ = module_list1[last_eq_index[0] + 1 : index[0]]
            module_list2_ = module_list2[last_eq_index[1] + 1 : index[1]]
            partial_eq_index.update(
                {
                    neq_index2[index_[1]]: neq_index1[index_[0]]
                    for index_ in lcs_one(module_list1_, module_list2_, partial_equal)
                }
            )
        last_eq_index = index

    delta = []
    last_index = [0, 0]
    for index in strict_eq_index:

        while last_index[0] < index[0]:
            delta.append(
                "{0: <3}{1: <5}{2}{3}{4: ^3}{5}".format(
                    last_index[0],
                    " ",
                    "- ",
                    module_list1[last_index[0]],
                    ":",
                    module_list1[last_index[0]].qualified_path,
                )
            )
            last_index[0] += 1

        while last_index[1] < index[1]:
            delta.append(
                "{1: <4}{0: <4}{2}{3}{4: ^3}{5}".format(
                    last_index[1],
                    " ",
                    "+ ",
                    module_list2[last_index[1]],
                    ":",
                    module_list2[last_index[1]].qualified_path,
                )
            )
            if last_index[1] in partial_eq_index.keys():
                pattern = "\['(.*)'\]"
                diff_ = list(
                    list(
                        DeepDiff(
                            module_list1[
                                partial_eq_index.get(last_index[1])
                            ].nn_attr_hashes,
                            module_list2[last_index[1]].nn_attr_hashes,
                        ).values()
                    )[0].keys()
                )
                diff = [re.search(pattern, str(item)).group(1) for item in diff_]
                delta.append(
                    "{0: <7}{1: ^3}mis-match due to difference in hashes of {2}".format(
                        " ", "*", diff
                    )
                )

            last_index[1] += 1

        if (
            module_list1[index[0]].qualified_path
            == module_list2[index[1]].qualified_path
        ):
            delta.append(
                "{0: <3}{1}{2: <3}{1: <3}{3}{4:^3}{5}".format(
                    index[0],
                    " ",
                    index[1],
                    module_list1[index[0]],
                    ":",
                    module_list1[index[0]].qualified_path,
                )
            )
        else:
            delta.append(
                "{0: <3}{1}{2: <3}{1: <3}{3}{4:^3}-{1}{5},{1}+{1}{6}".format(
                    index[0],
                    " ",
                    index[1],
                    module_list1[index[0]],
                    ":",
                    module_list1[index[0]].qualified_path,
                    module_list2[index[1]].qualified_path,
                )
            )

        last_index[0] += 1
        last_index[1] += 1

    return delta[1:-1]


def diff_lcs(loaded_models):
    submodule_list = [
        find_submodules(loaded_models[0]),
        find_submodules(loaded_models[1]),
    ]

    mgit_message = module_diff(submodule_list[0], submodule_list[1])
    for line in iter(mgit_message):
        if line[8] == "-":
            print(colored(line, "red"))
        elif line[8] == "+" or line[8] == "*":
            print(colored(line, "green"))
        else:
            print(line)
    return mgit_message
