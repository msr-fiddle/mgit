import inspect
import re
import networkx as nx
import copy
import torch

from .fx import *
from deepdiff import DeepDiff
from termcolor import colored
from pyvis.network import Network

from collections import defaultdict, OrderedDict, deque

__all__ = [
    "start",
    "end",
    "diff_terminal",
    "diff_graph",
    "module_diff",
    "find_submodules",
    "find_submodules_map",
    "find_basemodules",
    "find_subnodes",
    "bfs_iterative",
    "module_hash",
    "module_diff_contextual",
    "supported_hash_mode",
]

start = "input"
end = "output"
supported_hash_mode = ["structural", "contextual"]


class DFSNode(object):
    def __init__(self, name="", prev=""):
        self.name = name
        self.prev = prev


def module_hash(moduleinfo, mode="structural"):
    assert isinstance(
        moduleinfo, ModuleInfo
    ), "Must be of ModuleInfo type to use module content-based hashing"
    assert mode in supported_hash_mode, "hash type must be one of {}".format(
        supported_hash_mode
    )

    if mode == "structural":
        return moduleinfo.class_hash()
    else:
        return hash(moduleinfo)


# TODO: Need unit test
def bfs_iterative(graph, end, namespace, hash_mode="structural"):
    # all_node records all visited nodes, nn_graph records _from node as key and list of _to nodes as value
    queue, all_nodes, nn_graph = (
        deque([DFSNode(end, end)]),
        defaultdict(list),
        defaultdict(list),
    )
    while queue:
        vertex = queue.popleft()

        for n in graph[vertex.name]:
            if ("type" in n and ("embedding" not in n and "type_as" not in n and
                                 "einsum" not in vertex.name) ) or 'getattr' in n or 'size' in n:
                #We don't trace tensor.type, tensor.device and tensor.size as there is no dataflow
                continue

            if namespace.get(vertex.name):
                _to = vertex.name
            else:
                _to = vertex.prev

            if n in namespace or n == end:

                if nn_graph.get(n) is None or _to not in nn_graph.get(n):
                    nn_graph[n].append(_to)

            if all_nodes.get(n) is None or _to not in all_nodes.get(n):
                # prev field will always be node in namespace

                queue.append(DFSNode(n, _to))

                all_nodes[n].append(_to)

    assert len(nn_graph) == len(
        namespace
    ), "node is in namespace/nn graph but not found in nn graph/namespace"

    # update graph and ordered namespace with input/output nodes
    _namespace = OrderedDict()
    _namespace[start] = ModuleInfo(start, matched=True)
    for k, v in namespace.items():
        _namespace[k] = copy.deepcopy(v)
    _namespace[end] = ModuleInfo(end)

    node_index = {node: index for index, node in enumerate(_namespace.keys())}

    def execution_order(node):
        if node != end:
            return node_index[node]
        else:
            return len(node_index)

    for _from, _to in nn_graph.items():
        # sort _to node list in execution order
        _to.sort(key=execution_order)
        nn_graph[_from] = _to

    _to = set()
    for _to_list in nn_graph.values():
        _to.update(_to_list)

    edge_hash, node_hash = defaultdict(list), defaultdict(list)

    # find all start nodes and assign them a sudo input parent
    for start_node in set(nn_graph.keys()).difference(_to):
        _namespace[start_node].start = True

    for node in _namespace.keys():
        if node != start:
            node_hash[module_hash(_namespace[node], hash_mode)].append(node)

            for _to in nn_graph[node]:
                edge_hash[
                    (
                        module_hash(_namespace[node], hash_mode),
                        module_hash(_namespace[_to], hash_mode),
                    )
                ].append((node, _to))

    return nn_graph, _namespace, node_hash, edge_hash


def find_subnodes(reversed_exe_graph, namespace, mode="structural", used_names=False):
    _namespace = {}
    _reversed_exe_graph = {}
    if used_names:
        for name, moduleinfo in namespace.items():
            _namespace["_" + name] = copy.deepcopy(moduleinfo)
        for name, name_list in reversed_exe_graph.items():
            if name != end:
                _reversed_exe_graph["_" + name] = ["_" + _name for _name in name_list]
            else:
                _reversed_exe_graph[name] = ["_" + _name for _name in name_list]
    else:
        _namespace = {k: copy.deepcopy(v) for k, v in namespace.items()}
        _reversed_exe_graph = {k: v for k, v in reversed_exe_graph.items()}

    nn_exe_graph, _namespace, n, e = bfs_iterative(
        _reversed_exe_graph, end, _namespace, mode
    )
    return nn_exe_graph, _namespace, n, e


def find_submodules(
    module,
    mode="structural",
    input_names=None,
    used_names=None,
    tracing_module_pool=None,
):
    """
    Find all children modules in execution order
    :param tracing_module_pool:
    :param used_names:
    :param mode:
    :param input_names: list of input names
    :param module:the root module
    :return:list of ModuleInfo, a reversed execution graph
    """
    if hasattr(module, 'device'):
        device = module.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to("cpu")
    if input_names is None:
        if "mpt" in str(type(module)):
            input_names = ["input_ids"]
        elif "t5" in str(type(module)):
            input_names = ["input_ids", "decoder_input_ids"]
        else:
            input_names = ["input_ids", "attention_mask"]
    input_names = input_names
    sig = inspect.signature(module.forward)
    concrete_args = {
        p.name: None for p in sig.parameters.values() if p.name not in input_names
    }
    tracer = ModulePathTracer(used_names, tracing_module_pool)
    traced = tracer.trace(module, concrete_args)
    reversed_exe_graph = {
        node.name: [_from.name for _from in node._input_nodes.keys()]
        for node in traced.nodes
    }
    nn_exe_graph, namespace, n, e = bfs_iterative(
        reversed_exe_graph, end, tracer.get_submodule_info(), mode
    )
    module = module.to(device)
    return tracer.namespace.used_names(), nn_exe_graph, namespace, n, e



def find_submodules_map(
    module,
    mode="structural",
    input_names=None,
    used_names=None,
    tracing_module_pool=None,
):
    if hasattr(module, 'device'):
        device = module.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to("cpu")
    if input_names is None:
        if "mpt" in str(type(module)):
            input_names = ["input_ids"]
        else:
            input_names = ["input_ids","attention_mask"]
    input_names = input_names
    sig = inspect.signature(module.forward)
    concrete_args = {
        p.name: None for p in sig.parameters.values() if p.name not in input_names
    }
    tracer = ModulePathTracer(used_names, tracing_module_pool)
    traced = tracer.trace(module, concrete_args)
    reversed_exe_graph = {
        node.name: [_from.name for _from in node._input_nodes.keys()]
        for node in traced.nodes
    }
    nn_exe_graph, namespace, n, e = bfs_iterative(
        reversed_exe_graph, end, tracer.get_submodule_info(), mode
    )
    module_name_map = tracer.get_module_name_map()
    module = module.to(device)
    return {name: module_name_map.get(name, tracer.namespace.used_names()[name]) for name in tracer.namespace.used_names().keys()}, nn_exe_graph, namespace, n, e


def find_basemodules(
    module,
    mode="structural",
    input_names=None,
    used_names=None,
    tracing_module_pool=None,
):
    assert tracing_module_pool is not None, "list of base_modules is required"
    if hasattr(module, 'device'):
        device = module.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to("cpu")
    if input_names is None:
        input_names = ["input_ids", "attention_mask"]
    input_names = input_names
    sig = inspect.signature(module.forward)
    concrete_args = {
        p.name: None for p in sig.parameters.values() if p.name not in input_names
    }
    tracer = ModulePathTracer(used_names, tracing_module_pool)
    tracer.trace(module, concrete_args)
    traced_modules = list(tracer.get_submodule_info().keys())
    last_base = 0
    for name in traced_modules:
        for base_module in tracing_module_pool:
            if base_module in name:
                last_base = traced_modules.index(name)
    # filter out the heads, we don't care about them
    heads = traced_modules[last_base:]
    tracer = ModulePathTracer(used_names)
    traced = tracer.trace(module, concrete_args)

    # record the names of the base modules, share_weight is only valid if they share weights among these modules
    base_modules = {
        name: moduleinfo
        for name, moduleinfo in tracer.get_submodule_info().items()
        if name not in heads
        and "act" not in moduleinfo.class_type.casefold()
        and "dropout" not in moduleinfo.class_type.casefold()
    }
    reversed_exe_graph = {
        node.name: [_from.name for _from in node._input_nodes.keys()]
        for node in traced.nodes
    }
    nn_exe_graph, namespace, n, e = bfs_iterative(reversed_exe_graph, end, base_modules, mode)
    module = module.to(device)
    return tracer.namespace.used_names(), nn_exe_graph, namespace, n, e


# TODO: Need unit test
def module_diff(namespaces, nodes, edges):
    """
    :param namespaces: list of two OrderedDict, key is node name, value is moduleinfo, nodes are topological ordered
    :param nodes: list of two dict, key is hash, value is a list of nodes, nodes are topological ordered
    :param edges: list of two dict, key is hash, value is a list of edges, edges are topological ordered
    :return: list of tuple indexing matched node pairs, lists of names of node/edge to be added/deleted
    """
    # 1. Make copies of variables that will be changed later

    # copy since we don't want to alter the original ns
    namespace1 = OrderedDict(
        {name: copy.deepcopy(submodule) for name, submodule in namespaces[0].items()}
    )
    namespace2 = OrderedDict(
        {name: copy.deepcopy(submodule) for name, submodule in namespaces[1].items()}
    )
    # indexing node in node_matches
    submodules1 = {name: index for index, name in enumerate(namespace1.keys())}
    submodules2 = {name: index for index, name in enumerate(namespace2.keys())}
    # deepcopy as we don't want to alter the original edges
    (e1, e2) = (copy.deepcopy(edges[0]), copy.deepcopy(edges[1]))

    # 2. Order edge hashtable of graph1 by looking at the min length between two lists of edges,
    # corresponding to this hash in graph1 and graph2, e.g. if there is {1:[A-B, A-B], 2:[B-A,B-A]} in g1 and
    # {1:[A-B,A-B], 2:[B-A]} in g2 then the order of hashes in g1 is 1 followed by 2.
    # Tie should be broken by looking at the topological order of the first edge in these two lists.

    def order_func(hash_edges):
        if e2.get(hash_edges[0]):
            return tuple(
                [
                    min(len(hash_edges[1]), len(e2[hash_edges[0]])),
                    -1 * submodules1[hash_edges[1][0][0]],
                    -1 * submodules1[hash_edges[1][0][1]],
                ]
            )
        else:
            return tuple(
                [
                    min(len(hash_edges[1]), 0),
                    -1 * submodules1[hash_edges[1][0][0]],
                    -1 * submodules1[hash_edges[1][0][1]],
                ]
            )

    e1 = OrderedDict(
        {
            _hash: _edges
            for _hash, _edges in sorted(e1.items(), key=order_func, reverse=True)
        }
    )

    # 3. Iterate over edge hashtable of graph1, if hash exists in graph2, match each edge in two edge lists by
    # iterating over the list for graph1, picking candidates from the list for graph2 and greedily matching the first
    # appearing edge. Before deciding a matching, check nodes composing these edges and ony commit when corresponding
    # nodes have same matched status. Matching a node in graph1 with more than one node in graph2 is forbidden.

    node_matches = {0: 0}  # for input node
    matched_e1 = []  # edges matched in e1
    for _hash in e1:
        if _hash in e2:
            # candidates in topological ordering
            candidates = defaultdict(deque)
            for e in e2[_hash]:
                (source2, sink2) = (namespace2[e[0]], namespace2[e[1]])
                candidates[str(source2.matched) + str(sink2.matched)].append(e)

            for e in e1[_hash]:
                (source1, sink1) = (namespace1[e[0]], namespace1[e[1]])
                _type = str(source1.matched) + str(sink1.matched)

                while len(candidates[_type]) > 0:  # if there are candidates left

                    matched = candidates[_type].popleft()

                    if not (
                        (
                            source1.matched
                            and node_matches[submodules1[e[0]]]
                            != submodules2[matched[0]]
                        )
                        or (
                            sink1.matched
                            and node_matches[submodules1[e[1]]]
                            != submodules2[matched[1]]
                        )
                    ):
                        e2[_hash].remove(matched)  # remove matched edge
                        node_matches.update(
                            [
                                (submodules1[e[0]], submodules2[matched[0]]),
                                (submodules1[e[1]], submodules2[matched[1]]),
                            ]
                        )
                        matched_e1.append(e)

                        # update status of nodes appeared in the matched edges
                        namespace1[e[0]].update_matched(True)
                        namespace1[e[1]].update_matched(True)
                        namespace2[matched[0]].update_matched(True)
                        namespace2[matched[1]].update_matched(True)
                        break

                # reconstruct candidates
                candidates = defaultdict(deque)
                for _e in e2[_hash]:
                    (source2, sink2) = (namespace2[_e[0]], namespace2[_e[1]])
                    candidates[str(source2.matched) + str(sink2.matched)].append(_e)

    # add_edges are the unmatched edges in e2
    add_edges = []
    for _, e in e2.items():
        add_edges += e
    edges_e1 = []

    # del_edges are the unmatched edges in e1
    for _, e in e1.items():
        edges_e1 += e
    del_edges = list(set(edges_e1).difference(set(matched_e1)))

    # 4. While matching edges, nodes are also matched but there can be nodes who don't share any common source and sink
    # but still match. Greedily match these nodes in topological order.

    del_nodes = [
        name for name, submodule in namespace1.items() if not submodule.matched
    ]
    add_nodes = [
        name for name, submodule in namespace2.items() if not submodule.matched
    ]
    node_matches = [(index1, index2) for index1, index2 in node_matches.items()]

    n1 = {}
    for _hash, _nodes in nodes[0].items():
        for node in _nodes:
            n1[node] = _hash
    n2 = {}
    for _hash, _nodes in nodes[1].items():
        for node in _nodes:
            n2[node] = _hash

    _n1 = defaultdict(list)
    for n in del_nodes:
        _n1[n1[n]].append(n)
    _n2 = defaultdict(list)
    for n in add_nodes:
        _n2[n2[n]].append(n)

    del_nodes = []
    for n in _n1.keys():
        if n in _n2:
            if len(_n1[n]) > len(_n2[n]):
                node_matches += [
                    (submodules1[n_pair[0]], submodules2[n_pair[1]])
                    for n_pair in zip(_n1[n][: len(_n2[n])], _n2[n])
                ]
                del_nodes += _n1[n][-(len(_n1[n]) - len(_n2[n])) :]
            else:
                node_matches += [
                    (submodules1[n_pair[0]], submodules2[n_pair[1]])
                    for n_pair in zip(_n1[n], _n2[n][: len(_n1[n])])
                ]
        else:
            del_nodes += _n1[n]

    add_nodes = []
    for n in _n2.keys():
        if n not in _n1:
            add_nodes += _n2[n]
        else:
            if len(_n2[n]) > len(_n1[n]):
                add_nodes += _n2[n][-(len(_n2[n]) - len(_n1[n])) :]

    # 5. a node in graph1 can't be matched with a node in graph2 if one of its preceding node in graph1 has been matched
    # with a node that appears later than the suggested matching in graph2. Filter out these illegal inverse matches of
    # node and edges containing such node.

    edges_e2 = []
    for _, edge in edges[1].items():
        edges_e2 += edge
    _node_matches = []
    prev = -1
    node_matches = sorted(node_matches)
    for i in range(len(node_matches)):
        if prev < node_matches[i][1]:
            _node_matches.append(node_matches[i])
            prev = node_matches[i][1]
        else:
            del_nodes.append(list(namespace1.keys())[node_matches[i][0]])
            for edge in edges_e1:
                if (
                    list(namespace1.keys())[node_matches[i][0]] in edge
                    and edge not in del_edges
                ):
                    del_edges.append(edge)

            add_nodes.append(list(namespace2.keys())[node_matches[i][1]])
            for edge in edges_e2:
                if (
                    list(namespace2.keys())[node_matches[i][1]] in edge
                    and edge not in add_edges
                ):
                    add_edges.append(edge)

    # 6. handle input node separately
    node_matches1 = {match[0]: match[1] for match in _node_matches}
    node_matches2 = {match[1]: match[0] for match in _node_matches}
    start_nodes1 = [
        submodules1[name] for name, submodule in namespace1.items() if submodule.start
    ]
    start_nodes2 = [
        submodules2[name] for name, submodule in namespace2.items() if submodule.start
    ]
    for start_node in start_nodes1:
        if node_matches1.get(start_node):
            if node_matches1[start_node] not in start_nodes2:
                del_edges.append((start, list(namespace1.keys())[start_node]))
        else:
            del_edges.append((start, list(namespace1.keys())[start_node]))

    for start_node in start_nodes2:
        if node_matches2.get(start_node):
            if node_matches2[start_node] not in start_nodes1:
                add_edges.append((start, list(namespace2.keys())[start_node]))
        else:
            add_edges.append((start, list(namespace2.keys())[start_node]))

    return _node_matches, add_nodes, del_nodes, add_edges, del_edges


def module_diff_contextual(
    namespaces,
    node_matches_structural,
    edges,
    add_nodes,
    del_nodes,
    add_edges,
    del_edges,
):
    (namespace1, namespace2) = namespaces
    submodules1 = list(namespace1.keys())
    submodules2 = list(namespace2.keys())
    e1 = []
    for _, e in edges[0].items():
        e1 += e
    e2 = []
    for _, e in edges[1].items():
        e2 += e

    node_matches = []
    for n_pair in node_matches_structural:
        if hash(namespace1[submodules1[n_pair[0]]]) != hash(
            namespace2[submodules2[n_pair[1]]]
        ):
            del_nodes.append(submodules1[n_pair[0]])
            add_nodes.append(submodules2[n_pair[1]])
            del_edges += [
                e for e in e1 if submodules1[n_pair[0]] in e and e not in del_edges
            ]
            add_edges += [
                e for e in e2 if submodules2[n_pair[1]] in e and e not in add_edges
            ]
        else:
            node_matches.append(n_pair)

    # handle input node separately
    node_matches1 = {match[0]: match[1] for match in node_matches}
    node_matches2 = {match[1]: match[0] for match in node_matches}
    start_nodes1 = [
        submodules1.index(name)
        for name, submodule in namespace1.items()
        if submodule.start
    ]
    start_nodes2 = [
        submodules2.index(name)
        for name, submodule in namespace2.items()
        if submodule.start
    ]
    for start_node in start_nodes1:
        if node_matches1.get(start_node):
            if node_matches1[start_node] not in start_nodes2:
                del_edges.append((start, submodules1[start_node]))
        else:
            del_edges.append((start, submodules1[start_node]))

    for start_node in start_nodes2:
        if node_matches2.get(start_node):
            if node_matches2[start_node] not in start_nodes1:
                add_edges.append((start, submodules2[start_node]))
        else:
            add_edges.append((start, submodules2[start_node]))

    return node_matches, add_nodes, del_nodes, add_edges, del_edges


# TODO: Need unit test
def diff_graph(
    namespaces,
    graphs,
    node_matches,
    add_nodes,
    del_nodes,
    add_edges,
    del_edges,
    save_path="output/example.html",
):
    (namespace1, namespace2) = namespaces
    (graph1, graph2) = graphs
    graph = nx.DiGraph(graph2)

    # handle input node separately
    graph.add_node(start)
    input_edges = []
    for name, submodule in namespace2.items():
        if submodule.start:
            input_edges.append((start, name))
    graph.add_edges_from(input_edges)

    graph.remove_nodes_from(add_nodes)
    graph.remove_edges_from(add_edges)
    net = Network(
        notebook=True,
        height="100%",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
    )
    net.from_nx(graph)
    for n in add_nodes:
        net.add_node(n, color="green", size=10)
    for e in add_edges:
        net.add_edge(source=e[0], to=e[1], color="green")

    for n in del_nodes:
        net.add_node(n, color="red", size=10)

    node_matches_ = {match[0]: match[1] for match in node_matches}
    for e in del_edges:
        if net.node_map.get(e[0]) and net.node_map.get(e[1]):
            net.add_edge(source=e[0], to=e[1], color="red")
        elif net.node_map.get(e[0]):
            e1 = list(namespace2.keys())[
                node_matches_[list(namespace1.keys()).index(e[1])]
            ]
            net.add_edge(source=e[0], to=e1, color="red")

        elif net.node_map.get(e[1]):
            e0 = list(namespace2.keys())[
                node_matches_[list(namespace1.keys()).index(e[0])]
            ]
            net.add_edge(source=e0, to=e[1], color="red")

        else:
            e0 = list(namespace2.keys())[
                node_matches_[list(namespace1.keys()).index(e[0])]
            ]
            e1 = list(namespace2.keys())[
                node_matches_[list(namespace1.keys()).index(e[1])]
            ]

            net.add_edge(source=e0, to=e1, color="red")

    net.save_graph(save_path)
    return net


# TODO: Need unit test
def diff_terminal(
    namespaces,
    node_matches,
    mode="structural",
    node_matches_structural=None,
    coarse_grained=False,
    mute=False,
):
    if mode != "structural":
        structural_match_m2 = {pair[1]: pair for pair in node_matches_structural}

    (namespace1, namespace2) = namespaces
    submodules1 = {
        index: submodule for index, submodule in enumerate(namespace1.keys())
    }
    submodules2 = {
        index: submodule for index, submodule in enumerate(namespace2.keys())
    }
    delta = []
    last_index = [node_matches[0][0], node_matches[0][1]]
    n1 = [node[0] for node in node_matches]
    n2 = [node[1] for node in node_matches]
    for index in node_matches:

        while last_index[0] < index[0] and last_index[0] not in n1:
            delta.append(
                "{0: <3}{1: <5}{2}{3}{4: ^3}{5}".format(
                    last_index[0],
                    " ",
                    "- ",
                    namespace1[submodules1[last_index[0]]],
                    ":",
                    submodules1[last_index[0]],
                )
            )
            last_index[0] += 1

        while last_index[1] < index[1] and last_index[1] not in n2:
            delta.append(
                "{1: <4}{0: <4}{2}{3}{4: ^3}{5}".format(
                    last_index[1],
                    " ",
                    "+ ",
                    namespace2[submodules2[last_index[1]]],
                    ":",
                    submodules2[last_index[1]],
                )
            )
            if mode != "structural":
                if last_index[1] in structural_match_m2:

                    pattern = "\['(.*)'\]"

                    nn_attr_hashes1 = namespace1[
                        submodules1[structural_match_m2[last_index[1]][0]]
                    ].nn_attr_hashes
                    nn_attr_hashes2 = namespace2[
                        submodules2[structural_match_m2[last_index[1]][1]]
                    ].nn_attr_hashes
                    if coarse_grained:
                        _nn_attr_hashes2 = {}
                        for name, hashes in nn_attr_hashes2.items():
                            if name not in nn_attr_hashes1:
                                _nn_attr_hashes2[name[1:]] = hashes
                            else:
                                _nn_attr_hashes2[name] = hashes
                        nn_attr_hashes2 = _nn_attr_hashes2
                    diff_ = list(
                        list(DeepDiff(nn_attr_hashes1, nn_attr_hashes2).values())[
                            0
                        ].keys()
                    )
                    diff = [re.search(pattern, str(item)).group(1) for item in diff_]
                    delta.append(
                        "{0: <7}{1: ^3}mis-match due to difference in hashes of {2}".format(
                            " ", "*", diff
                        )
                    )

            last_index[1] += 1

        if index[0] == len(namespace1) or index[1] == len(namespace2):
            continue

        if submodules1[index[0]] == submodules2[index[1]]:
            delta.append(
                "{0: <3}{1}{2: <3}{1: <3}{3}{4:^3}{5}".format(
                    index[0],
                    " ",
                    index[1],
                    namespace1[submodules1[index[0]]],
                    ":",
                    submodules1[index[0]],
                )
            )
        else:
            delta.append(
                "{0: <3}{1}{2: <3}{1: <3}{3}{4:^3}-{1}{5},{1}+{1}{6}".format(
                    index[0],
                    " ",
                    index[1],
                    namespace1[submodules1[index[0]]],
                    ":",
                    submodules1[index[0]],
                    submodules2[index[1]],
                )
            )

        last_index[0] += 1
        last_index[1] += 1

    if (not mute):
        for line in iter(delta):
            if line[8] == "-":
                print(colored(line, "red"))
            elif line[8] == "+" or line[8] == "*":
                print(colored(line, "green"))
            else:
                print(line)

    return delta
