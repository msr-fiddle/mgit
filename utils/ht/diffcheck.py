from .difflib import *

__all__ = ["diff_ht", "diff_ht_test", "match_ht"]


def match_ht(namespaces, nodes, edges, mode="structural"):
    assert mode in supported_hash_mode, "hash type must be one of {}".format(
        supported_hash_mode
    )

    # structural matches (partial equal on class type and layer attrs)
    node_matches, add_nodes, del_nodes, add_edges, del_edges = module_diff(
        namespaces, nodes, edges
    )

    if mode == "contextual":
        (
            node_matches_s,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
        ) = module_diff_contextual(
            namespaces,
            node_matches,
            edges,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
        )
        return [
            (list(namespaces[0].keys())[i], list(namespaces[1].keys())[j])
            for (i, j) in node_matches_s
        ]

    return [
        (list(namespaces[0].keys())[i], list(namespaces[1].keys())[j])
        for (i, j) in node_matches
    ]

def diff_ht_helper(
    namespaces,
    nodes,
    edges,
    graphs,
    save_path=None,
    mode="structural",
    coarse=False,
    mute_terminal=False
):
    # structural matches (partial equal on class type and layer attrs)
    node_matches, add_nodes, del_nodes, add_edges, del_edges = module_diff(
        namespaces, nodes, edges
    )

    if mode == "contextual":
        (
            node_matches_s,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
        ) = module_diff_contextual(
            namespaces,
            node_matches,
            edges,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
        )
        if save_path:
            diff_graph(
                namespaces,
                graphs,
                node_matches_s,
                add_nodes,
                del_nodes,
                add_edges,
                del_edges,
                save_path,
            )
            delta = diff_terminal(namespaces, node_matches_s, mode, node_matches, coarse, mute=mute_terminal)

    elif save_path:
        diff_graph(
            namespaces,
            graphs,
            node_matches,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
            save_path,
        )
        delta = diff_terminal(namespaces, node_matches, mute=mute_terminal)

    return add_nodes, del_nodes, add_edges, del_edges, delta
    
def diff_ht(
    loaded_models,
    save_path=None,
    mode="structural",
    coarse=False,
    tracing_module_pool=None,
    input_names=None,
    mute_terminal=False
):
    assert mode in supported_hash_mode, "hash type must be one of {}".format(
        supported_hash_mode
    )
    if coarse:
        assert (
            tracing_module_pool is not None
        ), "module to be traced must be specified if using coarse-grained tracing."

    if isinstance(input_names, tuple):
        (input_name1, input_name2) = input_names
    else:
        input_name1 = input_name2 = input_names

    used_names, graph1, namespace1, n1, e1 = find_submodules(
        loaded_models[0],
        input_names=input_name1,
        tracing_module_pool=tracing_module_pool,
    )
    _, graph2, namespace2, n2, e2 = find_submodules(
        loaded_models[1],
        used_names=used_names,
        input_names=input_name2,
        tracing_module_pool=tracing_module_pool,
    )

    namespaces = [namespace1, namespace2]
    nodes = [n1, n2]
    edges = [e1, e2]
    graphs = [graph1, graph2]

    return diff_ht_helper(
            namespaces,
            nodes,
            edges,
            graphs,
            save_path=save_path,
            mode=mode,
            coarse=coarse,
            mute_terminal=mute_terminal
        )


# This function is only used for unit tests. Compared with diff_ht, it doesn't extract nodes from a tracer
# but from a graph instead. The output contains additional information of the hash tables of nodes and edges.
def diff_ht_test(
    reversed_exe_graphs,
    namespaces,
    save_path="output/tests/example.html",
    mode="structural",
):
    assert mode in supported_hash_mode, "hash type must be one of {}".format(
        supported_hash_mode
    )
    (reversed_exe_graph1, reversed_exe_graph2) = reversed_exe_graphs
    (namespace1, namespace2) = namespaces

    graph1, _namespace1, n1, e1 = find_subnodes(reversed_exe_graph1, namespace1)
    graph2, _namespace2, n2, e2 = find_subnodes(
        reversed_exe_graph2, namespace2, used_names=True
    )

    namespaces = [_namespace1, _namespace2]
    nodes = [n1, n2]
    edges = [e1, e2]

    node_matches, add_nodes, del_nodes, add_edges, del_edges = module_diff(
        namespaces, nodes, edges
    )

    if mode == "contextual":
        graph1_s, _, n1_s, e1_s = find_subnodes(reversed_exe_graph1, namespace1, mode)
        graph2_s, _, n2_s, e2_s = find_subnodes(
            reversed_exe_graph2, namespace2, mode, used_names=True
        )
        nodes = [n1_s, n2_s]
        edges = [e1_s, e2_s]
        graphs = [graph1_s, graph2_s]

        # contextual matches (strict equal)
        (
            node_matches_s,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
        ) = module_diff(namespaces, nodes, edges)
        diff_graph(
            namespaces,
            graphs,
            node_matches_s,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
            save_path,
        )
        diff_terminal(namespaces, node_matches_s, mode, node_matches)
    else:
        graphs = [graph1, graph2]
        diff_graph(
            namespaces,
            graphs,
            node_matches,
            add_nodes,
            del_nodes,
            add_edges,
            del_edges,
            save_path,
        )
        diff_terminal(namespaces, node_matches)

    return nodes, edges, add_nodes, del_nodes, add_edges, del_edges
