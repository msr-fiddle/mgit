from lineage.graph import *
from tabulate import tabulate


def print_result_table(all_nodes, all_tests, show_metrics, metric_list):
    show_results = set(metric_list)
    all_test_names = all_tests
    first_row = ["nodes/tests"] + [test_name for test_name in all_test_names]
    all_results = [first_row]
    for node in all_nodes:
        cur_node_results = [node.output_dir]
        for test_name in all_test_names:
            if test_name in node.test_result_dict:
                test_result = node.test_result_dict[test_name]
                if test_result["results"] is not None and show_results.intersection(
                    set(test_result["results"].keys())
                ):
                    cur_node_results.append(
                        dict(
                            (k, "%.3f" % v)
                            for k, v in test_result["results"].items()
                            if k in show_results
                        )
                    )
                elif test_result["results"] is not None and show_metrics is True:
                    cur_node_results.append("%.3f" % node.get_test_result(test_name))
                elif test_result["success"] is not None and show_metrics is False:
                    cur_node_results.append(test_result["success"])
                else:
                    cur_node_results.append("-")
            else:
                cur_node_results.append(None)
        all_results.append(cur_node_results)
    tab = tabulate(all_results, tablefmt="grid")
    print(tab)
    return tab

def show_result_table(
    graph,
    node_name_list=None,
    test_name_list=None,
    show_metrics=False,
    metric_list=[],
):
    if node_name_list is None:
        all_nodes = list(graph.nodes.values())
    else:
        all_nodes = [graph.nodes[node_name] for node_name in node_name_list]

    if test_name_list is None:
        all_tests = []
        for node in all_nodes:
            for test in node.test_name_list:
                all_tests.append(test)
        all_tests = set(all_tests)
    else:
        all_tests = set(test_name_list)
        nodes_to_remove = set()
        for test_name in all_tests:
            for node in all_nodes:
                if test_name not in node.test_name_list:
                    nodes_to_remove.add(node)
        for node in nodes_to_remove:
            all_nodes.remove(node)
    all_tests = list(all_tests)
    all_tests.sort()
    tab = print_result_table(
        all_nodes=all_nodes,
        all_tests=all_tests,
        show_metrics=show_metrics,
        metric_list=metric_list,
    )
    return tab


def get_chain(leaf_node, start_node=None, is_same_model_type=True):
    etype = "adapted"
    graph = leaf_node.graph
    node_chain = [leaf_node]
    cur_node = leaf_node
    if start_node is None:
        # get longest chain
        found_start = True
    else:
        found_start = False
    while len(cur_node.parent_dict[etype]) == 1:
        cur_node = graph.nodes[cur_node.parent_dict[etype][0]]
        if is_same_model_type and cur_node.model_type != leaf_node.model_type:
            break
        node_chain = [cur_node] + node_chain
        if start_node is not None and cur_node == start_node:
            found_start = True
            break
    if found_start is False:
        return None
    else:
        return node_chain


def bisect(test_name, leaf_node, start_node=None):
    node_chain = get_chain(leaf_node, start_node)
    assert node_chain is not None, "start and leaf node are not connected in a chain"
    print("found", len(node_chain), "total nodes in the bisect chain")
    print([node.output_dir for node in node_chain])
    if len(node_chain) == 1:
        return node_chain[0]
    test_success = node_chain[-1].run_test_by_name(test_name)
    node_chain[-1].unload_model(save_model=False)
    print(test_success)
    if test_success:
        print("leaf succeeds the test")
        return node_chain[-1]
    test_success = node_chain[0].run_test_by_name(test_name)
    node_chain[0].unload_model(save_model=False)
    if not test_success:
        print("root fails the test")
        return node_chain[0]
    low_idx = 0
    high_idx = len(node_chain) - 1
    while low_idx < high_idx:
        cur_idx = int((low_idx + high_idx) / 2)
        node = node_chain[cur_idx]
        test_success = node.run_test_by_name(test_name)
        node.unload_model(save_model=False)
        if test_success:
            high_idx = cur_idx - 1
        else:
            low_idx = cur_idx + 1
    cur_idx = int((low_idx + high_idx) / 2)
    return node_chain[cur_idx]
