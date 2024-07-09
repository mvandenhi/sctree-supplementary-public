import typing
from typing import Optional

import attr
import numpy
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.special import comb
from sctree.utils.utils import count_values_in_sequence


def leaf_purity(tree_root, ground_truth):
    values = []
    weights = []

    def get_leaf_purities(node):
        nonlocal values
        nonlocal weights

        if node is None:
            return

        if node.is_leaf:
            node_total_dp_count = len(node.dp_ids)
            node_per_label_counts = count_values_in_sequence(
                [ground_truth[id] for id in node.dp_ids]
            )
            if node_total_dp_count > 0:
                purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
            else:
                purity_rate = 1.0
            values.append(purity_rate)
            weights.append(node_total_dp_count)
        else:
            get_leaf_purities(node.left_child)
            get_leaf_purities(node.right_child)

    get_leaf_purities(tree_root)

    return numpy.average(values, weights=weights)


def dendrogram_purity(tree_root, ground_truth):
    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {
        k: comb(v, 2, exact=True) for k, v in total_per_label_frequencies.items()
    }
    total_n_of_pairs = sum(total_per_label_pairs_count.values())

    one_div_total_n_of_pairs = 1.0 / total_n_of_pairs

    purity = 0.0

    def calculate_purity(node, level):
        nonlocal purity
        if node.is_leaf:
            node_total_dp_count = len(node.dp_ids)
            node_per_label_frequencies = count_values_in_sequence(
                [ground_truth[id] for id in node.dp_ids]
            )
            node_per_label_pairs_count = {
                k: comb(v, 2, exact=True) for k, v in node_per_label_frequencies.items()
            }

        elif node.left_child is None or node.right_child is None:
            # We are in an internal node with pruned leaves and thus only have one child. Therefore no prunity calculation here!
            node_left, node_right = node.left_child, node.right_child
            child = node_left if node_left is not None else node_right
            node_per_label_frequencies, node_total_dp_count = calculate_purity(
                child, level + 1
            )
            return node_per_label_frequencies, node_total_dp_count

        else:  # it is an inner node
            left_child_per_label_freq, left_child_total_dp_count = calculate_purity(
                node.left_child, level + 1
            )
            right_child_per_label_freq, right_child_total_dp_count = calculate_purity(
                node.right_child, level + 1
            )
            node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
            node_per_label_frequencies = {
                k: left_child_per_label_freq.get(k, 0)
                + right_child_per_label_freq.get(k, 0)
                for k in set(left_child_per_label_freq)
                | set(right_child_per_label_freq)
            }

            node_per_label_pairs_count = {
                k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k)
                for k in set(left_child_per_label_freq)
                & set(right_child_per_label_freq)
            }

        for label, pair_count in node_per_label_pairs_count.items():
            label_freq = node_per_label_frequencies[label]
            label_pairs = node_per_label_pairs_count[label]
            purity += (
                one_div_total_n_of_pairs
                * label_freq
                / node_total_dp_count
                * label_pairs
            )
        return node_per_label_frequencies, node_total_dp_count

    calculate_purity(tree_root, 0)
    return purity


def prune_dendrogram_purity_tree(tree, n_leaves):
    """
    This function collapses the tree such that it only has n_leaves.
    This makes it possible to compare different trees with different number of leaves.

    Important, it assumes that the node_id is equal to the split order, that means the tree root should have the smallest split number
    and the two leaf nodes that are splitted the last have the highest node id. And that  max(node_id) == #leaves - 2

    :param tree:
    :param n_levels:
    :return:
    """
    max_node_id = n_leaves - 2

    def recursive(node):
        if node.is_leaf:
            return node
        else:  # node is an inner node
            if node.node_id < max_node_id:
                left_child = recursive(node.left_child)
                right_child = recursive(node.right_child)
                return DpNode(left_child, right_child, node.node_id)
            else:
                work_list = [node.left_child, node.right_child]
                dp_ids = []
                while len(work_list) > 0:
                    nc = work_list.pop()
                    if nc.is_leaf:
                        dp_ids = dp_ids + nc.dp_ids
                    else:
                        work_list.append(nc.left_child)
                        work_list.append(nc.right_child)
                return DpLeaf(dp_ids, node.node_id)

    return recursive(tree)


def to_dendrogram_purity_tree(children_array):
    """
    Can convert the children_ matrix of a  sklearn.cluster.hierarchical.AgglomerativeClustering outcome to a dendrogram_purity tree
    :param children_array:  array-like, shape (n_samples-1, 2)
        The children of each non-leaf nodes. Values less than `n_samples`
            correspond to leaves of the tree which are the original samples.
            A node `i` greater than or equal to `n_samples` is a non-leaf
            node and has children `children_[i - n_samples]`. Alternatively
            at the i-th iteration, children[i][0] and children[i][1]
            are merged to form node `n_samples + i`
    :return:
    """
    n_samples = children_array.shape[0] + 1
    max_id = 2 * n_samples - 2
    node_map = {dp_id: DpLeaf([dp_id], max_id - dp_id) for dp_id in range(n_samples)}
    next_id = max_id - n_samples

    for idx in range(n_samples - 1):
        next_fusion = children_array[idx, :]
        child_a = node_map.pop(next_fusion[0])
        child_b = node_map.pop(next_fusion[1])
        node_map[n_samples + idx] = DpNode(child_a, child_b, next_id)
        next_id -= 1
    if len(node_map) != 1:
        raise RuntimeError(
            "tree must be fully developed! Use ompute_full_tree=True for AgglomerativeClustering"
        )
    root = node_map[n_samples + n_samples - 2]
    return root


@attr.define()
class DpNode(object):
    """
    node_id should be in such a way that a smaller number means split before a larger number in a top-down manner
    That is the root should have node_id = 0 and the children of the last split should have node id
    2*n_dps-2 and 2*n_dps-1

    """

    left_child: typing.Any = None
    right_child: typing.Any = None
    node_id: Optional[int] = None

    @property
    def children(self):
        return [self.left_child, self.right_child]

    @property
    def is_leaf(self):
        return False


@attr.s(cmp=False)
class DpLeaf(object):
    dp_ids = attr.ib()
    node_id = attr.ib()

    @property
    def children(self):
        return []

    @property
    def is_leaf(self):
        return True


def modeltree_to_dptree(tree, y_predicted, n_leaves):
    i = 0
    root = DpNode(node_id=i)
    list_nodes = [{"node": tree, "id": 0, "parent": None, "dpNode": root}]
    labels_leaf = [i for i in range(n_leaves)]
    while len(list_nodes) != 0:
        current_node = list_nodes.pop(0)
        if current_node["node"].router is not None:
            node_left, node_right = (
                current_node["node"].left,
                current_node["node"].right,
            )
            i += 1
            if node_left.decoder is not None:
                y_leaf = labels_leaf.pop(0)
                ind = np.where(y_predicted == y_leaf)[0]
                current_node["dpNode"].left_child = DpLeaf(node_id=i, dp_ids=ind)
            else:
                current_node["dpNode"].left_child = DpNode(node_id=i)
                list_nodes.append(
                    {
                        "node": node_left,
                        "id": i,
                        "parent": current_node["id"],
                        "dpNode": current_node["dpNode"].left_child,
                    }
                )
            i += 1
            if node_right.decoder is not None:
                y_leaf = labels_leaf.pop(0)
                ind = np.where(y_predicted == y_leaf)[0]
                current_node["dpNode"].right_child = DpLeaf(node_id=i, dp_ids=ind)
            else:
                current_node["dpNode"].right_child = DpNode(node_id=i)
                list_nodes.append(
                    {
                        "node": node_right,
                        "id": i,
                        "parent": current_node["id"],
                        "dpNode": current_node["dpNode"].right_child,
                    }
                )

        else:
            # We are in an internal node with pruned leaves and will only add the non-pruned leaves
            node_left, node_right = (
                current_node["node"].left,
                current_node["node"].right,
            )
            child = node_left if node_left is not None else node_right
            i += 1

            if node_left is not None:
                if node_left.decoder is not None:
                    y_leaf = labels_leaf.pop(0)
                    ind = np.where(y_predicted == y_leaf)[0]
                    current_node["dpNode"].left_child = DpLeaf(node_id=i, dp_ids=ind)
                else:
                    current_node["dpNode"].left_child = DpNode(node_id=i)
                    list_nodes.append(
                        {
                            "node": node_left,
                            "id": i,
                            "parent": current_node["id"],
                            "dpNode": current_node["dpNode"].left_child,
                        }
                    )
            else:
                if node_right.decoder is not None:
                    y_leaf = labels_leaf.pop(0)
                    ind = np.where(y_predicted == y_leaf)[0]
                    current_node["dpNode"].right_child = DpLeaf(node_id=i, dp_ids=ind)
                else:
                    current_node["dpNode"].right_child = DpNode(node_id=i)
                    list_nodes.append(
                        {
                            "node": node_right,
                            "id": i,
                            "parent": current_node["id"],
                            "dpNode": current_node["dpNode"].right_child,
                        }
                    )

    return root


def tree_to_dptree(tree, labels):
    """
    This function reformats the output from a scanpy.tl.dendrogram to a DPdendrogram.
    This makes it possible to compute Leaf and Dendrogram Purity.
    The trick is to create the dendrogram of only the clusters and afterwards, replace each cluster_id with the sample_ids that are in the respective cluster.

    :param tree:
    :param labels:
    :return:
    """

    dptree_clusters = to_dendrogram_purity_tree(tree["linkage"][:, :2])

    def recursive(node):
        if node.is_leaf:
            cluster_id = node.dp_ids[0]
            sample_ids = np.where(labels == cluster_id)[0]
            return DpLeaf(list(sample_ids), node.node_id)
        else:  # node is an inner node
            left_child = recursive(node.left_child)
            right_child = recursive(node.right_child)
            return DpNode(left_child, right_child, node.node_id)

    return recursive(dptree_clusters)


def compute_metrics(adata, labels, pruned_tree, celltype_key, batch_key, run_time):
    results = {}
    if type(labels) is dict:
        for name, agg_labels, tree in zip(
            labels.keys(), labels.values(), pruned_tree.values()
        ):
            name = name.split("_")[0]
            results.update(
                {
                    f"{name}/NMI": normalized_mutual_info_score(
                        agg_labels, adata.obs[celltype_key].values
                    ),
                    f"{name}/ARI": adjusted_rand_score(
                        agg_labels, adata.obs[celltype_key].values
                    ),
                    f"{name}/Dendrogram purity": dendrogram_purity(
                        tree, adata.obs[celltype_key].values
                    ),
                    f"{name}/Leaf purity": leaf_purity(
                        tree, adata.obs[celltype_key].values
                    ),
                }
            )
            if batch_key:
                results[f"{name}/NMI_batch"] = 1 - normalized_mutual_info_score(
                    agg_labels, adata.obs[batch_key].values
                )

    else:
        results.update(
            {
                "NMI": normalized_mutual_info_score(
                    labels, adata.obs[celltype_key].values
                ),
                "ARI": adjusted_rand_score(labels, adata.obs[celltype_key].values),
                "Dendrogram purity": dendrogram_purity(
                    pruned_tree, adata.obs[celltype_key].values
                ),
                "Leaf purity": leaf_purity(pruned_tree, adata.obs[celltype_key].values),
            }
        )
        if batch_key:
            results["NMI_batch"] = 1 - normalized_mutual_info_score(
                labels, adata.obs[batch_key].values
            )
    results["run_time"] = run_time

    return results
