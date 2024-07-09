import logging

import numpy as np
import scanpy as sc
import scvi
import wandb
from sctree.models.sctree import scTree
from sklearn.cluster import AgglomerativeClustering

from eval import (
    to_dendrogram_purity_tree,
    prune_dendrogram_purity_tree,
    modeltree_to_dptree,
    tree_to_dptree,
)


_LOGGER = logging.getLogger(__name__)
_OFFSET = 100000
_NEW_BATCH_COL = "_ldvae_batch_col"


def run_model(
    adata,
    config,
    n_cluster,
    counts_layer,
    batch_key,
    random_seed: int,
    celltype_key: str,
):
    config = dict(config)
    model_name = config.pop("model")
    if model_name == "agg":
        return run_agg(adata, config, n_cluster=n_cluster, random_seed=random_seed)
    elif model_name == "scvi":
        return run_scvi(
            adata,
            config,
            n_cluster=n_cluster,
            counts_layer=counts_layer,
            batck_key=batch_key,
            random_seed=random_seed,
        )
    elif model_name == "ldvae":
        return run_ldvae(
            adata,
            config,
            n_cluster=n_cluster,
            counts_layer=counts_layer,
            batch_key=batch_key,
            random_seed=random_seed,
        )
    elif model_name == "sctree":
        return run_sctree(
            adata,
            config,
            n_cluster=n_cluster,
            counts_layer=counts_layer,
            batck_key=batch_key,
            celltype_key=celltype_key,
            random_seed=random_seed,
        )
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")


def run_agg(adata: sc.AnnData, config, n_cluster: int, random_seed: int):
    latent = sc.tl.pca(adata.X, n_comps=config["n_pcs"], random_state=random_seed)
    adata.obsm["X_pca"] = latent

    clusterer = AgglomerativeClustering(n_clusters=n_cluster)
    labels = clusterer.fit_predict(latent)
    tree = to_dendrogram_purity_tree(clusterer.children_)
    pruned_tree = prune_dendrogram_purity_tree(tree, n_cluster)

    sc.pp.neighbors(adata, use_rep="X_pca", random_state=random_seed)
    get_leiden_clustering(adata, n_cluster, random_seed)
    labels_leiden = adata.obs["leiden"].cat.codes
    tree_leiden = sc.tl.dendrogram(
        adata, groupby="leiden", use_rep="X_pca", inplace=False
    )
    dp_tree = tree_to_dptree(tree_leiden, labels_leiden)

    labels_dict = {"agg_labels": labels, "leiden_labels": labels_leiden}
    tree_dict = {"agg_tree": pruned_tree, "leiden_tree": dp_tree}

    return labels_dict, tree_dict


def run_scvi(
    adata: sc.AnnData,
    config,
    n_cluster: int,
    counts_layer: str,
    batck_key: str,
    random_seed: int,
):
    scvi.settings.seed = random_seed
    scvi.model.SCVI.setup_anndata(adata, layer=counts_layer, batch_key=batck_key)
    model = scvi.model.SCVI(adata, **config)
    model.train()
    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent

    # Agglomerative clustering
    clusterer = AgglomerativeClustering(n_clusters=n_cluster)
    labels = clusterer.fit_predict(latent)
    tree = to_dendrogram_purity_tree(clusterer.children_)
    pruned_tree = prune_dendrogram_purity_tree(tree, n_cluster)

    sc.pp.neighbors(adata, use_rep="X_scVI", random_state=random_seed)
    get_leiden_clustering(adata, n_cluster, random_seed)
    labels_leiden = adata.obs["leiden"].cat.codes
    tree_leiden = sc.tl.dendrogram(
        adata, groupby="leiden", use_rep="X_scVI", inplace=False
    )
    dp_tree = tree_to_dptree(tree_leiden, labels_leiden)

    labels_dict = {"agg_labels": labels, "leiden_labels": labels_leiden}
    tree_dict = {"agg_tree": pruned_tree, "leiden_tree": dp_tree}

    return labels_dict, tree_dict


def run_ldvae(
    adata: sc.AnnData,
    config,
    n_cluster: int,
    counts_layer: str,
    batch_key: str,
    random_seed: int,
):
    scvi.settings.seed = random_seed
    scvi.model.LinearSCVI.setup_anndata(adata, layer=counts_layer, batch_key=batch_key)
    model = scvi.model.LinearSCVI(adata, **config)
    model.train()
    latent = model.get_latent_representation()
    adata.obsm["X_ldvae"] = latent

    clusterer = AgglomerativeClustering(n_clusters=n_cluster)
    labels = clusterer.fit_predict(latent)
    tree = to_dendrogram_purity_tree(clusterer.children_)
    pruned_tree = prune_dendrogram_purity_tree(tree, n_cluster)

    sc.pp.neighbors(adata, use_rep="X_ldvae", random_state=random_seed)
    get_leiden_clustering(adata, n_cluster, random_seed)
    labels_leiden = adata.obs["leiden"].cat.codes
    tree_leiden = sc.tl.dendrogram(
        adata, groupby="leiden", use_rep="X_ldvae", inplace=False
    )
    dp_tree = tree_to_dptree(tree_leiden, labels_leiden)

    labels_dict = {"agg_labels": labels, "leiden_labels": labels_leiden}
    tree_dict = {"agg_tree": pruned_tree, "leiden_tree": dp_tree}

    return labels_dict, tree_dict


def run_sctree(
    adata: sc.AnnData,
    config,
    n_cluster: int,
    counts_layer: str,
    batck_key: str,
    celltype_key: str,
    random_seed: int,
):

    scvi.settings.seed = random_seed
    scTree.setup_anndata(
        adata, layer=counts_layer, batch_key=batck_key, labels_key=celltype_key
    )
    max_depth = int(np.ceil(np.log2(n_cluster))) + 3
    _LOGGER.info(f"Set max_dept to {max_depth}")
    training_config = config.pop("training")
    model = scTree(adata, n_cluster=n_cluster, max_depth=max_depth, **config)
    model.train(**training_config)
    labels, tree = model.get_tree(return_labels=True)
    tree = modeltree_to_dptree(model.module.tree, labels, n_cluster)
    return labels, tree


def get_leiden_clustering(
    adata,
    n_cluster,
    random_seed,
    start: float = 1e-4,
    end: float = 2.0,
    epsilon: float = 1e-8,
):
    for i in range(10):
        try:
            while end - start > epsilon:
                mid = (end + start) / 2.0
                sc.tl.leiden(
                    adata, resolution=mid, random_state=_OFFSET * i + random_seed
                )
                n_tmp = adata.obs["leiden"].nunique()
                if n_tmp == n_cluster:
                    _LOGGER.info(f"Found {n_tmp} cluster.")
                    break
                if n_tmp > n_cluster:
                    end = mid
                if n_tmp < n_cluster:
                    start = mid
            else:
                raise ValueError("Number of clusters doesn't match.")
        except ValueError:
            _LOGGER.info(
                f"Unsuccessful for random state {i}. Trying next random state."
            )
            pass
        else:
            break
