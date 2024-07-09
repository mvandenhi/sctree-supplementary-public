import logging
import os
import pathlib
import pickle
import time
import warnings

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import wandb
from omegaconf import DictConfig

from eval import compute_metrics
from models import run_model
from collections.abc import MutableMapping

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

_LOGGER = logging.getLogger(__name__)

_PHASE = "phase"
_G1 = "G1"


def load_data(config) -> sc.AnnData:
    _LOGGER.info("Loading data...")
    adata = sc.read_h5ad(config["data_path"])
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.strings_to_categoricals()

    if config.get("remove_cycling", False):
        adata = adata[adata.obs[_PHASE] == _G1].copy()
        _LOGGER.info("Removed cycling cells from the dataset.")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=config["n_hvgs"],
        subset=True,
        batch_key=config.get("batch_key", None),
    )
    _LOGGER.info(f"Subsetted data to {adata.n_vars} hvgs.")
    return adata


def reset_random_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    # Old # No determinism as nn.Upsample has no deterministic implementation
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    scvi.settings.seed = seed
    _LOGGER.info(f"Set random seed to {seed}")


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    wandb.init(
        project=cfg.logging.project,
        reinit=True,
        config=flatten(dict(cfg)),
        entity=cfg.logging.entity,
        mode=cfg.logging.mode,
        tags=cfg.model.get("tag", None),
    )

    results_dir = pathlib.Path(cfg["results_dir"])
    adata = load_data(cfg["data"])
    celltype_key = cfg["data"]["celltype_key"]
    counts_layer = cfg["data"]["counts_layer"]
    batck_key = cfg["data"].get("batch_key", None)
    n_cluster = (
        adata.obs[celltype_key].nunique()
        if (cfg.get("n_cluster", None) is None)
        else cfg["n_cluster"]
    )
    random_seed = cfg["random_seed"]
    reset_random_seeds(random_seed)
    start = time.time()
    _LOGGER.info("Start training model")
    labels, pruned_tree = run_model(
        adata,
        n_cluster=n_cluster,
        config=cfg["model"],
        batch_key=batck_key,
        counts_layer=counts_layer,
        celltype_key=celltype_key,
        random_seed=random_seed,
    )
    run_time = time.time() - start
    _LOGGER.info("Computing metrics.")

    results = compute_metrics(
        adata, labels, pruned_tree, celltype_key, batck_key, run_time
    )

    _LOGGER.info("logging results.")
    _LOGGER.info(results)
    wandb.log(results)
    pd.DataFrame(results, index=[0]).to_csv(
        results_dir.joinpath("results.csv"), index=None
    )
    np.save(results_dir.joinpath("labels.npy"), labels)
    pickle.dump(pruned_tree, results_dir.joinpath("tree.pkl").open("wb"), -1)


if __name__ == "__main__":
    main()
