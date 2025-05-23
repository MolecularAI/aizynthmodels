"""Module routines for evaluating multi-step retrosynthesis model"""

import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from scipy.spatial.distance import squareform

from aizynthmodels.route_distances.predict import main as distances_calculator
from aizynthmodels.template_based.utils import get_filename

# flake8: noqa: F401
from aizynthmodels.utils.configs.route_distances import predict
from aizynthmodels.utils.hydra import custom_config


def _create_config(model_path: str, templates_path: str, stock_path: str, properties: Dict[str, Any]) -> str:
    _, filename = tempfile.mkstemp(suffix=".yaml")
    dict_ = {
        "expansion": {"default": [model_path, templates_path]},
        "stock": {"default": stock_path},
        "search": properties,
    }
    with open(filename, "w") as fileobj:
        yaml.dump(dict_, fileobj)
    return filename


def _eval_finding(config: DictConfig, finder_config_path: str) -> Dict[str, List[Any]]:
    output_path = get_filename(config, "finder_output").replace(".json.gz", "_finding.json.gz")
    _run_finder(config.model_eval.target_smiles, finder_config_path, output_path, config)

    finder_output = pd.read_json(output_path, orient="table")
    stats = {
        "target": [str(x) for x in finder_output["target"].to_list()],
        "first solution time": [float(x) for x in finder_output["first_solution_time"].to_list()],
        "is solved": [bool(x) for x in finder_output["is_solved"].to_list()],
    }

    return stats


def _eval_recovery(
    config: DictConfig, route_distances_config: DictConfig, finder_config_path: str
) -> Dict[str, List[Any]]:
    with open(config.model_eval.reference_routes, "r") as fileobj:
        ref_trees = json.load(fileobj)
    smiles = [tree["smiles"] for tree in ref_trees]

    _, smiles_filename = tempfile.mkstemp(suffix=".txt")
    with open(smiles_filename, "w") as fileobj:
        fileobj.write("\n".join(smiles))

    output_path = get_filename(config, "finder_output").replace(".json.gz", "_recovery.json.gz")
    _run_finder(smiles_filename, finder_config_path, output_path, config)

    finder_output = pd.read_json(output_path, orient="table")
    route_distances_config.model_path = config.model_eval.distance_model
    route_distances_config.datamodule = {"type": "TreeListDataModule", "arguments": {}}
    route_distances_config.output_distances = False

    stats = defaultdict(list)
    for ref_tree, (_, row) in zip(ref_trees, finder_output.iterrows()):

        route_distances_config.datamodule["arguments"] = [{"route_list": [ref_tree] + list(row.trees)}]
        predictions = distances_calculator(route_distances_config)
        dists = squareform(predictions["predictions"])

        min_dists = float(min(dists[0, 1:]))
        stats["target"].append(ref_tree["smiles"])
        stats["is solved"].append(bool(row.is_solved))
        stats["found reference"].append(min_dists == 0.0)
        stats["closest to reference"].append(min_dists)
        stats["rank of closest"].append(float(np.argmin(dists)))

    return stats


def _run_finder(smiles_filename: str, finder_config_path: str, output_path: str, config: DictConfig) -> None:
    subprocess.run(
        [
            "conda",
            "run",
            "-p",
            config.model_eval.aizynthfinder_env,
            "aizynthcli",
            "--smiles",
            smiles_filename,
            "--config",
            finder_config_path,
            "--output",
            output_path,
        ]
    )


@hydra.main(version_base=None, config_name="expansion_pipeline")
@custom_config
def main(config: DictConfig) -> None:
    all_stats = {}
    route_distances_config = hydra.compose(config_name="predict")

    if config.model_eval.stock_for_finding and config.model_eval.target_smiles:
        if "model_path" in config and "templates_path" in config:
            finder_config_path = _create_config(
                config.model_path,
                config.templates_path,
                config.model_eval.stock_for_finding,
                dict(config.model_eval.search_properties_for_finding),
            )
        else:
            finder_config_path = _create_config(
                get_filename(config, "onnx_model"),
                get_filename(config, "unique_templates"),
                config.model_eval.stock_for_finding,
                dict(config.model_eval.search_properties_for_finding),
            )

        stats = _eval_finding(config, finder_config_path)
        all_stats["finding"] = stats

    if config.model_eval.stock_for_recovery and config.model_eval.reference_routes:
        if "model_path" in config and "templates_path" in config:
            finder_config_path = _create_config(
                config.model_path,
                config.templates_path,
                config.model_eval.stock_for_recovery,
                dict(config.model_eval.search_properties_for_recovery),
            )
        else:
            finder_config_path = _create_config(
                get_filename(config, "onnx_model"),
                get_filename(config, "unique_templates"),
                config.model_eval.stock_for_recovery,
                dict(config.model_eval.search_properties_for_finding),
            )

        stats = _eval_recovery(config, route_distances_config, finder_config_path)
        all_stats["recovery"] = stats

    with open(get_filename(config, "multistep_report"), "w") as fileobj:
        json.dump(all_stats, fileobj)

    if "finding" in all_stats:
        logging.info("\nEvaluation of route finding capabilities:")
        pd_stats = pd.DataFrame(all_stats["finding"])
        logging.info(f"Average first solution time: {pd_stats['first solution time'].mean():.2f}")
        logging.info(f"Average number of solved target: {pd_stats['is solved'].mean()*100:.2f}%")

    if "recovery" in all_stats:
        logging.info("\nEvaluation of route recovery capabilities:")
        pd_stats = pd.DataFrame(all_stats["recovery"])
        logging.info(f"Average number of solved target: {pd_stats['is solved'].mean()*100:.2f}%")
        logging.info(f"Average found reference: {pd_stats['found reference'].mean()*100:.2f}%")
        logging.info(f"Average closest to reference: {pd_stats['closest to reference'].mean():.2f}")
        logging.info(f"Average rank of closest: {pd_stats['rank of closest'].mean():.2f}")


if __name__ == "__main__":
    main()
