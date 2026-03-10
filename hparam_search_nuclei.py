"""
Hyperparameter search for CellposeSAM nuclei finetuning.

Loads training/eval data once, then runs all hyperparameter combinations
sequentially. After each trial the result is written to disk so progress
is never lost on a crash. Prints a ranked summary table at the end.

Usage:
    python hparam_search_nuclei.py [--config configs/hparam_search_config.json] [--device cuda]
"""

import argparse
import copy
import itertools
import json
import logging
import os
import shutil
import time

import numpy as np
from cellpose import models, train

# Reuse helpers from the main finetuning script
from finetune_nuclei import (
    collect_image_mask_pairs,
    evaluate_model,
    evaluate_model_3d,
    get_device,
    load_and_slice_pairs,
    load_finetune_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METRIC_KEY_MAP = {
    "2d_ap_0.50": ("2d", 0.5),
    "2d_ap_0.75": ("2d", 0.75),
    "2d_ap_0.90": ("2d", 0.9),
    "3d_ap_0.50": ("3d", 0.5),
    "3d_ap_0.75": ("3d", 0.75),
    "3d_ap_0.90": ("3d", 0.9),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_search_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Search config not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_trials(search_space: dict) -> list[dict]:
    """Cartesian product of all search axes -> list of trial dicts."""
    keys = list(search_space.keys())
    values = [search_space[k] if isinstance(search_space[k], list) else [search_space[k]] for k in keys]
    trials = []
    for combo in itertools.product(*values):
        trials.append(dict(zip(keys, combo)))
    return trials


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    trial: dict,
    trial_idx: int,
    base_cfg: dict,
    train_images, train_masks,
    eval_images, eval_masks, eval_volumes,
    ap_before_2d: dict, ap_before_3d: dict,
    device,
    results_dir: str,
) -> dict:
    """
    Train and evaluate one hyperparameter configuration.
    Returns a result dict with all metrics.
    """
    model_cfg = base_cfg["model"]
    training_cfg = base_cfg["training"]

    use_bfloat16 = device.type == "cuda"
    freeze_encoder = trial.get("freeze_encoder", model_cfg.get("freeze_encoder", True))
    lr = trial.get("learning_rate", training_cfg["learning_rate"])
    wd = trial.get("weight_decay", training_cfg["weight_decay"])
    epochs = trial.get("n_epochs", training_cfg["n_epochs"])
    batch_size = trial.get("batch_size", training_cfg.get("batch_size", 8))

    trial_name = f"trial_{trial_idx:03d}"
    trial_save_dir = os.path.join(results_dir, "models")
    os.makedirs(trial_save_dir, exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"Trial {trial_idx}: {trial}")
    logging.info(f"{'='*60}")

    # Fresh pretrained model for every trial
    model = models.CellposeModel(
        pretrained_model=model_cfg["base_model"],
        device=device,
        use_bfloat16=use_bfloat16,
    )

    if freeze_encoder:
        for param in model.net.encoder.parameters():
            param.requires_grad = False
        logging.info("Encoder frozen")
    else:
        logging.info("Full model finetuning (encoder unfrozen)")

    nimg_per_epoch = min(800, max(8, len(train_images)))

    t0 = time.time()
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=eval_images if eval_images else None,
        test_labels=eval_masks if eval_images else None,
        n_epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        weight_decay=wd,
        normalize=True,
        save_path=trial_save_dir,
        save_every=epochs,  # only save at end
        save_each=False,
        min_train_masks=training_cfg["min_train_masks"],
        model_name=trial_name,
        scale_range=training_cfg["scale_range"],
        bsize=training_cfg["bsize"],
        nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=len(eval_images) if eval_images else None,
    )
    elapsed = time.time() - t0
    logging.info(f"Training done in {elapsed:.0f}s | final train loss: {train_losses[-1]:.4f}")

    # Evaluate
    finetuned = models.CellposeModel(
        pretrained_model=str(model_path), device=device, use_bfloat16=use_bfloat16,
    )

    ap_after_2d, ap_after_3d = {}, {}

    if eval_images:
        logging.info("Evaluating 2D...")
        ap_after_2d, _ = evaluate_model(finetuned, eval_images, eval_masks)

    if eval_volumes:
        logging.info("Evaluating 3D...")
        ap_after_3d, _ = evaluate_model_3d(finetuned, eval_volumes)

    # Log per-threshold deltas
    for label, ap_before, ap_after in [("2D", ap_before_2d, ap_after_2d), ("3D", ap_before_3d, ap_after_3d)]:
        if not ap_before or not ap_after:
            continue
        for th in sorted(ap_before):
            delta = ap_after[th] - ap_before[th]
            sign = "+" if delta >= 0 else ""
            logging.info(f"  [{label}] AP@{th:.2f}  {ap_before[th]:.4f} -> {ap_after[th]:.4f}  ({sign}{delta:.4f})")

    # Clean up trial model file to save disk space
    try:
        if os.path.isfile(model_path):
            os.remove(model_path)
    except OSError:
        pass

    return {
        "trial_idx": trial_idx,
        "hparams": trial,
        "train_loss_final": float(train_losses[-1]),
        "train_time_s": round(elapsed, 1),
        "ap_after_2d": {str(k): v for k, v in ap_after_2d.items()},
        "ap_after_3d": {str(k): v for k, v in ap_after_3d.items()},
        "ap_before_2d": {str(k): v for k, v in ap_before_2d.items()},
        "ap_before_3d": {str(k): v for k, v in ap_before_3d.items()},
    }


# ---------------------------------------------------------------------------
# Scoring / ranking
# ---------------------------------------------------------------------------

def score_result(result: dict, primary_metric: str) -> float:
    """
    Extract the primary metric value from a trial result.
    primary_metric format: '2d_ap_0.75' or '3d_ap_0.50', etc.
    """
    dim, threshold = METRIC_KEY_MAP.get(primary_metric, ("2d", 0.75))
    ap_key = f"ap_after_{dim}"
    ap_dict = result.get(ap_key, {})
    # JSON keys are strings
    return ap_dict.get(str(threshold), ap_dict.get(threshold, 0.0))


def print_summary(results: list[dict], primary_metric: str):
    """Print a ranked table of all trial results."""
    ranked = sorted(results, key=lambda r: score_result(r, primary_metric), reverse=True)

    header = (
        f"\n{'='*100}\n"
        f"HYPERPARAMETER SEARCH RESULTS  (ranked by {primary_metric})\n"
        f"{'='*100}\n"
        f"{'Rank':>4}  {'Trial':>5}  {'Frz':>3}  {'LR':>8}  {'WD':>8}  {'Epochs':>6}  {'BS':>3}  "
        f"{'2D@.50':>7}  {'2D@.75':>7}  {'2D@.90':>7}  "
        f"{'3D@.50':>7}  {'3D@.75':>7}  {'3D@.90':>7}  {'Loss':>7}"
    )
    logging.info(header)
    logging.info("-" * 100)

    for rank, r in enumerate(ranked, 1):
        hp = r["hparams"]
        a2 = r.get("ap_after_2d", {})
        a3 = r.get("ap_after_3d", {})
        frz = "Y" if hp.get("freeze_encoder", True) else "N"

        def g(d, k):
            return f"{d.get(str(k), d.get(k, float('nan'))):.4f}"

        row = (
            f"{rank:>4}  {r['trial_idx']:>5}  {frz:>3}  "
            f"{hp.get('learning_rate', 0):>8.0e}  {hp.get('weight_decay', 0):>8.0e}  "
            f"{hp.get('n_epochs', 0):>6}  {hp.get('batch_size', 8):>3}  "
            f"{g(a2, 0.5):>7}  {g(a2, 0.75):>7}  {g(a2, 0.9):>7}  "
            f"{g(a3, 0.5):>7}  {g(a3, 0.75):>7}  {g(a3, 0.9):>7}  "
            f"{r.get('train_loss_final', float('nan')):>7.4f}"
        )
        logging.info(row)

    logging.info("=" * 100)

    best = ranked[0]
    logging.info(f"\nBest trial #{best['trial_idx']} ({primary_metric} = {score_result(best, primary_metric):.4f}):")
    for k, v in best["hparams"].items():
        logging.info(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter search for CellposeSAM nuclei finetuning")
    parser.add_argument("--config", default="configs/hparam_search_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip trials whose results already exist in results.json",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = get_device() if args.device is None else __import__("torch").device(args.device)
    logging.info(f"Using device: {device}")

    search_cfg = load_search_config(args.config)
    base_cfg = load_finetune_config(search_cfg["base_config"])
    primary_metric = search_cfg.get("primary_metric", "2d_ap_0.75")

    results_dir = search_cfg["output"]["results_dir"]
    results_json = search_cfg["output"]["results_json"]
    os.makedirs(results_dir, exist_ok=True)

    # Load any previously completed trials
    completed_results: list[dict] = []
    completed_indices: set[int] = set()
    if args.resume and os.path.isfile(results_json):
        with open(results_json) as f:
            completed_results = json.load(f)
        completed_indices = {r["trial_idx"] for r in completed_results}
        logging.info(f"Resuming: {len(completed_indices)} trials already done")

    # Build trial grid
    trials = build_trials(search_cfg["search"])
    logging.info(f"Search space: {len(trials)} total trials")

    # ------------------------------------------------------------------
    # Load data once (shared across all trials)
    # ------------------------------------------------------------------
    data_cfg = base_cfg["data"]
    training_cfg = base_cfg["training"]

    axis_map = {"Z": 0, "Y": 1, "X": 2}
    slice_axes = tuple(axis_map[a.upper()] for a in training_cfg.get("slice_axes", ["Z"]))

    logging.info("Collecting training pairs...")
    train_pairs = collect_image_mask_pairs(data_cfg["train_image_dirs"], data_cfg["gt_mapping"])
    logging.info(f"Found {len(train_pairs)} training pairs")

    logging.info("Collecting eval pairs...")
    eval_pairs = collect_image_mask_pairs(data_cfg["eval_image_dirs"], data_cfg["gt_mapping"])
    logging.info(f"Found {len(eval_pairs)} eval pairs")

    logging.info("Loading and slicing training data...")
    train_images, train_masks, _ = load_and_slice_pairs(
        train_pairs,
        min_masks_per_slice=training_cfg["min_masks_per_slice"],
        min_pixels=training_cfg["min_pixels"],
        slice_axes=slice_axes,
    )
    logging.info(f"Training set: {len(train_images)} slices")

    logging.info("Loading and slicing eval data...")
    eval_images, eval_masks, eval_volumes = load_and_slice_pairs(
        eval_pairs,
        min_masks_per_slice=training_cfg["min_masks_per_slice"],
        min_pixels=training_cfg["min_pixels"],
        slice_axes=slice_axes,
    )
    logging.info(f"Eval set: {len(eval_images)} slices, {len(eval_volumes)} 3D volumes")

    if not train_images:
        logging.error("No training slices found. Check config paths.")
        return

    # ------------------------------------------------------------------
    # Baseline evaluation (pretrained, no finetuning)
    # ------------------------------------------------------------------
    logging.info("Computing baseline metrics on pretrained model...")
    use_bfloat16 = device.type == "cuda"
    base_model = models.CellposeModel(
        pretrained_model=base_cfg["model"]["base_model"],
        device=device,
        use_bfloat16=use_bfloat16,
    )

    ap_before_2d, ap_before_3d = {}, {}
    if eval_images:
        ap_before_2d, _ = evaluate_model(base_model, eval_images, eval_masks)
        logging.info(f"Baseline 2D: " + "  ".join(f"AP@{t:.2f}={v:.4f}" for t, v in sorted(ap_before_2d.items())))
    if eval_volumes:
        ap_before_3d, _ = evaluate_model_3d(base_model, eval_volumes)
        logging.info(f"Baseline 3D: " + "  ".join(f"AP@{t:.2f}={v:.4f}" for t, v in sorted(ap_before_3d.items())))

    del base_model  # free memory

    # ------------------------------------------------------------------
    # Run trials
    # ------------------------------------------------------------------
    all_results = list(completed_results)

    for idx, trial in enumerate(trials):
        if idx in completed_indices:
            logging.info(f"Skipping trial {idx} (already completed)")
            continue

        try:
            result = run_trial(
                trial=trial,
                trial_idx=idx,
                base_cfg=base_cfg,
                train_images=train_images,
                train_masks=train_masks,
                eval_images=eval_images,
                eval_masks=eval_masks,
                eval_volumes=eval_volumes,
                ap_before_2d=ap_before_2d,
                ap_before_3d=ap_before_3d,
                device=device,
                results_dir=results_dir,
            )
            all_results.append(result)
        except Exception as e:
            logging.error(f"Trial {idx} failed: {e}", exc_info=True)
            all_results.append({
                "trial_idx": idx,
                "hparams": trial,
                "error": str(e),
            })

        # Save after every trial
        with open(results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Results saved to {results_json}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        print_summary(valid_results, primary_metric)

        best = max(valid_results, key=lambda r: score_result(r, primary_metric))
        best_hparams = best["hparams"]
        logging.info("\nTo use the best config, update finetune_nuclei_config.json with:")
        logging.info(json.dumps(best_hparams, indent=2))
    else:
        logging.warning("No successful trials to summarize.")


if __name__ == "__main__":
    main()
