"""Prepare training data by transforming DMRS to TCS representations"""

import os
import json
from collections import Counter
from tqdm import tqdm

from tcs_transformer.core.tcs_transform import TruthConditions


def read_transform_config(config_path):
    """Read transformation configuration"""
    with open(config_path, "r") as f:
        return json.load(f)


def prepare_training_data(
    data_dir="examples/dummy_data",
    output_dir="examples/dummy_data/transformed/TCS_f0-lgF-PAS-Gen_dummy",
    transform_config_path="configs/transform_config.json",
    min_pred_func_freq=0,
    min_content_pred_freq=0,
    filter_min_freq=False,
    relpron_dir="examples/eval_data_sets/RELPRON",
):
    """
    Transform DMRS data to TCS format for training

    Args:
        data_dir: Directory containing DMRS JSON files
        output_dir: Output directory for transformed data
        transform_config_path: Path to transform configuration file
        min_pred_func_freq: Minimum frequency for predicate functions
        min_content_pred_freq: Minimum frequency for content predicates
        filter_min_freq: Whether to filter by minimum frequency
        relpron_dir: Directory containing RELPRON evaluation data
    """
    print("=" * 80)
    print("TCS Training Data Preparation")
    print("=" * 80)

    # Load transform config
    transform_config = read_transform_config(transform_config_path)
    print(f"Loaded transform config from: {transform_config_path}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    info_dir = os.path.join(output_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

    # Load RELPRON predicates if provided
    keep_preds = set()
    if relpron_dir:
        word2pred_path = os.path.join(relpron_dir, "word2pred_premap.json")
        if os.path.exists(word2pred_path):
            with open(word2pred_path, "r") as f:
                word2pred = json.load(f)
                keep_preds = set(
                    [pred for preds in word2pred.values() for pred in preds]
                )
            print(f"Loaded {len(keep_preds)} RELPRON predicates")

    # First pass: Count frequencies
    print("\nFirst pass: Counting frequencies...")
    pred_func2cnt = Counter()
    content_pred2cnt = Counter()
    predarg2cnt = Counter()

    # Find all JSON files in data directory
    data_files = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".json") and not f.endswith("-checkpoint.json")
    ]

    if not data_files:
        print(f"Error: No JSON files found in {data_dir}")
        return

    print(f"Found {len(data_files)} data files")

    # Process files to count frequencies
    all_samples = []
    for data_file in data_files:
        file_path = os.path.join(data_dir, data_file)
        with open(file_path, "r") as f:
            samples = json.load(f)
            all_samples.extend(samples)

    print(f"Loaded {len(all_samples)} samples")

    # Count frequencies
    for sample in tqdm(all_samples, desc="Counting frequencies"):
        tc = TruthConditions(
            config=transform_config,
            to_ix=False,
            min_pred_func_freq=0,
            min_content_pred_freq=0,
            content_pred2cnt=None,
            pred_func2cnt=None,
            filter_min_freq=False,
            pred2ix=None,
            predarg2ix=None,
            pred_func2ix=None,
            keep_preds=keep_preds,
        )

        try:
            result = tc(sample)
            if not result["discarded"]:
                # Count predicate functions
                for pred_func in result["pred_func_used"]:
                    pred_func2cnt[pred_func] += 1
                # Count content predicates
                for pred in result["content_preds"]:
                    content_pred2cnt[pred] += 1
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            continue

    print(f"\nFound {len(pred_func2cnt)} unique predicate functions")
    print(f"Found {len(content_pred2cnt)} unique content predicates")

    # Create vocabularies
    print("\nCreating vocabularies...")

    # Filter by frequency and create indices
    pred_func2ix = {}
    pred_func_ix = 0
    for pred_func, cnt in pred_func2cnt.items():
        if cnt >= min_pred_func_freq or any(
            pred in keep_preds for pred in [pred_func.split("@")[0]]
        ):
            pred_func2ix[pred_func] = pred_func_ix
            pred_func_ix += 1

    pred2ix = {}
    pred_ix = 0
    for pred, cnt in content_pred2cnt.items():
        if cnt >= min_content_pred_freq or pred in keep_preds:
            pred2ix[pred] = pred_ix
            pred_ix += 1

    # Create predarg2ix
    predarg2ix = {}
    predarg_ix = 0
    for pred in pred2ix:
        for arg in [
            "NonARG",
            "ARG0",
            "ARG1",
            "ARG2",
            "ARG3",
            "ARG4",
            "ARG1-rvrs",
            "ARG2-rvrs",
            "ARG3-rvrs",
            "ARG4-rvrs",
        ]:
            predarg = f"{pred}@{arg}"
            predarg2ix[predarg] = predarg_ix
            predarg_ix += 1

    print("Vocabulary sizes:")
    print(f"  Predicate functions: {len(pred_func2ix)}")
    print(f"  Predicates: {len(pred2ix)}")
    print(f"  Predicate-arguments: {len(predarg2ix)}")

    # Second pass: Transform data
    print("\nSecond pass: Transforming data...")

    tc_transform = TruthConditions(
        config=transform_config,
        to_ix=True,
        min_pred_func_freq=min_pred_func_freq,
        min_content_pred_freq=min_content_pred_freq,
        content_pred2cnt=content_pred2cnt,
        pred_func2cnt=pred_func2cnt,
        filter_min_freq=filter_min_freq,
        pred2ix=pred2ix,
        predarg2ix=predarg2ix,
        pred_func2ix=pred_func2ix,
        keep_preds=keep_preds,
    )

    transformed_samples = []
    discarded_count = 0

    for sample in tqdm(all_samples, desc="Transforming samples"):
        try:
            result = tc_transform(sample)
            result["snt_id"] = sample["id"]
            transformed_samples.append(result)
            if result["discarded"]:
                discarded_count += 1
        except Exception as e:
            print(f"Error transforming sample {sample.get('id', 'unknown')}: {e}")
            discarded_count += 1
            continue

    print(f"\nTransformed {len(transformed_samples)} samples")
    print(f"Discarded {discarded_count} samples")

    # Save transformed data
    output_file = os.path.join(output_dir, "transformed_0.json")
    with open(output_file, "w") as f:
        json.dump(transformed_samples, f)
    print(f"Saved transformed data to: {output_file}")

    # Save vocabulary files
    print("\nSaving vocabulary files...")

    # pred_func2ix
    with open(os.path.join(info_dir, "pred_func2ix.txt"), "w") as f:
        for pred_func, ix in sorted(pred_func2ix.items(), key=lambda x: x[1]):
            f.write(f"{ix}\t{pred_func}\n")

    # pred_func2cnt
    with open(os.path.join(info_dir, "pred_func2cnt.txt"), "w") as f:
        for pred_func, cnt in sorted(pred_func2cnt.items(), key=lambda x: -x[1]):
            if pred_func in pred_func2ix:
                f.write(f"{pred_func2ix[pred_func]}\t{cnt}\n")

    # pred2ix
    with open(os.path.join(info_dir, "pred2ix.txt"), "w") as f:
        for pred, ix in sorted(pred2ix.items(), key=lambda x: x[1]):
            f.write(f"{ix}\t{pred}\n")

    # content_pred2cnt
    with open(os.path.join(info_dir, "content_pred2cnt.txt"), "w") as f:
        for pred, cnt in sorted(content_pred2cnt.items(), key=lambda x: -x[1]):
            if pred in pred2ix:
                f.write(f"{pred2ix[pred]}\t{cnt}\n")

    # content_predarg2ix
    with open(os.path.join(info_dir, "content_predarg2ix.txt"), "w") as f:
        for predarg, ix in sorted(predarg2ix.items(), key=lambda x: x[1]):
            f.write(f"{ix}\t{predarg}\n")

    print(f"Saved vocabulary files to: {info_dir}")
    print("\nDone!")
