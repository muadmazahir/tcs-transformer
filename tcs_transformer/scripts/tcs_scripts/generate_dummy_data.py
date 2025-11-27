"""Generate dummy data for testing TCS pipeline"""

import os
import json
from collections import defaultdict, Counter

from delphin import ace
from delphin.dmrs import from_mrs
from delphin.codecs import simplemrs, dmrsjson

from nltk.stem import WordNetLemmatizer
from networkx.readwrite.json_graph import node_link_data

from tcs_transformer.utils import dg_util, util


def lemmatize_unk(pred_lemma, pred_pos):
    """Lemmatize unknown predicates using WordNet"""
    pos_unk2pos_wn = {"j": "a", "v": "v", "n": "n", "r": "r"}
    wordnet_lemmatizer = WordNetLemmatizer()
    if pred_lemma[0] in pos_unk2pos_wn:
        norm_lemma = wordnet_lemmatizer.lemmatize(
            pred_lemma, pos_unk2pos_wn[pred_lemma[0]]
        )
    else:
        norm_lemma = wordnet_lemmatizer.lemmatize(pred_lemma)
    return norm_lemma


def normalize_pred(pred, unk2pos):
    """Normalize predicate names"""
    norm_pred = None
    if "_u_unknown" not in pred:
        norm_pred = pred
    else:
        pred_lemma, pred_pos = util.get_lemma_pos(pred)
        norm_prefix = "u"
        norm_lemma = lemmatize_unk(pred_lemma, pred_pos)
        norm_pos = unk2pos[pred_pos]
        norm_pred = "_".join([norm_prefix, norm_lemma, norm_pos])
    return norm_pred


def generate_dummy_data(
    erg_path="examples/erg/erg-1214-x86-64-0.9.34.dat",
    unk2pos_path="examples/erg/unk2pos.json",
    output_dir="examples/dummy_data",
    sentences=None,
):
    """
    Generate dummy data from example sentences

    Args:
        erg_path: Path to ERG grammar file
        unk2pos_path: Path to unk2pos.json mapping file
        output_dir: Output directory for generated data
        sentences: List of [sentence, parse_number] pairs. If None, uses default examples.
    """
    os.environ["LC_ALL"] = "en_US.UTF-8"
    os.environ["LC_CTYPE"] = "en_US.UTF-8"

    # Load unk2pos mapping
    with open(unk2pos_path) as f:
        unk2pos = json.load(f)

    # Default test sentences if none provided
    if sentences is None:
        sentences = [
            ["If bears and sad cats eat quickly or slowly, birds run.", 0],
            ["Lions are cats, animals and mammals", 0],
        ]

    # Create output directories
    dummy_data_fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dummy_data_fig_dir, exist_ok=True)

    # Initialize counters
    idx2instance = defaultdict(defaultdict)
    idx2file_path = defaultdict()
    err2cnt = Counter()
    pred2cnt = Counter()

    # Process each sentence
    for no_instance, (snt, no_parse) in enumerate(sentences):
        print(f"Processing sentence {no_instance}: {snt}")

        try:
            # Parse with ACE
            ace_results = ace.parse(erg_path, snt)

            if not ace_results["results"]:
                print(f"  No parse results for: {snt}")
                err2cnt["no_parse"] += 1
                continue

            if no_parse >= len(ace_results["results"]):
                print(f"  Parse {no_parse} not found, using first parse")
                no_parse = 0

            # Extract MRS and convert to DMRS
            mrs = ace_results["results"][no_parse]["mrs"]
            dmrs = from_mrs(simplemrs.decode(mrs))
            dmrs_json = dmrsjson.to_dict(dmrs)

            # Normalize predicates
            for node in dmrs_json["nodes"]:
                norm_pred = normalize_pred(node["predicate"], unk2pos)
                node["predicate"] = norm_pred
                pred2cnt[norm_pred] += 1

            # Create ERG digraphs and visualize
            erg_digraphs = dg_util.Erg_DiGraphs()
            erg_digraphs.init_dmrs_from_dict(dmrs_json)

            # Draw DMRS visualization
            fig_path = os.path.join(dummy_data_fig_dir, f"dmrs_{no_instance}.png")
            erg_digraphs.draw_dmrs(name=str(no_instance), save_path=fig_path)
            print(f"  Saved DMRS visualization to: {fig_path}")

            # Store instance data
            idx2instance[no_instance] = {
                "id": f"0_{no_instance:05d}",
                "snt": snt,
                "dmrs": node_link_data(erg_digraphs.dmrs_dg),
            }

        except Exception as e:
            print(f"  Error processing sentence: {e}")
            err2cnt["error"] += 1
            continue

    # Save data to JSON
    output_file = os.path.join(output_dir, "0_00000.json")
    instances = [instance for _, instance in sorted(idx2instance.items())]

    with open(output_file, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"\nSaved {len(instances)} instances to: {output_file}")

    # Save info files
    info_dir = os.path.join(output_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

    # Save idx2file_path
    idx2file_path[0] = output_file
    with open(os.path.join(info_dir, "idx2file_path.json"), "w") as f:
        json.dump(idx2file_path, f, indent=2)

    # Save err2cnt
    with open(os.path.join(info_dir, "err2cnt.txt"), "w") as f:
        for err, cnt in err2cnt.items():
            f.write(f"{err}\t{cnt}\n")

    # Save pred2cnt
    with open(os.path.join(info_dir, "pred2cnt.txt"), "w") as f:
        for pred, cnt in sorted(pred2cnt.items(), key=lambda x: -x[1]):
            f.write(f"{pred}\t{cnt}\n")

    print(f"Saved info files to: {info_dir}")
    print(f"Total predicates: {len(pred2cnt)}")
    print(f"Total errors: {sum(err2cnt.values())}")

    return instances
