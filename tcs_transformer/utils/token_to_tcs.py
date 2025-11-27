"""
Convert tokens to TCS data for hybrid training

This module provides functionality to convert token sequences back to text,
parse them with ERG to extract DMRS, and transform to TCS format for training.
"""

import os
import json
from typing import List, Dict, Any, Optional
import torch

from delphin import ace
from delphin.dmrs import from_mrs
from delphin.codecs import simplemrs, dmrsjson
from networkx.readwrite.json_graph import node_link_data
from nltk.stem import WordNetLemmatizer

from tcs_transformer.core.tcs_transform import TruthConditions
from tcs_transformer.utils import dg_util


def lemmatize_unk(pred_lemma: str, pred_pos: str) -> str:
    """Lemmatize unknown predicates using WordNet"""
    pos_unk2pos_wn = {"j": "a", "v": "v", "n": "n", "r": "r"}
    wordnet_lemmatizer = WordNetLemmatizer()
    if pred_lemma and pred_lemma[0] in pos_unk2pos_wn:
        norm_lemma = wordnet_lemmatizer.lemmatize(
            pred_lemma, pos_unk2pos_wn[pred_lemma[0]]
        )
    else:
        norm_lemma = wordnet_lemmatizer.lemmatize(pred_lemma)
    return norm_lemma


def normalize_pred(pred: str, unk2pos: Dict[str, str]) -> str:
    """Normalize predicate names"""
    if "_u_unknown" not in pred:
        return pred

    # Simple parsing of predicate format
    parts = pred.split("_")
    if len(parts) < 3:
        return pred

    pred_lemma = parts[1]
    pred_pos = parts[2] if len(parts) > 2 else "n"

    norm_prefix = "u"
    norm_lemma = lemmatize_unk(pred_lemma, pred_pos)
    norm_pos = unk2pos.get(pred_pos, pred_pos)
    norm_pred = "_".join([norm_prefix, norm_lemma, norm_pos])

    return norm_pred


def parse_text_to_dmrs(
    text: str, erg_path: str, unk2pos: Dict[str, str], parse_num: int = 0
) -> Optional[Dict[str, Any]]:
    """
    Parse text with ERG and convert to DMRS format

    Args:
        text: Input text to parse
        erg_path: Path to ERG grammar file
        unk2pos: Unknown to POS mapping
        parse_num: Which parse to use (default 0 = first)

    Returns:
        DMRS graph as dictionary, or None if parsing fails
    """
    try:
        # Parse with ACE
        ace_results = ace.parse(erg_path, text)

        if not ace_results["results"]:
            return None

        if parse_num >= len(ace_results["results"]):
            parse_num = 0

        # Extract MRS and convert to DMRS
        mrs = ace_results["results"][parse_num]["mrs"]
        dmrs = from_mrs(simplemrs.decode(mrs))
        dmrs_json = dmrsjson.to_dict(dmrs)

        # Normalize predicates
        for node in dmrs_json["nodes"]:
            norm_pred = normalize_pred(node["predicate"], unk2pos)
            node["predicate"] = norm_pred

        return dmrs_json

    except Exception as e:
        print(f"Parse error for text '{text}': {e}")
        return None


def dmrs_to_tcs_sample(
    dmrs_json: Dict[str, Any], tc_transform: TruthConditions, sample_id: str = "0"
) -> Optional[Dict[str, Any]]:
    """
    Transform DMRS to TCS format

    Args:
        dmrs_json: DMRS graph dictionary
        tc_transform: TruthConditions transformer
        sample_id: Sample identifier

    Returns:
        TCS sample dictionary, or None if transformation fails
    """
    try:
        # Create ERG digraph from DMRS
        erg_digraphs = dg_util.Erg_DiGraphs()
        erg_digraphs.init_dmrs_from_dict(dmrs_json)

        # Create sample in expected format
        sample = {"id": sample_id, "dmrs": node_link_data(erg_digraphs.dmrs_dg)}

        # Transform to TCS format
        result = tc_transform(sample)

        if result["discarded"]:
            return None

        return result

    except Exception as e:
        print(f"TCS transformation error: {e}")
        return None


class TokenToTCSConverter:
    """
    Converter for transforming token sequences to TCS training data

    This class maintains the necessary configuration and state for
    parsing text and transforming to TCS format during training.
    """

    def __init__(
        self,
        erg_path: str,
        unk2pos_path: str,
        transform_config_path: str,
        pred2ix: Optional[Dict[str, int]] = None,
        predarg2ix: Optional[Dict[str, int]] = None,
        pred_func2ix: Optional[Dict[str, int]] = None,
        keep_preds: Optional[set] = None,
        min_pred_func_freq: int = 0,
        min_content_pred_freq: int = 0,
        filter_min_freq: bool = False,
    ):
        """
        Initialize converter

        Args:
            erg_path: Path to ERG grammar file
            unk2pos_path: Path to unk2pos.json mapping
            transform_config_path: Path to transform configuration
            pred2ix: Predicate to index mapping (optional)
            predarg2ix: Predicate-argument to index mapping (optional)
            pred_func2ix: Predicate function to index mapping (optional)
            keep_preds: Set of predicates to always keep
            min_pred_func_freq: Minimum frequency for predicate functions
            min_content_pred_freq: Minimum frequency for content predicates
            filter_min_freq: Whether to filter by minimum frequency
        """
        # Set locale for ACE
        os.environ["LC_ALL"] = "en_US.UTF-8"
        os.environ["LC_CTYPE"] = "en_US.UTF-8"

        self.erg_path = erg_path

        # Load unk2pos mapping
        with open(unk2pos_path) as f:
            self.unk2pos = json.load(f)

        # Load transform config
        with open(transform_config_path) as f:
            self.transform_config = json.load(f)

        # Create TruthConditions transformer
        self.tc_transform = TruthConditions(
            config=self.transform_config,
            to_ix=True,
            min_pred_func_freq=min_pred_func_freq,
            min_content_pred_freq=min_content_pred_freq,
            content_pred2cnt=None,
            pred_func2cnt=None,
            filter_min_freq=filter_min_freq,
            pred2ix=pred2ix,
            predarg2ix=predarg2ix,
            pred_func2ix=pred_func2ix,
            keep_preds=keep_preds or set(),
        )

        # Statistics
        self.parse_failures = 0
        self.transform_failures = 0
        self.successes = 0

    def convert_tokens(
        self, token_ids: torch.Tensor, tokenizer, sample_id: str = "0"
    ) -> Optional[Dict[str, Any]]:
        """
        Convert token IDs to TCS data

        Args:
            token_ids: Token ID tensor (1D or 2D)
            tokenizer: Tokenizer for decoding
            sample_id: Sample identifier

        Returns:
            TCS sample dictionary with 'encoders' and 'decoders' keys,
            or None if conversion fails
        """
        # Decode tokens to text
        if token_ids.dim() == 2:
            # Batch of tokens - take first sequence
            token_ids = token_ids[0]

        text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)

        if not text.strip():
            return None

        # Parse to DMRS
        dmrs_json = parse_text_to_dmrs(text, self.erg_path, self.unk2pos)
        if dmrs_json is None:
            self.parse_failures += 1
            return None

        # Transform to TCS
        tcs_sample = dmrs_to_tcs_sample(dmrs_json, self.tc_transform, sample_id)
        if tcs_sample is None:
            self.transform_failures += 1
            return None

        self.successes += 1
        return tcs_sample

    def convert_batch(
        self,
        token_ids_batch: torch.Tensor,
        tokenizer,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert a batch of token sequences to TCS data

        Args:
            token_ids_batch: Token ID tensor (batch_size, seq_len)
            tokenizer: Tokenizer for decoding
            max_samples: Maximum number of samples to convert (None = all)

        Returns:
            List of TCS samples (may be shorter than batch if some fail)
        """
        tcs_samples = []
        batch_size = token_ids_batch.shape[0]

        # Limit number of conversions if specified
        num_to_convert = min(batch_size, max_samples) if max_samples else batch_size

        for i in range(num_to_convert):
            sample = self.convert_tokens(
                token_ids_batch[i], tokenizer, sample_id=f"batch_{i}"
            )
            if sample is not None:
                tcs_samples.append(sample)

        return tcs_samples

    def get_statistics(self) -> Dict[str, int]:
        """Get conversion statistics"""
        return {
            "successes": self.successes,
            "parse_failures": self.parse_failures,
            "transform_failures": self.transform_failures,
            "total_attempts": self.successes
            + self.parse_failures
            + self.transform_failures,
        }

    def reset_statistics(self):
        """Reset conversion statistics"""
        self.parse_failures = 0
        self.transform_failures = 0
        self.successes = 0


def create_converter(
    erg_path: str = "examples/erg/erg-1214-x86-64-0.9.34.dat",
    unk2pos_path: str = "examples/erg/unk2pos.json",
    transform_config_path: str = "configs/transform_config.json",
    vocab_dir: Optional[str] = None,
) -> TokenToTCSConverter:
    """
    Create a TokenToTCSConverter with optional pre-trained vocabularies

    Args:
        erg_path: Path to ERG grammar file
        unk2pos_path: Path to unk2pos.json mapping
        transform_config_path: Path to transform configuration
        vocab_dir: Directory containing vocabulary files (pred2ix.txt, etc.)

    Returns:
        Configured TokenToTCSConverter
    """
    pred2ix = None
    predarg2ix = None
    pred_func2ix = None

    # Load vocabularies if directory provided
    if vocab_dir:
        # Load pred2ix
        pred2ix_path = os.path.join(vocab_dir, "pred2ix.txt")
        if os.path.exists(pred2ix_path):
            pred2ix = {}
            with open(pred2ix_path) as f:
                for line in f:
                    ix, pred = line.strip().split("\t")
                    pred2ix[pred] = int(ix)

        # Load predarg2ix
        predarg2ix_path = os.path.join(vocab_dir, "content_predarg2ix.txt")
        if os.path.exists(predarg2ix_path):
            predarg2ix = {}
            with open(predarg2ix_path) as f:
                for line in f:
                    ix, predarg = line.strip().split("\t")
                    predarg2ix[predarg] = int(ix)

        # Load pred_func2ix
        pred_func2ix_path = os.path.join(vocab_dir, "pred_func2ix.txt")
        if os.path.exists(pred_func2ix_path):
            pred_func2ix = {}
            with open(pred_func2ix_path) as f:
                for line in f:
                    ix, pred_func = line.strip().split("\t")
                    pred_func2ix[pred_func] = int(ix)

    return TokenToTCSConverter(
        erg_path=erg_path,
        unk2pos_path=unk2pos_path,
        transform_config_path=transform_config_path,
        pred2ix=pred2ix,
        predarg2ix=predarg2ix,
        pred_func2ix=pred_func2ix,
    )
