"""
quantization/calibration.py — Calibration dataset for GPTQ.

Provides calibration data loading from:
  - WikiText-2 (default for language models)
  - C4 (Common Crawl)
  - Custom JSONL/text files
  - LLaVA-specific multimodal calibration (image captions)

The calibration dataset supplies the input activations used to compute
the Hessian matrix in GPTQ.
"""

import random
from typing import Optional, Iterator, List, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader


@dataclass
class CalibrationDataset:
    """
    Wraps calibration data for GPTQ Hessian computation.

    Usage:
        # From WikiText-2
        calib = CalibrationDataset.from_wikitext2(num_samples=128, seq_len=2048)

        # From custom text
        calib = CalibrationDataset.from_texts(texts, tokenizer)

        # Iterate
        for batch in calib.iter_batches(batch_size=4):
            model(batch)  # hooks capture layer inputs
    """

    # Input IDs [num_samples, seq_len]
    input_ids: torch.Tensor

    # Optional: attention mask
    attention_mask: Optional[torch.Tensor] = None

    # Optional: image features (for LLaVA calibration)
    image_features: Optional[torch.Tensor] = None

    def iter_batches(self, batch_size: int = 8) -> Iterator[torch.Tensor]:
        """Iterate over batches of input_ids."""
        for i in range(0, len(self.input_ids), batch_size):
            batch = self.input_ids[i:i + batch_size]
            if self.attention_mask is not None:
                mask = self.attention_mask[i:i + batch_size]
                yield {"input_ids": batch, "attention_mask": mask}
            else:
                yield batch

    def __len__(self) -> int:
        return len(self.input_ids)

    # ═══════════════════════════════════════════════════════════════════════
    # Factory Methods
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_wikitext2(
        cls,
        tokenizer=None,
        num_samples: int = 128,
        seq_len: int = 2048,
        split: str = "train",
    ) -> "CalibrationDataset":
        """
        Load WikiText-2 calibration data.

        WikiText-2 is the standard calibration dataset used by GPTQ.
        It contains high-quality Wikipedia articles with diverse topics.

        Args:
            tokenizer: HuggingFace tokenizer. If None, loads GPT-2 tokenizer.
            num_samples: Number of calibration samples (128 is standard)
            seq_len: Sequence length per sample
            split: 'train' or 'test'

        Returns:
            CalibrationDataset
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        except Exception:
            # Fallback: generate random data
            print("[Calib] WikiText-2 not available, using random calibration data")
            return cls._random_calibration(tokenizer, num_samples, seq_len)

        # Filter empty lines and tokenize
        texts = [item["text"] for item in dataset if len(item["text"].strip()) > 0]

        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)

        # Create fixed-length samples
        samples = []
        for i in range(num_samples):
            start = random.randint(0, max(0, len(all_tokens) - seq_len - 1))
            sample = all_tokens[start:start + seq_len]
            if len(sample) < seq_len:
                sample = sample + [tokenizer.pad_token_id or 0] * (seq_len - len(sample))
            samples.append(sample)

        input_ids = torch.tensor(samples, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        return cls(input_ids=input_ids, attention_mask=attention_mask)

    @classmethod
    def from_c4(
        cls,
        tokenizer=None,
        num_samples: int = 128,
        seq_len: int = 2048,
    ) -> "CalibrationDataset":
        """
        Load C4 (Common Crawl) calibration data.

        C4 provides more diverse web text than WikiText-2, which can
        lead to better generalization in quantization.

        Args:
            tokenizer: HuggingFace tokenizer
            num_samples: Number of calibration samples
            seq_len: Sequence length per sample

        Returns:
            CalibrationDataset
        """
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        try:
            from datasets import load_dataset
            dataset = load_dataset("c4", "en", split="validation", streaming=True)
        except Exception:
            return cls._random_calibration(tokenizer, num_samples, seq_len)

        samples = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = item["text"]
            tokens = tokenizer.encode(text, truncation=True, max_length=seq_len)
            if len(tokens) < seq_len:
                tokens = tokens + [tokenizer.pad_token_id or 0] * (seq_len - len(tokens))
            samples.append(tokens[:seq_len])

        # Pad to exactly num_samples
        while len(samples) < num_samples:
            samples.append([tokenizer.pad_token_id or 0] * seq_len)

        input_ids = torch.tensor(samples, dtype=torch.long)
        return cls(input_ids=input_ids)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        tokenizer,
        seq_len: int = 2048,
    ) -> "CalibrationDataset":
        """
        Create calibration dataset from custom text list.

        Useful for domain-specific quantization (e.g., medical, legal).

        Args:
            texts: List of text strings
            tokenizer: Tokenizer
            seq_len: Sequence length

        Returns:
            CalibrationDataset
        """
        samples = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=seq_len)
            if len(tokens) < seq_len:
                tokens = tokens + [tokenizer.pad_token_id or 0] * (seq_len - len(tokens))
            samples.append(tokens[:seq_len])

        input_ids = torch.tensor(samples, dtype=torch.long)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return cls(input_ids=input_ids, attention_mask=attention_mask)

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        tokenizer,
        seq_len: int = 2048,
    ) -> "CalibrationDataset":
        """
        Load calibration data from a text file (one line per prompt).

        Supports .txt, .jsonl formats.

        Args:
            file_path: Path to text or JSONL file
            tokenizer: Tokenizer
            seq_len: Sequence length

        Returns:
            CalibrationDataset
        """
        path = Path(file_path)

        if path.suffix == ".jsonl":
            import json
            with open(path) as f:
                texts = [json.loads(line).get("text", line) for line in f]
        else:
            with open(path) as f:
                texts = [line.strip() for line in f if line.strip()]

        return cls.from_texts(texts, tokenizer, seq_len)

    @classmethod
    def _random_calibration(
        cls,
        tokenizer,
        num_samples: int,
        seq_len: int,
    ) -> "CalibrationDataset":
        """
        Generate random calibration data as fallback.

        Uses tokenizer's vocabulary to create synthetic inputs.
        Note: This produces worse quantization quality than real data.
        """
        if tokenizer is not None:
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = 50000  # sensible default when no tokenizer given
        # Generate random token IDs (avoid special tokens)
        input_ids = torch.randint(0, min(vocab_size, 50000), (num_samples, seq_len))
        attention_mask = torch.ones(num_samples, seq_len)
        return cls(input_ids=input_ids, attention_mask=attention_mask)


def load_calibration_data(
    tokenizer,
    num_samples: int = 128,
    seq_len: int = 2048,
    source: str = "wikitext2",
) -> CalibrationDataset:
    """
    Convenience function to load calibration data.

    Args:
        tokenizer: HuggingFace tokenizer
        num_samples: Number of calibration samples
        seq_len: Sequence length
        source: "wikitext2", "c4", "random"

    Returns:
        CalibrationDataset
    """
    if source == "wikitext2":
        return CalibrationDataset.from_wikitext2(tokenizer, num_samples, seq_len)
    elif source == "c4":
        return CalibrationDataset.from_c4(tokenizer, num_samples, seq_len)
    elif source == "random":
        return CalibrationDataset._random_calibration(tokenizer, num_samples, seq_len)
    else:
        raise ValueError(f"Unknown calibration source: {source}")


# ═══════════════════════════════════════════════════════════════════════════
# LLaVA-Specific Calibration
# ═══════════════════════════════════════════════════════════════════════════

class LLaVACalibrationDataset(CalibrationDataset):
    """
    Calibration dataset for LLaVA-1.5 multimodal quantization.

    Uses image-caption pairs where images are processed through the
    vision tower to provide realistic input activations for both the
    multimodal projector and language model.
    """

    @classmethod
    def from_coco_captions(
        cls,
        tokenizer,
        vision_processor,
        num_samples: int = 128,
        seq_len: int = 2048,
    ) -> "LLaVACalibrationDataset":
        """
        Use COCO captions as calibration data for LLaVA.

        Args:
            tokenizer: LLaVA tokenizer
            vision_processor: CLIP image processor
            num_samples: Number of samples
            seq_len: Sequence length

        Returns:
            LLaVACalibrationDataset with image features
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("HuggingFaceM4/COCO", split="validation", streaming=True)
        except Exception:
            print("[Calib] COCO not available, using random data")
            return cls._random_llava_calibration(tokenizer, num_samples, seq_len)

        samples = []
        image_features_list = []

        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            # Tokenize caption
            caption = item.get("caption", item.get("text", ""))
            prompt = f"A chat between a curious human and an AI assistant. USER: <image>\n{caption} ASSISTANT:"
            tokens = tokenizer.encode(prompt, truncation=True, max_length=seq_len)
            if len(tokens) < seq_len:
                tokens = tokens + [0] * (seq_len - len(tokens))
            samples.append(tokens[:seq_len])

            # Process image if available
            if "image" in item:
                img = item["image"]
                img_tensor = vision_processor(images=img, return_tensors="pt")["pixel_values"]
                image_features_list.append(img_tensor.squeeze(0))

        input_ids = torch.stack([torch.tensor(s) for s in samples])

        inst = cls(input_ids=input_ids)
        if image_features_list:
            inst.image_features = torch.stack(image_features_list)
        return inst

    @classmethod
    def _random_llava_calibration(
        cls,
        tokenizer,
        num_samples: int,
        seq_len: int,
    ) -> "LLaVACalibrationDataset":
        """Random fallback for LLaVA calibration."""
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(100, min(vocab_size, 50000), (num_samples, seq_len))
        return cls(input_ids=input_ids)
