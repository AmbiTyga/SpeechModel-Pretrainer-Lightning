from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices, _sample_negative_indices
)
import torch
from typing import Union, List, Optional, Dict
from dataclasses import dataclass
import pandas as pd
from safetensors.torch import load_file

from torch.utils.data.dataloader import DataLoader, Dataset

@dataclass
class DataCollatorForWav2Vec2Pretraining:
    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    """
    Creates batch by padding features
    Creates masked indices for Masked Language Modeling training
    Creates negative samples for Contrastive Language Modeling

    args:
        features: List[Dict[str, Union[List[int], torch.Tensor]]]

    return:
        batch: Dict[str, torch.Tensor]
    """
    def __call__(
            self,
            features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )
        device = batch['input_values'].device
        batch_size = batch['input_values'].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(
            batch["input_values"].shape[-1]
        )
        mask_indices_seq_length = int(mask_indices_seq_length)

        if batch.get("attention_mask") is not None:
            batch['sub_attention_mask'] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )
        
        features_shape = (batch_size, mask_indices_seq_length)

        # Sample Randomly Masked Indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask")
        )

        # Sample Negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices
        )

        batch["mask_time_indices"] = torch.tensor(
            mask_time_indices, dtype=torch.long, device=device
        )
        batch['sampled_negative_indices'] = torch.tensor(
            sampled_negative_indices, dtype=torch.long, device=device
        )
        return batch

class AudioDataset(Dataset):
    """
    Dataset class by loading safetensors of preprocessed audio waveforms
    """
    def __init__(self, audio_paths):
        self.audio_sf_paths = [wav.replace('.wav', '.safetensors') for wav in audio_paths]
        
    def __len__(self):
        return len(self.audio_sf_paths)

    def __getitem__(self, idx):
        batch = load_file(self.audio_sf_paths[idx])
        return batch