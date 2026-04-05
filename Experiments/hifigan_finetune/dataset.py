"""
Dataset for HiFi-GAN fine-tuning on post-surgery speech.

Loads wav files from CUCO post-surgery directories, extracts WavLM-Large
layer-6 features on the fly, and returns (features, waveform) pairs.
"""

import random
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel
from torch.utils.data import Dataset


SAMPLE_RATE = 16000
WAVLM_LAYER = 6
HOP_SIZE = 320  # HiFi-GAN upsamples by 320x to go from feature frames to samples


class WavLMFeatureExtractor:
    """Wraps HuggingFace WavLM-Large for layer-6 feature extraction."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large", output_hidden_states=True
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, wav_16k):
        """
        Args:
            wav_16k: (1, T) tensor at 16kHz
        Returns:
            features: (T', 1024) tensor from layer 6
        """
        wav_16k = wav_16k.to(self.device)
        out = self.model(wav_16k)
        # hidden_states is tuple of (num_layers+1, batch, seq_len, dim)
        features = out.hidden_states[WAVLM_LAYER]  # (1, T', 1024)
        return features.squeeze(0).cpu()  # (T', 1024)


def collect_wav_paths(data_dir):
    """Recursively find all .wav files under data_dir/**/Speech/2/."""
    data_dir = Path(data_dir)
    wav_paths = []
    for wav_file in sorted(data_dir.rglob("Speech/2/*.wav")):
        wav_paths.append(wav_file)
    return wav_paths


class HiFiGANDataset(Dataset):
    """
    Dataset that returns (wavlm_features, waveform) pairs.

    Features are extracted on first access and cached to disk as .pt files
    alongside the original .wav files for speed on subsequent epochs.
    """

    def __init__(self, wav_paths, wavlm_extractor, segment_size=7040, split=True):
        """
        Args:
            wav_paths: list of Path objects to .wav files
            wavlm_extractor: WavLMFeatureExtractor instance
            segment_size: number of waveform samples per training segment
            split: if True, randomly crop segments; if False, return full utterance
        """
        self.wav_paths = wav_paths
        self.wavlm = wavlm_extractor
        self.segment_size = segment_size
        self.split = split

    def __len__(self):
        return len(self.wav_paths)

    def _get_features(self, wav_path):
        """Load or compute+cache WavLM features for a wav file."""
        cache_path = wav_path.with_suffix(".wavlm_l6.pt")
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        features = self.wavlm.extract(
            torchaudio.load(wav_path)[0]
        )
        torch.save(features, cache_path)
        return features

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]

        # Load waveform
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav[0]  # (T,)

        # Get WavLM features (T', 1024)
        features = self._get_features(wav_path)

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / HOP_SIZE)

            if wav.size(0) >= self.segment_size:
                # Random crop aligned to feature frames
                max_start = features.size(0) - frames_per_seg
                if max_start > 0:
                    feat_start = random.randint(0, max_start)
                else:
                    feat_start = 0
                features = features[feat_start : feat_start + frames_per_seg]
                wav_start = feat_start * HOP_SIZE
                wav = wav[wav_start : wav_start + frames_per_seg * HOP_SIZE]
            else:
                # Pad short utterances
                features = F.pad(features, (0, 0, 0, frames_per_seg - features.size(0)))
                wav = F.pad(wav, (0, self.segment_size - wav.size(0)))

            # Ensure exact length alignment
            expected_wav_len = features.size(0) * HOP_SIZE
            if wav.size(0) > expected_wav_len:
                wav = wav[:expected_wav_len]
            elif wav.size(0) < expected_wav_len:
                wav = F.pad(wav, (0, expected_wav_len - wav.size(0)))

        return features, wav.unsqueeze(0)  # (T', 1024), (1, T_wav)
