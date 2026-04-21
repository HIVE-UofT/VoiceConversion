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

WAVLM_LOCAL_PATH = "/lustre06/project/6086959/sepharfi/models/wavlm-large"


class WavLMFeatureExtractor:
    """Wraps HuggingFace WavLM-Large for layer-6 feature extraction."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = WavLMModel.from_pretrained(
            WAVLM_LOCAL_PATH, output_hidden_states=True, local_files_only=True
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


def get_patient_id(wav_path):
    """Extract patient ID from filename, e.g. 'Speech_0085.wav' -> '0085'."""
    return Path(wav_path).stem.split('_')[-1]


def collect_wav_paths(data_dir, surgery="Tonsill", exclude_patients=None):
    """
    Find all post-surgery .wav files for one surgery type, excluding held-out patients.

    Args:
        data_dir: path to CUCO Audios directory (contains Tonsill/, Fess/, etc.)
        surgery: surgery subdirectory to use (default: 'Tonsill')
        exclude_patients: collection of patient ID strings to exclude (e.g. ['0085', '0109'])

    Returns:
        sorted list of Path objects for included post-surgery wav files
    """
    data_dir = Path(data_dir)
    speech_dir = data_dir / surgery / "Speech" / "2"
    if not speech_dir.exists():
        raise FileNotFoundError(f"Post-surgery speech directory not found: {speech_dir}")

    exclude = set(exclude_patients) if exclude_patients else set()
    wav_paths = []
    for wav_file in sorted(speech_dir.glob("*.wav")):
        if get_patient_id(wav_file) not in exclude:
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
