"""
Prepare CUCO Tonsill data for FreeVC fine-tuning.

Steps:
  1. Collect all pre+post speech audio from training patients (excl. 5 test patients)
  2. Resample to 16 kHz and save as 16-bit PCM WAV into flat dir structure
     expected by FreeVC: dataset/cuco-16k/{pid}_{session}/{filename}.wav
  3. Precompute WavLM content features (preprocess_ssl.py-style)
  4. Precompute speaker embeddings (preprocess_spk.py-style)
  5. Precompute spectrograms (needed by training data loader)
  6. Write filelists {train,val}.txt

Output:
  <exp>/free_vc/data/cuco-16k/<pid_sess>/<file>.wav  — 16k wav
  <exp>/free_vc/data/wavlm/<pid_sess>/<file>.pt     — WavLM features
  <exp>/free_vc/data/spk/<pid_sess>/<file>.npy      — speaker embedding
  <exp>/free_vc/FreeVC/filelists/cuco_{train,val}.txt
"""
import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import torch
import torchaudio
import librosa
from pathlib import Path
from scipy.io.wavfile import write as wav_write
from tqdm import tqdm

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEVC_DIR = os.path.join(EXP_DIR, 'FreeVC')
sys.path.insert(0, FREEVC_DIR)
os.chdir(FREEVC_DIR)

import utils
from speaker_encoder.voice_encoder import SpeakerEncoder
from mel_processing import spectrogram_torch

TEST_PATIENTS = {'0045', '0085', '0110', '0122', '0132'}
CUCO_BASE = '/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios'

# Output dirs
DATA_ROOT = os.path.join(EXP_DIR, 'data')
WAV_DIR   = os.path.join(DATA_ROOT, 'cuco-16k')
SSL_DIR   = os.path.join(DATA_ROOT, 'wavlm')
SPK_DIR   = os.path.join(DATA_ROOT, 'spk')
FILELIST_DIR = os.path.join(FREEVC_DIR, 'filelists')

SR = 16000


def collect_pairs(surgery='Tonsill'):
    """Return dict {pid: {'pre': path, 'post': path}} for speech files."""
    def one(sess):
        pattern = os.path.join(CUCO_BASE, surgery, 'Speech', sess, '*.wav')
        return {Path(p).stem.split('_')[-1]: p
                for p in sorted(glob.glob(pattern))}
    pre = one('1')
    post = one('2')
    return {pid: {'pre': pre[pid], 'post': post[pid]}
            for pid in pre if pid in post}


def resample_save(src_path, dst_path, sr=SR):
    """Resample to 16 kHz, save as 16-bit PCM WAV."""
    sig, orig_sr = torchaudio.load(src_path)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        sig = torchaudio.functional.resample(sig, orig_sr, sr)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # scale to int16
    audio = sig.squeeze().numpy()
    audio = (audio * 32767).clip(-32768, 32767).astype('int16')
    wav_write(dst_path, sr, audio)


def extract_wavlm(cmodel, wav_path, out_path, device):
    wav, _ = librosa.load(wav_path, sr=SR)
    wav = torch.from_numpy(wav).unsqueeze(0).to(device)  # (1, T)
    with torch.no_grad():
        c = cmodel.extract_features(wav)[0].transpose(1, 2)  # (1, 1024, T')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(c.cpu(), out_path)


def extract_spk(smodel, wav_path, out_path):
    wav, _ = librosa.load(wav_path, sr=SR)
    emb = smodel.embed_utterance(wav)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, emb)


def extract_spec(wav_path, out_path, hps):
    audio, sr = torchaudio.load(wav_path)
    audio_norm = audio / hps.data.max_wav_value
    # FreeVC's code expects (1, T) input, without extra squeeze
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = torch.squeeze(spec, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(spec, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--config',  type=str, default='configs/freevc.json')
    parser.add_argument('--val_frac', type=float, default=0.1)
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pairs = collect_pairs(args.surgery)
    print(f'{len(pairs)} patients with pre+post Speech for surgery={args.surgery}')

    train_pids = sorted(p for p in pairs if p not in TEST_PATIENTS)
    test_pids  = sorted(p for p in pairs if p in TEST_PATIENTS)
    print(f'  Train: {len(train_pids)}  Test (held out): {len(test_pids)}')

    # Build list of (src_path, tag) where tag is unique identifier like "0007_pre"
    all_items = []
    for pid in train_pids:
        all_items.append((pairs[pid]['pre'],  f'{pid}_pre'))
        all_items.append((pairs[pid]['post'], f'{pid}_post'))

    # -------- 1. Resample + save 16k wav --------
    print('\n[1/4] Resampling to 16 kHz...')
    for src, tag in tqdm(all_items):
        dst_dir = os.path.join(WAV_DIR, tag)
        dst_file = os.path.join(dst_dir, f'{tag}.wav')
        if not os.path.exists(dst_file):
            resample_save(src, dst_file, SR)

    # -------- 2. Extract WavLM features --------
    print('\n[2/4] Extracting WavLM content features...')
    print('  Loading WavLM...')
    cmodel = utils.get_cmodel(0)
    for _, tag in tqdm(all_items):
        wav_p = os.path.join(WAV_DIR, tag, f'{tag}.wav')
        out_p = os.path.join(SSL_DIR, tag, f'{tag}.pt')
        if not os.path.exists(out_p):
            extract_wavlm(cmodel, wav_p, out_p, device)
    del cmodel
    torch.cuda.empty_cache()

    # -------- 3. Extract speaker embeddings --------
    print('\n[3/4] Extracting speaker embeddings...')
    print('  Loading speaker encoder...')
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    for _, tag in tqdm(all_items):
        wav_p = os.path.join(WAV_DIR, tag, f'{tag}.wav')
        out_p = os.path.join(SPK_DIR, tag, f'{tag}.npy')
        if not os.path.exists(out_p):
            extract_spk(smodel, wav_p, out_p)
    del smodel
    torch.cuda.empty_cache()

    # -------- 4. Precompute spectrograms (saved next to wav as .spec.pt) --------
    print('\n[4/4] Precomputing spectrograms (next to wavs as .spec.pt)...')
    for _, tag in tqdm(all_items):
        wav_p  = os.path.join(WAV_DIR, tag, f'{tag}.wav')
        spec_p = os.path.join(WAV_DIR, tag, f'{tag}.spec.pt')
        if not os.path.exists(spec_p):
            extract_spec(wav_p, spec_p, hps)

    # -------- 5. Write filelists --------
    print('\n[5] Writing filelists...')
    random.seed(42)
    shuffled = [t for _, t in all_items]
    random.shuffle(shuffled)
    n_val = max(2, int(args.val_frac * len(shuffled)))
    val_tags = shuffled[:n_val]
    train_tags = shuffled[n_val:]

    # Filelist format: FreeVC expects lines like "DUMMY/speaker/file.wav"
    # We'll use absolute paths to be safe.
    os.makedirs(FILELIST_DIR, exist_ok=True)
    def write_list(path, tags):
        with open(path, 'w') as f:
            for tag in tags:
                wav_p = os.path.join(WAV_DIR, tag, f'{tag}.wav')
                f.write(f'{wav_p}\n')
    train_list = os.path.join(FILELIST_DIR, 'cuco_train.txt')
    val_list   = os.path.join(FILELIST_DIR, 'cuco_val.txt')
    write_list(train_list, train_tags)
    write_list(val_list,   val_tags)
    print(f'  train: {len(train_tags)} lines → {train_list}')
    print(f'  val:   {len(val_tags)} lines → {val_list}')

    # -------- 6. Save paths manifest --------
    manifest = {
        'train_patients': train_pids,
        'test_patients':  test_pids,
        'pairs':          pairs,
        'wav_dir':        WAV_DIR,
        'ssl_dir':        SSL_DIR,
        'spk_dir':        SPK_DIR,
    }
    with open(os.path.join(DATA_ROOT, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('\nDone. Data ready for fine-tuning.')


if __name__ == '__main__':
    main()
