"""
Zero-shot FreeVC evaluation on Tonsill test patients.

For each test patient:
  - Source audio: their pre-surgery Speech recording
  - Reference audio: AVG speaker embedding over all training post audio
  - Output: FreeVC conversion of pre -> "post-like" voice
  - Metric: ECAPA cosine similarity to actual test post audio (vs baseline pre-vs-post)
"""
import argparse
import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path

# Add FreeVC to path and cd into it (FreeVC expects to run from its own dir)
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEVC_DIR = os.path.join(EXP_DIR, 'FreeVC')
sys.path.insert(0, FREEVC_DIR)
os.chdir(FREEVC_DIR)

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder

TEST_PATIENTS = ['0045', '0085', '0110', '0122', '0132']
CUCO_BASE = '/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios'


def get_speech_paths(surgery='Tonsill', session='1'):
    """Return dict {pid: path} for speech files."""
    import glob
    pattern = os.path.join(CUCO_BASE, surgery, 'Speech', session, '*.wav')
    result = {}
    for p in sorted(glob.glob(pattern)):
        stem = Path(p).stem
        pid = stem.split('_')[-1]
        result[pid] = p
    return result


def load_audio(path, device, target_sr=16000):
    sig, sr = torchaudio.load(path)
    if sr != target_sr:
        sig = torchaudio.functional.resample(sig, sr, target_sr)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    return sig.to(device)  # (1, T)


def load_audio_np(path, target_sr=16000):
    wav, _ = librosa.load(path, sr=target_sr)
    return wav


def compute_avg_spk_embedding(smodel, train_post_paths):
    """Compute averaged speaker embedding over all training post audio."""
    embs = []
    for p in train_post_paths:
        wav = load_audio_np(p)
        emb = smodel.embed_utterance(wav)
        embs.append(emb)
    avg = np.mean(np.stack(embs, axis=0), axis=0)
    # L2 normalize (SpeakerEncoder embeddings are typically normalized)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return torch.from_numpy(avg).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--config',  type=str, default='configs/freevc.json')
    parser.add_argument('--ckpt',    type=str,
                        default=os.path.join(EXP_DIR, 'checkpoints/freevc.pth'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(EXP_DIR, 'converted_test_zeroshot'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    hps = utils.get_hparams_from_file(args.config)

    print('Loading FreeVC synthesiser...')
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_g.eval()
    _ = utils.load_checkpoint(args.ckpt, net_g, None, True)

    print('Loading WavLM content encoder...')
    cmodel = utils.get_cmodel(0)

    if hps.model.use_spk:
        print('Loading speaker encoder...')
        smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    else:
        smodel = None

    pre_paths  = get_speech_paths(args.surgery, '1')
    post_paths = get_speech_paths(args.surgery, '2')
    train_pids = sorted(p for p in pre_paths if p not in TEST_PATIENTS)
    test_pids  = [p for p in TEST_PATIENTS if p in pre_paths and p in post_paths]

    # Reference speaker embedding (mean over training post)
    print(f'\nComputing avg speaker embedding over {len(train_pids)} training patients...')
    train_post = [post_paths[p] for p in train_pids if p in post_paths]
    if smodel is not None:
        g_tgt = compute_avg_spk_embedding(smodel, train_post).unsqueeze(0).to(device)
        print(f'Reference speaker embedding shape: {tuple(g_tgt.shape)}')
    else:
        # use_spk=False: model uses mel of reference (tgt) directly, not embedding
        g_tgt = None

    # Load ECAPA for evaluation metric
    print('\nLoading ECAPA-TDNN for metric...')
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})
    for p in ecapa.mods.parameters():
        p.requires_grad = False

    def ecapa_emb(path_or_wav):
        if isinstance(path_or_wav, str):
            sig, sr = torchaudio.load(path_or_wav)
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, sr, 16000)
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
        else:
            sig = path_or_wav
        with torch.no_grad():
            return ecapa.encode_batch(sig.to(device)).squeeze().cpu()

    # --------- Evaluate on test patients ---------
    print('\n' + '=' * 60)
    print(f'  FreeVC (zero-shot) — ECAPA Eval on Test Patients')
    print('=' * 60)
    print(f'     Patient    Baseline   Converted     Delta')
    print(f'  ----------------------------------------------')

    rows = []
    for pid in test_pids:
        src_path  = pre_paths[pid]
        post_path = post_paths[pid]

        # Load source audio
        wav_src = load_audio(src_path, device)  # (1, T)

        # Extract content via WavLM
        with torch.no_grad():
            c = utils.get_content(cmodel, wav_src)  # (1, 1024, T')

        # Generate
        with torch.no_grad():
            if hps.model.use_spk:
                audio_gen = net_g.infer(c, g=g_tgt)
            else:
                # use reference mel (mean mel of training post)
                raise NotImplementedError('use_spk=False path not implemented here')
        audio_gen = audio_gen.squeeze().cpu()

        # Save to disk
        out_path = os.path.join(args.out_dir, f'{pid}_freevc_zeroshot.wav')
        torchaudio.save(out_path, audio_gen.unsqueeze(0),
                        hps.data.sampling_rate)

        # Metric
        emb_post = ecapa_emb(post_path)
        emb_conv = ecapa_emb(audio_gen.unsqueeze(0))
        emb_pre  = ecapa_emb(src_path)
        sim_base = F.cosine_similarity(emb_pre.unsqueeze(0),
                                        emb_post.unsqueeze(0)).item()
        sim_conv = F.cosine_similarity(emb_conv.unsqueeze(0),
                                        emb_post.unsqueeze(0)).item()
        delta = sim_conv - sim_base
        print(f'        {pid}      {sim_base:.4f}      {sim_conv:.4f}   {delta:+.4f}')
        rows.append((pid, sim_base, sim_conv, delta))

    print(f'  ----------------------------------------------')
    bases = np.array([r[1] for r in rows])
    convs = np.array([r[2] for r in rows])
    deltas = np.array([r[3] for r in rows])
    print(f'        Mean      {bases.mean():.4f}      {convs.mean():.4f}   {deltas.mean():+.4f}')
    print(f'         Std      {bases.std():.4f}      {convs.std():.4f}')

    # Also eval on training patients for reference
    print('\n' + '=' * 60)
    print(f'  FreeVC (zero-shot) [TRAIN SET] — ECAPA Eval')
    print('=' * 60)
    print(f'     Patient    Baseline   Converted     Delta')
    print(f'  ----------------------------------------------')
    train_rows = []
    for pid in train_pids:
        if pid not in post_paths:
            continue
        src_path  = pre_paths[pid]
        post_path = post_paths[pid]
        wav_src = load_audio(src_path, device)
        with torch.no_grad():
            c = utils.get_content(cmodel, wav_src)
            audio_gen = net_g.infer(c, g=g_tgt)
        audio_gen = audio_gen.squeeze().cpu()
        emb_post = ecapa_emb(post_path)
        emb_conv = ecapa_emb(audio_gen.unsqueeze(0))
        emb_pre  = ecapa_emb(src_path)
        sim_base = F.cosine_similarity(emb_pre.unsqueeze(0),
                                        emb_post.unsqueeze(0)).item()
        sim_conv = F.cosine_similarity(emb_conv.unsqueeze(0),
                                        emb_post.unsqueeze(0)).item()
        delta = sim_conv - sim_base
        print(f'        {pid}      {sim_base:.4f}      {sim_conv:.4f}   {delta:+.4f}')
        train_rows.append((pid, sim_base, sim_conv, delta))
    print(f'  ----------------------------------------------')
    tbases = np.array([r[1] for r in train_rows])
    tconvs = np.array([r[2] for r in train_rows])
    tdeltas = np.array([r[3] for r in train_rows])
    print(f'        Mean      {tbases.mean():.4f}      {tconvs.mean():.4f}   {tdeltas.mean():+.4f}')
    print(f'         Std      {tbases.std():.4f}      {tconvs.std():.4f}')

    print(f'\nConverted files saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
