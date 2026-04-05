import os
import csv
import glob
import librosa
import numpy as np
import pickle
import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURATION ---
DATASET_ROOT = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech"
CLINICAL_DIR = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Clinical"
FINAL_PATH = Path("/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data")
FOLDERS = {'1': 0, '2': 1}
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

# Mel Spectrogram Params
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80

# 13 clinical features:
#   age_norm, gender, height_norm, weight_norm, smoker, osa,
#   tonsillar_grade_norm, nasality_norm, G/3, R/3, A/3, B/3, S/3
META_DIM = 13

TONSILLAR_MAP = {'i': 0.25, 'ii': 0.5, 'iii': 0.75, 'iv': 1.0}
WEIGHT_COL  = {1: 'WEIGHT',  2: 'WEIGHT2',  3: 'WEIGHT3'}
NASALITY_COL = {1: 'NASALITY TEST', 2: 'NASALITY TEST2', 3: 'NASALITY TEST3'}


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_metadata_lookup(clinical_dir):
    """Build {(patient_id_int, session_int): np.float32 array of shape (META_DIM,)}."""
    lookup = {}
    for ses in [1, 2, 3]:
        csv_path = os.path.join(clinical_dir, f"clinical_Ses{ses}.csv")
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pid = int(row['ID'])
                except (ValueError, KeyError):
                    continue

                age    = _safe_float(row.get('AGE'))
                gender = 1.0 if row.get('GENDER', '').strip().lower() == 'female' else 0.0
                height = _safe_float(row.get('SIZE'))
                weight = _safe_float(row.get(WEIGHT_COL[ses]))
                smoker = 1.0 if row.get('SMOKER', '').strip().upper() == 'TRUE' else 0.0
                osa    = 1.0 if row.get('OSA', '').strip().upper() == 'TRUE' else 0.0
                tg_raw = row.get('TONSILLAR GRADE', '').strip().lower()
                tg     = TONSILLAR_MAP.get(tg_raw, 0.0)
                nas    = _safe_float(row.get(NASALITY_COL[ses]))
                g      = _safe_float(row.get('G'))
                r      = _safe_float(row.get('R'))
                a      = _safe_float(row.get('A'))
                b      = _safe_float(row.get('B'))
                s      = _safe_float(row.get('S'))

                feat = np.array([
                    age / 100.0,
                    gender,
                    height / 200.0,
                    weight / 150.0,
                    smoker,
                    osa,
                    tg,
                    nas / 100.0,
                    g / 3.0,
                    r / 3.0,
                    a / 3.0,
                    b / 3.0,
                    s / 3.0,
                ], dtype=np.float32)

                lookup[(pid, ses)] = feat

    return lookup


def get_patient_id(filename):
    """Extracts zero-padded ID string from filename like 'Tonsill_ses1_speech_0007.wav'."""
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    return name_without_ext.split('_')[-1]


def get_session(filename):
    """Extracts session number (1,2,3) from filename like 'Tonsill_ses1_speech_0007.wav'."""
    base = os.path.basename(filename)
    parts = base.split('_')
    for part in parts:
        if part.startswith('ses') and len(part) > 3:
            try:
                return int(part[3:])
            except ValueError:
                pass
    return 1  # default to session 1 if not found


def process_file(file_path, status, metadata_lookup):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=30)

        pid_str = get_patient_id(file_path)
        pid_int = int(pid_str)
        session = get_session(file_path)

        # Fetch metadata; fall back to Ses1 then zeros if missing
        meta = metadata_lookup.get((pid_int, session),
               metadata_lookup.get((pid_int, 1),
               np.zeros(META_DIM, dtype=np.float32)))

        step = SAMPLES_PER_SEGMENT
        file_data = []

        for i, start_idx in enumerate(range(0, len(y), step)):
            end_idx = start_idx + step
            segment_audio = y[start_idx:end_idx]

            if len(segment_audio) < step:
                if len(segment_audio) < (SAMPLE_RATE * 2):
                    continue
                segment_audio = np.pad(segment_audio, (0, step - len(segment_audio)), mode='constant')

            mel_spec = librosa.feature.melspectrogram(
                y=segment_audio, sr=sr, n_fft=N_FFT,
                hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            mel_spec_db   = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = np.clip((mel_spec_db + 80) / 80, 0, 1)

            file_data.append({
                "segment_id": f"{pid_str}_seg_{i}",
                "mel_spectrogram": mel_spec_norm.astype(np.float32),
                "label": status,
                "patient_id": pid_str,
                "metadata": meta,          # shape (META_DIM,)
            })

        return file_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def main():
    print("Loading clinical metadata...")
    metadata_lookup = build_metadata_lookup(CLINICAL_DIR)
    print(f"  Loaded metadata for {len(metadata_lookup)} (patient, session) pairs.")

    patient_to_files = {}

    print("Scanning directories...")
    for folder_name, status in FOLDERS.items():
        folder_path = os.path.join(DATASET_ROOT, folder_name)
        wav_files = glob.glob(os.path.join(folder_path, "**/*.wav"), recursive=True)

        for f in wav_files:
            pid = get_patient_id(f)
            if pid not in patient_to_files:
                patient_to_files[pid] = []
            patient_to_files[pid].append((f, status))

    all_pids = list(patient_to_files.keys())
    if not all_pids:
        print("No files found! Check your DATASET_ROOT path.")
        return

    train_pids, temp_pids = train_test_split(all_pids, test_size=0.3, random_state=42)
    val_pids, test_pids   = train_test_split(temp_pids, test_size=0.5, random_state=42)

    split_map = {'train': train_pids, 'val': val_pids, 'test': test_pids}

    os.makedirs(FINAL_PATH, exist_ok=True)

    for set_name, pid_list in split_map.items():
        print(f"\n--- Processing {set_name} set ({len(pid_list)} patients) ---")
        set_data = []

        for pid in tqdm.tqdm(pid_list):
            for file_path, status in patient_to_files[pid]:
                segments = process_file(file_path, status, metadata_lookup)
                set_data.extend(segments)

        save_file = FINAL_PATH / f"{set_name}_dataset.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(set_data, f)
        print(f"Saved {len(set_data)} segments to {save_file}")

    print("\nProcessing Complete!")


if __name__ == "__main__":
    main()
