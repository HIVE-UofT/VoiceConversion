import os
import glob
import librosa
import numpy as np
import pickle
import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURATION ---
DATASET_ROOT = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech"
FINAL_PATH = Path("/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data")
FOLDERS = {'1': 0, '2': 1}  
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5  
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

# Mel Spectrogram Params
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80

def get_patient_id(filename):
    """Extracts ID from filename like 'audio_0147.wav'"""
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    return name_without_ext.split('_')[-1]

def process_file(file_path, status):
    try:
        # Load audio (mono=True is default and usually best for speech)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Gentle Trim
        y, _ = librosa.effects.trim(y, top_db=30) 

        # We want segments of exactly 5 seconds
        step = SAMPLES_PER_SEGMENT
        file_data = []
        
        # Loop through the audio in 5-second chunks
        for i, start_idx in enumerate(range(0, len(y), step)):
            end_idx = start_idx + step
            segment_audio = y[start_idx:end_idx]

            # Handle the last segment: if it's > 2 seconds, pad it. If shorter, discard.
            if len(segment_audio) < step:
                if len(segment_audio) < (SAMPLE_RATE * 2): # Less than 2 seconds
                    continue 
                segment_audio = np.pad(segment_audio, (0, step - len(segment_audio)), mode='constant')
            
            # Compute Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment_audio, sr=sr, n_fft=N_FFT, 
                hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            
            # Convert to Decibels
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalization to [0, 1] based on -80dB floor
            mel_spec_norm = (mel_spec_db + 80) / 80
            mel_spec_norm = np.clip(mel_spec_norm, 0, 1)

            file_data.append({
                "segment_id": f"{get_patient_id(file_path)}_seg_{i}",
                "mel_spectrogram": mel_spec_norm.astype(np.float32),
                "label": status,
                "patient_id": get_patient_id(file_path)
            })
            
        return file_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    # 1. Gather all files
    patient_to_files = {} 

    print("Scanning directories...")
    for folder_name, status in FOLDERS.items():
        folder_path = os.path.join(DATASET_ROOT, folder_name)
        # Use recursive search to be safe
        wav_files = glob.glob(os.path.join(folder_path, "**/*.wav"), recursive=True)
        
        for f in wav_files:
            pid = get_patient_id(f)
            if pid not in patient_to_files:
                patient_to_files[pid] = []
            patient_to_files[pid].append((f, status))

    all_pids = list(patient_to_files.keys())
    if not all_pids:
        print("❌ No files found! Check your DATASET_ROOT path.")
        return

    # 2. Split Patients (70/15/15)
    train_pids, temp_pids = train_test_split(all_pids, test_size=0.3, random_state=42)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)

    split_map = {
        'train': train_pids,
        'val': val_pids,
        'test': test_pids
    }

    # 3. Process and Save
    os.makedirs(FINAL_PATH, exist_ok=True)

    for set_name, pid_list in split_map.items():
        print(f"\n--- Processing {set_name} set ({len(pid_list)} patients) ---")
        set_data = []
        
        for pid in tqdm.tqdm(pid_list):
            for file_path, status in patient_to_files[pid]:
                segments = process_file(file_path, status)
                set_data.extend(segments)
        
        save_file = FINAL_PATH / f"{set_name}_dataset.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(set_data, f)
        print(f"Saved {len(set_data)} segments to {save_file}")

    print("\n✅ Processing Complete!")

if __name__ == "__main__":
    main()