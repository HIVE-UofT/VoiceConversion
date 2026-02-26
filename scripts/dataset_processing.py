import os
import glob
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURATION ---
DATASET_ROOT = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech"  # Root folder containing '1' and '2'
FINAL_PATH = Path("/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data")
FOLDERS = {'1': 0, '2': 1}  # Folder Name -> Surgery Status Label
SAMPLE_RATE = 16000
SEGMENT_DURATION = 5  # seconds
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

# Mel Spectrogram Params
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80

def get_patient_id(filename):
    """
    Extracts 0147 from 'Tonsill_ses1_speech_0147.wav'
    Assumes the ID is the last part before the extension.
    """
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    return name_without_ext.split('_')[-1]

def process_file(file_path, status):
    """
    Loads audio, splits into 5s segments, computes Mel Specs, 
    and returns a list of data dictionaries.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Calculate necessary padding
        total_samples = len(y)
        remainder = total_samples % SAMPLES_PER_SEGMENT
        
        if remainder != 0:
            pad_length = SAMPLES_PER_SEGMENT - remainder
            y = np.pad(y, (0, pad_length), mode='constant')
            
        num_segments = len(y) // SAMPLES_PER_SEGMENT
        file_data = []

        for i in range(num_segments):
            start = i * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            segment_audio = y[start:end]
            
            # Compute Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment_audio, 
                sr=sr, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                n_mels=N_MELS
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Create data object
            data_point = {
                "audio_path": file_path,
                "mel_spectrogram": mel_spec_db, # Shape: (n_mels, time_steps)
                "sequence_id": i, # 0 for 1st segment, 1 for 2nd, etc.
                "surgery_status": status, # 0 or 1
                "patient_id": get_patient_id(file_path)
            }
            file_data.append(data_point)
            
        return file_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    # 1. Gather all files and group by Patient ID
    patient_files = {} # { '0147': [ ('path/to/file1.wav', 0), ('path/to/file2.wav', 1) ] }

    print("Scanning files...")
    for folder_name, status in FOLDERS.items():
        folder_path = os.path.join(DATASET_ROOT, folder_name)
        # Search for .wav files (recursive search if needed, currently flat)
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        for f in wav_files:
            pid = get_patient_id(f)
            if pid not in patient_files:
                patient_files[pid] = []
            patient_files[pid].append((f, status))

    all_patients = list(patient_files.keys())
    print(f"Found {len(all_patients)} unique patients.")

    # 2. Split Patients (70/15/15)
    # First split: Train (70%) vs Temp (30%)
    train_ids, temp_ids = train_test_split(all_patients, test_size=0.3, random_state=42)
    # Second split: Validation (15% of total) vs Test (15% of total) -> split Temp in half
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 3. Process Data and Assign to Sets
    datasets = {
        'train': [],
        'val': [],
        'test': []
    }

    # Helper to process list of IDs
    def process_id_list(id_list, set_name):
        print(f"Processing {set_name} set...")
        for pid in id_list:
            files = patient_files[pid]
            for file_path, status in files:
                segments = process_file(file_path, status)
                datasets[set_name].extend(segments)

    process_id_list(train_ids, 'train')
    process_id_list(val_ids, 'val')
    process_id_list(test_ids, 'test')

    # 4. Save the datasets
    # Using pickle for simplicity, but for huge datasets consider HDF5 or saving individually
    os.makedirs(FINAL_PATH, exist_ok=True)
    
    for set_name, data in datasets.items():
        save_path = FINAL_PATH / f"{set_name}_dataset.pkl"
        print(f"Saving {set_name} dataset with {len(data)} segments to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    print("Done!")

if __name__ == "__main__":
    main()