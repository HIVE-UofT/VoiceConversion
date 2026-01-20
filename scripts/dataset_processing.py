import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, Audio, load_from_disk
import math
import librosa
import numpy as np

# data_root = Path("../CUCO/data_final/Audios")

# data_list = []

# surgeries = ["Fess", "Sept", "Tonsill"]

# for surgery in surgeries:
#     surgery_path = data_root / surgery / "Speech"
#     for folder_num in ["1"]:
#         folder_path = surgery_path / folder_num
        
#         if not folder_path.exists():
#             continue
            
#         for audio_file in folder_path.glob("*.wav"):   
#             data_list.append({
#                 "audio_pre": str(audio_file),
#                 "audio_post": str(audio_file).replace("_ses1_", "_ses2_").replace("/1/", "/2/"),
#                 "audio_dir_pre": str(audio_file),
#                 "audio_dir_post": str(audio_file).replace("_ses1_", "_ses2_").replace("/1/", "/2/"),
#                 "surgery_type": surgery.lower(),
#                 "file_id": audio_file.stem 
#             })

# ds = Dataset.from_list(data_list)

# ds = ds.cast_column("audio_pre", Audio(sampling_rate=16000))
# ds = ds.cast_column("audio_post", Audio(sampling_rate=16000))

# def compute_dual_mels(batch):
    
#     pre_arrays = [x["array"] for x in batch["audio_pre"]]
#     post_arrays = [x["array"] for x in batch["audio_post"]]
#     sr = batch["audio_pre"][0]["sampling_rate"] 
    
#     mels_pre = []
#     mels_post = []
    
#     for y_pre, y_post in zip(pre_arrays, post_arrays):
#         S_pre = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
#         mels_pre.append(librosa.power_to_db(S_pre, ref=np.max))
        
#         S_post = librosa.feature.melspectrogram(y=y_post, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
#         mels_post.append(librosa.power_to_db(S_post, ref=np.max))
        
#     batch["mel_pre"] = mels_pre
#     batch["mel_post"] = mels_post
#     return batch

# ds_processed = ds.map(compute_dual_mels, batched=True, batch_size=8)

# ds_processed.save_to_disk("../CUCO/processed_dataset")


def process_and_segment_5s(batch):
    sr = 16000
    segment_len = 5 * sr  # 80,000 samples
    
    new_batch = {
        "mel_pre": [], "mel_post": [],
        "wav_pre": [], "wav_post": [],
        "surgery_type": [], "file_id": []
    }
    
    for i in range(len(batch["audio_pre"])):
        # Convert to numpy arrays
        y_pre = np.array(batch["audio_pre"][i]["array"])
        y_post = np.array(batch["audio_post"][i]["array"])
        
        # Determine segments based on the longer file to ensure we don't miss data
        max_len = max(len(y_pre), len(y_post))
        num_segments = math.ceil(max_len / segment_len)
        
        for s in range(num_segments):
            start = s * segment_len
            end = start + segment_len
            
            # Slice and Pad (Fixed length of 80,000)
            slice_pre = y_pre[start:end]
            if len(slice_pre) < segment_len:
                slice_pre = np.pad(slice_pre, (0, segment_len - len(slice_pre)))
                
            slice_post = y_post[start:end]
            if len(slice_post) < segment_len:
                slice_post = np.pad(slice_post, (0, segment_len - len(slice_post)))
            
            # Compute Log-Mel Spectrograms
            S_pre = librosa.feature.melspectrogram(
                y=slice_pre, sr=sr, n_mels=128, n_fft=1024, hop_length=512
            )
            mel_pre = librosa.power_to_db(S_pre, ref=1.0)
            
            S_post = librosa.feature.melspectrogram(
                y=slice_post, sr=sr, n_mels=128, n_fft=1024, hop_length=512
            )
            mel_post = librosa.power_to_db(S_post, ref=1.0)
            
            # Store everything
            new_batch["mel_pre"].append(mel_pre)
            new_batch["mel_post"].append(mel_post)
            new_batch["wav_pre"].append(slice_pre)
            new_batch["wav_post"].append(slice_post)
            new_batch["surgery_type"].append(batch["surgery_type"][i])
            new_batch["file_id"].append(f"{batch['file_id'][i]}_seg{s}")
            
    return new_batch


ds = load_from_disk("../CUCO/processed_dataset")
ds_segmented = ds.map(
    process_and_segment_5s, 
    batched=True, 
    batch_size=4,
    remove_columns=ds.column_names 
)
ds_segmented.save_to_disk("../CUCO/segmented_dataset")