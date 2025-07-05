import os
import shutil
import numpy as np
from scipy.io import wavfile
import webrtcvad
import glob

def read_wav(filepath):
    sample_rate, audio = wavfile.read(filepath)
    return sample_rate, audio

def write_wav(filepath, sample_rate, audio):
    wavfile.write(filepath, sample_rate, audio)

def detect_voice_activity(audio, sample_rate, vad, frame_duration_ms=30):
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    num_frames = len(audio) // frame_size

    voiced_segments = []
    for i in range(num_frames):
        start_index = i * frame_size
        end_index = start_index + frame_size
        frame = audio[start_index:end_index]

        if len(frame) < frame_size:
            break

        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if is_speech:
            voiced_segments.append(frame)

    if voiced_segments:
        return np.concatenate(voiced_segments)
    else:
        return np.array([])

def process_and_save_files(file_list, source_folder, target_folder, vad_level=2):
    vad = webrtcvad.Vad(vad_level)
    os.makedirs(target_folder, exist_ok=True)

    for filepath in file_list:
        rel_path = os.path.relpath(filepath, source_folder)
        target_path = os.path.join(target_folder, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        sample_rate, audio = read_wav(filepath)
        duration = len(audio) / sample_rate

        if duration > 0:
            cropped_audio = detect_voice_activity(audio, sample_rate, vad)
            if cropped_audio.size > 0:
                write_wav(target_path, sample_rate, cropped_audio)
            else:
                write_wav(target_path, sample_rate, audio)
        else:
            shutil.copyfile(filepath, target_path)

        print(f"Processed: {filepath} -> {target_path}")

# Base directory with language folders
base_dir = "indicsuperb_qbe_testset/qbe_indicsuperb"
target_folders = {"Audio", "eval_queries"}

# Process each language
for language in os.listdir(base_dir):
    language_path = os.path.join(base_dir, language)
    if not os.path.isdir(language_path):
        continue

    for subfolder in os.listdir(language_path):
        if subfolder not in target_folders:
            continue

        subfolder_path = os.path.join(language_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        wav_files = glob.glob(os.path.join(subfolder_path, "*.wav"))
        if not wav_files:
            continue

        # Output folder with _vad suffix
        target_subfolder = os.path.join(language_path, subfolder + "_vad")
        process_and_save_files(wav_files, subfolder_path, target_subfolder)
