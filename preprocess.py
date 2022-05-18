import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
import librosa
from config import Config
import numpy as np
from tqdm import tqdm




def _compute_melspectrogram(wav):
    """Compute the mel-spectrogram
    """
    # Apply pre-emphasis
    wav = librosa.effects.preemphasis(wav, coef=0.97)

    # Compute the mel spectrogram
    mel = librosa.feature.melspectrogram(y=wav,
                                         sr=Config.sampling_rate,
                                         hop_length=Config.hop_length,
                                         win_length=Config.win_length,
                                         n_fft=Config.n_fft,
                                         n_mels=Config.num_mels,
                                         fmin=Config.fmin,
                                         norm=1,
                                         power=1)

    # Convert to log scale
    mel = librosa.core.amplitude_to_db(mel, top_db=None) - Config.ref_db

    # Normalize
    mel = np.maximum(mel, -Config.max_db)
    mel = mel / Config.max_db

    return mel


# Compressing waveform using mu-law compression
def _mulaw_compression(wav):
    wav = np.pad(wav, (Config.win_length // 2, ), mode="reflect")
    wav = wav[:((wav.shape[0] - Config.win_length) // Config.hop_length + 1) * Config.hop_length]
    wav = 2**(Config.num_bits - 1) + librosa.mu_compress(wav, mu=2 ** Config.num_bits - 1)

    return wav


def process_wav(mel_dir, qwav_dir, wav_path):
    """Process a single wav file
    This writes the mel spectrogram as well as the quantized wav to disk and returns a tuple to write to the train.txt
    file
    """
    filename = os.path.splitext(os.path.basename(wav_path))[0]

    # Load wav file from disk
    wav, _ = librosa.load(wav_path, sr=Config.sampling_rate)

    peak = np.abs(wav).max()
    if peak >= 1:
        wav = wav / peak * 0.999

    # Compute mel spectrogram
    mel = _compute_melspectrogram(wav)

    # Quantize the wavform
    qwav = _mulaw_compression(wav)

    # Save to disk
    mel_path = os.path.join(mel_dir, filename + ".npy")
    qwav_path = os.path.join(qwav_dir, filename + ".npy")
    np.save(mel_path, mel)
    np.save(qwav_path, qwav)

    return filename, mel.shape[-1]


def write_metadata(metadata, out_dir):
    """Write the metadata to train.txt file
    """
    with open(os.path.join(out_dir, "train.txt"), "w") as file_writer:
        for m in metadata:
            file_writer.write(m[0] + "\n")

    frames = sum([m[1] for m in metadata])
    frame_shift_ms = Config.hop_length / Config.sampling_rate * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)

    print(
        f"Wrote {len(metadata)} utterances, {frames} frames, {hours:2f} hours")


def build_from_path_ljspeech(input_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    """Preprocess the LJSpeech dataset from a given input path into a given output directory
    """
    mel_dir = os.path.join(out_dir, "mel")
    qwav_dir = os.path.join(out_dir, "qwav")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(qwav_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    with open(os.path.join(input_dir, "metadata.csv"), "r") as file_reader:
        for line in file_reader:
            parts = line.strip().split("|")
            wav_path = os.path.join(input_dir, "wavs", f"{parts[0]}.wav")
            futures.append(
                executor.submit(
                    partial(process_wav, mel_dir, qwav_dir, wav_path)))

    return [future.result() for future in tqdm(futures)]


# Dataset Preprocessing
def preprocess(input_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)

    if Config.dataset == "ljspeech":
        metadata = build_from_path_ljspeech(input_dir, out_dir, num_workers, tqdm=tqdm)
    else:
        raise NotImplementedError

    write_metadata(metadata, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")

    parser.add_argument("--dataset_dir", help="Directory containing the Dataset", required=True)
    parser.add_argument("--out_dir", help="Directory to store output", required=True)

    args = parser.parse_args()
    num_workers = cpu_count()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir

    preprocess(dataset_dir, out_dir, num_workers)
