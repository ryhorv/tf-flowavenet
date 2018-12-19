from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import librosa
from multiprocessing import cpu_count
import argparse
from hparams import hparams
import audio


def build_from_path(in_dir, out_dir, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    
    book_folders = [f for f in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, f))]
    for book in book_folders:
        with open(os.path.join(in_dir, book, 'metadata.csv'), encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                parts = line.strip().split('|')
                wav_path = os.path.join(in_dir, book, 'wavs', '%s.wav' % parts[0])
                try:
                    text = parts[2]
                except:
                    print(os.path.join(in_dir, book, 'metadata.csv'))
                    print(parts)
                futures.append(executor.submit(
                    partial(_process_utterance, out_dir, index, wav_path, text)))
                index += 1
    return [future.result() for future in futures]


def _process_utterance(out_dir, index, wav_path, text):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path, hparams.sample_rate)
    
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    
    wav = audio.pad_wav(wav, hparams)
    out = audio.preemphasis(wav, hparams.preemphasis, preemphasize=hparams.preemphasize)
    constant_values = 0.
    out_dtype = np.float32
    hop_size = audio.get_hop_size(hparams)
    
     # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(out_dtype).T
    mel_frames = mel_spectrogram.shape[0]
    
    if hparams.normalize_spectr:
        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)
        mel_spectrogram = np.interp(mel_spectrogram, T2_output_range, (0, 1)).astype(np.float32)  
        
    out = out[:mel_frames * hop_size]

    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_filename = 'dataset-audio-%05d.npy' % index
    mel_filename = 'dataset-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return audio_filename, mel_filename, timesteps, text


def preprocess(in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, num_workers)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames / hparams.sample_rate / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dir', '-i', type=str, default='./', help='In Directory')
    parser.add_argument('--out_dir', '-o', type=str, default='./', help='Out Directory')
    args = parser.parse_args()

    num_workers = cpu_count()
    preprocess(args.in_dir, args.out_dir, num_workers)
