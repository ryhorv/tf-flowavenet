from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import librosa
from multiprocessing import cpu_count
import argparse
from hparams import hparams
from tqdm import tqdm
from tfrecord import TFRecordCreator


def build_from_path(in_dir, out_dir, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    
    if hparams.gin_channels > 0:
        speakers = [f for f in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, f))]
        books = []
        with open('speakers.txt', 'wt', encoding='utf-8') as f:
            for i, speaker in enumerate(speakers):
                f.write('%s - %i\n' % (speaker, i))
                speaker_books = [f for f in os.listdir(os.path.join(in_dir, speaker)) if os.path.isdir(os.path.join(in_dir, speaker, f))]
                for speaker_book in speaker_books:
                    book_path = os.path.join(in_dir, speaker, speaker_book)
                    books.append((i, book_path))
            f.flush()
    else:
        books = [(0, os.path.join(in_dir, f)) for f in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, f))]
                
    for speaker_id, book in books:
        with open(os.path.join(book, 'metadata.csv'), encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                parts = line.strip().split('|')
                wav_path = os.path.join(book, 'wavs', '%s.wav' % parts[0])
                try:
                    text = parts[2]
                except:
                    print(os.path.join(book, 'metadata.csv'))
                    print(parts)
                futures.append(executor.submit(
                    partial(_process_utterance, out_dir, index, wav_path, text, speaker_id)))
                index += 1
    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(out_dir, index, wav_path, text, speaker_id):
    wav, sr = librosa.load(wav_path, sr=hparams.sample_rate)

    wav = wav / np.abs(wav).max() * hparams.rescaling_max
    out = wav
    constant_values = 0.0
    out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = librosa.feature.melspectrogram(wav, 
                                                     sr=sr, 
                                                     n_fft=hparams.n_fft, 
                                                     hop_length=hparams.hop_size, 
                                                     n_mels=hparams.num_mels, 
                                                     fmin=hparams.fmin, 
                                                     fmax=hparams.fmax).T

    # mel_spectrogram = np.round(mel_spectrogram, decimals=2)
    mel_spectrogram = 20 * np.log10(np.maximum(1e-4, mel_spectrogram)) - hparams.ref_level_db
    mel_spectrogram = np.clip((mel_spectrogram - hparams.min_level_db) / (-hparams.min_level_db), 0, 1)

    pad = (out.shape[0] // hparams.hop_size + 1) * hparams.hop_size - out.shape[0]
    pad_l = pad // 2
    pad_r = pad // 2 + pad % 2

    # zero pad for quantized signal
    out = np.pad(out, (pad_l, pad_r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * hparams.hop_size

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * hparams.hop_size]
    assert len(out) % hparams.hop_size == 0

    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_filename = 'dataset-audio-%05d.npy' % index
    mel_filename = 'dataset-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, 'audios', audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, 'mels', mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return audio_filename, mel_filename, timesteps, speaker_id, text


def preprocess(in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'audios'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mels'), exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, num_workers)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames / hparams.sample_rate / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[4]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))

    print('Creating tfrecords...')
    creator = TFRecordCreator(os.path.join(out_dir, 'train.txt'), hparams)
    creator.create_tfrecords()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dir', '-i', type=str, default='./', help='In Directory')
    parser.add_argument('--out_dir', '-o', type=str, default='./', help='Out Directory')
    args = parser.parse_args()

    num_workers = cpu_count()
    preprocess(args.in_dir, args.out_dir, num_workers)
    
