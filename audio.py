import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import windows


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)
    
def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def pad_wav(y, hparams):
    n_frames = (y.shape[0] - hparams.win_size) // hparams.hop_size + 1
    H = (n_frames - 1) * hparams.hop_size + hparams.win_size
    padding = hparams.hop_size - (y.shape[0] - H)
    return np.pad(y, (0, padding), mode='constant', constant_values=0.)

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def _stft(y, hparams):
    win_size = hparams.win_size if hparams.win_size is not None else hparams.n_fft
    window = windows.hann(win_size)
    frames = librosa.util.frame(y, frame_length=win_size, hop_length=get_hop_size(hparams)) # (win_size, n_frames)
    weighted_frames = window[:, np.newaxis] * frames
    fft_res = fft.fft(weighted_frames, n=hparams.n_fft, axis=0)
    output_size = int(1 + hparams.n_fft // 2)
    
    return fft_res[:output_size]
        #return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)


def _istft(S, hparams):
    win_size = hparams.win_size if hparams.win_size is not None else hparams.n_fft
    hop_size = get_hop_size(hparams)
    window = windows.hann(win_size)
    
    S = np.concatenate([S, S[-2:0:-1]], axis=0)
    ifft = fft.ifft(S, n=hparams.n_fft, axis=0)[:win_size].real
    weighted_ifft = window[:, np.newaxis] * ifft 
    
    res_size = (weighted_ifft.shape[1] - 1) * hop_size + weighted_ifft.shape[0]
    
    result = np.zeros((res_size,), dtype=np.float32)
    i = 0
    for frame in weighted_ifft.T:
        result[i:i+frame.shape[0]] += frame
        i += hop_size
    
    return result

def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
             -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)