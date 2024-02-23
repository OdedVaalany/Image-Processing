from scipy.io import wavfile
import numpy as np


def stft(samples, window_size, window_shift, weights=0):
    if weights == 0:
        weights = np.ones(window_size)
    start = 0
    _samples = np.zeros(
        ((samples.shape[0]-window_size)//window_shift+1, window_size), dtype='complex')
    while start + window_size < samples.shape[0]:
        _samples[int(start//window_shift),
                 :] = np.fft.fft(samples[start:start+window_size]*weights)
        start += window_shift
    return _samples, window_size, window_shift


def istft(samples, window_size, window_shift):
    _samples = np.zeros(
        (samples.shape[0]-1)*window_shift+window_size, dtype='complex')
    start = 0
    while start + window_size < _samples.shape[0]:
        _samples[start:start +
                 window_size] += np.fft.ifft(samples[int(start//window_shift), :])
        start += window_shift
    _samples[window_shift:-window_shift+1] /= 2
    return _samples, window_size, window_shift


def q1(audio_path) -> np.array:
    """
    :param audio_path: path to q1 audio file
    :return: return q1 denoised version
    """
    file = wavfile.read(filename=audio_path)
    a, b, c = stft(np.asarray(file[1], dtype='complex'), 128, 64)
    a[:, 36:37] = 0
    a[:, 92:93] = 0
    a[:, 50:79] = 0
    aa, bb, cc = istft(a, b, c)
    return np.real(aa).astype('float32')


def q2(audio_path) -> np.array:
    """
    :param audio_path: path to q2 audio file
    :return: return q2 denoised version
    """
    file = wavfile.read(filename=audio_path)
    a, b, c = stft(np.asarray(file[1], dtype='complex'), 100, 50)
    a[110:350, 14:20] /= 2
    a[111:349, 15:19] = 0
    a[110:350, 81:87] /= 2
    a[111:349, 82:86] = 0
    aa, bb, cc = istft(a, b, c)
    return np.real(aa).astype('float32')
