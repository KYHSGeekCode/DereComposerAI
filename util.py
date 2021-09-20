import os
import re
from os import listdir
from os.path import isfile, join

import librosa.display
import numpy as np
from scipy.io.wavfile import read as read_wav

wavfile_root = './wavs'
feature_root = './features'


def wav_name(music_id):
    return os.path.join(wavfile_root, f'song_{music_id:04}.wav')


def feature_name(music_id):
    return os.path.join(feature_root, f'song_{music_id:04}.npy')


def prepare_audio_feature(music_id):
    print(f"Preparing {music_id}")
    path = wav_name(music_id)
    sampling_rate, _ = read_wav(path)

    x = librosa.load(path, sampling_rate)[0]
    S = librosa.feature.melspectrogram(x, sr=sampling_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    rms = librosa.feature.rms(S=S, frame_length=255)
    print(len(rms[0]))
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=12, )
    mfcc_e = np.append(mfcc, rms, axis=0)
    delta_mfcc_e = librosa.feature.delta(mfcc_e, order=1)
    delta2_mfcc_e = librosa.feature.delta(mfcc_e, order=2)

    # plt.figure(figsize=(96, 4))
    #    librosa.display.specshow(delta2_mfcc_e, sr=sampling_rate, x_axis='time')
    #    plt.ylabel('MFCC coeffs')
    #    plt.xlabel('Time')
    #    plt.title('MFCC')
    #    plt.colorbar()
    #    plt.tight_layout()
    #    plt.show()

    print(mfcc[0])
    total_feature = np.append(mfcc_e, delta_mfcc_e, axis=0)
    total_feature = np.append(total_feature, delta2_mfcc_e, axis=0)
    print(len(total_feature))
    for line in total_feature:
        print(len(line))
        print(line)
    print(total_feature)
    feature_vectors = total_feature.transpose()
    with open(feature_name(music_id), 'wb') as f:
        np.save(f, feature_vectors)
    return feature_vectors


def read_audio_feature(music_id):
    with open(feature_name(music_id), 'rb') as f:
        return np.load(f)


def read_or_fallback_feature(music_id):
    if not isfile(feature_name(music_id)):
        return prepare_audio_feature(music_id)
    else:
        return read_audio_feature(music_id)


# 해당 폴더에 있는 음악 파일들을 반환한다.
def get_music_files():
    onlyfiles = [os.path.join(wavfile_root, f) for f in listdir(wavfile_root) if isfile(join(wavfile_root, f))]
    # ids = [re.findall("\d+", name)[0] for name in onlyfiles]
    # print(onlyfiles)
    return onlyfiles


# wavs 폴더 안의 music_ids에 해당하는 모든 파일들에 대해서 특징을 추출하여 저장한다.
def prepare_all_features(music_files):
    features = []
    for file in music_files:
        music_id = int(re.findall("\d+", file)[0])
        feature = read_or_fallback_feature(music_id)
        features.append(feature)
    return features


# # 특징들을 전부 집합에 집어넣고 글자 집합을 만든다.
# def audio_vector_set(features):
#     # 글자 집합 구축
#     src_vocab = set()
#     for one_features in features:
#         for vec in one_features:
#             src_vocab.add(tuple(vec))
#     src_vocab_size = len(src_vocab) + 1
#     print(src_vocab_size)
#     return src_vocab


def generate_or_load_audio_vec():
    files = get_music_files()
    features = prepare_all_features(files)
    # src_vocab = audio_vector_set(features)
    # with open('audio_vec.npy', 'wb') as f:
    #     np.save(f, src_vocab)
    # return features, src_vocab


def fna(f):
    music_id = int(re.findall("\d+", f)[0])
    return music_id > 1000


def load_random_vectors(files, n=60):
    files = list(filter(fna, files))
    selected = np.random.choice(files, n)
    features = []
    labels = []
    for file in selected:
        music_id = int(re.findall("\d+", file)[0])
        feature = read_or_fallback_feature(music_id)
        features.append(feature)
        labels.append(music_id)
    return labels, features
