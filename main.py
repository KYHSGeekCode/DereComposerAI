import os
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
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
    print(onlyfiles)
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


def setup_folders():
    Path("./features").mkdir(parents=True, exist_ok=True)



if __name__ == '__main__':
    setup_folders()
    generate_or_load_audio_vec()

# total_feature: 37 * 10869
# feature_vectors: 10869 * 37


# # JSON 글자 집합 준비
#
#
# # JSON 글자 집합 구축
# tar_vocab = set()
# for vec in tar_vectors:
#     tar_vocab.add(tuple(vec))
# tar_vocab_size = len(tar_vocab) + 1
# print(tar_vocab_size)
#
# encoder_inputs = Input(shape=(None, src_vocab_size))
# encoder_lstm = LSTM(units=256, return_state=True)
# encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# # encoder_outputs도 같이 리턴받기는 했지만 여기서는 필요없으므로 이 값은 버림.
# encoder_states = [state_h, state_c]
# # LSTM은 바닐라 RNN과는 달리 상태가 두 개. 바로 은닉 상태와 셀 상태
#
# decoder_inputs = Input(shape=(None, tar_vocab_size))
# decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# # 디코더의 첫 상태를 인코더의 은닉 상태, 셀 상태로 합니다.
# decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
# decoder_outputs = decoder_softmax_layer(decoder_outputs)
#
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
#
# model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=50, validation_split=0.2)
#
# encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
#
# # 이전 시점의 상태들을 저장하는 텐서
# decoder_state_input_h = Input(shape=(256,))
# decoder_state_input_c = Input(shape=(256,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# # 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
# decoder_states = [state_h, state_c]
# # 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
# decoder_outputs = decoder_softmax_layer(decoder_outputs)
# decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
