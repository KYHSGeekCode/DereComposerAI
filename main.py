from pathlib import Path

from util import generate_or_load_audio_vec


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
