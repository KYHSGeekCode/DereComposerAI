import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def read_wav(filename):
    desired_samples = 16000 * 60
    wav_loader = tf.io.read_file(filename)
    # [?, 1] 형태의 Int 배열을 -1 ~ 1 사이의 Float 형태, [16000, 1]의 형태로 정규화 해줍니다
    # ?가 16000보다 작으면 0으로 채워지고 넘으면 뒤에는 버려집니다
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1,
                                      desired_samples=desired_samples)
    # 고속 푸리에 변환을 통해 일렬로 된 음성데이터 파일의 시간축을 주파수축으로 바꾸어서 3차원 배열을 만들어줍니다
    # 아래의 코드는 [1, 98, 257]로 변환됩니다
    # 첫번째 1은 Channel을 뜻하며 스테레오 음성은 2, 5.1 채널 음성데이터는 그에 맞는 숫자가 나올것입니다
    # 두번째 98은 시간축을 의미하며 1초의 음성데이터를 10ms 간격으로 자르고 맨앞과 맨뒤를 버리고 98개가 남은것입니다
    # 세번째 257은 주파수별 음압(dB)의 크기를 담고 있습니다 257종류의 주파수별 dB 크기를 담고있다고 보면 되겠습니다
    spectrogram = contrib_audio.audio_spectrogram(wav_decoder.audio,
                                                  window_size=window_size_samples,
                                                  stride=window_stride_samples,
                                                  magnitude_squared=True)

    # 고속 푸리에 변환을 통해 얻은 스펙트로그램을 MFCC 기법을 통해 머신러닝에 유리한 정보만을 추려냅니다
    # 아래의 코드는 [1, 98, 257]의 배열을 받아서, [1, 98, 40]의 배열로 변합니다
    # 사람은 고주파의 소리는 잘 못듣고 저주파의 소리엔 민감하다고 합니다
    # 이를 통해 좀더 효율적으로 소리의 특징을 추출하기 위해 연구한것이 MFCC라고 보면 됩니다
    # 257종류의 주파수중에 40개만의 주파수만 쓰려고 골라냈다고 생각하시면 편합니다
    output_ = contrib_audio.mfcc(spectrogram,
                                 wav_decoder.sample_rate,
                                 dct_coefficient_count=fingerprint_width)

    # 신경망에 데이터를 1열로 넣어주기 위해서 길게 펴줍니다
    # [1, 98, 40]의 데이터가 들어가서 [1, 3920] 형태의 데이터가 나옵니다
    out = tf.compat.v1.layers.Flatten()(output_)
