# Timbre2Vec
7th deep daiv. 은근예민 프로젝트 : 내 음색의 정체

본 프로젝트는 오디오의 다양한 세부 분야 중 **Source Separation,** **Music Information Retrieval**을 적용하여 최종적으로 **Self-Supervised Contrastive Learning을 사용한 Representation Learning**을 진행함. 크게 3가지 태스크로 나눌 수 있다.

1. 노래에서 가수의 목소리만을 추출하는 전처리 과정을 거침
2. 전처리를 거친 mel-spectrogram을 인풋으로 넣었을 때 아웃풋의 벡터값이 음색을 나타내도록 함
3. 이를 기반으로 사전에 준비된 가수 22명 중 가장 유사한 음색의 가수 1명을 추천함


## Data Preprocessing

데이터셋을 구축하는 방법은 다음과 같다. song_df.csv를 mp3_to_mel(mp3).py 이용하여 mp3 다운, HT Demucs 사용하여 보컬 추출, 공백 제거한 파일로부터 npy 만든 후 mp3 삭제가 진행된다.
![전처리](https://github.com/Jin1025/Timbre2Vec/assets/111305638/ce29b471-f63b-4ca8-bd7e-84ba39cf8984)

input_data.csv : train 용 데이터셋 정보들
test_input_data.csv : test 및 추천에 사용되는 데이터셋 정보들

용량 제한으로 인해 깃허브에 업로드 불가하나 전처리가 완료된 npy 파일들이 필요할 경우 text me (nixnox10@naver.com)


## Model

모델 아키텍처는 다음과 같으며, 코드는 model 폴더 안에서 확인 가능하다. 
<img width="314" alt="model" src="https://github.com/Jin1025/Timbre2Vec/assets/111305638/79f3f257-e346-4e41-8fce-f1471fd36dd2">

본 프로젝트에서 사용한 3 epoch 동안 학습이 완료된 모델을 사용하기 위해서는 trained_model 폴더의 please.pt를 이용하여 아래와 같이 불러올 수 있다. 다만 개발 환경이 Mac의 mps 였기 때문에, 동일 환경에서만 사용 가능하다.

  model_path = 'please.pt'
  device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
  extract_model = Extract_Model()
  extract_model.load_state_dict(torch.load(model_path))
  extract_model.to(device)


## Demo

모델의 아웃풋을 이용해 음색 기반 가수를 추천한다. 이때 추천에 사용된 가수 목록은 test_input_data.csv에서 확인할 수 있다. 각 가수들의 아웃풋 평균과 코사인 유사도를 이용해 유사도를 측정하며, threshold가 0.7을 넘기지 못하면 추천을 진행하지 않는다.
