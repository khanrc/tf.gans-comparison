# Generative adversarial networks comparison without cherry-picking

- 비슷한 레포가 몇개 있지만, 여기서는 최대한 논문의 구조를 그대로 구현하려 노력함
- 물론 100% 그대로 재현한 건 아님. 디테일한 부분은 생략했고 - 예를들면 weight init - 논문에서 구조가 정확히 안 나온 부분 등은 임의로 구현함
- 큰 구조는 동일하게 맞추었기는 하지만 GAN 이 파라메터에 민감하여 사소한 부분에서도 결과 차이가 날 수 있다는 점은 생각하고 볼 것

[TOC]

## ToDo

- LSUN dataset
- flexible input shape
- multiple results show on tensorboard 

## 특징

- 논문의 구조를 그대로 구현하려 노력함
- 약간의 실험은 하였으나 디테일한 hyperparams-tuning 은 하지 않음
- tf queue runner 를 사용 (input pipeline)
- 동일한 trainer 하에 model 만 바꿔가며 실험할 수 있도록 구조적으로 설계함
    - 다만 설계에 실패한 듯… -.-;
- 텐서보드를 최대한 활용함
    - text summary 는 tf 1.2 에서 warning 을 냄 - tf bug
    - 아마 tf 1.3 에선 괜찮을거라고 생각하지만 테스트해보진 않음
    - 단 graph structure 는 별로 신경쓰지 않음 (scoping 신경 X)

### 유사 레포들

- 유사 레포들: [wiseodd/generative-models](https://github.com/wiseodd/generative-models), [hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections), [sanghoon/tf-exercise-gan](https://github.com/sanghoon/tf-exercise-gan), [YadiraF/GAN_Theories](https://github.com/YadiraF/GAN_Theories)
- 하지만 위 레포들은 논문에서 제안한 아키텍처를 그대로 쓰지 않고 전부 동일한 아키텍처를 사용함
- 이렇게 하면 각 모델의 특성을 제대로 볼 수 없음
- 따라서 여기서는 각 논문에서 제안한 모델을 최대한 구현함 - 디테일은 제외하고.
- https://ajolicoeur.wordpress.com/cats/
    - 요것도 비슷한 작업임 - for cats, with tuning, pytorch.

## Requirements

- python 2.7
- tensorflow 1.2
- tqdm
- (optional) pynvml - for auto gpu selection

## Models

Conditional GAN 류는 포함하지 않음 (CGAN, acGAN, SGAN 등)

- DCGAN
- LSGAN
- WGAN
- WGAN-GP
- EBGAN
- BEGAN
- DRAGAN

## Results

### DCGAN

- 네트워크 구조가 가장 간단함
- G 의 lr 을 조정했을 때 더 좋은 결과가 나옴
  - 아마 더 디테일하게 조정하면 더 좋은 결과를 볼 수 있지 않을까 싶음
  - https://ajolicoeur.wordpress.com/cats/ 에서는 5e-5 for D, 2e-4 for G 를 제안함 (for 64x64)

DCGAN.origin vs. DCGAN.G1e-3

### EBGAN

- 개인적으로 좋아하는 논문 - energy concept 이 매력적
- 다만 딱히 energy-based 라고 할만한 점이 없다는 비판도 있긴 함
- 아무튼 결과는 꽤 괜찮음 
- pt regularizer 의 효용에 대해 좀 의문이 있음 (weight=0 으로 줘도 점점 줄어듬)
  - 나는 pt weight = 0 으로 하면 mode collapse 가 발생하고 pt regularizer 가 이를 방지할거라고 생각했으나 별로 그런 느낌이 없는듯함 

ebgan.pt vs. ebgan.nopt

### LSGAN

- 특이하게 LSGAN 에서는 z_dim 을 크게 씀: 1024
- 오히려 z_dim=100 일때 결과가 더 좋았음

LSGAN.100 vs. LSGAN.1024

### WGAN

- 이론적인 논문이라서 결과가 엄청 좋지는 않음
- 네트워크 구조도 특별히 제안하지 않음

wgan.dcgan

### WGAN-GP

- DCGAN architecture / appendix C 의 ResNet architecture
- resnet 결과가 더 좋은데 기대만큼의 성능향상이 나오지는 않음
  - 특이한 점은 굉장히 빨리 수렴하고 더 학습하면 결과가 나빠짐
  - skip-connection 의 영향으로 보임
- DRAGAN 논문에서는 WGAN(-GP) 의 constraint 가 너무 restrict 해서 poor G 를 학습한다고 함

wgan-gp.dcgan vs. wgan-gp.resnet

### BEGAN

- celebA 에 대해서는 결과가 좋음 (사람이 보기에)
  - optional improvement 부분은 구현하지 않았음에도!
- 그러나 디테일이 없어지는듯한 느낌이 있음
- Q. LSUN 등 다른 데이터셋에 대해서도 결과가 잘 나올까? - ToDo

### DRAGAN

- Game theory 에서의 접근이 굉장히 흥미로움
- DCGAN architecture
- 동일한 아키텍처인 DCGAN, WGAN, WGAN-GP 와 비교해봤을 때 결과가 좋음
- 특히 WGAN-GP 와 알고리즘이 비슷 (WGAN-GP + DCGAN 느낌)
- 논문에 따르면 WGAN-GP 는 restriction 이 너무 강하여 poor G 를 생성한다고 함

## Conclusion

- BEGAN이 제일 인상적이긴 하나 LSUN 에서도 그럴지 궁금
- DCGAN도 learning rate 을 잘 조절해주면 좋은 결과를 보임
- DRAGAN 도 꽤나 인상적

## Usage

- download
- convert
- train
  - 만약 warning 을 보기 싫으면 text_summary 부분을 주석처리해주면 됨
  - 대신 tensorboard 에서 config text 를 볼 수 없음
  - 이는 텐서플로 버그로 아마 tensorflow 1.3 에서는 괜찮을거라고 생각함
- eval