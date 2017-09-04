# GANs with structured

* 네이밍을 하기가 애매했음
* 목표는 구조화된 GAN 프로젝트를 만들고 이를 기반으로 다양한 모델에 대해서 실험해 보는 것
    * 잘 구조화시켜서 만들어서 모델만 슥슥 만들어주면 다 실험이 가능하도록
* MNIST 에 대해서는 해 봤지만 사실 MNIST 는 데이터셋이 간단해서 구별이 잘 안 감
* 따라서 CelebA 나 LSUN 등 좀 복잡한 데이터셋에 대해서 해 보면서 몇몇 코드들 리팩토링도 하고...

## ToDo

* [x] GAN 은 sess.run 을 두 번 해줘야 하므로 Input pipeline 을 떼어내서 명시적으로 input 을 fetch 해주고 다시 feed_dict 로 넣어줘야 함
    * 이건 했는데 image_shape 관련해서 좀 마음에 안 든다. 근데 일단 놔두자...
* [x] config.py, utils.py, ops.py 쓸데없이 많은 것 같다. 이거 정리좀 해줘야 할 듯 - refactoring => 이것도 딱히 손대기가 좀 애매함. 그래서 일단 놔둔다.
* [x] inputpipe 에서도 shape 지정하는 부분 잘 구조화해야함 => 구조화하기가 좀 애매한 것 같음. 오히려이렇게 그냥 바꾸고 싶으면 코드 자체를 수정하도록 놔두는 게 더 나을것같음
* [ ] LSUN dataset - 이건 좀 진짜 필요할 것 같음. BEGAN 같은 게 얼굴 말고 다른데도 잘 될지를 볼 수 있을 듯.
* [ ] Flexible input shape - 64/64 에 최적화시켜 놨는데 유동적으로 바꿀 수 있게 하자!
    * MNIST 도 커버할 수 있으면 참 좋을텐데 그건 어렵겠지...?
* [ ] add text summary to TensorBoard
* Flexible learning-times - G/D 가 각각 1번씩만 도는데 지정할 수 있도록 하자!
	* 일단 lr 로 컨트롤 해 두었는데 바꿔야 할까?

## GANs

* [x] DCGAN
* [x] LSGAN
* [x] WGAN
* [x] WGAN-GP
* [x] EBGAN
* [x] BEGAN
* [x] DRAGAN
* Additional
    * BGAN, CramerGAN, GoGAN, MDGAN
    * SGAN 은 CGAN 계열임

## Datasets

* CelebA
* Additional: LSUN, Flower, MNIST? ...


## Resuable code

* `utils.py`
* `inputpipe.py`


---

# Generative adversarial networks comparison without cherry-picking

- 비슷한 레포가 몇개 있지만, 여기서는 최대한 논문의 구조를 그대로 구현하려 노력함
- 물론 100% 그대로 재현한 건 아님. 디테일한 부분은 생략했고 - 예를들면 weight init - 논문에서 구조가 정확히 안 나온 부분 등은 임의로 구현함
- 큰 구조는 동일하게 맞추었기는 하지만 GAN 이 파라메터에 민감하여 사소한 부분에서도 결과 차이가 날 수 있다는 점은 생각하고 볼 것

## 다른 레포와의 차이점

- 유사 레포들: wiseodd, hwalsuk, sanghoon
- 논문의 구조를 그대로 구현하려 노력함
- tf queue 를 사용하여 celebA 에 대해서 수행함 
- 동일한 trainer 하에 model 만 바꿔가며 실험할 수 있도록 구조적으로 설계함
    - 다만 구조는 마음에 들지 않음... ㅜ.ㅜ

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
- DCGAN 에 PatchGAN 컨셉을 더해보았으나 특별히 좋은 결과가 나오진 않음 

DCGAN vs. DCGAN.G1e-3 vs. DCGAN.G1e-3.patchgan

### LSGAN

- 특이하게 LSGAN 에서는 z_dim 을 크게 씀: 1024
- 100으로 했을 때보다 1024로 했을 때 결과가 더 좋음

LSGAN.100 vs. LSGAN.1024

### WGAN

- 이론적인 논문이라서 결과가 엄청 좋지는 않음
- 네트워크 구조도 특별히 제안하지 않음

### WGAN-GP

- DCGAN architecture / appendix C 의 ResNet architecture
- resnet 결과가 더 좋기는 하나 딱히 눈에 띄는 성능향상은 보이지 않음

wgan.dcgan vs. wgan.resnet

### EBGAN

- 개인적으로 좋아하는 논문 - energy concept 이 매력적
- 다만 딱히 energy-based 라고 할만한 점이 없다는 비판도 있긴 함
- 아무튼 결과는 꽤 괜찮음 
- pt regularizer 의 효용에 대해 좀 의문이 있음 (weight=0 으로 줘도 점점 줄어듬)
    - 나는 pt weight = 0 으로 하면 mode collapse 가 발생하고 pt regularizer 가 이를 방지할거라고 생각했으나 별로 그런 느낌이 없는듯함 

### BEGAN

- celebA 에 대해서는 결과가 좋음 (사람이 보기에)
- 그러나 디테일이 없어지는듯한 느낌이 있음
- LSUN 등 다른 데이터셋에 대해서도 결과가 잘 나올까?

### DRAGAN

- DCGAN architecture
- 동일한 아키텍처인 DCGAN, WGAN, WGAN-GP 와 비교해봤을 때 결과가 좋음
- 특히 WGAN-GP 와 알고리즘이 비슷 (WGAN-GP + DCGAN 느낌)
- 논문에 따르면 WGAN-GP 는 restriction 이 너무 강하여 poor G 를 생성한다고 함
