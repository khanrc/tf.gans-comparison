# GANs with structured

* 네이밍을 하기가 애매했음
* 목표는 구조화된 GAN 프로젝트를 만들고 이를 기반으로 다양한 모델에 대해서 실험해 보는 것
    * 잘 구조화시켜서 만들어서 모델만 슥슥 만들어주면 다 실험이 가능하도록
* MNIST 에 대해서는 해 봤지만 사실 MNIST 는 데이터셋이 간단해서 구별이 잘 안 감
* 따라서 CelebA 나 LSUN 등 좀 복잡한 데이터셋에 대해서 해 보면서 몇몇 코드들 리팩토링도 하고...

## GANs

* [ ] DCGAN
* [ ] LSGAN
* [ ] EBGAN
* [ ] WGAN
* [ ] WGAN-GP
* [ ] BEGAN
* Additional
    * BGAN, DRAGAN, CramerGAN

## Datasets

* CelebA
* Additional: LSUN, Flower, ...


## Resuable code

* `utils.py`
* `inputpipe.py`