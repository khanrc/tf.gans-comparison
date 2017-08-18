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
* [ ] EBGAN
* [ ] BEGAN
* Additional
    * BGAN, DRAGAN, CramerGAN

## Datasets

* CelebA
* Additional: LSUN, Flower, MNIST? ...


## Resuable code

* `utils.py`
* `inputpipe.py`
