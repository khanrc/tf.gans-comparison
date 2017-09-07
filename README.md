# GANs comparison without cherry-picking

Implementions of some theoretical generative adversarial nets: DCGAN, EBGAN, LSGAN, WGAN, WGAN-GP, BEGAN, and DRAGAN. 

I implemented the structure of model equal to the structure in paper and compared it on the CelebA dataset.


[TOC]

## ToDo

- LSUN dataset
- flexible input shape
- modulation of D/G networks
- Other models - CramerGAN, GoGAN

## Features

- Model structure is copied from each paper
    - But some details are ignored
    - I admit that a little details make great differences in the results due to the very unstable GAN training
- No hard tuning for each model - so the results can be improved
- Well-structured - was my goal at the start, but the result is not satisfactory :(
    - TensorFlow queue runner is used for inpue pipeline
    - Single trainer (and single evaluator) - multi model structure
    - Logs in training and configuration are recorded on the TensorBoard

## Models

- DCGAN
- LSGAN
- WGAN
- WGAN-GP
- EBGAN
- BEGAN
- DRAGAN

The family of conditional GANs are excluded (CGAN, acGAN, SGAN, and so on).

## Dataset 

### CelebA

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

- All experiments were performed on 64x64 CelebA dataset
- The dataset has 202599 images
- 1 epoch consists of about 15.8k iterations for batch size 128

## Results

- I implemented the same as the proposed model in each paper, but ignored some details (or the paper did not describe details of model)
  - For example, I used all weight initialization as default which is xavier init
- Default batch_size=128 and z_dim=100 (from DCGAN)

### DCGAN

- Simple networks
- Learning rate for discriminator (D_lr) is 2e-4

|                G_lr=2e-4                 |                G_lr=1e-3                 |
| :--------------------------------------: | :--------------------------------------: |
|                   50k                    |                   30k                    |
| ![dcgan.G2e-4.50k](assets/dcgan.G2e-4.50k.png) | ![dcgan.G1e-3.30k](assets/dcgan.G1e-3.30k.png) |

Higher learning rate for generator makes better results. I used G_lr=1e-3 and D_lr=2e-4 which is the same as the paper suggested. In this case, however, the generator has been collapsed sometimes due to its large learning rate. Lowering both learning rate can bring stability like https://ajolicoeur.wordpress.com/cats/ in which suggested D_lr=5e-5 and G_lr=2e-4.

### EBGAN

- 개인적으로 좋아하는 논문 - energy concept 이 매력적
- 다만 딱히 energy-based 라고 할만한 점이 없다는 비판도 있긴 함
- 아무튼 결과는 꽤 괜찮음 
- pt regularizer 의 효용에 대해 좀 의문이 있음 (weight=0 으로 줘도 점점 줄어듬)
  - 나는 pt weight = 0 으로 하면 mode collapse 가 발생하고 pt regularizer 가 이를 방지할거라고 생각했으나 별로 그런 느낌이 없는듯함 

ebgan.pt vs. ebgan.nopt

|             pt weight = 0.1              |                No pt loss                |
| :--------------------------------------: | :--------------------------------------: |
|                   30k                    |                   30k                    |
| ![ebgan.pt.30k](assets/ebgan.pt.30k.png) | ![ebgan.nopt.30k](assets/ebgan.nopt.30k.png) |

pt 를 쓴게 더 결과가 좋긴 함

근데 pt 를 쓰는게 mode collapse 를 잡기 위함인데 그런 효과는 잘 모르겠음


|             pt weight = 0.1              |                No pt loss                |
| :--------------------------------------: | :--------------------------------------: |
| ![ebgan.pt.graph](assets/ebgan.pt.graph.png) | ![ebgan.nopt.graph](assets/ebgan.nopt.graph.png) |

pt loss 를 쓰는 경우 pt_loss 가 조금 더 빠르게 줄어들기는 하나 큰 차이가 나지 않고 결국 비슷하게 됨

pt loss 를 쓰지 않아도 pt 값이 똑같이 줄어든다는 점이 재미있음

위 샘플 퀄리티를 보면 pt_loss 를 쓰면 더 좋아지는것 같기는 하지만 실제로 역할이 있는지는 의문임

### LSGAN

- Unusually, LSGAN used large dimension for latent space (z_dim=1024)
- But in my experiments, z_dim=100 makes better results than z_dim=1024 which is originally used in paper

|                z_dim=100                 |                z_dim=1024                |
| :--------------------------------------: | :--------------------------------------: |
|                   30k                    |                   30k                    |
| ![lsgan.100.30k](assets/lsgan.100.30k.png) | ![lsgan.1024.30k](assets/lsgan.1024.30k.png) |


### WGAN

- Very theoretical paper, so the results are not impressive (the theory is very impressive!)
- Also no specific network structure proposed, so DCGAN architecture was used for experiments

|               30k                |               W distance               |
| :------------------------------: | :------------------------------------: |
| ![wgan.30k](assets/wgan.30k.png) | ![wgan.w_dist](assets/wgan.w_dist.png) |


### WGAN-GP

- DCGAN architecture / appendix C 의 ResNet architecture
- resnet 결과가 더 좋은데 기대만큼의 성능향상이 나오지는 않음
  - 특이한 점은 굉장히 빨리 수렴하고 더 학습하면 결과가 나빠짐
  - skip-connection 의 영향으로 보임
- DRAGAN 논문에서는 WGAN(-GP) 의 constraint 가 너무 restrict 해서 poor G 를 학습한다고 함
- wgan-gp 는 에퐄이 올라가면 얼굴이 뭉개지는 현상이 있는데, 이는 w_dist 가 예쁘게 떨어지는 것과 상반됨
  - Q. 왤까?

wgan-gp.dcgan vs. wgan-gp.resnet

|            DCGAN architecture            |           ResNet architecture            |
| :--------------------------------------: | :--------------------------------------: |
|                   30k                    |           7k, batch size = 64            |
| ![wgan-gp.dcgan.30k](assets/wgan-gp.dcgan.30k.png) | ![wgan-gp.good.7k](assets/wgan-gp.good.7k.png) |

wgan-gp.dcgan 은 에퐄이 올라가면 얼굴이 뭉개지는 현상이 있음

|                   10k                    |                   20k                    |                   30k                    |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| ![wgan-gp.dcgan.10k](assets/wgan-gp.dcgan.10k.png) | ![wgan-gp.dcgan.20k](assets/wgan-gp.dcgan.20k.png) | ![wgan-gp.dcgan.30k](assets/wgan-gp.dcgan.30k.png) |


wgan-gp.resnet 도 비슷함

|                    5k                    |                    7k                    |                   10k                    |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
| ![wgan-gp.good.5k](assets/wgan-gp.good.5k.png) | ![wgan-gp.good.7k](assets/wgan-gp.good.7k.png) | ![wgan-gp.good.10k](assets/wgan-gp.good.10k.png) |
|                   15k                    |                   20k                    |                   25k                    |
| ![wgan-gp.good.15k](assets/wgan-gp.good.15k.png) | ![wgan-gp.good.20k](assets/wgan-gp.good.20k.png) | ![wgan-gp.good.25k](assets/wgan-gp.good.25k.png) |
|                   30k                    |                   40k                    |                   50k                    |
| ![wgan-gp.good.30k](assets/wgan-gp.good.30k.png) | ![wgan-gp.good.40k](assets/wgan-gp.good.40k.png) | ![wgan-gp.good.50k](assets/wgan-gp.good.50k.png) |

그래프는 이쁘게 떨어짐

|            DCGAN architecture            |           ResNet architecture            |
| :--------------------------------------: | :--------------------------------------: |
| ![wgan-gp.dcgan.w_dist](assets/wgan-gp.dcgan.w_dist.png) | ![wgan-gp.good.w_dist](assets/wgan-gp.good.w_dist.png) |
| ![wgan-gp.dcgan.w_dist.expand](assets/wgan-gp.dcgan.w_dist.expand.png) | ![wgan-gp.good.w_dist.expand](assets/wgan-gp.good.w_dist.expand.png) |

W_dist < 0 이 되는게 흥미로운데, discriminator 를 generator 가 압도하고 있다고 볼 수 있음. E[fake] > E[real]. n_critic 을 늘려야 하나?


### BEGAN

- celebA 에 대해서는 결과가 좋음 (사람이 보기에)
  - optional improvement 부분은 구현하지 않았음에도!
- 그러나 디테일이 없어지는듯한 느낌이 있음
- Q. LSUN 등 다른 데이터셋에 대해서도 결과가 잘 나올까? - ToDo

batch size = 16, z_dim=64, gamma=0.5

|                30k                 |                50k                 |                75k                 |
| :--------------------------------: | :--------------------------------: | :--------------------------------: |
| ![began.30k](assets/began.30k.png) | ![began.50k](assets/began.50k.png) | ![began.75k](assets/began.75k.png) |

|     Convergence measure M      |
| :----------------------------: |
| ![began.M](assets/began.M.png) |



### DRAGAN

- Game theory 에서의 접근이 굉장히 흥미로움
- DCGAN architecture
- 동일한 아키텍처인 DCGAN, WGAN, WGAN-GP 와 비교해봤을 때 결과가 좋음
  - hyperparam tuning 한 DCGAN 과는 비슷한 느낌
- 특히 WGAN-GP 와 알고리즘이 비슷 (WGAN-GP + DCGAN 느낌)
- 논문에 따르면 WGAN-GP 는 restriction 이 너무 강하여 poor G 를 생성한다고 함

|          DCGAN architecture          |
| :----------------------------------: |
|                 30k                  |
| ![dragan.30k](assets/dragan.30k.png) |



## Conclusion

- BEGAN이 제일 인상적이긴 하나 LSUN 에서도 그럴지 궁금
  - BEGAN 은 learning rate decay 등등 implementation 자체가 신경써서 되어 있기는함 - 즉 엔지니어링빨이라고 할수도
  - 그렇다 쳐도 결과가 매우 인상적
- DCGAN도 learning rate 을 잘 조절해주면 좋은 결과를 보임
  - BEGAN 을 제외하고 제일 좋은 결과인 것 같기는 한데 다른 모델들은 tuning 을 전혀 안했기때문에 무조건 그렇다고 할 순 없음
- DRAGAN 도 꽤나 인상적

## Usage

Download CelebA dataset:

```
$ python download.py celeba
```

Convert images to tfrecords format:

```
$ python convert.py
```

Train:

```
$ python train.py --model model --name name
```

Monitor through TensorBoard:

```
$ tensorboard --logdir=summary/name
```

Evaluate:

```
$ python eval.py --model model --name name
```

### Requirements

- python 2.7
- tensorflow 1.2
- tqdm
- (optional) pynvml - for auto gpu selection


### Similar works

- [wiseodd/generative-models](https://github.com/wiseodd/generative-models)
- [hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)
- [sanghoon/tf-exercise-gan](https://github.com/sanghoon/tf-exercise-gan)
- [YadiraF/GAN_Theories](https://github.com/YadiraF/GAN_Theories)
- https://ajolicoeur.wordpress.com/cats/

