# EfficientNet_pytorch

paper : https://arxiv.org/abs/1905.11946 


![1](https://user-images.githubusercontent.com/13817715/116502051-747d0d00-a8ed-11eb-8409-80214b6ded46.PNG)


논문에서는 알파는 1.2, 베타는 1.1, 감마는 1.15를 사용했다고 한다.

![2](https://user-images.githubusercontent.com/13817715/116502025-60391000-a8ed-11eb-84f2-d696c65991cf.PNG)

우선 efficientnet의 구조이다. 차근차근 하나씩 살펴보도록 하겠다.


## SEBlock

기존의 cnn은 channel의 global feature을 고려하지 못 한다. 이를 해결하기 위한 방법.

![image](https://user-images.githubusercontent.com/13817715/116505396-c32ea500-a8f5-11eb-909e-b77be81bb25b.png)

F<sub>tr</sub>는 단순한 convolution 연산이다.

### squeeze

이는 채널들의 정보를 짜내는(squeeze)하는 연산이다.

F<sub>sq</sub>는 H x W x C를 1 x 1 x C(평균)로 바꿔준다. pytorch 에서 AdaptiveAvgPool2d(1)로 보면 된다.

### excitation

채널 간의 의존성을 계산하는 과정이다.

앞에서 구한 것을 fullyconnected layer에 집어넣어서 input channel을 reudciton ratio r로  나누어서 크기를 줄인다.

그리고 이를 다시 input_channel의 크기로 늘려준다(코드로 보는 것이 이해가 더 빠를 것이다.)

이렇게 나온 것을 그림처럼 곱해준다.



## depthwise convolution

![image](https://user-images.githubusercontent.com/13817715/116570951-9d30f100-a945-11eb-960f-55046fe2f127.png)

기존의 convolution으로 위 그림처럼 출력하려면 H x W x 3 x 3 개의 필터가 필요하다. 하지만 위 그림같은 방법을 통하여 연산량을 줄일 수 있다.

이를 왜 설명하냐면, MBConv에 쓰이기 때문이다.



## stochastic depth


stochastic depth기법도 이용되었다. 이는 일정 확률로 block 자체를 건너뛴다.


## MBConv


논문을 보면 MBConv를 이용한다.

MBConv는 inverted residual을 이용한다.

expand ratio 만큼 채널을 키우고, depthwise convoloution을 이용한다. 그리고 앞에서 언급했던 SEBlock을 이용해서 다시 원래 채널 크기로 되돌린다.

그 다음은 커널 1짜리로 channel의 크기를 변경시켜주고, batch normalization을 해준다.

끝으로 resnet처럼 input을 다시 더 해준다.



코드는 이 유튜브를  작성하였다.

https://www.youtube.com/watch?v=fR_0o25kigM



