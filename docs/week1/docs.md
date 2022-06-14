# Week 1:  Motivations and Basics
편집자: Jungduri & gwleee

----------
## Motivation

인터넷의 등장으로 생긴 방대한 데이터와, 처리 할 수 있는 능력(computation power)의 확보로 데이터와 관련된 여러가지 task를 수행하는 학문

###### 대표적인 기계 학습의 tasks
* Documentaton classification(e.g. 스팸 처리)
* Stock market prediction 
* Plate number classification
* Social media recomandation
* Manupulator control

###### 머신러닝의 큰 분류
* Supervised learning: **know ture answers**
  * Classification: discrete data에 접근
  * Regression: continuous data에 접근

* Unsupervised learning: **unknow ture answers**
  * Clustering
  * Filtering 

* Reinforcement learning: pass

## Maximum Llikehood Estimation
###  Case study: thumbtask question

###### Overview
* 압정으로 내기
* 앞/뒤 어디에 배팅해야 더 승률이 좋을지 아는 것이 목적

###### Simple approach

* 5번 던졌더니
  * 앞: 3/5
  * 뒤: 2/5
* 앞에 배팅하는 것이 더 확률이 높음

###### Binomial distribution
* 이산적 확률 분포
* iid: 독립적인 이벤트
* 베르누이 시행(Bernoulli trial): 통계학에서 임의의 결과가 '성공' 또는 '실패'의 두 가지 중 하나인 실험을 의미 

* 위의 approach애서 ... 
$P(H) = \theta$
$P(T) = 1-\theta  \qquad  \because (Bernoulli \; trial) $
$P(HHTHT) = \theta^3 (1-\theta)^2$

* Let)
$ D = HHTHT$
$ n = 5 $
$ k = a_H = 3 $
$p = \theta $
$ \therefore \; P(D| \theta) =\theta^{ a_H} (1-\theta)^a_T $

###### Task definition
* Data: head와 tail로 구성된 관측된 데이터
* 가정: 압적의 도박 결과는 theta를 매개변수로 갖는 이산적 확률 분포를 따름
* 목적: 가정을 어떻게 하면 더 강하게 만들 수 있나?
  1. 더 좋은 가정은 만듬
  2. $\theta$ 를 최적화

###### Maximum Llikehood Estimation(MLP)
* IDEA: 가정안에서  **관측한 데이터**가 등장할 확률을 최대화
> 가정 = 확률분포 = P
> 관측한 데이터 = observatoin = D
> 실험을 통해 얻어진 observation이 등장하는 확률이 최대가 되는 특정 모델($\hat{\theta}$)을 찾는 것이 목표
* 즉, 아래 수식을 만족하는 $\hat{\theta}$를 찾는 것이 목적
$$ \hat{\theta} =\argmax\limits_{\theta}P(D|\theta)  $$

* 편의를 위해 로그를 적용
$ \hat{\theta} = \argmax\limits_{\theta}P(D|\theta)  $
 $\quad = \argmax\limits_{\theta} \{ a_H \ln \theta + a_T \ln (1-\theta) \}$

* minima를 찾기 위해서 derivative
$ \cfrac{d}{d\theta} ( a_H \ln \theta + a_T \ln (1-\theta) ) = 0$ 
$ \therefore \; \hat{\theta} = \cfrac{a_H}{a_H + a_T} $

* 시행을 더 진행해서 50번을 하였다. 이때 어떤 변화가 생김?
  * Hoeffding's inequality 
$$ P(|\hat{\theta} - \theta^*|  \geq e) \leq 2e^{-2Ne^2}$$ 
  * true값($\theta^*$)과 추정 값 $\hat{\theta}$ 가 오차 범위 안에 들어오는 것은 $N$에 영향을 받음
따라서 시행을 늘리는 것은 에러 bound를 줄이는 효과가 있음



### Jungduri 
Test

### gwleee
Test