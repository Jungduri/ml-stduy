## 생성모델과 판별모델
### 추론과 결정
* 분류 문제의 두가지 스텝
	1. 학습 단계(추론 단계): 훈련 집단을 활용하여 $p(C_k | \mathbf{x})$ 를 학습 시키는 단계
	2. 추론 단계(결정 단계): 학습된 Posterior를 이용하여 최적의 클래스를 할당하는 것
		이때, $\mathbf{x}$에 대해서 결정값($C_k$)를 return해주는 것을 판별함수(discriminant function)라고 부름
* 결정 문제를 풀기 위한 3가지 접근 방법
	* 확률적 생성모델
	  * 각각의 클래스에 대해서 조건부 확률 밀도(Likelihood: $p(\mathbf{x} | C_k)$)를 구하고 사전확률($p(C_k)$)로 따로 구한다. 그런 다음 Bayes' rule을 적용.
				$$p(C_k | \mathbf{x}) = {{p(\mathbf{x} | C_k)p(C_k)}\over{p(\mathbf{x})}}$$
		* 분자에 있는 likelihood와 prior를 이용하여 데이터셋 분포를 계산 할 수 있음
				$$p(\mathbf{x}) = \sum{p(\mathbf{x} | C_k)p(C_k)}$$
		*  직간접적으로 입력값과 출력값의 분포를 모델링하는 이러한 방식을 *생성 모델(Generative model)*이라고 함.
	* 확률적 판별모델
		*  각각의 클래스에 대해서 사후 확률($p(C_k | \mathbf{x})$)를 구하는 추론 문제를 풀어낸 후 결정 이론을 적용하여 입력 변수 $\mathbf{x}$에 대한 클래스를 구한다. 사후 확률을 직접 모델링하는 이러한 방식을 판별 모델(discriminative model)이라고 한다. 
	
	* 비확률적 모델 
		* 각각의 입력값  $\mathbf{x}$를 클래스에 사상하는 판별함수 $f(x)$를 찾는다. 예를 들어 두개의 클래스를 가진 문제의 경우에$f(\;\dot{}\;)$은 $f=0$일때 $C_0$, $f=1$일때 $C_1$을 표현 할 수 있다. 이때는 확률론이 사용되지 않음.

* ![[generative vs discriminative.png]]

### 확률적 생성모델
* _generative model_이란 데이터 $\mathbf{x}$가 생성되는 과정을 두 개의 확률모형, 즉  $p(C_k)$,  $p(\mathbf{x}|C_k)$으로 정의하고, 베이즈룰을 사용해 $p(C_k | \mathbf{x})$를 간접적으로 도출하는 모델을 가리킴
* 클래스가 2개인 경우, $p(C_1 | \mathbf{x})$에 대해서 아래 수식을 이용하여 정의 됌
$$
\begin{align}
p(C_1 | \mathbf{x}) &= {{p(\mathbf{x} | C_1)p(C_1)}\over{p(\mathbf{x})}} \\
&= {{p(\mathbf{x} | C_1)p(C_1) } \over {p(\mathbf{x} | C_1)p(C_1) + p(\mathbf{x} | C_2)p(C_2)}} \\
&= {1\over1+\exp{(-a)}} = \sigma(a)  \;\;\;\;\;\; where, \; a =\ln{{p(\mathbf{x} | C_1)p(C_1) } \over {p(\mathbf{x} | C_2)p(C_2)}}
\end{align}
$$

> remark) 
	> * 이때 함수 $\sigma(a)$ 를 sigmoid(S자 곡선을 가졌다는 뜻), squashing function이라고 부름. 왜냐면 전체 실수축을 유한한 범위 안에 사상(mapping)하기 때문. 

* 클래스가 2개를 넘는  경우, 아래 수식으로 정의 됨
$$
\begin{align}
p(C_k | \mathbf{x}) &= {{p(\mathbf{x} | C_k)p(C_k)}\over{ \sum_j {p(\mathbf{x} | C_j)p(C_j)}}} \\
&= {{\exp{(a_k)}}\over{ \sum_j {{\exp{(a_j)}}}}} \;\;\;\;\;\; where, \; a_k =\ln{p(\mathbf{x} | C_k)p(C_k) } 
\end{align}
$$

> remark) 
	> * 위 함수를 정규화된 지수 함수(normalized exponectial function)이라고 함. 혹은 익숙한 말로는 softmax function이라고 하는데,  이는 모든 평활화(normalized) 중에서 최대값을 가장 두드러지게 표현해주기 때문(exp의 특성)
* 딥러닝을 이용한 생성 모델
* ![[generative_models.png]]
	-   Tractable Density : 데이터 X를 보고 확률분포를 ‘직접’ 구하는 방법
	-   Approximate Density: 데이터 X를 보고 확률분포를 ‘추정’하는 방법
	-   Implicit Density : 데이터 X의 분포를 몰라도 되는 방법


---
reference: 
교재 4.1 ~ 4.3
https://minsuksung-ai.tistory.com/12
