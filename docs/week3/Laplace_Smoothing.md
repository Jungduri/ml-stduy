## Laplace Smoothing
-  bayes risk를 줄이기 위한 Optimal Classifier
	 $$f^{*}(x) = argmax_{Y=y} \ P(Y=y|X=x)$$
	 (bayes theorem)
	 $$= argmax_{Y=y} \quad P(X=x|Y=y) \ P(Y=y)$$
	 
 - Naive Bayes Classifier
  $$ f^{ \* }(x)=argmax_{Y=y} \quad P(X=x|Y=y) \ P(Y=y)$$
 (condition independce assumption)
 $$=argmax_{Y=y} \  P(Y=y)\prod _{i}P(X_{i}=x_{i}|Y=y) \ $$
	 - Problem_1 : Naive assumption, 현실에는 너무 많은 유기적인 사건이 존재
	 - Problem_2: Incorrect Probability Estimation, 관측하지 못한 경우 MLE는 0이 되고, MAP는 prior를 측정하기에 부족한 조건  

- Laplace Smoothing (=Additive Smotthing)
	:  훈련된 분류기에 훈련 데이터셋에 없던 데이터가 입력되었을때, Likelihood가 0이 되어, 정상적인 분류가 되지 않은 경우를 보완하기 위해서, 가중치 $\alpha$  를 주어 계산하는 방법.
	$$P(X_{i}|Y_c)= \frac{ \sum tf(x_i,d \in Y_c)+ \alpha}{ \sum{N_{d\in{Y_c}}}+\alpha V}$$
- $X_i$ : feature vector X안의 단어중 특정 샘플 한개
- $\sum tf({x_i,d \in Y_c})$: document d의 $Y_c$를 가지는 단어$X_i$의 출현 빈도 수 의 합
- $\sum{N_{d \in Y_c}}$ : 특정 Y에 해당하는 모든 문서의 용어출현의 수
- $\alpha$: smoothing parameter 
-  $V$: training set에 있는 단어들의 수

### 예제

|           | Doc(d) | words                               | class |
| --------- | ------ | ----------------------------------- | ----- |
| Trainging | 1      | Chinese beijing chinese             | c     |
|           | 2      | Chinese Chinese shanghai            | c     |
|           | 3      | Chinese Macao                       | c     |
|           | 4      | Tokyo Japan Chinese                 | j     |
| Test      | 5      | Bejing Beijing tokyo Macao          | ?     |

      
- bag of words

|          |     | Chinese | beijing | shanghai | Macao | Tokyo | Japan | Class |
| -------- | --- | ------- | ------- | -------- | ----- | ----- | ----- | ----- |
| Training | 1   | 2       | 1       | 0        | 0     | 0     | 0     | c     |
|          | 2   | 2       | 0       | 1        | 0     | 0     | 0     | c     |
|          | 3   | 1       | 0       | 0        | 1     | 0     | 0     | c     |
|          | 4   | 1       | 0       | 0        | 0     | 1     | 1     | j     |
| Test     | 5   | 0       | 2       | 0        | 1     | 1     | 0     | ?     |


- Test 5 without laplace smoothing:
$$P(c|d_5)=P(c)P(Beijing|c)P(Beijing|c)P(tokyo|c)P(macao|c)$$
$$=\frac{3}{4} \cdot \frac{1}{8} \cdot \frac{1}{8} \cdot \frac{1}{8} \cdot \frac{1}{8} = 0.0018311$$
$$P(j|d_5)=P(j)P(Beijing|j)P(Beijing|j)P(tokyo|j)P(macao|j)$$
$$=\frac{1}{4} \cdot \frac{0}{3} \cdot \frac{0}{3} \cdot \frac{1}{3} \cdot \frac{0}{3} = 0$$
- Test 5 with Laplace Smoothing $\alpha =1$
$$P(c|d_5)=P(c)P(Beijing|c)P(Beijing|c)P(tokyo|c)P(macao|c)$$
$$=\frac{3}{4} \cdot \frac{1+1}{8+6 \* 1} \cdot \frac{1+1}{8+6 \* 1} \cdot \frac{1+1}{8+6 \* 1} \cdot \frac{1+1}{8+6 \* 1}$$
$$= 0.00031237$$
$$P(j|d_5)=P(j)P(Beijing|j)P(Beijing|j)P(tokyo|j)P(macao|j)$$
$$=\frac{1}{4} \cdot \frac{0+1}{3+1 \* 6} \cdot \frac{0+1}{3+1 \* 6} \cdot \frac{1+1}{3+1 \* 6} \cdot \frac{0+1}{3+1 \* 6}$$
$$= 0.00007621$$
