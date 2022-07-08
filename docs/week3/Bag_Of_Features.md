# Bag Of Features

- Bag Of Words에 영감을 받아 이름이 지어진 것
- Bag Of Words 는 문서 검색을 위해 단어의 위치는 고려하지 않고 모든 문서를 bag of words 로 간주함. 이에 영감을 받아 Visual Words 혹은 Features 라는 형태로 image 를 BOW 와 유사하게 representation 을 하는 것.


### Image representation based on the BoW model

![[Pasted image 20220707220920.png]]

1. Feature Extraction and Clustering
   ![[스크린샷 2022-07-07 오후 10.06.11.png]]

   - 먼저 영상에서 feature 들을 추출(using SIFT)
     SIFT
     		- 이미지 크기와 회전에 불변하는 특징을 추출하는 알고리즘
     		- 이미지 내에서 위와 같은 환경 변화에 강인한 부분, edge 부분을 찾아 feature로써 추출함
     		![[Pasted image 20220707223246.png]]
   - 추출된 feature 들을 가지고 clustering 을 수행하여 cluster center 인 codeword 를 찾아냄 

2. Codebook generation

   - feature 들을 대표할 수 있는 값(code) 들로 구성되는 codebook 생성
   - 일종의 dictionary 로 모든 종류의 feature 가 포함된 것이 아니라 주요하다고 생각되는 feature 들만 담는것임. (using k-means clustering)
     - 여기서 추출된 center == codeword
   - 이 codebook 을 몇개의 codeword로 구성할지는 hyperparams 로써 주어짐
     - 몇개의 cluster 로 군집화 시킬지에 따라 몇개의 codeword로 구성될지가 달라지는 것

3. Image Representation

   - 각각의 이미지들을 codeword들의 histogram 으로 표현함
   - image 하나당 하나의 histogram이 나오며 이 히스토그램의 크기는 codebook의 크기(codeword들의 개수)와 동일하다.
     ![[Pasted image 20220708104644.png]]

   ![[Pasted image 20220707220812.png]]

   - 이 Histogram 이 2D Image 를 표현하는 1D Vector 가 되는 것임. (== Feature Vector)

4. Learning and Recognition

   - **Learning**
     - Unsupervised Learning 인 Naive Bayes 확률을 이용한 방법
       - class 별 histogram 값을 확률로서 해석하여 물체를 분류
       - ![[스크린샷 2022-07-07 오후 10.56.07.png]]
     - Supervised Learning 인 SVM 과 같은 Classifier 을 이용한 방법
       - 1D vector 로 표현된 Histogram 값을 feature vector로 보고 SVM 에 넣어 decision boundary 를 추출해냄.
       - 이를 통해 학습 데이터 간의 decision boundary 를 찾게 된다면 test scene의 class 도 분류할 수 있게 될 것임.
         ![[Pasted image 20220708130526.png]]
   - **Recognition**
     - Test Image 의 feature 추출 (using SIFT)
     - Train dataset 으로 생성된 Codebook 과 Test Feature 들을 비교해 Histogram 생성(가장 유사한 Image 에 bin을 가장 높게)
     - Test Data의 Histogram 을 이용해 SVM 이나 Naive Bayes 에 넣어 가장 확률이 높은 class 를 추출해 냄



참고자료
https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision
https://ai.stackexchange.com/questions/21914/what-are-bag-of-features-in-computer-vision
https://darkpgmr.tistory.com/125
https://kr.mathworks.com/help/vision/ug/image-classification-with-bag-of-visual-words.html
https://mingtory.tistory.com/88

실습 https://github.com/maponti/imageprocessing_course_icmc/blob/master/07c_bag_of_features.ipynb

---

### Further ..

- Spatial Pyramid Matching
  - Histogram 으로 표현한 것이 image 의 위치적인 특성을 고려하지 못한다.
    - 어떤게 문제가 되는것인가??
      ![[Pasted image 20220708121725.png]]
      계곡 사진을 보고 이 사진의 중심은 결국 계곡인데, 숲으로 판단할 수도 있는 것
  - 이러한 문제를 극복하기 위해 image 를 분할해 각 분면마다 BOF 를 적용하는 것
    ![[Pasted image 20220708121841.png]]
  - 이렇게 나온 vector를 Histogram Vector 대신에 사용하여 local 정보도 가져갈 수 있게 하는 것.


	- 이 방법을 착안한 Network 가 SPPNet (원본 이미지에서 추출된 Selective Search 들을 Feature Extractor 에 넣어 FeatureMap 을 추출할 수 있게, Feature Extractor 와 FCN 중간에서 유연하게 가져갈 수 있는 Layer)


- Beyond Sliding Windows :  **Efficient Subwindow Search 기법**
