---
title: Ch 4. Classification
date: "2021-10-13"
description: "Introduction to Statistical Learning Chapter 4 Notes"
tags: ["ISL"]
---

- Chapter 3에서 살펴보았던 linear regression 모델의 *Y*는 quantitative이다. 반대로 **qualitative인 *Y*를 예측하는 과정을 classification**이라고 한다.
- 많은 경우 데이터가 주어졌을 때 **Y의 각 카테고리에 속할 확률을 먼저 계산하고, 이 확률을 바탕으로 결정**을 내린다. 이러한 측면에서 classiciation은 regression 방법과 비슷하다고 할 수 있다.

## 4.1 An Overview of Classification

- 실제로는 regression보다 classification 문제를 더 많이 마주치게 된다.
- 이번 장에서 주로 살펴볼 데이터는 **Default**이다. 이 데이터에서는 연 소득과 월 카드 사용금액으로 이 사람이 카드빚을 갚을 수 있을지 또는 없을지를 예측하고자 한다.

## 4.2 Why Not Linear Regression?

- Y가 qualitative 일 때 왜 linear regression 모델이 적절하지 않은 걸까?
- 3가지 범주 중 하나의 값을 가지는 Y가 있다고 하자. 각 카테고리를 1, 2, 3으로 나타내면 linear regression 모델을 만들 수 있다. 하지만, 이와 같은 방법을 사용하면 독립적인 카테고리가 수치적으로 연관 있게 되어 각 카테고리가 어떤 값을 가지는지에 따라 모델 결과가 달라지게 된다.
- 그렇다면 카테고리가 2개만 있을 때는 괜찮을까? $\hat{Y}$가 0.5 이상이면 첫 번째 카테고리에, 그 외에는 두 번째 카테고리에 속한다고 하자. Linear regression을 사용하면 $\hat{Y}$이 0 보다 작거나 1 보다 클 수 있어 확률의 정의에서 벗어난다.
- 위와 같은 문제들로 Y가 qualitative 이면 classification 방법을 사용하는게 좋다.

## 4.3 Logistic Regression

- Logistic regression은 *Y*를 직접적으로 사용하기보단 *Y*가 특정 카테고리에 속할 확률을 알고자 한다. 즉, *X*와 P(X)의 관계를 알고 싶다.
- 예를 들어, **Default** 데이터에서 logistic regressioin은 카드값을 갚을 수 없는 확률을 계산한다. 이 확률 값이  특정 값(예. 0.5) 이상이면 카드값을 갚을 수 없다고 이하이면 갚을 수 있다고 한다.

### 4.3.1 The Logistic Model

- 4.2에서 보았듯이 P(X)를 X의 linear model로 나타내면 문제가 생길 수 있다. 이와 같은 문제를 피하기 위해 logistic model은 logistic function을 사용한다. Logistic function은 S 모양의 함수로 어떠한 데이터를 넣든지 0에서 1 사이의 값이 나온다.

$$p(X)={e^{\beta_{0}+\beta_{1}X} \over 1+e^{\beta_{0}+\beta_{1}X}}$$

- Logistic function을 정리하면 아래와 같은 식을 만들 수 있다.
    
    $${p(X) \over 1-p(X)}=e^{\beta_{0}+\beta_{1}X}$$
    
    $p(X)/(1-p(X))$는 odds라 한다. Odds는 0에서 무한대의 값을 갖는데 0에 가까울수록 *Y*가 1일 확률이 매우 낮고, 무한대에 가까울수록 1일 확률이 매우 높다는 걸 의미한다.
    
- 위 식 양쪽에 로그를 씌어 정리하면 다음과 같다.
    
    $$log({p(X) \over 1-p(X)})=\beta_{0}+\beta_{1}X$$
    
    왼쪽에 있는 부분을 log-odds 또는 logit이라 한다. Logistic regression의 logit은 *X*와 선형 관계이다.
    
- X가 한 단위 증가하면 log odds가 $\beta_{1}$만큼 또는 odds가 $e^{\beta_{1}}$배만큼 변화한다고 할 수 있다.
- p(X)는 *X*가 어떤 값을 가지고 있느냐에 따라 X의 한 단위 증가에 따라 변화하는 정도가 다르다. 그렇지만 *X* 값에 상관없이 $\beta_{1}$이 양수이면 *X*가 증가함에 p(X)도 증가하고 $\beta_{1}$이 음수이면 *X*가 증가함에 p(X)는 감소한다.

### 4.3.2 Estimating the Regression Coefficients

- $\beta$ 추정을 위해 linear regression처럼 least squares 방법을 사용할 수도 있지만, 많은 경우에 maximum likelihood 방법이 선호된다.
- Maximum likelihood의 기본은 모든 데이터에 대해서 예측한 확률이 실제 확률과 유사하도록 하는 $\beta$를 추정하는 것이다. 다른 말로 하면 $\hat{\beta}$를 모델에 사용했을 때 Y가 1의 값을 가지는 데이터에 대해서는 1에 가까운 값을, 0의 값을 가지는 데이터에는 0에 가까운 값을 주어야 한다.
- 정리하면 maximum likelihood는 likelihood function을 최대화하는 $\hat{\beta}$을 찾는 방법이다. Likelihood function은 다음과 같다.
    
    $$l(\beta_{0},\beta_{1})=\prod_{i:y_{i}=1}p(x_{i})\prod_{i':y_{i'}=1}(1-p(x_{i}))$$
    
- Maximum likelihood는 책에서 살펴볼 다양한 비선형 모델에 사용될 수 있을 만큼 일반적인 방법이다. 정확하게 말하면 least squares는 maximum likelihood의 특별한 케이스이다.
- Linear regression에서 살펴보았던 계수의 정확도 측정, 가설 검정 등 많은 것들이 logistic regression 결과에도 적용할 수 있다.

### 4.3.3 Making Predictions

- $\hat{\beta}$를 추정했으면 궁금한 데이터를 *X*에 대입해 $\hat{Y}$를 얻을 수 있다.
- *X*가 qualitative라면 3.3.1에서 보았던 거처럼 dummy 변수로 변환해 모델에 포함시키면 된다.

### 4.3.4 Multiple Logistic Regression

- 변수가 두 개 이상이면 multiple linear regression 에서처럼 모델에 변수를 추가해 주면 된다. 식으로 표현하면 다음과 같다.
    
    $$log({p(X) \over 1-p(X)})=\beta_{0}+\beta_{1}X_{1}+...+\beta_{p}X_{p}$$
    
- Simple logistic regression처럼 $\beta_{0}, \beta_{1}, ..., \beta_{p}$는 maximum likelihood method를 이용해 추정할 수 있다.
- 모델에 어떤 변수가 포함되었냐에 따라 해당 변수가 통계적으로 유의미하지 않을 수도 있고, 계수의 부호가 바뀔 수도 있다. 이를 confounding이라 한다.

### 4.3.5 Logistic Regression for > 2 Response Classes

- *Y*가 3개 이상의 값을 가진다면 logistic regression을 확장하면 된다. 하지만 이 방법을 책에서 다루지 않을 건데, 다음 장에서 공부할 linear discriminant analysis가 훨씬 많이 사용되기 때문이다.

## 4.4 Linear Discriminant Analysis

- Logistic regression은 *X*가 주어졌을 때 *Y*의 조건부 확률을 계산한다. Linear discriminant analysis는 각 *Y*마다 *X*의 분포를 추정한다. 그리고 Bayes' theorem을 이용해 $P(Y=k|X=x)$를 계산한다. X가 정규분포를 따른다고 가정하면 logistic regression과 매우 유사한 모델이 된다.
- Linear discriminant analysis는 몇 가지 경우에 logistic regression 보다 결과가 더 좋다.
    - 데이터가 *Y*의 카테고리별로 잘 나누어져 있을 때
    - 데이터 수가 적고 각 카테고리의 *X*가 정규분포에 근사할 때
    - *Y*의 카테고리가 세 개 이상일 때

### 4.4.1 Using Bayes' Theorem for Classification

- K 개 중 하나의 값을 가지는 *Y*가 있다고 하자. $\pi_{k}$를 어떠한 *X*가 k 카테고리에 속할 piror 확률, $f_{k}(X)=P(X=x|Y=k)$를 *X*가 k 카테고리에서의 분포라고 하자. $f_{k}(X)$는 k 카테고리에 있는 *X*가 x와 가까운 값을 가질수록 높은 값을 가진다.
- 위에서 정의한 $\pi_{k}$와 $f_{k}(X)$를 이용하면 Bayes' theorm으로 인해 $P(Y=k|X=x)$, posterior 확률을 계산할 수 있다.

$$P(Y=k|X=x)=p_{k}(X)={\pi_{k}f_{k}(x)\over\sum_{l=1}^{K}\pi_{l}f_{l}(x)}$$

- Train data에서 Y의 카테고리 분포를 알 수 있다면 $\pi_{k}$는 쉽게 추정할 수 있다. 하지만 $f_{k}(X)$는 분포에 간단한 가정을 하지 않는 이상 추정이 어렵다.
- 2장에서 모델을 잘 추정한다면 Bayes classifier에서의 오분류율이 가장 낮았다. 따라서 $f_{k}(X)$를 추정할 방법만 안다면 Bayes classifier에 근사하는 모델을 만들 수 있을 거다.

### 4.4.2 Linear Discriminant Analysis for p = 1

- $p_{k}(x)$을 알기 위해서는  $f_{k}(X)$를 추정해야 한다. 그다음에는 $p_{k}(x)$ 값이 가장 큰 k로 데이터를 분류하면 된다.
- $f_{k}(X)$ 추정을 위해서는 $f_{k}(X)$이 normal 또는 Gaussian 분포이고, 각 카테고리의 데이터가 등분산이라는 가정이 필요하다.
- 위 가정을 하면 $p_{k}(x)$를 다음과 같이 쓸 수 있다.
    
    $$p_{k}(x)={\pi_{k}{1\over\sqrt{2\pi}\sigma}exp(-{1\over2\sigma^{2}}(x-\mu_{k})^2)\over\sum_{l=1}^{K}\pi_{l}{1\over\sqrt{2\pi}\sigma}exp(-{1\over2\sigma^{2}}(x-\mu_{k})^2)}$$
    
    양쪽에 로그를 취하고 식을 정리하면 discriminant function $\delta_{k}(x)$를 얻을 수 있다. $\delta_{k}(x)$ 값이 가장 큰 K로 데이터를 분류하면 된다.
    
    $$\delta_{k}(x)=x{\mu_{k}\over\sigma^{2}}-{\mu_{k}^{2}\over2\sigma^{2}}+log(\pi_{k})$$
    
- Discriminant function을 계산하기 위해서는 $\mu_{1},...,\mu_{k}$, $\pi_{1},...,\pi_{k}$, $\sigma^2$를 추정해야 하는데, linear discriminant analysis는 train 데이터를 이용한다.

### 4.4.3 Linear Discriminant Analysis for p > 1

- 이제 *X*가 두 개 이상인 경우를 살펴보자. 비슷하게 *X*는 multivariate Gaussian 분포를 따르며, 모든 카테고리에서 분산이 같다는 가정이 필요하다. *X*가 multivariate Gaussian 분포를 따른다는 건 각각의 *X*는 normal 분포를 따른다는 걸 의미한다.
- Discriminant function $\delta_{k}(x)$은 다음과 같이 정리할 수 있다.

$$\delta_{k}(x)=x^{T}\Sigma^{-1}\mu_{k}-{1\over2}\mu_{k}^{T}\Sigma^{-1}\mu_{k}+log\pi_{k}$$

- **Default** 데이터에 LDA를 적용해 보면 training 오분류율은 2.75%이다. 이 값이 작아보이긴 전체 데이터에서 3.33%만이 *Y*가 1이기 때문에 모든 데이터에서 카드값을 갚지 못한다고 분류하는 아무 의미 없는 모델의 오분류율도 3.33%으로 큰 차이가 없다.
- 카테고리가 2개인 문제에서 일어날 수 있는 오류는 두 개이다. 실제로 Y가 0인데 1이라고 예측하는 오류와 Y가 1인데 0이라고 예측하는 오류. 두 개의 오류 중 어떤 오류가 더 큰지 알고 싶다면 confusion matrix를 사용하면 된다.
- 풀고자 하는 문제에 따라서 둘 중 하나의 오류를 줄이는 게 더 중요할 수도 있다.
- 실제로 Y가 1의 값을 가지는 데이터 중 Y는 1이라고 예측 한 비율을 sensitivity, 실제로 Y가 0의 값을 가지는 데이터 중 Y는 0이라고 예측 한 비율을 specificity라 한다.
- Bayes classifier와 LDA는 기본적으로 posterior probability 확률 $p_{k}(X)$이 가장 높은 카테고리로 분류한다. K=2일 때는 그 기준이 0.5이다. 즉 카테고리가 2개일 때 해당 카테고리에 속할 확률이 0.5 이상인 카테고리에 분류하는 것이다.
- 하지만, 만약 문제에서 결과의 sensitivity와 specificity를 조절하고 싶다면 분류 기준 값인 threshold value를 0.5에서 낮추거나 높이면서 조절해 볼 수 있다.
- Default 데이터에서 $p_{k}(X)$이 0.3 이상이면 Y가 1 아니면 0이라고 하자. 이렇게 하면 오분류율은 높아지고 specificity는 감소하지만 문제에서 가장 중요한 sensitivity를 0.24에서 0.58로 높일 수 있다.
- Threshold value는 풀고자 하는 문제에서 중요하게 생각하는 부분이 어디인지에 따라 달라진다.
- Threshold value에 따라 sensitivity와 specificity가 어떻게 달라지는지 한눈에 보기 위해서는 ROC(Receiver Operating Characteristics) curve를 그려보면 된다. ROC는 여러 분류 모델의 성능을 비교하고 싶을 때 자주 사용된다.
- ROC curve에서 AUC(Area Under the Curve)를 계산할 수 있는데, AUC는 0에서 1의 값을 가지며 1에 가까울수록 모델 전체적인 성능이 좋다는 걸 의미한다.

### 4.4.4 Quadratic Discriminant Analysis

- QDA는 각 카테고리는 분산이 다르다는 가정을 한다는 측면에서 LDA와 다르다.
- QDA는 LDA와 비교했을 때 유연한 모델이다. 따라서 데이터 수가 많거나 카테고리의 등분산 가정이 명확하게 위배되는 데이터에서는 결과가 더 좋다. 반대로 데이터 수가 적어 QDA에서 필요로 하는 많은 변수를 추정하기 어렵거나 카테고리가 등분산이라고 가정할 수 있는 문제에서는 LDA 성능이 더 좋다.

## 4.5 A Comparison of Classification Methods

- 지금까지 KNN, logistic regression, LDA, QDA을 살펴보았다.
- Logistic regression과 LDA는 decision boundary가 선형이라는 점에서 공통점을 가진다. 유일한 차이는 logistic regression은 maximum likelihood를 이용해 $\beta$를 추정하는 반면에 LDA는 데이터가 정규 분포를 따른다고 가정한 후 추정한 평균과 분산을 사용한다는 것이다.
- 만약 각 카테고리의 데이터가 Gaussian 분포에 근사하면 LDA를 이용한 결과가 더 좋게 나올 확률이 높고, 반대의 경우에는 logistic regression을 사용하는 게 좋다.
- KNN은 decision boundary의 형태에 아무런 가정을 하지 않는 non-parametric 방법이다. 그렇기 때문에 decision boundary가 비선형인 문제에서 logistic regression과 LDA 보다 결과가 좋다. 하지만, 어떤 *X*가 중요한지와 같은 정보는 알 수 없다.
- QDA는 KNN과 logistic regression, LDA의 장단점을 합친 모델이라고 할 수 있다. 선형 모델보다는 다양한 문제에서 사용될 수 있으며, KNN보다는 덜 유연하기 때문에 상대적으로 데이터 수가 적어도 좋은 결과를 낼 수 있다.
- 어떤 모델의 성능이 가장 좋을지는 데이터에 따라 다르다. Decision boundary가 선형이면 LDA와 logistic regression이 대체적으로 결과가 좋다. 비선형 decision boundary에서는 QDA가, 그리고 정말 복잡한 decision boundary에서는 KNN과 같은 non-parametric 모델을 사용하는 게 좋다.
- Regression 문제에서처럼 logistic regression와 LDA 모델에 변형된 *X*를 추가해 비선형 모델로 만들 수도 있다.