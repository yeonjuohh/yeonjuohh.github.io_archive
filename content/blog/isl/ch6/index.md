---
title: Ch 6. Linear Model Selection and Regularization
date: "2021-11-23"
description: "Introduction to Statistical Learning Chapter 6 Notes"
tags: ["ISL"]
---

- Linear model은 *X*와 *Y*의 관계를 설명하기 위해 주로 사용되는 방법이다. 앞으로 이 책에서는 linear model에서 확장된 다양한 비선형 모델을 살펴볼 예정이다. 하지만, linear model이 가진 장점이 분명히 있기에 다른 모델을 공부하기 전에 simple linear model의 성능을 높일 수 있는 방법에 대해 알아보고자 한다.
- 지금까지는 least squares 방법으로 linear model을 구했다. 하지만, n이 p와 큰 차이가 나지 않는다면 least squares로 추정한 모델의 variance는 증가하고, 이는 overfitting 문제로 이어질 수 있다. 또한, n이 p보다 작다면 variacen는 계속 증가해 least squares를 아예 사용할 수 없다. 이때, 추정치의 크기를 제한하면 bias를 어느 정도 올리고 variance를 많이 낮출 수 있다.
- Least squares의 또 다른 단점은 모든 *X*의 계수를 추정한다는 것이다. 하지만, 특히 multiple regression 모델에서는 모든 *X*가 *Y*와 연관되어 있지 않을 수 있다. 의미 없는 *X*를 모델에 추가하는 건 모델의 복잡성을 높일 뿐이다. 따라서 이런 경우에는 연관성이 낮은 *X*의 계수를 0으로 만들어 주는 작업이 필요하다.
- Least squares의 단점을 해결할 수 있는 여러 가지 방법 중 6장에서는 subset selection, shrinkage, dimension reduction에 대해 알아볼 것이다.
    - Subset selection
        - *Y*와 관련 있는 일부 *X*만을 선택해 모델을 만드는 방법이다.
    - Shrinkage / Regularization
        - Shrinkages는 *X*를 모두 모델에 이용하지만, 계수를 0으로 축소한다. 이는 모델의 variance를 감소시킨다. 어떤 shrinkage 방법을 사용하는지에 따라 어떤 계수는 아예 0이 되기도 해 variable selection처럼 작동한다고 생각할 수도 있다.
    - Dimension Reduction
        - 이 방법은 p개의 *X*를 M차원의 공간에 projecting 한 후 얻은 M projections을 모델 적합에 사용한다. M projections은 *X*들의 linear combinations으로 계산할 수 있다.

## 6.1 Subset Selection

- 의미 있는 X를 선택하는 방법에 대해 알아본다.

### 6.1.1 Best Subset Selection

- Best subset selection은 p개의 가능한 모든 조합으로 모델을 만들고 각 모델의 성능을 비교해 가장 좋은 모델을 선택하는 방법이다.
- 알고리즘을 정리하면 다음과 같다.
    1. $M_{0}$를 *X*가 없는 null model이라고 하자. 이 모델은 각 데이터에 단순히 평균값을 예측한다.
    2. $k=1, 2, ..., p$ 에 대해:
        1. $k$개의 *X*를 가지고 있는 모델을 ${p \choose k}$ 경우의 수만큼 만든다.
        2. $k$개의 *X*를 사용한 모델 중 가장 좋은 모델을 선택해 $M_{k}$라 한다. 가장 좋은 모델이란 RSS가 가장 작거나 $R^{2}$가 가장 큰 모델이다.
    3. 2번에서 구한 $M_{0}, ..., M_{p}$ 중에 cross validation error, $C_{p}$, AIC, BIC 또는 adjusted $R^{2}$을 기준으로 마지막 모델을 선택한다.
- 마지막 모델을 선택할 때 RSS 또는 $R^{2}$를 사용하지 않는 이유는 이 지표들은 *X*가 많아질수록 값이 좋아지기 때문이다. 따라서 RSS 또는 $R^{2}$을 기준으로 모델을 선택한다면 항상 모든 *X*를 사용하는 모델이 선택된다.
- Logistic regression에서는 RSS 대신 deviance를 사용한다. Deviance는 작을수록 좋은 모델이다.

### 6.1.2 Stepwise Selection

- Best subset selection은 p가 크면 계산량이 너무 많아지기 때문에 사용하기 어렵다. 또한, 모든 경우의 수를 고려하다 보면 overfittiing된 모델을 선택할 수도 있는 단점이 있다.
- 위와 같은 이유 때문에 재한 된 모델을 탐색하는 stepwise 방법이 선호된다.

**Forward Stepwise Selection**

1. 아무런 *X*가 없는 모델을 $M_{0}$라 하자.
2. $k=0, ..., p-1$에 대해
    1. 하나의 *X*가 추가로 들어간 모델 $M_{k}$를 $p-k$개의 만든다.
    2. $p-k$개의 모델 중 RSS가 가장 작거나 $R^{2}$가 가장 큰 모델을 선택해 $M_{k+1}$라 한다.
3. 2번에서 구한 $M_{0}, ..., M_{p}$ 중에 cross validation error, $C_{p}$, AIC, BIC 또는 adjusted $R^{2}$을 기준으로 마지막 모델을 선택한다.
- 실제 데이터에서 forward stepwise selection의 성능이 좋기는 하지만 모든 경우의 수를 탐색하는 게 아니기 때문에 가장 최고의 모델을 선택하지 않을 수도 있다.

**Backward Stepwise Selection**

1. $p$개의 모든 *X*가 포함된 모델을 $M_{p}$라 하자.
2. $k=p, p-1, ..., 1$에 대해
    1. *X*가 하나 빠진 모델 $M_{k}$를 $k$개 만든다.
    2. $k$개의 모델 중 RSS가 가장 작거나 $R^{2}$가 가장 큰 모델을 선택해 $M_{k-1}$라 한다.
3. 2번에서 구한 $M_{0}, ..., M_{p}$ 중에 cross validation error, $C_{p}$, AIC, BIC 또는 adjusted $R^{2}$을 기준으로 마지막 모델을 선택한다.
- Backward stepwise selection 역시 모든 경우의 수를 탐색하는 게 아니기 때문에 가장 최고의 모델을 선택하지 않을 수도 있다.
- Backward stepwise selection은 데이터 수가 p보다 큰 경우에 사용할 수 있는 반면에 forward stepwise selection은 이런 제한 조건이 없어 더 많은 문제에서 사용할 수 있다.

**Hybrid Approaches**

- 위에서 살펴본 세 가지 방법은 대체적으로 비슷하지만 똑같지는 않은 모델을 결과로 준다.
- 또 다른 방법인 hybrid는 forward stepwise와 backward stepwise를 같이 사용한다. Hybrid는 forward stepwise처럼 유의미한 변수를 모델에 하나씩 추가한다. 하지만 추가 후 쓸모없는 변수가 있으면 backward stepwise처럼 하나씩 뺄 수 있다.

### 6.1.3 Choosing the Optimal Model

- 많은 문제에서 best subset selection, forward selection, backward selection의 마지막 단계에서 test error가 가장 작은 모델은 선택하고자 한다. Test error는 training error에 조정을 해 간접적으로 추정하거나 validation set 또는 cross-validation 방법으로 직접적으로 추정할 수 있다.

$**C_{p}$, AIC, BIC, adjusted $R^{2}$**

- RSS 또는 $R^{2}$는 변수가 많아질수록 더 좋아지기 때문에 사용하지 않는 게 좋다. 하지만, training error에 변수가 많아짐에 따른 페널티를 준다면 test error 대신 사용할 수도 있다. 대표적으로 $C_{p}$, AIC, BIC, adjusted $R^{2}$가 있다.
- $C_{p}$, AIC, BIC는 값이 작을수록 adjusted $R^{2}$는 값이 클수록 좋은 모델이라고 할 수 있다.
- BIC는 $C_{p}$, AIC 보다 변수 수에 따른 페널티를 더 많이 주기 때문에 BIC를 기준으로 한다면 더 작은 모델이 선택되기도 한다.

**Validation and Cross-validation**

- Validation 또는 cross-validation 방법은 $C_{p}$, AIC, BIC, adjusted $R^{2}$보다 모델에 가정을 덜 하고, 더 많은 데이터에서 사용할 수 있다는 장점이 있다.
- 단순히 추정한 test error가 가장 작은 모델을 선택하기도 하지만, one-standard-error 방법은 추정한 test error의 가장 작은 값의 1 standard deviation 범위에 있는 모델 중 가장 간단한 모델을 선택한다.

## 6.2 Shrinkage Methods

- Shrinkage는 모든 변수를 이용해 모델을 추정하고, 일부 변수의 계수를 제한하는 방법이다. 다른 말로 계수를 0에 가까운 값으로 축소하는 방법이다.
- Shrinkage 방법을 사용하면 모델의 분산이 현저하게 감소해 전체적인 성능이 좋아진다.
- 회귀 계수를 제한하는 대표적인 방법으로는 ridge regression과 lasso regression이 있다.

### 6.2.1 Ridge Regression

- Least squares는 RSS를 최소화하는 계수를 추정하는 방법인 반면에 ridge regression은 아래와 같이 RSS에 페널티가 더해진 값을 최소화하는 계수를 찾고자 한다.
    
    $$RSS+\lambda\Sigma^{p}_{j=1}\beta_{j}^{2}$$
    
- $\lambda\Sigma^{j=1}_{p}\beta_{j}^{2}$는 shrinkage penalty로 $\beta$가 0에 가까울수록 작아진다. 즉, 이 부분으로 인해 $\beta$가 0에 가까운 값이 될 수 있다.
- $\lambda$는 tuning parameter로 따로 추정해야 하는 값이다. $\lambda$가 0이면 least squares와 똑같은 모델이 나오고, 커질수록 페널티가 커져 계수는 0에 더 가까워진다. $\lambda$에 따라 $\beta$가 달라지기 때문에 어떤 $\lambda$값을 선택하는지 중요하다.
- Shrinkage의 목적은 *Y*와 *X*의 관계를 조정하는 데 있기 때문에 $\beta_{0}$에는 페널티를 부여하지 않는다.
- Ridge regression은 *X*를 표준편차가 1이 되도록 정규화하고 적용하는 게 좋다.

**Why Does Ridge Regression Improve Over Least Squares?**

- Ridge regression이 least squares 보다 성능 좋은 모델을 얻을 수 있는 건 bias-variance trade-off와 관련 있다.
- $\lambda$가 커지면 ridge regression의 유동성은 작아지는데, 이로 인해 예측값의 bias가 증가하지만 variance가 훨씬 많이 더 감소하게 된다.

### 6.2.2 The Lasso

- Ridge regression의 단점은 마지막 모델에 모든 변수가 포함된다는 점이다. Ridge regression에서 계수는 0에 가까이 가지 0이 되지는 않는다. 이런 점이 예측 성능에는 문제가 없을지라도 p가 많은 경우 모델 해석에 어려움이 있을 수 있다.
- Lasso regression은 ridge regression의 단점을 해결할 수 있는 방법으로, 아래의 값을 최소화하는 계수를 찾고자 한다.
    
    $$RSS+\lambda\Sigma^{p}_{j=1}|\beta_{j}|$$
    
- $\lambda$가 충분히 크면 어떤 계수들은 0의 값을 가지기도 한다. 즉, lasso regression은 variable selection으로서의 기능도 수행한다.

**Another Formulation for Ridge Regression and the Lasso**

- 수학적으로 ridge regression과 lasso regression은 best subset과 비슷하다. 따라서 이 두 가지 방법을 best subset의 많은 계산량 문제를 해결할 수 있는 대체재라고도 이해할 수 있다.

**The Variable Selection Property of the Lasso**

**Comparing the Lasso and Ridge Regression**

- Lasso regression은 ridge regression에 비해 간단하고 해석이 용이하다는 점에서 장점을 가진다.
- 대부분의 문제에서 lasso regression의 예측력이 더 좋기는 하지만 모든 *X*가 *Y*와 관련 있다면 ridge regression의 결과가 더 좋을 수도 있다.
- 모든 데이터에서 좋은 하나의 모델은 없다. 그렇기 때문에 cross-validation 같은 방법을 사용해 특정 데이터에서 예측력이 좋은 모델을 선택하는게 좋다.

**A Simple Special Case for Ridge Regression and the Lasso**

- Ridge regression과 lasso regression의 차이를 간단히 설명하자면, ridge regression은 모든 계수를 동일한만큼 제한한다. 반면 lasso regression은 *Y*값과 lambda를 비교한 후 조건에 맞게 제한하는데 특정 조건을 만족하면 0으로 만들어버린다.

**Bayesian Interpretation for Ridge Regression and the Lasso**

- Regression에서 $\beta$를 추정하는 문제를 Bayesian 방법으로 접근하면 ridge regression과 lasso regression은 각각 $\beta$의 사전분포로 Gaussian, Laplace를 가정했을 때 얻을 수 있는 결과이다.

### 6.2.3 Selecting the Tuning Parameter

- 간단하게 cross-validation 방법으로 $\lambda$의 값을 정할 수 있다. 가장 먼저 $\lambda$의 후보값들을 정의한다. 다음으로 후보값을 하나씩 이용해 모델을 추정해보고, cross-validation error가 가장 작은 $\lambda$를 선택하면 된다.

## 6.3. Dimension Reduction Methods

- Dimension reduction method는 *X*를 변환시킨 후, 새로운 *X*를 이용해 모델을 추정하는 방법이다.
- $Z_{1}, Z_{2}, ..., Z_{m}$을 *X*의 선형 합이라 하자.

$$Z_{m}=\Sigma_{j=1}^{p}\phi_{jm}X_{j}$$

$\phi_{1m}, ..., \phi_{pm}$은 상수이다. *Z*를 이용해 아래와 같이 선형 모델을 구할 수 있다.

$$y_{i}=\theta_{0}+\Sigma_{m=1}^{M}\theta_{m}z_{im}+\epsilon_{i}$$

- $\phi_{1m}, ..., \phi_{pm}$ 값을 잘 선택하면 least squares 방법으로 얻은 모델보다 좋은 예측력을 보이기도 한다. $\phi$를 추정하는 다양한 방법이 있지만 책에서는 principal components와 partial least squares를 살펴볼 거다.
- Dimension reduction method는 p + 1개의 $\beta$를 추정하는 문제를 M + 1개의 $\theta$를 추정하는 간단한 문제로 감소시키는 데에서 이름이 붙여졌다.
- Dimension reduction method는 $\beta$에 제한을 가하는데 이로 인해 bias는 증가하지만 variance가 훨씬 더 많이 감소한다.

### 6.3.1 Partial Components Regression

- PCA(Principal Components Analysis)는 많은 변수들 중 차원이 낮은 변수를 얻을 수 있는 대표적인 방법이다. Unsupervised learning으로 사용되기도 한다.

 

**An Overview of Principal Component Analysis**

- PCA는 데이터의 분산이 큰 방향 순으로 *X*를 축소시킨다. 분산이 크다는 건 데이터를 가장 잘 설명할 수 있는 방향을 의미한다.
- PCA로 얻은 *Z*는 principal component이라 하고, *Z*는 서로 독립이다.

**The Principal Components Regression Approach**

- PCR(Principal Component Regression)은 PCA로 얻은 M개의 *Z*를 이용해 linear regression model을 추정한다. 많은 경우 작은 수의 principal components만으로 데이터의 분산과 *Y*와의 관계가 설명된다.
- 즉, PCR은 데이터의 분산이 가장 큰 방향이 *Y*와 연관성도 높다고 가정한다. 모든 문제에서 이 가정이 성립하지는 않지만 좋은 결과를 주는 충분히 합리적인 가정이다.
- 만약 위 가정이 충족된다면 PCR은 *X*를 사용한 모델보다 결과가 좋을 수밖에 없다. 왜냐하면 *Z*에 *X*와 *Y*의 관계에 관한 대부분의 데이터가 있고, 더 적은 수의 계수를 추정해 overfitting을 완화시킬 수 있기 때문이다.
- 몇 개의 *Z*를 사용하는지에 따라 결과가 달라지기 때문에 *M*값을 신중히 선택해야 한다. 주로 cross-validation 방법을 이용해 선택한다.
- PCR이 더 적은 수의 계수를 추정하다고 해서 feature selection method는 아니다. 왜냐하면 각각의 principal component는 모든 *X*의 선형 합이기 때문이다. 이런 측면에서 PCR은 lasso regression보다 ridge regression와 더 관련 있다고 할 수 있다.
- 만약 *X*의 분산이 많이 다르다면 PCA를 하기 전에 비슷한 크기로 표준화하는 게 좋다.

### 6.3.2 Partial Least Squares

- Principal component를 얻는 문제는 *X*만 사용하기 때문에 unsupervised라고 할 수 있다. 이로서 생기는 명확한 단점은 principal component가 *X*를 설명하는 가장 좋은 방법일 수는 있지만 *Y*를 가장 잘 예측하는 건 아닐 수 있다는 것이다.
- PLS(Partial least squares)은 PCR 대신 사용할 수 있는 supervised 방법이다. PCR과 비슷하게, PLS도 dimension reduction 방법이다. 다른 점이 있다면 PLS는 *X*의 선형합인 m개의 *Z*를 이용해 least squares 모델을 만든다. 즉, PLS는 데이터와 *Y*를 모두 잘 설명할 수 있는 *Z*를 찾고자 한다.
- 많은 문제에서 PLS를 적용하기 전에 *X*와 *Y*를 표준화하는 게 좋다. 또한, PCR처럼 *Z*의 수가 결과에 영향을 미칠 수 있기 때문에 cross-validation 방법으로 M값을 정한다.
- *Z*는 *Y*와 상관관계가 높은 *X*에게 가중치를 주는 방법으로 구한다. 그리고 다음 *Z*는 *X*와 지금 구한 *Z*를 반응 변수로 하는 회귀식의 잔차를 이용한다.

## 6.4 Considerations in High Dimensions

### 6.4.1 High-Dimensional Data

- 전통적인 통계 모델들은 변수의 수가 데이터 수가 훨씬 작은 낮은 차원의 데이터를 해결하는 데 목적이 있다.
- 지난 20년 동안 새로운 기술의 발달로 인해 다양한 데이터를 얻을 수 있게 되었다. 하지만, 비용 등의 이유로 상대적으로 얻을 수 있는 데이터의 수는 많이 증가하지 않았다.

### 6.4.2 What Goes Wrong in High Dimensions?

- 고차원의 데이터를 위한 방법이 아닌 모델을 사용하게 되면 다양한 문제가 일어날 수 있다. Overfitting은 흔히 일어나는 문제로 변수의 수가 많다면 모델을 만드는데 쓰이지 않은 데이터에서 모델의 성능을 평가하는 과정을 꼭 거치는 게 좋다.
- 또한, $\hat{\sigma}^2$를 구하는데 문제가 생기기 때문에 모델의 성능을 평가하는 $C_{p}$, AIC, BIC를 신뢰하기 어려워진다.

### 6.4.3 Regression in High Dimensions

- Forward stepwise selection, ridge regression, lasso regression, PCR 등은 least squares 보다 덜 유여한 모델로 고차원의 데이터에서 overfitting 될 위험이 적기 때문에 대신 사용할 수 있다.
- 보통 변수의 수가 많을수록 더 좋은 모델을 얻을 수 있다고 생각할 수 있지만, 새로운 변수가 *Y*와 연관이 있지 않는 이상 test error는 감소한다. 이를 curse of dimensionality라 한다.

### 6.4.4 Interpreting Results in High Dimensions

- 고차원 데이터에서는 multicollinearity 문제가 일어날 가능성이 매우 높고, dimension reduction 방법을 적용해서 얻은 모델이 가장 좋은 모델이 아닐 수도 있다. 또한, overfitting으로 새로운 데이터에서는 성능이 안 좋은 모델을 통계적으로 좋다고 말해 신뢰를 줄 수 있기 있기 때문에 항상 결과를 전달하는데 조심해야 한다. 가장 좋은 방법은 독립적인 데이터에서 계산한 test error 또는 cross validation error를 제공하는 것이다.