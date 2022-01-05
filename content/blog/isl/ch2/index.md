---
title: Ch2. Why Estimate f?
date: "2021-08-17"
description: "Introduction to Statistical Learning Chapter 2 Notes"
tags: ["ISL"]
---

함수 *f*를 알고 싶은 이유는 크게 두 개로 나눌 수 있다, **예측(prediction)과 추론(inference)**.

**Prediction**

- 대부분의 문제에서 *X*는 쉽게 알 수 있지만 *Y*는 얻기 힘들다.
- Random error의 평균은 0이기 때문에, 아래 식으로 Y를 예측할 수 있다. $\hat{f}$는 *f*의 추정 함수, $\hat{Y}$는 *Y*의 예측값이다.
$$
\hat{Y}=\hat{f}(X)
$$
- *f*가 *Y*를 정확하게 예측할 수 있다면 *f*가 어떤 함수인지는 중요하지 않다.
- $\hat{Y}$의 정확도는 두 개의 오류, **reducible error**와 **irreducible error**에 의해 정해진다.
- 대체적으로 $\hat{f}$는 실제 *f*의 정확한 추정치가 아니다. 이 차이로 생기는 reducible error는 *f* 추정에 더 정확한 모델을 사용함으로써 해결할 수 있다.
- 하지만, 아무리 *f*를 정확하게 추정한다고 해도 예측값에는 오류가 생길 수 있다. 왜냐하면 *Y*는 $\epsilon$ 의 함수이기도 한데, $\epsilon$ 는 X와 독립이기 때문이다. $\epsilon$ 의 변동성으로 생기는 오류가 있고, 이 오류는 f를 잘 추정해도 줄일 수 없기 때문에 irreducible error라 한다.
- $\epsilon$ 는 *Y* 예측에 유의미한 정보를 가지고 있을 수도, 그냥 측정 불가능한 변동성을 가지고 있을 수도 있다.
- 예를 들어, *X*를 환자들의 혈액 샘플 특징, *Y*를 특정 약에 대한 심각한 부작용 위험이라고 하자. Reducible error는 *f*를 정확하게 측정할수록 낮아진다. 하지만, 부작용의 위험은 환자의 그날 컨디션, 생산 과정에서 특정 약들의 문제 등에 의해 생길 수도 있다. 이와 같이 *X*와 독립적으로 *Y*에 영향을 미치는 오류를 irreducible error라 한다.
- 이 책의 목적은 **reducible error를 줄이면서 *f*를 추정하는 방법**을 알려주는 데 있다.

**Inference**

- **추론은 *X*가 변함에 따라 *Y*가 어떻게 영향을 받는지 이해**하는 거다. 이제는 X와 Y의 관계를 알아야 하기 때문에 $\hat{f}$는 black box가 될 수 없다.
- 추론 문제에서 우리는 다음과 같은 문제들을 알고 싶어 할 수 있다.
1. 여러 개의 *X* 중에서 *Y*와 연관 있는 건 어떤 걸까? 많은 경우에 변수로 사용하는 *X*중 일부만 *Y*와 통계적으로 유의미한 관계를 가지고 있기 때문에, 중요한 몇몇의 변수를 알아내는 건 중요한 문제이다.
2. 각 *X*와 *Y*는 어떤 관계일까? *X*가 증가할수록 *Y*도 증가하는 양의 관계일 수도 있고 반대일 수도 있다. 또한, 복잡한 함수에서는 *X*와 *Y*의 관계가 다른 변수의 값에 의해 달라지기도 한다.
3. *X*와 *Y는* 선형 관계인가, 아니면 비선형 관계여서 더 복잡한 모델을 사용해야 할까?

- 우리는 이 책에서 예측 문제, 추론 문제 그리고 예측과 추론을 동시에 하는 문제도 만나볼 것이다.
- **문제의 목적이 예측인지 추론인지에 따라 *f*를 추정할 때 다른 방법을 사용**하는 게 적절하다. 예측 문제의 목적은 정확한 예측을 하는 거이기 때문에 *f*를 이해하지 못하더라도 정확도만 높으면 괜찮다. 하지만, 추론 문제에서는 *X*와 *Y*의 관계를 이해해야 하기 때문에 정확도가 떨어지더라도 해석 가능한 함수를 사용하는 게 좋다.

### 2.1.2 Ho De We Estimate *f*?

- *f*를 추정할 때 사용하는 데이터를 **training data**라 한다. 우리의 목적은 **통계 방법을 training data에 적용해 알지 못하는 *f*를 추정**하는 거다. 다른 말 로하면, 우리는 **모든 (*X*, *Y*)에 대해 *Y*$\approx$ *f*(*X*)를 만족하는 *f* 함수를 찾고 싶다.**
- 대부분 통계 방법은 **parametric method과 non-parametric method**으로 나눌 수 있다.

**Parametric Methods**

- Parametric method은 크게 2단계로 이루어진 **모델 중심 접근 방식**이다.

1) 가장 먼저 ***f*의 함수적 형태(모델)를 가정**한다. 모델을 사용함으로써 알 수 없는 *f*를 추정하는 문제에서 모델의 모수를 찾는 문제로 단순화되었다.

2) **Training data를 이용해 모델을 학습**하며 모델의 모수를 추정한다.

- 이 방법은 예측 문제를 모수 추정 문제로 바꿈으로써 문제를 더 간단하게 만든다.
- 하지만, 가정한 *f*의 형태가 진짜 *f*와는 다른 경우가 많다. 더 복잡한 모델을 사용해 문제를 해결할 수 있지만, 복잡한 모델일수록 추정해야 할 모수의 개수는 늘어난다. 그리고 이는 overfitting 문제로 이어질 수 있다.

**Non-parametric Methods**

- Non-parametric method에서는 **데이터에 아무런 가정을 하지 않고, 그저 데이터를 가장 잘 따라가는 *f***를 알고자 한다.
- Parametric method와 비교했을 때 상대적으로 다양한 형태의 *f*를 더 잘 추정할 수 있지만, 훨씬 더 많은 데이터가 필요하다.

### 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

- 어떤 모델은 다른 모델과 비교했을 때 유동적이지 않고 제한적이어서 데이터의 일부분만 추정할 수 있다. 대표적인 예가 선형 모델이다.
- 반면에 thin plate splines처럼 데이터의 많은 부분을 설명해 줄 수 있는 더 유동적인 모델도 있다.
- 데이터를 더 잘 따라가는 모델이 있는데도 제한적인 모델 역시 아직도 사용하는 이유는 **덜 유동적인 모델의 결과가 더 이해하기 쉽기** 때문이다. 특히, 분석의 목적이 추론일 때에는 간단한 모델을 사용하는 게 좋다.
- 반대로 데이터에 대한 가정을 하지 않는 **유연한 모델은 예측력이 더 좋을 수 있어 분석의 목적이 예측인 상황에서 많이 사용**된다. 하지만, 해석이 어려우며 overfitting 문제가 생길 가능성이 높아 **어떤 경우에는 간단한 모델보다 예측력이 안 좋을 수도 있다.**

### 2.1.4 Supervised Versus Unsupervised Learning

- 대부분의 statistical learning 문제는 **supervised와 unsupervised**로 나눌 수 있다.
- **Supervised 문제에서는 *X*를 이용해 *Y*를 **예측하거나 설명**하고자 한다.
- **Unsupervised 문제에는** *X*에 대응하는 *Y*가 없다. 대신 ***X* 변수와의 관계나 데이터 값의 관계를 알고 싶을 때** 사용한다.
- 대표적인 unsupervised 모델에는 cluster anlaysi가 있는데, 분석의 목적은 *X*의 특징을 이용해 데이터를 **그룹화하는 거다.
- 많은 문제들은 명확하게 supervised 또는 unsupervised로 나눌 수 있다. 하지만, *Y* 데이터를 얻기 어려워 모든 *X*에 대응하는 *Y*가 없는 경우도 있는데, 이를 semi-supervised라고 한다.

### 2.1.5 Regression Versus Classification Problems

- 변수는 **quantitative와 qualitative**로 나눌 수 있다.
- Quantitative 변수는 나이, 키, 집값처럼 숫자 값을 가진다. Qualitative 변수는 성별, 구입한 상품의 브랜드처럼 K 개의 다른 카테고리 중 하나의 값을 가진다.
- ***Y*가 quantitative이면 regression 문제**, **qualitative 이면 classification 문제**라고 한다.
- 대부분 모델은 모든 경우에 사용할 수 있지만, 몇몇 모델은 *Y*가 quantitative 인지 qualitative 인지에 따라 적용 불가능하기도 하다.

## 2.2 Assessing Model Accuracy

- 어떤 데이터에서 한 모델의 성능이 가장 좋다고 해도, 비슷하지만 다른 데이터에서는 그렇지 않을 수 있다. 즉, **하나의 모델로 모든 데이터를 설명하기는 어렵다.**
- 따라서 데이터를 분석하는 다양한 방법을 알고 있어야 할 뿐 아니라, 어떤 모델의 성능이 가장 좋은지 결정할 수 있어야 한다.

### 2.2.1 Measuring the Quality of Fit

- 주어진 데이터에서 모델의 성능을 평가하기 위해서는 **예측값이 실제 값과 얼마나 비슷한지 측정**할 수 있어야 한다.
- Regression 문제에서는 대표적으로 **MSE(Mean Squared Error)**를 사용한다. **예측값과 실제 값이 가까울수록 MSE는 작다.**
$$
MSE=1/n(\sum_{i=1}^{n}(y_{i}-\hat{f}(x_{i}))^2
$$
- MSE와 같이 모델의 예측 정확도를 나타내는 지표는 모델을 학습할 때 사용한 training data가 아니라 아직 사용하지 않은 test data로 계산한다. 왜냐하면 우리는 **모델이 아직 보지 않은 미래 데이터에서의 정확도**가 궁금하기 때문이다. 그리고 training MSE가 아닌 test MSE가 가장 낮은 모델을 선택해야 한다.
- Test data가 없다면 train MSE를 사용할 수도 있다. 하지만, 대부분의 경우에서 training MSE가 가장 작은 모델의 test MSE는 큰 편이다.
- **유연한 모델일수록 train data에서 MSE가 작다**. **Test MSE도 모델의 자유도가 커질수록 작아지긴 하지만 어느 순간을 기준으로 커진다.**
- 주어진 모델의 **train MSE는 작지만 test MES는 크다면 모델이 데이터에 overfitting** 된 것이다. 이러한 문제는 모델이 train data를 학습하면서 함수 *f*의 진짜 특성이 아닌 우연으로 생긴 데이터의 패턴을 같이 학습했기 때문이다. 그리고 이 패턴은 test data에 없을 가능성이 높기 때문에 test MSE는 클 가능성이 높다.
- 모델은 train data로 학습하기 때문에 **train MSE은 항상 test MSE보다 작다.** **Overfitting은 덜 유연한 모델로 더 작은 test MSE를 얻을 수 있는 상황**에서 생기는 문제를 의미한다.

### 2.2.2 The Bias-Variance Trade-off

- $x_{0}$에서 **test MSE의 평균은 $\hat{f(x_{0})}$의 variance, $\hat{f(x_{0})}$ bias의 제곱 그리고 $\epsilon$  variance의 합으로 나눌 수 있다.**
$$E(y_{0}-\hat{f}(x_{0}))^2=Var(\hat{f}(x_{0}))+[Bias(\hat{f}(x_{0}))]^2+Var(\epsilon)
$$
- Var($\epsilon$ )는 irreducible error로 줄일 수 없기 때문에 **낮은 test MSE를 얻으려면 variance와 bias가 모두 작아야 한다.**
- **Variance는 다른 train data를 사용했을 때 $\hat{f}$가 얼마나 변하는지**를 나타낸다. 즉, 모델의 variance가 높다면 어떤 training data를 사용했는지에 따라 결과가 많이 다르다는 것을 의미한다. 많은 경우에 유연한 모델일수록 variance가 높다.
- **Bias는 모델의 데이터에 대한 가정 때문에 생기는 오류**이다. 즉, 실제 데이터의 복잡한 패턴을 고려하지 않은 간단한 모델을 적용했을 때 생긴다. 유연한 모델일수록 bias가 낮다.
- **모델의 자유도가 높을수록 variance는 커지고 bias는 작아진다.** 그리고 **두 값이 얼마나 상승/하락하는지에 따라 test MSE가 증가 또는 감소**하는데, ****이를 **bias-variance trade off**라 한다.
- 처음에는 모델의 자유도가 증가할수록 variance의 상승폭보다 bias의 하락폭이 더 커 test MSE는 감소한다. 하지만 어느 시점 이상에서는 bias가 하락하는 것보다 variance가 더 상승하게 되고 test MSE는 커진다. 우리의 목표는 variance와 bias가 작은 값을 가지는 모델을 찾는 것이다.

### 2.2.3 The Classification Setting

- 지금까지는 regression 문제에서 정확도를 측정하는 방법을 살펴보았다. **Classification 문제**에서는 *Y*가 숫자가 아니기 때문에 약간 달라지는 것 외에는 **bias-variance와 같은 개념들은 똑같이 적용 가능**하다.
- Classification 문제에서 $\hat{f}$의 정확도를 측정하는 대표적인 지표는 **error rate**이다. 모델을 적용했을 때 잘 못 분류한 비율을 계산한다. 아래 식에서 I($\cdot$ )는 indicator 함수로 안에 있는 조건이 참이면 1, 거짓이면 0의 값을 가진다.
$$
Avg(I(y_{0}\ne\hat{y_{0}}))
$$

**The Bayes Classifier**

- *X*가 주어졌을 때, 각 *X*를 **가장 확률이 높은 *Y*의 class로 분류**하는 간단한 모델을 사용하면 평균 test error rate을 최소화할 수 있다. 이를 **Bayes classifier**라고 한다.
- **Bayes error rate는 irreducible error와 같다.**

**K-Nearest Neighbors**

- Bayes classifier를 사용하기 위해서는 *X*가 주어졌을 때 *Y*의 조건부 확률 값을 알아야 한다. 하지만, **실제 상황에서는 이 값을 모르는 경우가 많아 bayes classifier를 사용하기 어렵다.**
- 이 문제를 해결하기 위해 많은 방법들이 **조건부 확률을 추정해 추정 확률 값이 가장 높은 class로 분류**하고자 한다. 대표적인 방법이 **K-nearest neighbors**이다.
- K-nearest neighbors은 test 데이터 $x_{0}$와 가까운 K 개의 train data를 찾는다. 그리고 K 개 train data의 *Y*로 조건부 확률의 추정 값을 계산하고 추정치가 가장 큰 class로 $x_{0}$를 분류한다.
- **K 값에 따라 KNN의 결과는 큰 차이를 보이는데,** K가 커질수록 자유도가 작아지면서 variance는 낮아지고 bias는 커진다. 반대로 K가 작아지면 test data에 없는 train data의 패턴을 따라가게 되면서 bias는 낮아지고 variance는 커진다.