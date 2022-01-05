---
title: Ch 5. Resampling Methods
date: "2021-10-26"
description: "Introduction to Statistical Learning Chapter 5 Notes"
tags: ["ISL"]
---

- 통계학에서 resampling methods는 training 데이터에서 샘플을 여러 번 추출한 후 이 샘플을 이용해 모델을 적합해 모델에 대한 추가 정보를 얻는 방법을 말한다.
- 예를 들어, linear regression의 변동성을 예측하고 싶다고 하자. Training 데이터에서 샘플을 여러 번 추출한 후, 각 샘플에 linear regression을 적합한다. 그리고 모델이 샘플 데이터에 따라 얼마나 달라지는지 확인하면 된다.
- Chapter 5에서는 resampling methods 중 가장 많이 사용되는 cross-validation과 bootstrap에 대해 다룰 거다.
- Cross-validation은 test error를 계산해 모델의 성능을 측정하거나, 모델의 유연성을 결정하기 위해 사용할 수 있다.
- 모델의 성능을 평가하는 건 model assessment라 하고, 모델의 적절한 유연성을 결정하는 과정은 model selection이라 한다.
- Bootstrap은 주로 추정된 모수나 모델의 적확도를 측정할 때 사용한다.

## 5.1 Cross-Validation

- Test error는 모델 training에 사용하지 않은 데이터를 모델에 넣었을 때의 평균 오류 값이다.
- Test error는 test 데이터가 있다면 계산하는 건 문제가 안 되지만 많은 문제에서 training 데이터만 주어진다.
- Cross-validation 방법을 사용하면 test 데이터가 없을 때 test error를 추정할 수 있다.
- Training error에 수학적인 조정을 해 test error를 추정하는 방법도 있다. 이는 6장에서 살펴볼 예정이다.

### 5.1.1 The Validation Set Approach

- Validation set 방법은 데이터를 랜덤으로 training set과 validation set 또는 hold-out set으로 나눈다. 그리고 training 데이터를 이용해 얻은 모델을 이용해 validation set의 데이터를 예측하고, 여기서 얻은 오류값 을 test error rate의 추정 값이라 한다.
- 모델에 어떤 변수가 추가하는 게 의미가 있는지 확인하고 싶다고 하자. 해당 변수의 p-value를 확인할 수도 있지만, 변수가 있는 모델과 없는 모델의 validation error rate를 비교해 결론을 내릴 수도 있다.
- 데이터를 어떻게 랜덤으로 나누는지에 따라 validation error rate은 달라진다.
- Validation set approach는 단순해 쉽게 적용할 수 있지만, 두 가지 단점이 있다. 첫 번째는 데이터가 training과 validation set 중 어디에 속하는지에 따라 validation error 값이 달라질 수 있다는 것이다. 두 번째는 데이터가 많을수록 모델의 성능이 좋아질 가능성이 높은데, 모델 적합에 training 데이터를 다 사용하지 않아 validation set error가 test error를 높게 예측할 수 있는 문제가 있다.

### 5.1.2 Leave-One-Out Cross-Validation

- LOOCV(Leave-One-Out Cross-Validation)은 validation set 방법과 비슷하게 데이터를 training과 validation set으로 나눈다. 하지만 validation set 방법과 다른 점은 LOOCV은 하나의 데이터만 validation set으로, 나머지를 training set으로 분류한다는 점이다.
- 첫 번째 데이터를 validation set으로 사용해 validation error를 구한다. 이 과정을 모든 데이터에 대해서 적용하면 n개의 validation error를 얻을 수 있다. LOOCV는 이렇게 얻은 validation error의 평균을 test error의 추정치로 사용한다.
- LOOCV는 validation set 방법과 달리 주어진 데이터를 거의 다 사용해 모델을 추정하기 때문에 bias가 적고 test error를 더 높게 예측할 확률도 나다. 또한, 데이터를 랜덤으로 나누지 않기 때문에 몇 번 반복하든 똑같은 추정치를 얻을 수 있다.
- 하지만 LOOCV는 모델을 데이터 수만큼 적합해야 하기 때문에 계산량이 많다는 단점이 있기도 하다.

### 5.1.3 k-Fold Cross-Validation

- K-fold cross-validation은 데이터를 균등하게 k개의 그룹으로 나눈다. k개 중 하나의 그룹을 validation set, 나머지 k - 1 그룹을 traning set으로 분류해 모델을 적합하고 validation set error를 구한다. 이 과정을 k번 반복하면 k개의 validation set error를 얻을 수 있고, 이를 평균 내 test error 추정치로 사용할 수 있다.
- LOOCV는 k-fold cross-validation에서 k = n일 때의 특별한 경우이다.
- Cross-validation은 모델이 새로 보는 데이터에서 얼마나 잘 작동할지 알고 싶을 때 사용하기도 하지만, 많은 경우에는 모델끼리 비교를 하거나 모델의 모수를 결정하기 위해서도 사용한다. 두 번째 경우에서는 test error의 값 자체보다는 언제 test error가 가장 작은지가 중요하다.

### 5.1.4 Bias-Variance Trade-Off for k-Fold Cross-Validation

- LOOCV와 비교했을 때 k-fold CV는 계산량이 적다는 장점이 있지만, 더 중요한 건 test error를 더 잘 예측한다는 것이다. 이는 bias-variance trade-off와 관련 있다.
- Bias 측면에서는 LOOCV가 k-fold CV 보다 좋은 방법이 될 수 있다. 하지만, LOOCV에서는 모든 training set이 비슷해 이 데이터로 계산한 어떠한 값은 서로 상관관계가 높을 수밖에 없고, 이는 높은 variance로 이어진다.
- 정리하면 어떤 k를 사용하는지에 따라 bias-variance trade off가 일어난다. 경험적으로 k = 5 또는 10일 때 variance와 bias가 적당하다고 알려져 있다.

### 5.1.5 Cross-Validation on Classification Problems

- Regression에서는 test error로 주로 MSE를, classication에서는 오분류율을 사용한다.

## 5.2 The Bootstrap

- Bootstrap은 추정치의 불확실성을 측정하는 하나의 방법이다. 예를 들어, bootstrap 방법으로 linear regression의 계수의 오차를 추정할 수 있다. Linear regression 외에도 통계 프로그램으로도 계산하기 쉽지 않은 다양한 모델에 적용할 수 있기 때문에 많이 사용된다.
- 모집단의 데이터를 알고 있다면 모집단에서 샘플링한 데이터로 추정치를 계산할 수 있다. 이 과정을 n번 반복하면 n개의 추정치를 얻을 수 있고, n개의 추정치로 계산한 표준편차가 문제에서 알고 싶어 하는 불확실성이다.
- 하지만, 대부분의 경우에 모집단 데이터는 알기 어렵다. 이런 상황에서 bootstrap은 샘플 데이터에서 랜덤으로 데이터를 복원 추출하는 방법이다.
- 구체적인 예시로 설명하자면, 데이터에서 n개의 데이터를 복원 추출하고 이를 bootstrap data라 한다. 그리고 bootstrap data로 계산한 추정치는 bootstrap estimate라 한다. 이 과정을 충분히 큰 B번 반복하면 B개의 bootstrap estimate를 얻게 되고, 표준편차를 계산할 수 있다.