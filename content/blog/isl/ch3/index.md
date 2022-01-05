---
title: Ch 3. Linear Regression
date: "2021-09-21"
description: "Introduction to Statistical Learning Chapter 3 Notes"
tags: ["ISL"]
---

- Linear regression은 supervised learning 종류의 매우 간단한 모델이다.
- 앞으로 배울 모델에 비해 단순하기는 하지만 아직도 많은 분야에서 사용되고 있으며, 복잡한 모델 중에 linear regression에서 확장 또는 일반화된 것들이 많다. 따라서, linear regression에 대한 이해 없이 다른 모델을 공부하기는 어렵다.
- Advertising 데이터를 이용해 TV, 라디오, 뉴스에 얼마를 투자해야 내년에 높은 매출을 낼 수 있는지 마케팅 계획을 세우고 싶다고 하자. 이런 상황에서는 아마도 다음과 같은 질문을 할 수 있을거다.
1. 마케팅 비용과 매출간에 유의미한 관계가 있나?

가장 먼저 매출이 마케팅 비용에 따라 달라지는지 확인해야 한다. 만약 상관없다면 마케팅에 아무런 비용을 쓸 필요가 없다고 할 수도 있다.

2. 마케팅 비용과 매출의 관계는 얼마나 유의미한가?

만약 두 변수 사이에 유의미한 관계가 있다면, 이 관계가 얼마나 강한지 알고 싶은 거다. 즉, 마케팅 비용이 주어졌을 때 매출을 정확하게 예측할 수 있는지 알고 싶다.

3. 어떤 매체가 매출에 영향을 미치나?

TV, 라디오, 신문이 모두 매출에 영향을 미치는지, 아니면 이 중 하나 또는 두 개만 유의미한 변수인지 확인해야 한다. 이 질문에 답하기 위해서는 모든 매체에 비용을 지불했을 때 각 매체가 매출에 미치는 영향을 분리할 수 있어야 한다.

4. 각 매체가 매출에 미치는 영향을 얼마나 정확하게 측정할 수 있나?

특정 매체에 1달러를 투자했을 시 매출이 얼마큼 변하나요? 그리고 이 변화를 얼마나 정확하게 예측할 수 있나요?

5. 미래 매출을 얼마나 정확하게 예측할 수 있나?

6. 마케팅 비용과 매출은 선형 관계인가?

마케팅 비용과 매출이 선형 관계라면 linear regression이 적절한 방법이 될 수 있다. 하지만, 그렇지 않더라도 변수에 적당한 변형을 하면 linear regression을 적용할 수도 있다.

7. 두 개 이상의 매체에 마케팅 비용을 투자했을 때 시너지 효과가 있나?

두 매체에 각각 $5,000를 투자했을 때 한 매체에 $10,000를 사용하는 거보다 매출이 더 높을 수 있다. 이를 마케팅 용어로는 시너지 효과, 통계학에서는 interaction effect이라고 한다.

## 3.1. Simple Linear Regression

- Simple linear regression은 *X*와 *Y*가 선형 관계라고 가정한다. 수학적으로 표현하면 다음과 같다.

$$
Y\approx\beta_{0}+\beta_{1}X
$$

- $\beta_{0}$, $\beta_{1}$은 아직 알지 못하는 상수로 각각 linear regression의 intercept와 slope이다. 또한, 이 둘을 합쳐 **coefficients이나 parameter**라 한다.
- Training data 추정한 $\hat{\beta_{0}}$, $\hat{\beta_{1}}$를 이용하면 새로운 $x$가 주어졌을 때 아래 식을 이용해 $\hat{y}$를 예측할 수 있다.

$$
\hat{y}=\hat{\beta_{0}}+\hat{\beta_{1}}x
$$

### 3.1.1. Estimating the Coefficients

- 실제로 $\beta_{0}$, $\beta_{1}$는 알 수 없는 값이기 때문에 training data로 이 값을 추정해야 한다. 즉, 모델이 데이터를 잘 설명할 수 있는 $\hat{\beta_{0}}$, $\hat{\beta_{1}}$를 구해야 한다.
- 여러 가지 방법이 있지만, 가장 대표적으로 **오차 제곱의 합(residual sum of squares)을 최소**로 하는 추정값을 찾는 **least squares 방법**이 있다.
- $x_{i}$를 모델에 넣어 $\hat{y_{i}}$를 얻었다고 하자. 오차는 $e_{i}=y_{i}-\hat{y_{i}}$로, 예측값과 실제 값의 차이다. 모든 *X*에 대해 계산한 오차의 합을 RSS(residual sum of squares)라고 하는데, least squares 방법을 이용하면 RSS가 가장 작은 $\hat{\beta_{0}}$, $\hat{\beta_{1}}$를 얻을 수 있다. 그리고 이렇게 추정한 paramerter는 least squares coefficient estimates라 한다.
- Advertising data에서 *X*가 TV 마케팅 비용, *Y*가 매출이라고 하자. $\hat{\beta_{0}}$=7.03, $\hat{\beta_{1}}$=0.0475이면, TV 광고에 $1,000를 추가로 투자할때마다 상품이 47.5개 더 팔린다는 의미이다.

### 3.1.2 Assessing the Accuracy of the Coefficient Estimates

- 2.1에서 *X*와 *Y*는 아직 알려지지 않은 함수 *f*와 평균이 0인 랜덤 오차 $\epsilon$로 나타낼 수 있다는 걸 보았다. 만약 *f*가 선형 모델에 가깝다면 두 변수의 관계는 다음과 같이 표현할 수 있다.

$$
Y=\beta_{0}+\beta_{1}X+\epsilon
$$

- $\beta_{0}$는 intercept으로 *X*가 0일 때 *Y*의 추정 값이고, $\beta_{1}$는 slope으로 *X*가 한 단위 증가할 때 Y의 평균 변화량이다. *X*와 *Y*는 실제로 완벽한 선형 관계가 아닐 가능성이 높고, 이로 인해 생기는 오차 $\epsilon$는 보통 *X*와 독립이라고 가정한다.
- Linear regression의 **parameter $\beta_{0}$, $\beta_{1}$는 어떤 데이터를 이용해 추정하는지에 따라 값이 달라진다.** 즉, 모수 데이터를 알고 있는 게 아니라면 실제 값과 정확하게 같을 수는 없다. 하지만, 여러 데이터로 **추정 값을 구해 평균을 내면 실제 값에 근사**한다. 이러한 특징을 가진 변수를 **unbiased**하다고 한다.
- 그렇다면 그다음에 궁금할 수 있는 건 parameter 추정 값 하나는 실제 값에 얼마나 근사하는지이다. 이는 오차의 분산인 standard error로 계산할 수 있다.
- Standard error로는 parameter의 신뢰 구간을 계산할 수도 있다. 95% 신뢰구간이라는 의미는 95%의 확률로 실제 값이 추정한 구간 사이에 있다는 걸 의미한다.
- 또한, standard error는 가설검정에도 사용된다. 대표적으로 가장 많이 하는 검정은 다음과 같다.

> 귀무가설: $\beta_{1}$= 0 (*X*와 *Y* 사이에는 아무런 관계도 없다.)
대립가설: $\beta_{0}\not$= 0 (*X*와 *Y* 사이에는 어떠한 관계가 있다.)

- 위 가설을 검증하기 위해서는 $**\hat{\beta_{1}}$가 0에서 충분히 멀리 떨어져 있어** 0이 아니라는 결론을 내릴 수 있는지 확인하면 된다. 그렇다면 0에서 얼마나 많이 떨어져 있어야 할까? 이는 $\hat{\beta_{1}}$의 오차 값, SE($\hat{\beta_{1}}$)에 의해 결정된다. 만약 SE($\hat{\beta_{1}}$)이 작다면 $\hat{\beta_{1}}$의 값이 작더라도 $\beta_{1}$는 0이 아닐 가능성이 높다. 반대로 SE($\hat{\beta_{1}}$)이 크다면 귀무가설을 기각하기 위해선 $\hat{\beta_{1}}$도 커야 한다.
- 가설 검증 위해서는 t-statistics이라는 지표를 계산한다. T-statistics는 $\hat{\beta_{1}}$이 0으로부터 SE($\hat{\beta_{1}}$)의 몇 배 떨어져 있는지 나타낸다.

$t=(\hat{\beta_{1}}-0)/\hat{SE(\hat{\beta_{1}})}$

- 만약에 *X*와 *Y* 사이에 아무런 관계가 없다면 t-statistics는 자유도가 n-2인 t 분포를 따른다. T 분포는 데이터 수가 많아질수록 정규분포에 근사하는 벨 모양의 분포이다.
- 이때, **|t|와 같거나 큰 값을 가지는 확률을 p-value**라 한다. 간단히 설명하면 귀무가설이 사실일 때 t-statistics 값을 얻을 확률이다. 즉, **p-value가 충분히 작으면 귀무가설을 기각하고 *X*와 *Y* 사이에는 어떠한 관계가 있다고 결론**내릴 수 있다. 보통 0.05 또는 0.001보다 작으면 충분히 작다고 한다.

### 3.1.3 Assessing the Accuracy of the Model

- 귀무가설을 기각함으로써 '*X*와 *Y*가 어떠한 관계를 가지고 있다'라는 결론을 내린다면, 그 후에는 **모델이 데이터를 얼마나 설명하는지** 알고 싶을 것이다.
- Linear regression의 적합도(lack of fit)를 수치화하는 대표적인 지표로는 **Residual Standard Error와 $R^{2}$**가 있다.

**Residual Standard Error**

- Regression 모델의 모수 값을 알고 있다고 해도 $\epsilon$  때문에 *X*로 *Y*를 완벽히 예측하는 건 불가능하다. RSE는 이 $**\epsilon$  표준편차의 추청치**다. 간단히 설명하면, y 예측값과 y 값 차이의 평균이다.
- RSE의 값은 데이터에 따라 너무 클 수도 적절할 수도 있다.
- RSE는 모델의 lack of fit을 측정하는 지표로 사용된다. RSE가 작으면 $\hat{y}$는 $y$와 근사하다는 것으로 모델이 데이터를 잘 설명하고 있다는 의미이다.

**$R^{2}$ Statistic**

- RSE는 *Y*의 단위로 모델의 lack of fit을 측정하고, 절대값이기 때문에 많은 경우에서 이 RSE 값이 괜찮은지 알기 어렵다.
- 반면에 $**R^{2}$는 *Y*의 분산 중 *X*로 설명 가능한 분산의 비율**로 Y의 단위와 무관하며 **0에서 1 사이의 값을 가진다.** 그렇기 때문에 모델끼리 비교가 가능해 상대적으로 해석이 쉽다는 장점은 있지만 어느 정도가 좋은 값인지는 역시 문제에 따라 다르다. 모델이 *Y* 분산을 많이 설명할수록 1에 가까운 값을 가진다.
- $R^{2}$는 *X*와 *Y*의 **선형 관계를 측정**하는 지표이고, input이 하나면 두 변수의 선형 관계를 측정하는 상관계수의 제곱과 똑같은 값을 가진다.

## 3.2 Multiple Linear Regression

- Input이 2개 이상인 경우에 우리가 시도할 수 있는 방법 중 하나는 변수 개수만큼 simple linear regression 모델을 만드는 거다. 하지만, 이 방법을 사용하면 각각의 모델로 하나의 output을 어떻게 계산해야 되는지 애매해질 뿐만 아니라 모델의 계수를 추정할 때 다른 input을 고려하지 않는 문제가 있다.
- 더 좋은 방법은 simple linear regression에 변수를 추가해 확장하는 것이다.
- *p*개의 *X*가 있다고 하면 multiple linear regression은 다음과 같다. $X_{j}$는 j번째 변수이고, $\beta_{j}$는 *Y*와 $X_{j}$의 관계를 수치로 나타낸 값이다.

$$
Y=\beta_{0}+\beta_{1}X_{1}+...+\beta_{p}X_{p}+\epsilon
$$

- 또한, **$\beta_{j}$는 다른 변수가 고정되었을 때 $X_{j}$의 1 증가에 따른 *Y*의 평균 변화량**이라고도 할 수 있다.

### 3.2.1 Estimating the Regression Coefficients

- $\beta$는 알 수 없는 값이기 때문에 추정해야 하는데, simple linear regression에서처럼 **RSS를 최소화하는 값**을 찾으면 된다.
- Input을 하나씩 모델에 적용했을 때와 하나의 모델에 두 개 이상의 input을 적용했을 때 똑같은 *X*라고 해도 추정된 계수의 값이 다를 수 있다. 이는 multiple linear regression은 *X*간의 관계도 모델에 포함시키기 때문이다.
- 또한, *X*가 다른 *X*들과 correlation이 높을 때 simple linear regression에서의 결과와 달리 multiple linear regression에서는 변수가 유의미하지 않다는 결론이 날 수도 있다.

### 3.2.2 Some Important Questions

- Multiple linear regression 모델을 사용할 때 우리는 아래 네 가지 질문에 대한 답을 찾고 싶어 한다.
1. 최소 한 개의 *X*가 *Y*를 예측하는데 유효한가?
2. 모든 *X*가 *Y*를 설명하는 데 도움을 주는가, 아니면 이 중 몇 개의 *X*만 유효한가?
3. 모델이 데이터를 얼마나 잘 설명해 주는가?
4. *X*로 예측한 *Y*가 얼마나 정확한가?

**One: Is There a Relationship Between the Response and Predictors?**

- Simple linear regression에서 X와 Y가 연관 있는지 알고 싶으면 $\beta_{1}$가 0인지 확인해 보았다. 비슷하게 multiple linear regression에서도 $\beta$가 0인지 검정해 볼 수 있다.

> $H_{0}:\beta_{1}=\beta_{2}=...=\beta_{p}=0$
$H_{1}$=$\beta_{j}$ 중 하나 이상이 0이 아니다.

- 위 **가설을 검정하기 위해 F statisitcs을 사용**한다. 귀무가설이 참일 때 F statistics는 F 분포를 따르는데, 이때 계산한 p-value 값을 이용해 귀무가설을 기각할지 말지 결정한다.
- 일부 *X*에 대해서만 F 검정을 할 수도 있다.
- 하나의 *X*에 대해 t 검정한 결과는 F 검정 결과와 같게 나온다.
- 최소 한 개의 X가 Y와 연관 있다는 결론을 내리기 위해 왜 모든 X에 각각 t 검정을 하지 않고 전체 X에 F 검정을 하는 걸까? X 중 하나라도 p-value가 매우 작으면, 최소 한 개의 X가 Y를 설명할 수 있다는 결론을 내릴 수 있지 않을까 하는 의문이 들 수 있다. 하지만, X의 수가 많으면 모든 개별 X의 계수가 유의미하더라도 100번 중 5번은 오차에 의해 p-value가 0.05보다 작게 나올 수 있다. 대신 F-statistics는 변수의 개수도 고려해 값을 계산하기 때문에 이러한 문제를 피할 수 있다.

**Two: Deciding on Important Variables**

- 최소한 하나의 *X가* *Y*와 유의미한 관계를 가진다는 결론이 나오면 그다음에는 어떤 변수가 *Y*에 가장 영향을  많이 미치는지를 알고 싶을 거다.
- 여러 개의 ***X* 중에서 의미 있는 일부를 고르는 작업을 variable selection**이라 한다.
- *X*를 사용하는 모든 경우의 수를 고려해 모델을 비교하는 게 가장 좋은 방법이다. 하지만 p 값이 커짐에 따라 계산량이 너무 많아지기 때문에 다른 방법이 필요하다.
- **Forward selection은 *X*가 없는 모델에서 시작**한다. **Input을 하나씩 넣은 p개의 모델을 만들고, 각 모델의 RSS를 계산한다. 그리고 가장 작은 RSS 값을 가진 모델의 *X*를 추가**한다. 이 과정을 미리 정한 중단 조건을 만족할때까지 반복한다. 이때, 처음에 유의미하지 않다고 판단한 변수가 나중에 모델에 추가될 수도 있다.
- **Backward selection은 모든 *X*가 들어간 모델부터 시작한다. P-value가 큰 변수부터 하나씩 제거**하는데, 미리 정한 중단 조건을 만족할 때까지 반복한다. 데이터 수보다 변수의 수가 많으면 사용할 수 없다.
- **Mixed selection은** forward selection과 backward selection을 합친 방법이다. ***X*가 없는 모델부터 시작해 forward selection 방법처럼 *X*를 하나씩 추가**한다. 그 과정에서 **p-value가 미리 정한 기준보다 큰 변수가 생기면 해당 변수는 제외**한다. 이 과정을 모델에 포함된 모든 변수의 p-value는 작고, 모델에 포함되지 않은 변수는 추가했을 때 p-value가 커질 때까지 반복한다.
- 모델을 비교하는 기준이 되는 지표로는 **Mallow's $C_{p}$, AIC, BIC, adjusted-$R^{2}$** 등이 있다. 또한, residuals을 그래프로 그려보는 거도 좋은 방법이 될 수 있다.

**Three: Model Fit**

- **모델이 데이터를 얼마나 잘 설명하는지** 알기 위해서 확인하는 지표로는 대표적으로 **$R^{2}$와 RSE**가 있다.
- *X*가 *Y*와 유의미한 관계가 아닐지라도 모델에 추가되면 **$R^{2}$**는 커진다. 어찌 되었든 변수를 추가하면 모델은 train data를 더 잘 따르게 되기 때문이다.
- **변수를 추가했을 때 $R^{2}$가 거의 증가하지 않는다면, 이 변수는 유의미하지 않다**고 결론 내릴 수 있다. 그리고 해당 변수를 사용한 모델은 overfitting 문제가 있을 수도 있다. 반대로 **$R^{2}$**가 많이 증가한다면 이 변수는 *Y*를 설명하는데 유의미한 변수일 가능성이 높다.
- RSE는 p가 아무리 커져도 RSS의 감소 폭이 작다면 값이 커질 수도 있다.
- 그래프를 이용하면 수치적으로 확인할 수 없는 문제(예. 비선형 관계)를 알 수 있기도 하다.

**Four: Predictions**

- 추정한 모델을 이용해 *Y* 값을 예측하고자 한다. 이때 3가지 불확실성이 생긴다.
1. $\beta$의 추정값 $\hat{\beta}$에 대한 불확실성이 있다. 이는 reducible error이고, confidence interval을 계산해 $\hat{Y}$이 f(X)와 얼마나 가까운지 수치화할 수 있다.
2. 선형 모델에서는 *X*와 *Y*가 선형 관계라고 가정하는데, 여기서 생기는 또 다른 reducible error인 model bias가 있다.
3. f(X)의 진짜 값을 안다고 해도 $\epsilon$ 때문에 Y를 완벽하게 예측하는건 불가능하다. 그리고 이는 irreducible error이다. 이 불확실성을 수치적으로 나타내기 위해 prediction interval을 계산할 수 있는데, prediction interval은 confidence interval 보다 항상 크다. 왜냐하면 예측구간은 reducible error와 irreducible error를 모두 포함하기 때문이다.

## 3.3 Other Considerations in the Regression Model

### 3.3.1 Qualitative Predictors

- 성별, 학력처럼 데이터가 정해진 몇 개의 범주 중 하나의 값을 가지는 변수를 qualitative라 한다.

**Predictors with Only Two Levels**

- Qualitative 변수가 두 가지 중 하나의 값을 가지면, **0 또는 1 값을 가지는 dummpy 변수**를 이용하면 된다.
- 예를 들어, *X*가 'female' 또는 'male' 값을 가지는 변수라면 아래와 같이 새로운 변수를 정의할 수 있다.

    $$  x_{i} =
        \begin{cases}
          1 & \text{if $i$th person is female}\\
          0 & \text{if $i$th person is male}
        \end{cases}       
    $$

    그리고 새롭게 정의한 *X*를 모델에 적용하면 *X* 값에 따라 다른 모델을 얻게 된다.

    $$  x_{i} = \beta_{0}+\beta_{1}x_{i}+\epsilon_{i} =
        \begin{cases}
          \beta_{0}+x_{i}+\epsilon_{i} & \text{if $i$th person is female}\\    
          \beta_{0}+\epsilon_{i} & \text{if $i$th person is male}
        \end{cases}       
    $$

    여기서 $**\beta_{0}$은 남성의 평균 *Y* 값*, $\beta_{0}+\beta_{1}$*은 여성의 평균 *Y* 값, $\beta_{1}$은 남성과 여성의 *Y* 값의 평균 차이**라고 설명할 수 있다.

- Dummy 변수를 어떻게 정의하든 결과는 같게 나온다. 하지만, 계수에 대한 해석이 달라질 수 있으니 유의해야 한다.

**Qualitative Predictors with More than Two Levels**

- **Qualitative 변수가 셋 이상의 값을 가지면** dummy 변수 한 개로는 모든 경우의 수를 나타낼 수 있다. 이런 경우에는 단순히 **여러 개의 dummy 변수를 사용**하면 된다.
- 예를 들어, 'Asian', 'Caucasian', 'African American' 중 하나의 값을 가지는 변수가 있다고 하자. 이 경우에는 아래와 같이 두 개의 변수가 필요하다

    $$  x_{i1} =
        \begin{cases}
          1 & \text{if $i$th person is Asian}\\
          0 & \text{if $i$th person is not Asian}
        \end{cases}       
    $$

    $$  x_{i2} =
        \begin{cases}
          1 & \text{if $i$th person is Caucasian}\\
          0 & \text{if $i$th person is not Caucasian}
        \end{cases}       
    $$

    $x_{i1}$, $x_{i2}$를 사용하면 다음과 같은 모델을 얻을 수 있다.

    $$  x_{i} = \beta_{0}+\beta_{1}x_{i1}+\beta_{2}x_{i2}+\epsilon_{i} =
        \begin{cases}
       \beta_{0}+\beta_{1}+\epsilon_{i} & \text{if $i$th person is Asian}\\    
       \beta_{0}+\beta_{2}+\epsilon_{i} & \text{if $i$th person is Caucasian} \\
          \beta_{0}+\epsilon_{i} & \text{if $i$th person is African American}
        \end{cases}       
    $$

- 항상 범주 수보다 하나 적은 dummy 변수가 필요하다.
- Dummy 변수를 어떻게 정의하든 모델에 대한 해석 결과는 같게 나오지만, **개별 변수의 t 검정 결과는 달라질 수 있다.** 따라서 이 경우에는 **F-test를 사용**하는 게 좋다.

### 3.3.2 Extensions of the Linear Model

- 선형모델은 해석 가능한 결과를 제공해 주고 실제 상황에서도 잘 작동하기는 하지만, 데이터가 additctive 하고 linear 하다는 매우 제한적인 가정을 하고 있다.
- **Additive는 $X_{j}$의 변화가 *Y*에 미치는 영향이 다른 *X*의 값과 상관없이 일어난다**는 것이고, **linear는 $X_{j}$가 한 단위 증가하면 어떤 값에 있든 *Y*의 변화가 일정하다**는 것이다

**Removing the Additive Assumption**

- **$X_{j}$의 변화가 *Y*에 미치는 영향이 다른 *X* 값에 의해 영향을 받을 때 interaction effect이 있다**고 한다.
- 모델에 interaction effect을 추가하고 싶다면 서로 영향을 주고받는 두 개 *X*의 곱을 변수로 추가하면 된다.

    $$Y=\beta_{0}+\beta_{1}X_{1}+\beta_{2}X_{2}+\beta_{3}X_{1}X_{2}+\epsilon$$

    이 모델에서 **$X_{1}$이 한 단위 증가하면 *Y*는 $\beta_{0}+\beta_{1}+\beta_{3}X_{2}$만큼 증가**한다. 즉, **$X_{2}$값에 의해 영향을 받게 된다.** 

- Interaction 변수의 계수 $\beta_{3}$은 $X_{2}$가 한 단위 증가할 때 $X_{1}$가 Y에 미치는 영향력이라고 해석할 수 있다.
- 어떤 모델에서는 interaction 변수만 통계적으로 유의미할 때가 있는데, **hierarchical principle에 의해 interaction 변수를 사용한다면 main effect 변수가 유의미하지 않더라도 모델에 포함**시켜야 한다.

**Non-linear Relationships**

- *X*와 *Y*가 선형관계가 아닐때 **선형 모델을 비선형 모델로 확장**하는 가장 간단한 방법은 ***X*를 변형해 (예. $X^{2}$, $X^{3}$) 모델에 추가**하는 **polynomial regression**을 사용하는 것이다.

### 3.3.3 Potential Problems

- Linear regression 모델을 사용할 때 많은 문제가 나타날 수 있는데, 문제점을 알아채고 해결하는 것은 과학이라기 보단 예술에 가깝다.

**1.** **Non-linearity of the Data**

- Linear regression 모델은 *X*와 *Y*가 선형 관계라고 가정하기 때문에 실제로 두 변수가 비선형 관계라면 모델을 통해 내리는 모든 결론은 신뢰할 수 없게 된다. 또한, 모델의 예측 정확도도 낮아진다.
- ***X* 또는 $\hat{Y}$와 잔차 $e$간의 그래프**를 그렸을 때 어떠한 패턴이 보인다면 선형 가정에 문제가 있을 가능성이 높다.
- 문제를 해결하기 위한 가장 간단한 방법으로는 *X*에 변형을 한 변수를 모델에 추가하면 된다.

**2. Correlation of Error Terms**

- 오차가 독립 관계라는 선형모델의 가정 중 하나가 깨지게 되면 추정 오차 값이 실제 값에 비해 작은 경향을 보이게 된다.
- 그리고 이는 낮은 p-value과 좁은 신뢰구간으로 이어져 유의미하지 않은 변수를 통계적으로 의미 있다고 판단할 수 있게 한다. 즉, 모델의 신뢰성이 낮아진다.
- 이 문제는 시계열 데이터에서 주로 나타나는데, 근접한 시간 내에서 데이터가 양의 관계를 가지는 경우가 많다.
- 만약에 시간에 따른 잔차 그래프를 그렸을 때 근처에 있는 오차들이 비슷한 값을 가지는 패턴을 보인다면 문제가 있을 가능성이 높다.
- 시계열 데이터 외에도 데이터가 모델에 포함하지 않는 변수에 영향을 받는다면 오차의 독립성 가정에 문제가 생길 수 있다.

**3. Non-constant Variance of Error Terms**

- 선형모델의 중요한 또 다른 가정은 오차의 분산이 일정하다는 것이다. 이는 잔차와 **$\hat{Y}$** 그래프에서 확인할 수 있는다.
- 오차의 변수가 일정하지 않은 문제(heteroscedasticity)는 Y의 변형 (예를 들어 logY, $\sqrt{Y}$) 또는 weighted leat squares으로 문제를 해결할 수 있다.

**4. Outliers**

- **$x_{i}$를 모델에 넣었을때 $y_{i}$와 매우 다른 값을 가질때** $x_{i}$를 outlier라 한다.
- Outlier의 유무에 따라 모델의 결과가 크게 달라지지는 않지만, RSE, p-value 등에 영향을 미친다.
- 잔차와 **$\hat{Y}$** 그래프에서 다른 데이터들과 다른 값을 가지면 outlier 일 가능성이 높다. 하지만, 얼마나 달라야 outlier라고 결정해야 하는지 명확하지 않을 때가 있는데, 이런 경우에는 studentized 잔차를 사용하는 게 좋다. **Studentized 잔차가 3보다 크면 outlier라고 한다.**
- 만약 outlier가 데이터 수집상의 오류라면 단순히 제거하면 되지만, 모델이 데이터를 잘 설명하지 못하고 있어서 나타나는 값일 수도 있으니 조심해야 한다.

**5. High Leverage Points**

- **나머지 *X*와 다른 값을 가지는 $x_{i}$**를 leverage라 한다.
- Outlier와 달리 leverage의 유무에 따라 모델의 결과가 크게 달라진다. 그렇기 때문에 leverage가 있는지 유심히 살펴야 한다.
- Simple linear regression에서는 단순히 평범한 범위 내에서 벗어나 있는 값을 확인하면 되지만, multiple linear regression에서는 조금 복잡하다. 왜냐하면 하나의 *X*에서는 leverage가 아닌데, 다차원의 *X*로 봤을 때 leverage 일 수 있기 때문이다.
- 다차원에서 leverage를 찾아내는 방법 중 하나는 **각 데이터의 leverage statistic를 계산**하는 것이다. 만약 leverage statistic이 **(p+1)/n보다 매우 높다면 해당 데이터가 leverage 일 가능성이 있다.**

**6. Collinearity**

- ***X* 변수끼리 상관관계가 높을 때** collinearity라고 한다.
- Collinearity가 있으면 한 변수가 증가/감소함에 따라 다른 변수도 같이 증가/감소하기 때문에 각각의 변수와 *Y*와의 관계를 알아내기 어렵다는 문제가 있다. 또한, $\hat{\beta_{j}}$의 오류를 증가시키고 변수가 유의하지 않다는 검정을 기각하기 어렵게 만든다.
- Collinearity를 확인하는 가장 간단한 방법은 *X*끼리 상관계수를 계산해 보는 것이다. 하지만, 이 방법으로는 3개 이상 변수끼리의 collinearity(multicollinearity)를 확인하기는 어렵다.
- 이런 경우에는 각 변수의 **VIF(Variance Inflation Factor)**를 계산해 볼 수 있다. 통상적으로 **VIF가 5 이상이면 문제가 있다**고 한다.
- 어떤 변수끼리 상관관계가 높은지 확인했다면 해당 변수 중 하나 이상을 제거하거나 상관관계가 높은 변수를 합쳐(예를 들어, 평균을 내거나) 문제를 해결할 수 있다.

## 3.4 The Marketing Plan

- 이제 이 장의 가장 처음에 했던 질문에 답을 할 수 있다.
1. 마케팅 비용과 매출 간에 유의미한 관계가 있나?

이 문제는 multiple regression model을 적합 후, $\beta_{j}$가 0이 아닌지 검정함으로써 답할 수 있다. F-statistics의 p-value가 유의수준보다 낮다면 귀무가설을 기각하고 마케팅 비용과 매출은 유의미한 관계라고 결론 내리면 된다.

2. 마케팅 비용과 매출의 관계는 얼마나 유의미한가?

모델의 정확도는 RSE와 $R^{2}$으로 확인할 수 있다.

3. 어떤 매체가 매출에 영향을 미치나?

모든 변수에 대해 $\beta_{j}$=0인지 통계적 검정을 해보면 된다. T-statistics의 p-value가 유의수준보다 낮다면 귀무가설을 기각하고 해당 매체가 매출에 영향을 미친다고 할 수 있다.

4. 각 매체가 매출에 미치는 영향을 얼마나 정확하게 측정할 수 있나?

각 변수의 confidence interval을 계산해 보거나, 변수를 하나씩만 넣은 simple linear regression을 이용해 각각의 매체가 매출에 얼마나 영향을 미치는지 알아볼 수 있다.

5. 미래 매출을 얼마나 정확하게 예측할 수 있나?

목적이 Y의 개별 데이터 예측이라면 prediction interval, Y의 평균 예측이라면 confidence interval을 계산하면 된다.

6. 마케팅 비용과 매출은 선형 관계인가?

잔차 그래프로 확인할 수 있다.

7. 두 개 이상의 매체에 마케팅 비용을 투자했을 때 시너지 효과가 있나?

Interaction 변수를 모델에 추가해 보고 통계적으로 유의미한지 확인해 본다.

## 3.5 Comparison of Linear Regression with K-Nearest Neighbors

- Linear regression은 f(X)가 선형이라는 가정을 하는 parametric 방법 중 하나이다. Parametric 방법은 상대적으로 적은 수의 계수를 추정하면 되기 때문에 모델을 만들고 결과를 해석하는 게 쉽다. 하지만, 데이터에 대한 가정이 틀릴 경우 모델의 신뢰도가 하락한다는 문제가 있다. 반대로 non-parametric 방법은 데이터에 대한 가정을 하지 않기 때문에 좀 더 유연한 접근 방식을 가지고 있다. K-Nearest Neighbors는 non-parametric의 대표적인 방법 중 하나이다.
- 만약 parametric 모델의 데이터에 대한 가정이 실제 데이터를 잘 설명해 줄 수 있다면 parametric 방법이 nonparametric 방법보다 더 좋은 결과를 낼 수 있다.
- 또한, 다차원의 데이터에서 nonparametric 방법의 결과가 안 좋은 경우가 많다. 예를 들어, KNN에서 test $x_{0}$와 가장 가깝다고 여겨지는 K 개의 데이터는 차원이 증가할수록 $x_{0}$와 멀어지는 문제가 발생할 수 있다. 이를 **curse of dimensionality**라 한다. 정리하면 **변수당 데이터 수가 적을 때 parametric 방법이 더 잘 작동**한다.
- 다차원이 아니더라도 만약 정확도에서 큰 차이가 나지 않는다면 결과 해석 가능하고 p-value를 계산할 수 있는 parametric 방법이 선호된다.