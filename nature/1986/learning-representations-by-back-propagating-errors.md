# Learning representations by back-propagating errors
David E. Rumelhart, Geoffrey E. Hinton & Ronald J. Williams

## Summary
신경망의 가중치를 조정해 실제값과 예측값 간의 오차를 최소화하는 방식으로 input unit과 output unit 사이에 hidden unit을 사용해 입력 데이터의 규칙성과 중요한 feature를 학습하는 방안을 제시한다. 

## Related Work
### Forward Pass
input unit과 output unit이 있을 때, 학습이 시작되면 아래(input layer)에서부터 위(output layer)로 진행된다.

- $x_j=\sum_i y_i w_{ji}$
    - $x_j$: $j$번째 뉴런으로 들어오는 총 입력값(이전 뉴런의 출력값과 가중치의 선형 결합)
    - $y_i$: 이전 $i$번째 뉴런의 출력값
    - $w_{ji}$: $i$번째에서 $j$번째 뉴런으로 연결된 가중치
- $y_j=\frac{1}{1+e^{-x_j}}$
    - $y_j$: $x_j$를 비선형 활성화 함수(sigmoid)를 통해 변환한 출력값
- $E=\frac{1}{2}\sum_c \sum_j (y_{j,c}-d_{j,c})^2$
    - $y_{j,c}$: 예측값
    - $d_{j,c}$: 실제값

## Method
### Backward Pass
가중치를 조금 변화시켰을 때 해당 변화가 전체 에러($E$)에 어느 정도 영향이 있는지 확인하고, 이를 최소화시키는 방향으로 위에서부터 아래로 내려오면서 가중치를 업데이트하는 과정을 제안한다.

1. $E$를 $y_j$에 대해 미분
    - 출력 유닛에서 오차($E$)가 출력값($y_j$)에 따라 어떻게 변화하는지 계산
    - $\delta E / \delta y_j=y_j-d_j$
2. $E$를 $x_j$에 대해 미분(식.1에 chain rule 적용)
    - 총 입력($x_j$)에 따른 오차 변화량 계산
    - $\delta E / \delta x_j=\delta E / \delta y_j \cdot dy_j/dx_j$
3. forward pass $y_j$식을 $x$에 대해 미분
    - 활성화 함수(sigmoid)의 도함수를 사용해 출력값($y_j$)이 총 입력($x_j$)에 따라 어떻게 변화하는지 계산
    - $y_j=\frac{1}{1+e^{-x_j}}$ &rarr; $dy_j/dx_j=y_j(1-y_j)$
    - $\delta E / \delta x_j=\delta E / \delta y_j \cdot y_j(1-y_j)$
4. $E$를 $w_{ji}$에 대해 미분
    - 가중치($w_{ji}$)가 오차($E$)에 미치는 영향을 계산
    - $\delta E / \delta w_{ji}=\delta E / \delta x_j \cdot \delta x_j / \delta w_{ji}$
    - $\delta E / \delta w_{ji}=\delta E / \delta x_j \cdot y_i$

위 1-4 과정을 반복하며 학습을 진행하고, gradient descent 알고리즘을 통해 다음과 같이 가중치를 업데이트한다

- $\Delta w = -\epsilon \delta E/ \delta w$
    - 간단한 버전으로 누적된 가중치에 비례해 가중치를 변경
- $\Delta w(t) = -\epsilon \delta E/ \delta w(t) + \alpha \Delta w(t-1)$
    - 학습 안정성과 수렴 속도를 개선하기 위한 방법
    - 모멘텀 계수 $\alpha$를 사용하여 이전 기울기를 반영함으로써 학습 속도를 조절
    