# 운동 동작 분류 AI 경진대회
### code submitted to the contest(월간 데이콘 11)
for data sets and contest description, check out [DACON overview](https://dacon.io/competitions/official/235689/overview/description)

![image](https://user-images.githubusercontent.com/80621384/141674009-34dcae5e-c1ee-45d6-a213-a0d67fc5872d.png)


## explanation
**order: load data -> feature engineering -> scaling -> model train -> stacking ensemble -> predict & submit**

feature engineering, scaling, model training parts are processed into ```.py``` file after submission (elaborated)

- feature engineering: 주어진 피처만으로는 만족할만한 성능을 내지 못해, 데이터를 대변할 수 있을만한 피처를 추가적으로 생성할 수 있을까 고민하다가 아래와 같은 파생변수를 추가했습니다.

  - 가속도와 자이로스코프 제곱합 'acc_t' 와 'gy_t'
  - 자이로스코프의 경우 이전 time에서 얼마나 달라졌는지를 나타내는 'gy_x_diff' 등

- scaling: feature가 가지는 범위의 영향을 줄이기 위해 feature 값들이 표준정규분포를 따르도록 변환하는 StandardScaler를 사용했습니다. [machinemastery](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/) 사이트 참고

- models: cnn 모델을 여러 개 만들어 앙상블했습니다. LSTM이나 GRU 층을 끼고도 만들어보았지만, 성능 향상에 도움이 되지 않아 Conv1D 층을 중심으로 만들었습니다. 데이터 수가 부족하다고 판단해 validation data 없이 모델을 학습시켰습니다. (fit시킬 때 validation_split=0)


*check ```dacon_final.ipynb``` file, or our team's shared code in [DACON site](https://dacon.io/codeshare/2408) for original codes and korean explanation*

## result
총 1,097팀 중 12위 (최종 코드 공유하지 않은 팀 제외)

![result](https://user-images.githubusercontent.com/80621384/141674504-e217c822-1018-49c8-80cc-437b1ff10dae.png)
