# %% [markdown]
# # DecisionTree 분류 모델을 이용한 타이타닉 데이터 분류

# %% [markdown]
# ## 데이터 불러오기

# %%
### 필요한 라이브러리 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
### 데이터 불러오기

# 파일 경로 설정하기
file_path='/content/drive/MyDrive/데이터 분석/titanic_preprocessed2.csv'

# csv file --> DataFrame 자료형 생성
df = pd.read_csv(file_path)

# 결과 확인하기
print(df)

# %% [markdown]
# ## 데이터 전처리

# %%
### 인코딩 --> 범주형 컬럼 : 문자열 --> 숫자

# np.unique() --> 범주형 컬럼 : 문자열 항목 --> 알파벳 순으로 정렬
arr = df.loc[:,'title'].values
kinds= np.unique(ar=arr)
print(f'title 컬럼의 항목 확인 : \n{kinds}')

print('-'*80)

# replace() 함수 --> label 인코딩
df = df.replace({'female':0,
            'male':1,
            kinds[0]:0,
            kinds[1]:1,
            kinds[2]:2,
            kinds[3]:3,
            kinds[4]:4})
print(df)

# %% [markdown]
# ## 학습용 데이터와 평가용 데이터 생성하기

# %%
### X_data / y_data 생성

# X_data 생성
X_data = df.drop(columns=['Survived', 'Embarked'])

# y_data 생성
y_data = df.loc[:,'Survived']

# 결과 확인하기
print(f'X_data 확인 : \n{X_data}')
print('-'*80)
print(f'y_data 확인 : \n{y_data}')

# %%
### 70 : 30의 비율로 학습용 데이터와 평가용 데이터 생성

# 필요한 함수 import
from sklearn.model_selection import train_test_split

# train_test_split() 함수 사용
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y_data)

# %%
### 학습용 데이터 확인

# 모양 확인
print(f'X_train의 모양 : {X_train.shape}')
print('-'*80)
print(f'y_train의 모양 : {y_train.shape}')

print('-'*80)

# 인덱스 확인
print(f'X_train의 인덱스 : \n{X_train.index}')
print('-'*80)
print(f'y_train의 인덱스 : \n{y_train.index}')

# %%
### 학습용 데이터 --> 정답 레이블의 분포 확인
ratio = y_train.value_counts(normalize=True)
print(f'학습용 데이터의 정답 레이블의 항목별 비율 : \n{ratio}')

# %%
### 평가용 데이터 --> 정답 레이블의 분포 확인
ratio = y_test.value_counts(normalize=True)
print(f'평가용 데이터의 정답 레이블의 항목별 비율 : \n{ratio}')

# %% [markdown]
# ## 모델 생성

# %%
### 필요한 함수 import
from sklearn.tree import DecisionTreeClassifier

# %%
### 모델 생성 함수 호출, 모델 생성
dt = DecisionTreeClassifier(random_state=0)

# %% [markdown]
# ## 모델 학습

# %%
dt.fit(X_train, y_train)

# %% [markdown]
# ## 학습 결과 시각화

# %%
### 기본 모델 학습 시 max_depth 확인
depth = dt.get_depth()
print(f'기본 모델의 최대 깊이 = {depth}')

# %%
### 필요한 함수 import
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 이미지의 크기 재설정
plt.figure(figsize=(12, 8))

# 시각화
plot_tree(dt, max_depth=3, feature_names=X_train.columns, filled=True)
plt.show()

# %% [markdown]
# ## 평가용 데이터를 이용한 예측

# %%
pred_test = dt.predict(X_test)
print(pred_test)

# %% [markdown]
# ## 모델 평가

# %%
# 필요한 함수 import
from sklearn.metrics import accuracy_score

# 평가용 데이터에 대한 정확도 평가
accuracy = accuracy_score(y_test, pred_test)

# 결과 학인하기
print(f'학습용 데이터에 대한 정확도 = {accuracy}')

# %% [markdown]
# ## GridSearchCV를 이용한 모델 성능 최적화

# %%
### GridSearchCV 함수 실행

# 필요한 함수 import
from sklearn.model_selection import GridSearchCV

# 최적화할 기본 모델 생성
base_dt = DecisionTreeClassifier(random_state=0)

# 튜닝할 매개 변수 설정
params = {'max_depth':[3,4,5,6,7,8,9]}

# GridSearchCV() 함수 호출, 모델 생성
grid_dt = GridSearchCV(estimator=base_dt,
                       param_grid=params,
                       scoring='accuracy',
                       cv=10)

# 학습 및 평가
grid_dt.fit(X_train, y_train)

# %%
### 최적의 하이퍼파라미터 확인
print(grid_dt.best_params_)

# %%
### best 모델 완성
best_dt = DecisionTreeClassifier(max_depth=3, random_state=0)

# %% [markdown]
# ### 모델 학습

# %%
best_dt.fit(X_train, y_train)

# %% [markdown]
# ### 학습 결과 시각화

# %%
# 필요한 라이브러리 / 함수 import
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 이미지 크기 재설정
plt.figure(figsize=(12,8))

# 시각화
plot_tree(best_dt, feature_names=X_train.columns, filled=True)
plt.show()

# %% [markdown]
# ### 평가용 데이터를 이용한 예측

# %%
pred_test = best_dt.predict(X_test)

# %% [markdown]
# ### 모델 평가

# %%
# 필요한 함수 import
from sklearn.metrics import accuracy_score

# 평가용 데이터에 대한 정확도 평가
accuracy = accuracy_score(y_test, pred_test)

# 결과 확인하기
print(f'평가용 데이터를 이용한 정확도 = {accuracy}')

# %% [markdown]
# # RandomForest 분류 모델을 이용한 타이타닉 데이터 분류

# %% [markdown]
# ## 모델 생성

# %%
### GridSearchCV 함수 사용

# 필요한 함수 import
from sklearn.ensemble import RandomForestClassifier

# 최적화 할 기본 모델 생성
base_rf = RandomForestClassifier(random_state=0)

# 튜닝할 매개 변수 설정
params = {'n_estimators':[100,200,300,400],
          'max_depth':[3,4,5,6,7]}

# GridSearchCV 함수 호출, 모델 생성
grid_rf = GridSearchCV(estimator=base_rf,
                       param_grid=params,
                       scoring='accuracy',
                       cv=10)

# 학습 및 평가
grid_rf.fit(X_train, y_train)

# %%
### 최적의 하이퍼파라미터 조합 확인
print(grid_rf.best_params_)

# %%
### best 모델 생성
best_rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0)

# %% [markdown]
# ## 모델 학습

# %%
best_rf.fit(X_train, y_train)

# %% [markdown]
# ## 평가용 데이터를 이용한 예측

# %%
pred_test = best_rf.predict(X_test)

# %% [markdown]
# ## 모델 평가

# %%
# 필요한 함수 import
from sklearn.metrics import accuracy_score

# 평가영 데이터에 대한 정확도 평가
accuracy = accuracy_score(y_test, pred_test)

# 결과 확인하기
print(f'평가용 데이터에 대한 정확도 = {accuracy}')

# %% [markdown]
# # LightGBM 분류 모델을 이용한 타이타닉 데이터 분류

# %% [markdown]
# ## 모델 생성

# %%
## GridsearchCV 함수 사용

# 필요한 함수 import
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

# 최적화 할 기본 모델 생성
base_lgbm = LGBMClassifier(random_state=0)

# 튜닝할 매개 변수 설정
params = {'n_estimators':[50,100,150,200],
          'learning_rate':[0.05,0.1,0.3],
          'max_depth':[3,4,5]}

# GridSearchCV 함수 호출, 모델 생성
grid_lgbm = GridSearchCV(estimator=base_lgbm,
                         param_grid=params,
                         scoring='accuracy',
                         cv=10)

# 모델 학습 및 평가
grid_lgbm.fit(X_train, y_train)

# %%
### 최적의 하이퍼파라미터 조합 확인
print(grid_lgbm.best_params_)

# %%
### best 모델 생성
best_lgbm = LGBMClassifier(learning_rate=0.05,
                           max_depth=3,
                           n_estimators=50)

# %% [markdown]
# ## 모델 학습

# %%
best_lgbm.fit(X_train, y_train)

# %% [markdown]
# ## 평가용 데이터를 이용한 예측

# %%
pred_test = best_lgbm.predict(X_test)

# %% [markdown]
# ## 모델 평가

# %%
# 필요한 함수 import
from sklearn.metrics import accuracy_score

# 평가용 데이터에 대한 정확도 평가
accuracy = accuracy_score(y_test, pred_test)

# 결과 확인하기
print(f'평가용 데이터에 대한 평가 결과 : {accuracy}')

# %% [markdown]
# # XGBoost 모델을 이용한 타이타닉 데이터 분류

# %% [markdown]
# ## 모델 생성

# %%
## GridsearchCV 함수 사용

# 필요한 함수 import
from xgboost import XGBClassifier

# 최적화 할 기본 모델 생성
base_xgb = XGBClassifier(random_state=0)

# 튜닝할 매개 변수 설정
params = {'n_estimators':[50,100,150,200],
          'learning_rate':[0.05,0.1,0.3],
          'max_depth':[3,4,5]}

# GridSearchCV 함수 호출, 모델 생성
grid_xgb = GridSearchCV(estimator=base_xgb,
                         param_grid=params,
                         scoring='accuracy',
                         cv=10)

# 학습 및 성능 평가
grid_xgb.fit(X_train, y_train)

# %%
### 최적의 하이퍼파라미터 조합 확인
print(grid_xgb.best_params_)

# %%
### best 모델 생성
best_xgb = XGBClassifier(learning_rate=0.05,
                         n_estimators=50,
                         max_depth=3,
                         random_state=0)

# %% [markdown]
# ## 모델 학습

# %%
best_xgb.fit(X_train, y_train)

# %% [markdown]
# ## 평가용 데이터를 이용한 예측

# %%
pred_test = best_xgb.predict(X_test)

# %% [markdown]
# ## 모델 평가

# %%
# 평가용 데이터에 대한 정확도 평가
accuracy = accuracy_score(y_test, pred_test)

# 결과 확인하기
print(f'평가용 데이터에 대한 평가 결과 : {accuracy}')


