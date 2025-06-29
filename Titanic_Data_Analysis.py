# %% [markdown]
# # 타이타닉 데이터 분석

# %% [markdown]
# ## 함수 정리

# %% [markdown]
# ### value_counts(normalize=True)

# %%
### value_counts(normalize=True) : 항목별 비율 추출

# 필요한 라이브러리 import
import seaborn as sns
import pandas as pd

# 데이터 생성
df_tips = sns.load_dataset('tips')

# 결과 확인하기
print(df_tips)

print('-'*80)

# day 컬럼 --> 항목별 빈도수 추출 --> value_counts()
day_counts = df_tips.loc[:,'day'].value_counts()
print(f'day컬럼의 항목별 빈도수 : \n{day_counts}')

print('-'*80)

# day 컬럼 --> 항목별 비율 추출 --> value_counts(normalize=True)
day_ratio = df_tips.loc[:,'day'].value_counts(normalize=True)
print(f'day컬럼의 항목별 비율 : \n{day_ratio}')

# %% [markdown]
# ### apply() 사용

# %%
### 사용자 정의 함수 --> 데이터프레임 / 시리즈 데이터에 적용

'''
1. df,apply(func)
2.df.loc[:,'col'].apply(func)
'''

# DataFrame 자료형 생성
data = {'A':[4,1,5],
        'B':[9,4,6]}

df = pd.DataFrame(data=data)
print(df)

print('-'*80)

# plus_one 함수 정의 --> 사용자 정의 함수
def plus_one(x):
    return x+1

# pandas에서 제공하지 않는 함수 --> df.func() --> 불가능 (에러 발생)
# df.apply(func) --> 가능
df = df.apply(plus_one)
print(df)

print('-'*80)

# 특정 컬럼 --> df.loc[:,'col'].apply(func)
df.loc[:,'A'] = df.loc[:,'A'].apply(plus_one)
print(df)

# %% [markdown]
# ### apply(lambda)

# %%
# 사용 데이터 --> df
print(df)

print('-'*80)

# DataFrame --> apply(lambda) 적용
df = df.apply(lambda x:x+1)
print(df)

# %% [markdown]
# ## 데이터 불러오기

# %%
### 필요한 라이브러리 import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
### 데이터 불러오기

# 파일 경로 설정
train_path = '/content/drive/MyDrive/데이터 분석/titanic_train.csv'
test_path = '/content/drive/MyDrive/데이터 분석/titanic_test.csv'

# pd.read_csv() --> DataFrame 자료형 생성
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 결과 확인하기
print(df_train)
print('-'*80)
print(df_test)

# %%
### 학습용 데이터와 평가용 데이터의 컬럼 비교
print(df_train.columns == df_test.columns)

# %%
### 데이터 병합하기
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# ignore_index --> pandas version 문제로 작동(X) --> df.reset_index(drop=True, inplace=True) 사용

# 결과 확인하기
print(df)

# %% [markdown]
# ### 데이터 전처리

# %% [markdown]
# #### 누락 데이터 처리

# %%
### 각 컬럼별 누락 데이터 확인

# isnull() --> 각 컬럼별 누락 여부 확인
df_bool = df.isnull()
print(df_bool)

print('-'*80)

# isnull().sum() --> 각 컬럼별 누락 데이터의 수 확인
num_nulls = df_bool.sum()
print(num_nulls)

# %% [markdown]
# ##### Cabin 컬럼

# %%
### Cabin 컬럼 제거 --> df.drop(columns=['col'])
cleaned_df = df.drop(columns=['Cabin'])

# 결과 확인하기
print(cleaned_df)

# %% [markdown]
# ##### Age 컬럼

# %%
### Age 컬럼 값의 분포 확인 --> 요약 통계량 --> describe()
print(cleaned_df.loc[:,'Age'].describe())

# %%
### Age 컬럼 --> 최빈값 추출 --> 변수(series).mode() / 변수(series).value_counts()
counts = cleaned_df.loc[:,'Age'].value_counts()
print(f'Age 컬럼의 항목별 빈도수 : \n{counts.iloc[:10]}')

# %%
### Age 컬럼 --> 누락 데이터 --> 중간값으로 대체
age_median = cleaned_df.loc[:,'Age'].median()
cleaned_df.loc[:,'Age'].fillna(age_median, inplace=True)

# 중간값으로 대체 후 누락 데이터의 수 확인
print(cleaned_df.loc[:,'Age'].isnull().sum())


# %% [markdown]
# ###### Binning

# %%
### Age 컬럼 --> 8단계 구간화

# pd.cut() 함수 사용 --> 매개 변수 설정
x = cleaned_df.loc[:,'Age']
bins = 8
labels = [0, 1, 2, 3, 4, 5, 6, 7]

# 데이터 변환
cleaned_df.loc[:,'Age'] = pd.cut(x=x, bins=bins, labels=labels)

# 결과 확인하기
print(f'전체 데이터 확인 : \n{cleaned_df}')

# %% [markdown]
# ##### Embarked 컬럼

# %%
### 승선 항구의 항구별 빈도수 / 비율 추출

# 빈도수 추출
counts = cleaned_df.loc[:,'Embarked'].value_counts()
print(f'승선 항구의 항구별 빈도수 : \n{counts}')

print('-'*80)

# 비율 추출
ratio = cleaned_df.loc[:,'Embarked'].value_counts(normalize=True)
print(f'승선 항구의 항구별 비율 : \n{ratio}')

# %%
### 누락 데이터 처리 --> 최빈값으로 채우기
cleaned_df.loc[:,'Embarked'].fillna('S', inplace=True)

# 결과 확인하기
print(cleaned_df.loc[:,'Embarked'].isnull().sum())

# %% [markdown]
# ##### Fare 컬럼

# %%
### Fare 컬럼의 항복별 빈도수 확인
print(cleaned_df.loc[:,'Fare'].value_counts(normalize=True))

# %%
### Fare 컬럼에서 누락 데이터 확인 --> 불리언 배열 생성 --> loc[불리언배열,:]

# Fare 컬럼 --> 누락 여부 --> isnull() --> 불리언 배열 생성
condition = cleaned_df.loc[:,'Fare'].isnull()
print(condition)

print('-'*80)

# 불리언 배열 --> loc[condition, 열]
null = cleaned_df.loc[condition,:]
print(f'Fare 컬럼에서 누락된 데이터 확인 :\n{null}')

# %%
### 3등석 & 승선 항구 S & SibSp 0 % Parch 0 --> 요금의 종류 확인
condition1 = cleaned_df.loc[:,'Pclass'] == 3
condition2 = cleaned_df.loc[:, 'Embarked'] == 'S'
condition3 = cleaned_df.loc[:, 'SibSp'] == 0
condition4 = cleaned_df.loc[:, 'Parch']  == 0
condition = condition1 & condition2 & condition3 & condition4

# 행의 조건을 만족하는 데이터 추출
data = cleaned_df.loc[condition, 'Fare']
print(f'4가지 조건을 모두 만족하는 승객의 요금 : \n{data}')
print('-'*80)
print(f'4가지 조건을 모두 만족하는 승객 요금의 요약 통계량 : \n{data.describe()}')
print('-'*80)
print(f'4가지 조건을 모두 만족하는 승객의 요금의 항목별 빈도수 : \n{data.value_counts()}')

# %%
### 4가지 조건을 모두 만족하는 요금 데이터 --> boxplot --> 이상치 확인
data.plot(kind='box')
plt.show()

# %%
### 4가지 조건을 모두 만족하는 요금 데이터 --> 이상치의 조건 확인

# 1사분위 값 / 3 사분위 값
q1 = data.quantile(q=0.25)
q3 = data.quantile(q=0.75)

# iqr 추출
iqr = q3 - q1

# 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'정상 범위의 최소값 : \n{min}')

print('-'*80)

# 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'정상 범위의 최대값 : \n{max}')

print('-'*80)

# 이상치의 인덱스 추출
condition = (data<min) | (data>max)
print(condition)
outlier_index = data.loc[condition].index
print(f'4가지 조건을 모두 만족하는 요금 데이터에서 이상치의 인덱스는 : \n{outlier_index}')

print('-'*80)

# 이상치 제거
cleaned_data = data.drop(index=outlier_index)
print(f'첫번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_data}')

print('-'*80)

# 첫번째 이상치 제거 후 남은 데이터 --> boxplot --> 이상치 확인
cleaned_data.plot(kind='box')
plt.show()

# %%
# 두번째 1사분위 값 / 3 사분위 값
q1 = cleaned_data.quantile(q=0.25)
q3 = cleaned_data.quantile(q=0.75)

# iqr 추출
iqr = q3 - q1

# 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'두번째 정상 범위의 최소값 : \n{min}')

print('-'*80)

# 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'두번째 정상 범위의 최대값 : \n{max}')

print('-'*80)

# 이상치의 인덱스 추출
condition = (cleaned_data<min) | (cleaned_data>max)
print(condition)
outlier_index = cleaned_data.loc[condition].index
print(f'4가지 조건을 모두 만족하는 요금 데이터에서 두번째 이상치의 인덱스는 : \n{outlier_index}')

print('-'*80)

# 두번째 이상치 제거
cleaned_data2 = cleaned_data.drop(index=outlier_index)
print(f'첫번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_data2}')

print('-'*80)

# 두번째 이상치 제거 후 남은 데이터 --> boxplot --> 이상치 확인
cleaned_data2.plot(kind='box')
plt.show()

# %%
### 이상치 제거 후 4가지 조건을 모두 만족하는 요금 데이터 --> 요약 통계량 추출
stats = cleaned_data2.describe()
print(stats)

# %%
### 이상치 제거 후 4가지 조건을 모두 만족하는 요금 데이터 --> 요금별 빈도수 추출
counts = cleaned_data2.value_counts()
print(counts)

# %%
### 누락 데이터 대체
price = 8.0
cleaned_df.loc[:,'Fare'].fillna(price, inplace=True)

# 결과 확인하기
print(cleaned_df.iloc[1043,:])

# %%
### 이상치 처리 전 생존자 / 사망자 수 분포 확인

# 빈도수 확인
counts = cleaned_df.loc[:,'Survived'].value_counts()
print(f'이상치 처리 전 생존자 / 사망자 빈도수 확인 : \n{counts}')

print('-'*80)

# 비율 확인
ratio = cleaned_df.loc[:,'Survived'].value_counts(normalize=True)
print(f'이상치 처리 전 생존자 / 사망자 비율 확인 : \n{ratio}')

# %% [markdown]
# #### 이상치 처리

# %%
### Fare 컬럼 --> boxplot --> 이상치 확인
cleaned_df.loc[:,'Fare'].plot(kind='box')
plt.show()

# %%
### Fare 컬럼 --> 첫번째 이상치 확인 및 제거

# 1사분위 값 / 3 사분위 값
q1 = cleaned_df.loc[:,'Fare'].quantile(q=0.25)
q3 = cleaned_df.loc[:,'Fare'].quantile(q=0.75)

# 첫번째 iqr 추출
iqr = q3 - q1

# 첫번째 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'첫번째 정상 범위의 최대값 : \n{min}')

print('-'*80)

# 첫번째 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'첫번째 정상 범위의 최대값 : \n{max}')

print('-'*80)

# 첫번째 이상치 인덱스 추출
condition = (cleaned_df.loc[:,'Fare'] < min) | (cleaned_df.loc[:,'Fare'] > max)
print(f'첫번째 이상치 데이터의 불리언 배열 생성 : \n{condition}')
print('-'*80)
outlier_index = cleaned_df.loc[:,'Fare'].loc[condition].index
print(f'첫번째 이상치 데이터의 인덱스 : \n{outlier_index}')

print('-'*80)

# 첫번째 이상치 제거
cleaned_df2 = cleaned_df.drop(index=outlier_index)
print(f'첫번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_df2}')

print('-'*80)

# 첫번째 이상치 제거 후 Fare 컬럼 --> boxplot --> 이상치 확인
cleaned_df2.loc[:,'Fare'].plot(kind='box')
plt.show()

# %%
### Fare 컬럼 --> 두번째 이상치 확인 및 제거

# 1사분위 값 / 3 사분위 값
q1 = cleaned_df2.loc[:,'Fare'].quantile(q=0.25)
q3 = cleaned_df2.loc[:,'Fare'].quantile(q=0.75)

# 두번째 iqr 추출
iqr = q3 - q1

# 두번째 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'두번째 정상 범위의 최대값 : \n{min}')

print('-'*80)

# 두번째 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'두번째 정상 범위의 최대값 : \n{max}')

print('-'*80)

# 두번째 이상치 인덱스 추출
condition = (cleaned_df2.loc[:,'Fare'] < min) | (cleaned_df2.loc[:,'Fare'] > max)
print(f'두번째 이상치 데이터의 불리언 배열 생성 : \n{condition}')
print('-'*80)
outlier_index = cleaned_df2.loc[:,'Fare'].loc[condition].index
print(f'두번째 이상치 데이터의 인덱스 : \n{outlier_index}')

print('-'*80)

# 두번째 이상치 제거
cleaned_df3 = cleaned_df2.drop(index=outlier_index)
print(f'두번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_df3}')

print('-'*80)

# 두번째 이상치 제거 후 Fare 컬럼 --> boxplot --> 이상치 확인
cleaned_df3.loc[:,'Fare'].plot(kind='box')
plt.show()

# %%
### Fare 컬럼 --> 세번째 이상치 확인 및 제거

# 1사분위 값 / 3 사분위 값
q1 = cleaned_df3.loc[:,'Fare'].quantile(q=0.25)
q3 = cleaned_df3.loc[:,'Fare'].quantile(q=0.75)

# 세번째 iqr 추출
iqr = q3 - q1

# 세번째 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'세번째 정상 범위의 최대값 : \n{min}')

print('-'*80)

# 세번째 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'세번째 정상 범위의 최대값 : \n{max}')

print('-'*80)

# 세번째 이상치 인덱스 추출
condition = (cleaned_df3.loc[:,'Fare'] < min) | (cleaned_df3.loc[:,'Fare'] > max)
print(f'세번째 이상치 데이터의 불리언 배열 생성 : \n{condition}')
print('-'*80)
outlier_index = cleaned_df3.loc[:,'Fare'].loc[condition].index
print(f'세번째 이상치 데이터의 인덱스 : \n{outlier_index}')

print('-'*80)

# 세번째 이상치 제거
cleaned_df4 = cleaned_df3.drop(index=outlier_index)
print(f'세번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_df4}')

print('-'*80)

# 세번째 이상치 제거 후 Fare 컬럼 --> boxplot --> 이상치 확인
cleaned_df4.loc[:,'Fare'].plot(kind='box')
plt.show()

# %%
### Fare 컬럼 --> 네번째 이상치 확인 및 제거

# 1사분위 값 / 3 사분위 값
q1 = cleaned_df4.loc[:,'Fare'].quantile(q=0.25)
q3 = cleaned_df4.loc[:,'Fare'].quantile(q=0.75)

# 네번째 iqr 추출
iqr = q3 - q1

# 네번째 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'네번째 정상 범위의 최대값 : \n{min}')

print('-'*80)

# 네번째 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'네번째 정상 범위의 최대값 : \n{max}')

print('-'*80)

# 네번째 이상치 인덱스 추출
condition = (cleaned_df4.loc[:,'Fare'] < min) | (cleaned_df4.loc[:,'Fare'] > max)
print(f'네번째 이상치 데이터의 불리언 배열 생성 : \n{condition}')
print('-'*80)
outlier_index = cleaned_df4.loc[:,'Fare'].loc[condition].index
print(f'네번째 이상치 데이터의 인덱스 : \n{outlier_index}')

print('-'*80)

# 네번째 이상치 제거
cleaned_df5 = cleaned_df4.drop(index=outlier_index)
print(f'네번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_df5}')

print('-'*80)

# 네번째 이상치 제거 후 Fare 컬럼 --> boxplot --> 이상치 확인
cleaned_df5.loc[:,'Fare'].plot(kind='box')
plt.show()

# %%
### Fare 컬럼 --> 5번째 이상치 확인 및 제거

# 1사분위 값 / 3 사분위 값
q1 = cleaned_df5.loc[:,'Fare'].quantile(q=0.25)
q3 = cleaned_df5.loc[:,'Fare'].quantile(q=0.75)

# 5번째 iqr 추출
iqr = q3 - q1

# 5번째 정상 범위의 최소
min = q1 - (1.5*iqr)
print(f'5번째 정상 범위의 최대값 : \n{min}')

print('-'*80)

# 5번째 정상 범위의 최대
max = q3 + (1.5*iqr)
print(f'5번째 정상 범위의 최대값 : \n{max}')

print('-'*80)

# 5번째 이상치 인덱스 추출
condition = (cleaned_df5.loc[:,'Fare'] < min) | (cleaned_df5.loc[:,'Fare'] > max)
print(f'5번째 이상치 데이터의 불리언 배열 생성 : \n{condition}')
print('-'*80)
outlier_index = cleaned_df5.loc[:,'Fare'].loc[condition].index
print(f'5번째 이상치 데이터의 인덱스 : \n{outlier_index}')

print('-'*80)

# 5번째 이상치 제거
cleaned_df6 = cleaned_df5.drop(index=outlier_index)
print(f'5번째 이상치 제거 후 남은 데이터 확인 : \n{cleaned_df6}')

print('-'*80)

# 5번째 이상치 제거 후 Fare 컬럼 --> boxplot --> 이상치 확인
cleaned_df6.loc[:,'Fare'].plot(kind='box')
plt.show()

# %%
### 이상치 제거 후 남은 데이터 --> 인덱스 재설정
cleaned_df6.reset_index(drop=True, inplace=True)

# 결과 확인하기
print(cleaned_df6)

# %% [markdown]
# ### Feature Engineering

# %% [markdown]
# #### 가족 인원수 컬럼 추가

# %%
### SibSp 컬럼 값과 Parch 컬럼 값을 합 --> 가족 인원수 컬럼 생성 --> 파생 변수

# SibSp 컬럼 값과 Parch 컬럼 값 --> 합
data = cleaned_df6.loc[:,'SibSp'] + cleaned_df6.loc[:,'Parch']
print(data)

print('-'*80)

# 새로운 컬럼 추가
cleaned_df6.loc[:,'num_family'] = data

# 결과 확인하기
print(cleaned_df)

# %%
### Fare 컬럼 --> 이상치 제거(X)
### SibSp 컬럼 값과 Parch 컬럼 값을 합 --> 가족 인원수 컬럼 생성 --> 파생 변수

# SibSp 컬럼 값과 Parch 컬럼 값 --> 합
data = cleaned_df.loc[:, 'SibSp'] + cleaned_df.loc[:, 'Parch']
print(data)

print('-'*80)

# 새로운 컬럼 추가
cleaned_df.loc[:, 'num_family'] = data

# 결과 확인하기
print(cleaned_df)

# %% [markdown]
# #### Name 컬럼 세분화

# %%
### Name 컬럼 확인하기

'''
### 이름의 구성 : 성, 호칭, 이름
'''

# Name 컬럼의 값 확인
print(cleaned_df6.loc[:,'Name'])

# %%
### '호칭' 추출 --> split() 함수

'''
### split() 함수
(예시)
name = 'Braund, Mr. Owen Harris'
name.split(',')[1].split('.')[0]
'''

# test name 생성
test_name = cleaned_df6.loc[:,'Name'].iloc[0]
print(test_name)

print('-'*80)

# 첫번째 분할 --> 기준 : ','
result1 = test_name.split(',')
print(f'첫번째 분할의 결과 : {result1}')

print('-'*80)

# '성'을 제외한 나머지 부분 추출 --> 인덱싱 --> test_name.split(',')[1]
print(test_name.split(',')[1])

# 나머지 부분 --> '호칭' 추출
result2 = test_name.split(',')[1].split('.')
print(f'두번째 분할의 결과 : {result2}')
print('-'*80)
title = test_name.split(',')[1].split('.')[0]
print(f'추출한 호칭 확인 : {title}')

# %%
### 문자열의 앞/뒤 공백 제거 --> strip() 함수

# 추출된 호칭에 strip 함수 적용
result = title.strip()
print(result)

# %%
### Name 컬럼 --> 호칭 추출 함수 정의
def extract_title(name):
    # ','로 split & 두번째 성분 원소 선택 & '.'로 split & 첫번째 성분 원소 선택 & strip 함수
    title = name.split(',')[1].split('.')[0].strip()
    return title

# %%
### Name 컬럼으로부터 호칭을 추출하여 새로운 컬럼 생성 --> 파생 변수

# Name 컬럼 --> apply(extract_title) 적용
data = cleaned_df6.loc[:,'Name'].apply(extract_title)
print(f'호칭 추출의 결과 확인 : \n{data}')

# 새로운 컬럼 생성
cleaned_df6.loc[:,'title'] = data

# 결과 확인하기
print(cleaned_df6)

# %%
### Fare 컬럼 --> 이상치(X)
### Name 컬럼으로부터 호칭을 추출하여 새로운 컬럼 생성 --> 파생 변수

# Name 컬럼 --> apply(extract_title) 적용
data = cleaned_df.loc[:,'Name'].apply(extract_title)
print(f'호칭 추출의 결과 확인 : \n{data}')

# 새로운 컬럼 생성
cleaned_df.loc[:,'title'] = data

# 결과 확인하기
print(cleaned_df)

# %%
### title 컬럼의 항목별 빈도수 / 비율 확인

'''
# 호칭 정리
  1) 'Rev', 'Col', 'Major', 'Dr', 'Capt', 'Sir' : 직위 표현
  2) 'Ms', 'Mme', 'Mrs', 'Dona' : 여성 표현
  3) 'Miss', 'Mlle', 'Lady' : 젊은 여성 표현
  4) 'Mr', 'Don' : 남성 표현
  5) 'Master' : 주로 청소년 이하 결혼하지 않은 남성
  6) 'Jonkheer', 'the Countess' : 귀족 표현
'''

# 빈도수 추출
counts = cleaned_df6.loc[:,'title'].value_counts()
print(f'title 컬럼의 항목별 빈도수 : \n{counts}')

print('-'*80)

# 비율 추출
ratio = cleaned_df6.loc[:,'title'].value_counts(normalize=True)
print(f'title 컬럼의 항목별 비율 : \n{ratio}')


# %%
### Fare 컬럼 --> 이상치(X)
### title 컬럼의 항목별 빈도수 / 비율 확인

'''
# 호칭 정리
  1) 'Rev', 'Col', 'Major', 'Dr', 'Capt', 'Sir' : 직위 표현
  2) 'Ms', 'Mme', 'Mrs', 'Dona' : 여성 표현
  3) 'Miss', 'Mlle', 'Lady' : 젊은 여성 표현
  4) 'Mr', 'Don' : 남성 표현
  5) 'Master' : 주로 청소년 이하 결혼하지 않은 남성
  6) 'Jonkheer', 'the Countess' : 귀족 표현
'''

# 빈도수 추출
counts = cleaned_df.loc[:,'title'].value_counts()
print(f'title 컬럼의 항목별 빈도수 : \n{counts}')

print('-'*80)

# 비율 추출
ratio = cleaned_df.loc[:,'title'].value_counts(normalize=True)
print(f'title 컬럼의 항목별 비율 : \n{ratio}')


# %% [markdown]
# #### title 컬럼 정리

# %%
### 목표 : title 컬럼에서 Mr 또는 Miss 또는 Mrs 또는 Master가 아닌 값 --> others로 변환

# 비교 연산자 --> 불리언 배열 생성
condition1 = (cleaned_df6.loc[:,'title'] != 'Mr')
condition2 = (cleaned_df6.loc[:,'title'] != 'Miss')
condition3 = (cleaned_df6.loc[:,'title'] != 'Mrs')
condition4 = (cleaned_df6.loc[:,'title'] != 'Master')

# 논리 연산자 --> 교집합
condition = condition1 & condition2 & condition3 & condition4

# df.loc[condition,'title'] = 'others' 적용
cleaned_df6.loc[condition,'title'] = 'others'

# 결과 확인하기 --< 'title' 컬럼 --> 항목별 빈도수 / 비율 추출
counts = cleaned_df6.loc[:,'title'].value_counts()
print(f'title 컬럼을 정리한 결과 학인(빈도수) : \n{counts}')
print('-'*80)
ratio = cleaned_df6.loc[:,'title'].value_counts(normalize=True)
print(f'title 컬럼을 정리한 결과 학인(비율) : \n{ratio}')

# %%
### Fare 컬럼 --> 이상치(X)
### 목표 : title 컬럼에서 Mr 또는 Miss 또는 Mrs 또는 Master가 아닌 값 --> others로 변환

# 비교 연산자 --> 불리언 배열 생성
condition1 = (cleaned_df.loc[:,'title'] != 'Mr')
condition2 = (cleaned_df.loc[:,'title'] != 'Miss')
condition3 = (cleaned_df.loc[:,'title'] != 'Mrs')
condition4 = (cleaned_df.loc[:,'title'] != 'Master')

# 논리 연산자 --> 교집합
condition = condition1 & condition2 & condition3 & condition4

# df.loc[condition,'title'] = 'others' 적용
cleaned_df.loc[condition,'title'] = 'others'

# 결과 확인하기 --< 'title' 컬럼 --> 항목별 빈도수 / 비율 추출
counts = cleaned_df.loc[:,'title'].value_counts()
print(f'title 컬럼을 정리한 결과 학인(빈도수) : \n{counts}')
print('-'*80)
ratio = cleaned_df.loc[:,'title'].value_counts(normalize=True)
print(f'title 컬럼을 정리한 결과 학인(비율) : \n{ratio}')

# %%
### 보충 : title 컬럼에서 Mr 또는 Miss 또는 Mrs 또는 Master가 아닌 값 --> others로 변환 (2)

# 데이터프레임 복사
df_test = cleaned_df6.iloc[:,:]
# print(df_test)

# replace() 함수 사용 --> 호칭 변환
df_test.loc[:,'title'].replace({'Rev':'others',
                                'Dr':'others',
                                'Col':'others',
                                'Ms':'others',
                                'Major':'others',
                                'Don':'others',
                                'Lady':'others',
                                'Jonkheer':'others'})

# 결과 확인하기
print(df_test.loc[:,'title'].value_counts())

# %% [markdown]
# ### 불필요한 컬럼 삭제

# %%
### 삭제의 대상 : PassengerId, Name, Ticket
df = cleaned_df6.drop(columns=['PassengerId', 'Name', 'Ticket'])

# 결과 학인하기
print(df)


# %%
### Fare 컬럼 --> 이상치 제거 (X)
### 삭제의 대상 : PassengerId, Name, Ticket
df = cleaned_df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# 결과 학인하기
print(df)

# %% [markdown]
# ### 결과 저장하기

# %%
### 전처리한 결과를 csv 파일로 저장하기

# 이상치 제거 버전 저장 경로 설정하기
file_path1 = '/content/drive/MyDrive/데이터 분석/titanic_preprocessed.csv'

# 이상치 제거를 하지 않은 버전 저장 경로 설정하기
file_path2 = '/content/drive/MyDrive/데이터 분석/titanic_preprocessed2.csv'

# df.to_csv() 함수 사용
df.to_csv(file_path2, index=False)

# %%
### 저장된 결과 불러오기

# 이상치 제거 버전 파일 경로 설정하기
file_path1 = '/content/drive/MyDrive/데이터 분석/titanic_preprocessed.csv'

# 이상치 제거를 하지 않은 버전 저장 경로 설정하기
file_path2 = '/content/drive/MyDrive/데이터 분석/titanic_preprocessed2.csv'

# pd.read_csv() 함수 사용
df = pd.read_csv(file_path2)

# 결과 확인하기
print(df)

# %% [markdown]
# ## 데이터 탐색

# %%
### 이미지의 크기 재설정
plt.rcParams['figure.figsize'] = [4, 3]

# %% [markdown]
# ### Survived 컬럼

# %%
### 사망자 수 / 생존자 수 분포 시각화

# 빈도수 추출
counts = df.loc[:,'Survived'].value_counts()
print(f'이상치 제거 후 사망자 / 생존자 빈도수 확인 : \n{counts}')

print('-'*80)

# 비율 추출
ratio = df.loc[:,'Survived'].value_counts(normalize=True)
print(f'이상치 제거 후 사망자 / 생존자 비율 확인 : \n{ratio}')


# %%
### 사망자 / 생존자 분포 시각화

# 범주형 데이터의 항목별 빈도수 시각화 --> 막대 그래프
sns.countplot(data=df, x='Survived')
plt.show()

# %% [markdown]
# ### 생존 여부와 승객의 등급 분석

# %%
### 탑승객의 등급 분석

# 시각화 --> 범주형 데이터 --> 항목별 빈도수 --> sns.countplot(data, x)
sns.countplot(data=df, x='Pclass')
plt.show()

print('-'*80)

# 통계 데이터 - 승객의 등급별 빈도수 추출
counts = df.loc[:,'Pclass'].value_counts()
print(f'승객의 등급별 빈도수 : \n{counts}')

print('-'*80)

# 통계 데이터 - 승객의 등급별 비율 추출
ratio = df.loc[:,'Pclass'].value_counts(normalize=True)
print(f'승객의 등급별 비율 : \n{ratio}')

# %%
### Survived 컬럼과 Pclass 컬럼의 관계 분석
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.show()

# %%
### 승객의 등급별 사망자 / 생존자 수와 비율 분석

'''
전체 탑승객 중 사망자 / 생존자 비율 --> 67.8 : 32.2
'''

## 1등석의 사망자 / 생존자 수와 비율 분석

# 1등석 --> 사망자 / 생존자 수 분석
# 비교 연산자 --> 불리언 배열 생성 --> loc[불리언 배열, 'Survived']
condition1 = (df.loc[:,'Pclass']==1)
pclass_survived = df.loc[condition1,'Survived'].value_counts()
print(f'1등석 승객중 사망자 수 / 생존자 수 분석 : \n{pclass_survived}')

print('-'*80)

# 1등석 --> 사망자 / 생존자 비율 분석
# 비교 연산자 --> 불리언 배열 생성 --> loc[불리언 배열, 'Survived']
condition1 = (df.loc[:,'Pclass']==1)
pclass_survived_ratio = df.loc[condition1,'Survived'].value_counts(normalize=True)
print(f'1등석 승객중 사망자 수 / 생존자 비율 분석 : \n{pclass_survived_ratio}')

print('-'*80)

# 2등석 --> 사망자 / 생존자 수 분석
# 비교 연산자 --> 불리언 배열 생성 --> loc[불리언 배열, 'Survived']
condition1 = (df.loc[:,'Pclass']==2)
pclass2_survived = df.loc[condition1,'Survived'].value_counts()
print(f'2등석 승객중 사망자 수 / 생존자 수 분석 : \n{pclass2_survived}')

print('-'*80)

# 2등석 --> 사망자 / 생존자 비율 분석
# 비교 연산자 --> 불리언 배열 생성 --> loc[불리언 배열, 'Survived']
condition1 = (df.loc[:,'Pclass']==2)
pclass2_survived_ratio = df.loc[condition1,'Survived'].value_counts(normalize=True)
print(f'2등석 승객중 사망자 수 / 생존자 비율 분석 : \n{pclass2_survived_ratio}')

print('-'*80)

# 3등석 --> 사망자 / 생존자 수 분석
# 비교 연산자 --> 불리언 배열 생성 --> loc[불리언 배열, 'Survived']
condition1 = (df.loc[:,'Pclass']==3)
pclass3_survived = df.loc[condition1,'Survived'].value_counts()
print(f'3등석 승객중 사망자 수 / 생존자 수 분석 : \n{pclass3_survived}')

print('-'*80)

# 3등석 --> 사망자 / 생존자 비율 분석
# 비교 연산자 --> 불리언 배열 생성 --> loc[불리언 배열, 'Survived']
condition1 = (df.loc[:,'Pclass']==3)
pclass3_survived_ratio = df.loc[condition1,'Survived'].value_counts(normalize=True)
print(f'3등석 승객중 사망자 수 / 생존자 비율 분석 : \n{pclass3_survived_ratio}')

# %% [markdown]
# ### 생존 여부와 성별의 관계 분석

# %%
### 전체 탑승객 --> 남여의 빈도수 분석

# 시각화
sns.countplot(data=df, x='Gender')
plt.show()

print('-'*80)

# 통계 분석 --> 성별 빈도수 추출
counts = df.loc[:,'Gender'].value_counts()
print(f'Gender 컬럼의 남여 빈도수 : \n{counts}')

print('-'*80)

# 통계 분석 --> 성별 비율 추출
ratio = df.loc[:,'Gender'].value_counts(normalize=True)
print(f'Gender 컬럼의 남여 비율 : \n{ratio}')

# %%
### Survived 컬럼과 Gender 컬럼의 관계 (1)
sns.countplot(data=df, x='Gender', hue='Survived')
plt.show()

# %%
### 성별 생존자 / 사망자의 수와 비율 분석


## 대상 : 여성
# 여성 --> 사망자 / 생존자수 추출
condition1 = (df.loc[:,'Gender']=='female')
female_survived_counts = df.loc[condition1,'Survived'].value_counts()
print(f'여성 중 사망자 / 생존자 수 : \n{female_survived_counts}')

print('-'*80)

# 여성 --> 사망자 / 생존자 비율 추출
condition1 = (df.loc[:,'Gender']=='female')
female_survived_ratio = df.loc[condition1,'Survived'].value_counts(normalize=True)
print(f'여성 중 사망자 / 생존자 비율 : \n{female_survived_ratio}')

print('-'*80)

## 대상 : 남성
# 남성 --> 사망자 / 생존자수 추출
condition2 = (df.loc[:,'Gender']=='male')
male_survived_counts = df.loc[condition2,'Survived'].value_counts()
print(f'남성 중 사망자 / 생존자 수 : \n{male_survived_counts}')

print('-'*80)

# 남성 --> 사망자 / 생존자 비율 추출
condition2 = (df.loc[:,'Gender']=='male')
male_survived_ratio = df.loc[condition2,'Survived'].value_counts(normalize=True)
print(f'남성 중 사망자 / 생존자 비율 : \n{male_survived_ratio}')

# %%
### Survived 컬럼과 Gender 컬럼의 관계 (2)
sns.countplot(data=df, x='Survived', hue='Gender')
plt.show()

# %%
### 생존자 / 사망자의 성별 분석

## 대상 : 사망자

# 사망자 --> 남여의 빈도수 추출
condition1 = (df.loc[:,'Survived']==0)
dead_gender_counts = df.loc[condition1,'Gender'].value_counts()
print(f'사망자 중 남여의 수 : \n{dead_gender_counts}')

print('-'*80)

# 사망자 --> 남여의 비율 추출
condition1 = (df.loc[:,'Survived']==0)
dead_gender_ratio = df.loc[condition1,'Gender'].value_counts(normalize=True)
print(f'사망자 중 남여의 비율 : \n{dead_gender_ratio}')

## 대상 : 생존자

# 생존자 --> 남여의 빈도수 추출
condition2 = (df.loc[:,'Survived']==1)
alive_gender_counts = df.loc[condition2,'Gender'].value_counts()
print(f'생존자 중 남여의 수 : \n{alive_gender_counts}')

print('-'*80)

# 생존자 --> 남여의 비율 추출
condition2 = (df.loc[:,'Survived']==1)
alive_gender_ratio = df.loc[condition2,'Gender'].value_counts(normalize=True)
print(f'사망자 중 남여의 비율 : \n{alive_gender_ratio}')

# %% [markdown]
# ### 생존 여부와 연령의 관계 분석

# %%
### 탑승객의 연령 시각화
sns.countplot(data=df, x='Age')
plt.show()

print('-'*80)

# 통계 분석 --> 탑승객의 연령별 빈도수 추출
counts = df.loc[:,'Age'].value_counts()
print(f'탑승객의 연령별 빈도수 : \n{counts}')

print('-'*80)

# 통계 분석 --> 탑승객의 연령별 비율 추출
ratio = df.loc[:,'Age'].value_counts(normalize=True)
print(f'탑승객의 연령별 비율 : \n{ratio}')

# %%
### Survived 컬럼과 Age 컬럼의 관계 확인 (1)
sns.countplot(data=df, x='Survived', hue='Age')
plt.show()

# %%
### 사망자 / 생존자의 연령 분석

## 대상 : 사망자

# 사망자의 연령별 빈도수 추출
condition1 = (df.loc[:,'Survived']==0)
dead_age_counts = df.loc[condition1, 'Age'].value_counts()
print(f'사망자의 연령별 빈도수 : \n{dead_age_counts}')

print('-'*80)

# 사망자의 연령별 비율 추출
condition1 = (df.loc[:,'Survived']==0)
dead_age_ratio = df.loc[condition1, 'Age'].value_counts(normalize=True)
print(f'사망자의 연령별 비율 : \n{dead_age_ratio}')

print('-'*80)

## 대상 : 생존자

# 생존자의 연령별 빈도수 추출
condition2 = (df.loc[:,'Survived']==1)
alive_age_counts = df.loc[condition2, 'Age'].value_counts()
print(f'생존자의 연령별 빈도수 : \n{alive_age_counts}')

print('-'*80)

# 생존자의 연령별 비율 추출
condition2 = (df.loc[:,'Survived']==1)
alive_age_ratio = df.loc[condition2, 'Age'].value_counts(normalize=True)
print(f'생존자의 연령별 비율 : \n{alive_age_ratio}')

# %%
### 사망자 / 생존자 연령 분석의 결과 통합
df_compare = pd.concat([ratio, dead_age_ratio, alive_age_ratio], axis=1)
df_compare.columns = ['total','dead','alive']
print(df_compare)

# %%
### Survived 컬럼과 Age 컬럼의 관계 확인 (2)
sns.countplot(data=df, x='Age', hue='Survived')
plt.show()

# %%
### 20대 / 30대 / 10대의 생존 여부 분석

### 대상 : 20대

# 20대 --> 사망자 / 생존자 수 추출
condition1 = (df.loc[:,'Age']==2)
age20_survived_counts = df.loc[condition1,'Survived'].value_counts()
print(f'20대 중 사망자 / 생존자 수 : \n{age20_survived_counts}')

print('-'*80)

# 20대 --> 사망자 / 생존자 비율 추출
condition1 = (df.loc[:,'Age']==2)
age20_survived_ratio = df.loc[condition1,'Survived'].value_counts(normalize=True)
print(f'20대 중 사망자 / 생존자 비율 : \n{age20_survived_ratio}')

print('-'*80)

### 대상 : 30대

# 30대 --> 사망자 / 생존자 수 추출
condition2 = (df.loc[:,'Age']==3)
age30_survived_counts = df.loc[condition2,'Survived'].value_counts()
print(f'30대 중 사망자 / 생존자 수 : \n{age30_survived_counts}')

print('-'*80)

# 30대 --> 사망자 / 생존자 비율 추출
condition2 = (df.loc[:,'Age']==3)
age30_survived_ratio = df.loc[condition2,'Survived'].value_counts(normalize=True)
print(f'30대 중 사망자 / 생존자 비율 : \n{age30_survived_ratio}')

print('-'*80)

### 대상 : 10대

# 10대 --> 사망자 / 생존자 수 추출
condition3 = (df.loc[:,'Age']==1)
age10_survived_counts = df.loc[condition3,'Survived'].value_counts()
print(f'10대 중 사망자 / 생존자 수 : \n{age10_survived_counts}')

print('-'*80)

# 10대 --> 사망자 / 생존자 비율 추출
condition3 = (df.loc[:,'Age']==1)
age10_survived_ratio = df.loc[condition3,'Survived'].value_counts(normalize=True)
print(f'10대 중 사망자 / 생존자 비율 : \n{age10_survived_ratio}')

# %% [markdown]
# ### 성별과 연령의 관계 분석

# %%
### Gender 컬럼과 Age 컬럼의 관계 확인 (1) : x='Gender', hue='Age'
sns.countplot(data=df, x='Gender', hue='Age')
plt.show()

# %%
### Gender 컬럼과 Age 컬럼의 관계 확인 (2) : x='Age', hue='Gender'
sns.countplot(data=df, x='Age', hue='Gender')
plt.show()

# %%
### 20대의 성별 분석

# 20대의 성별 빈도수 추출
condition1 = (df.loc[:,'Age']==2)
age20_gender_counts = df.loc[condition1,'Gender'].value_counts()
print(f'20대 중 남녀의 수 : \n{age20_gender_counts}')

print('-'*80)

# 20대의 성별 비율 추출
condition1 = (df.loc[:,'Age']==2)
age20_gender_ratio = df.loc[condition1,'Gender'].value_counts(normalize=True)
print(f'20대 중 남녀의 비율 : \n{age20_gender_ratio}')

# %%
### 30대의 성별 분석

# 30대의 성별 빈도수 추출
condition2 = (df.loc[:,'Age']==3)
age30_gender_counts = df.loc[condition2,'Gender'].value_counts()
print(f'30대 중 남녀의 수 : \n{age30_gender_counts}')

print('-'*80)

# 30대의 성별 비율 추출
condition2 = (df.loc[:,'Age']==3)
age30_gender_ratio = df.loc[condition2,'Gender'].value_counts(normalize=True)
print(f'30대 중 남녀의 비율 : \n{age30_gender_ratio}')

# %%
### 10대의 성별 분석

# 10대의 성별 빈도수 추출
condition3 = (df.loc[:,'Age']==1)
age10_gender_counts = df.loc[condition3,'Gender'].value_counts()
print(f'10대 중 남녀의 수 : \n{age10_gender_counts}')

print('-'*80)

# 10대의 성별 비율 추출
condition3 = (df.loc[:,'Age']==1)
age10_gender_ratio = df.loc[condition3,'Gender'].value_counts(normalize=True)
print(f'10대 중 남녀의 비율 : \n{age10_gender_ratio}')

# %% [markdown]
# ### 생존 여부와 요금의 관계 분석

# %%
### Fare 컬럼의 분포 확인

# 이미지 크기 재설정
plt.figure(figsize=(12,6))

sns.histplot(data=df, x='Fare', bins=100, kde=True, hue='Survived')
plt.show()

# %% [markdown]
# #### Pclass 컬럼과 Fare 컬럼의 관계 분석

# %%
### Pclass 컬럼과 Fare 컬럼의 관계 시각화
sns.scatterplot(data=df, x='Pclass', y='Fare')
plt.show()

# %%
### 2등석, 3등석 요금의 최대값 추출

# 2등석 요금의 최대값 추출
condition1 = (df.loc[:,'Pclass']==2)
pclass2_fare_max = df.loc[condition1,'Fare'].max()
print(f'2등석 요금의 최대값 : {pclass2_fare_max}')

print('-'*80)

# 3등석 요금의 최대값 추출
condition2 = (df.loc[:,'Pclass']==3)
pclass3_fare_max = df.loc[condition2,'Fare'].max()
print(f'3등석 요금의 최대값 : {pclass3_fare_max}')

# %%
###  요금이 73.5 달러보다 큰 승객 --> 사망자 / 생존자 비율 확인

# 비교 연산 적용
condition = (df.loc[:,'Fare']>73.5)
# print(condition)

# 빈도수 추출
counts = df.loc[condition,'Survived'].value_counts()
print(f'요금이 73.5 달러보다 큰 승객의 사망자 / 생존자 수 : \n{counts}')

print('-'*80)

# 비율 추출
ratio = df.loc[condition,'Survived'].value_counts(normalize=True)
print(f'요금이 73.5 달러보다 큰 승객의 사망자 / 생존자 비율 : \n{ratio}')

# %% [markdown]
# ### 생존 여부와 가족 인원수의 관계 분석

# %%
### 탑승객의 가족 인원수 컬럼 분포 분석

# 시각화
sns.countplot(data=df, x='num_family')
plt.show()

print('-'*80)

# 통계 분석 --> 가족 인원수별 빈도수 추출
counts = df.loc[:,'num_family'].value_counts()
print(f'가족 인원수별 빈도수 : \n{counts}')

print('-'*80)

# 통계 분석 --> 가족 인원수별 비율 추출
ratio = df.loc[:,'num_family'].value_counts(normalize=True)
print(f'가족 인원수별 비율 : \n{ratio}')

# %%
### Survived 컬럼과 num_family 컬럼의 관계 확인 (1)
sns.countplot(data=df, x='num_family', hue='Survived')
plt.show()

# %%
### 생존 여부와 가족 인원수 분포 분석

## 대상 : 가족 인원수 = 0 --> 사망자 / 생존자 비율
condition1 = (df.loc[:,'num_family']==0)
num0_ratio = df.loc[condition1,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 0인 경우 사망자 / 생존자 비율 : \n{num0_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 1 --> 사망자 / 생존자 비율
condition2 = (df.loc[:,'num_family']==1)
num1_ratio = df.loc[condition2,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 1인 경우 사망자 / 생존자 비율 : \n{num1_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 2 --> 사망자 / 생존자 비율
condition3 = (df.loc[:,'num_family']==2)
num2_ratio = df.loc[condition3,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 2인 경우 사망자 / 생존자 비율 : \n{num2_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 3 --> 사망자 / 생존자 비율
condition4 = (df.loc[:,'num_family']==3)
num3_ratio = df.loc[condition4,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 3인 경우 사망자 / 생존자 비율 : \n{num3_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 4 --> 사망자 / 생존자 비율
condition5 = (df.loc[:,'num_family']==4)
num4_ratio = df.loc[condition5,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 4인 경우 사망자 / 생존자 비율 : \n{num4_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 5 --> 사망자 / 생존자 비율
condition6 = (df.loc[:,'num_family']==5)
num5_ratio = df.loc[condition6,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 5인 경우 사망자 / 생존자 비율 : \n{num5_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 6 --> 사망자 / 생존자 비율
condition7 = (df.loc[:,'num_family']==6)
num6_ratio = df.loc[condition7,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 6인 경우 사망자 / 생존자 비율 : \n{num6_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 7 --> 사망자 / 생존자 비율
condition8 = (df.loc[:,'num_family']==7)
num7_ratio = df.loc[condition8,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 7인 경우 사망자 / 생존자 비율 : \n{num7_ratio}')

print('-'*80)

## 대상 : 가족 인원수 = 10 --> 사망자 / 생존자 비율
condition9 = (df.loc[:,'num_family']==10)
num10_ratio = df.loc[condition9,'Survived'].value_counts(normalize=True)
print(f'동반 가족 인원수가 10인 경우 사망자 / 생존자 비율 : \n{num10_ratio}')

# %%
### Survived 컬럼과 num_family 컬럼의 관계 확인 (2)
sns.countplot(data=df, x='Survived', hue='num_family')
plt.show()

# %% [markdown]
# ### 생존 여부와 호칭의 관계 분석

# %%
### 탐승객의 호칭 분포 분석

'''
### title 컬럼 ==> 'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'others':4
'''

sns.countplot(data=df, x='title')
plt.show()

# 통계 분석 --> 호칭별 빈도수 추출
counts = df.loc[:,'title'].value_counts()
print(f'호칭별 빈도수 :\n{counts}')

print('-'*80)

# 통계 분석 --> 호칭별 비율 추출
ratio = df.loc[:,'title'].value_counts(normalize=True)
print(f'호칭별 비율 :\n{ratio}')

# %%
### Survived 컬럼과 title 컬럼의 관계 확인
sns.countplot(data=df, x='Survived', hue='title')
plt.show()

# %%
### 생존 여부와 호칭별 분포 분석

'''
호칭별 비율 :
Mr        0.578304
Miss      0.198625
Mrs       0.150497
Master    0.046600
others    0.025974
'''

# 대상 : 사망자 --> 호칭별 비율 추출
condition1 = (df.loc[:,'Survived']==0)
dead_title = df.loc[condition1,'title'].value_counts(normalize=True)
print(f'호칭별 사망자 비율 : \n{dead_title}')

print('-'*80)

# 대상 : 생존자 --> 호칭별 비율 추출
condition2 = (df.loc[:,'Survived']==1)
alive_title = df.loc[condition2,'title'].value_counts(normalize=True)
print(f'호칭별 생존자 비율 : \n{alive_title}')

# %%
### 생존 여부와 호칭별 분포 --> 데이터프레임 생성

df_title_survive = pd.concat([ratio, dead_title, alive_title], axis=1)

# 컬럼 이름 변경
df_title_survive.columns = ['total', 'dead', 'alive']

# 결과 확인하기
print(df_title_survive)

# %% [markdown]
# ### 생존 여부와 승선 항구의 관계 분석

# %%
### 탐승객의 승선 항구 분석
sns.countplot(data=df, x='Embarked')
plt.show()

print('-'*80)

# 통계 분석 --> 승선 항구별 빈도수 추출
counts = df.loc[:,'Embarked'].value_counts()
print(f'승선 항구별 빈도수 :\n{counts}')

print('-'*80)

# 통계 분석 --> 승선 항구별 비율 추출
ratio = df.loc[:,'Embarked'].value_counts(normalize=True)
print(f'승선 항구별 비율 :\n{ratio}')

# %%
### 탐승객의 승선 항구 분석
sns.countplot(data=df, x='Embarked', hue='Survived')
plt.show()

# %%
### 생존 여부와 승선 항구별 분포 분석

## 대상자 : 사망자

# 사망자의 항구별 비율 추출
condition1 = (df.loc[:,'Survived']==0)
dead_ratio = df.loc[condition1,'Embarked'].value_counts(normalize=True)
print(f'항구별 사망자 비율 : \n{dead_ratio}')

print('-'*80)

## 대상 : 생존자

# 생존자의 항구별 비율 추출
condition2 = (df.loc[:,'Survived']==1)
alive_ratio = df.loc[condition2,'Embarked'].value_counts(normalize=True)
print(f'항구별 생존자 비율 : \n{alive_ratio}')

# %%
### 생존 여부와 승선 항구별 분포 --> 데이터프레임 생성

# pd.concat() 사용 --> 데이터 프레임 생성
df_sur_embarked = pd.concat([ratio, dead_ratio, alive_ratio], axis=1)

# 컬럼 이름 변경
df_sur_embarked.columns = ['total', 'dead', 'alive']

# 결과 확인하기
print(df_sur_embarked)

# %%
'''
어우 힘들엇다...
'''


