import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# 데이터 로드
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# 특성과 타겟 분리
X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# 수치형과 범주형 컬럼 분리
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# 수치형 결측치는 평균으로 대체
imputer_num = SimpleImputer(strategy='mean')
X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])
test[numeric_cols] = imputer_num.transform(test[numeric_cols])

# 범주형 결측치는 최빈값으로 대체
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
test[categorical_cols] = imputer_cat.transform(test[categorical_cols])

# 범주형 인코딩 (LabelEncoder)
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])
    test[col] = label_encoder.transform(test[col])

# 특성 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)

# 데이터 분할 (검증용)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor 모델 학습
rf = RandomForestRegressor(random_state=42)

# 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 최적 모델로 검증 데이터 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)

# 검증 데이터에 대한 MSE 출력
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse:.4f}")

# 테스트 데이터 예측
pred = best_model.predict(test)

# 결과 저장
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['probability'] = pred
sample_submission.to_csv('./improved_submission.csv', index=False)

# 최적의 하이퍼파라미터 출력
print(f"Best Parameters: {grid_search.best_params_}")
