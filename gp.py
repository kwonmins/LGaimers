import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import optuna

# 데이터 로드
train = pd.read_csv('/mnt/data/train.csv').drop(columns=['ID'])
test = pd.read_csv('/mnt/data/test.csv').drop(columns=['ID'])

X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# 카테고리형 컬럼
categorical_columns = [...]
# 수치형 컬럼
numeric_columns = [...]

# 카테고리형 데이터를 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# Ordinal Encoding 적용
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 결측값 처리
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
test[numeric_columns] = test[numeric_columns].fillna(X[numeric_columns].median())

# 최적의 하이퍼파라미터 찾기
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 0, 5),
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
        scores.append(model.score(X_val, y_val))
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# 최적의 모델 학습
best_params = study.best_params
final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
final_model.fit(X, y)

# 예측 확률
pred_proba = final_model.predict_proba(test)[:, 1]

# 제출 파일 생성
sample_submission = pd.read_csv('/mnt/data/sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('/mnt/data/optimized_submit.csv', index=False)
