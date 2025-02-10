import pandas as pd
import numpy as np
import os
import tempfile
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# Windows 환경에서 joblib의 임시 폴더를 강제로 설정
temp_dir = tempfile.gettempdir()  # 기본 OS 임시 폴더 가져오기
os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir  # 임시 폴더 설정

# 파일 존재 여부 확인
def check_file_exists(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} 파일이 존재하지 않습니다.")

for file in ['train.csv', 'test.csv', 'sample_submission.csv']:
    check_file_exists(file)

# 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ID 컬럼 제거
train.drop(columns=['ID'], errors='ignore', inplace=True)
test.drop(columns=['ID'], errors='ignore', inplace=True)

X = train.drop(columns=['임신 성공 여부'], errors='ignore')
y = train['임신 성공 여부']

# 카테고리형 및 수치형 컬럼 자동 감지
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 카테고리형 데이터를 category 타입으로 변환 후 코드화
for col in categorical_columns:
    X[col] = X[col].astype('category')
    test[col] = test[col].astype('category')

# 수치형 데이터 변환 및 결측값 처리
for df in [X, test]:
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# LGBM 하이퍼파라미터 튜닝 (RandomizedSearchCV 사용)
param_dist = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': np.logspace(-3, -1, 5),
    'subsample': np.linspace(0.5, 1.0, 5),
    'colsample_bytree': np.linspace(0.5, 1.0, 5),
    'lambda_l1': np.logspace(-3, 1, 5),
    'lambda_l2': np.logspace(-3, 1, 5)
}

lgbm = LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42)

random_search = RandomizedSearchCV(
    lgbm, param_distributions=param_dist,
    n_iter=20, scoring='roc_auc', cv=5, verbose=1, n_jobs=1, random_state=42  # n_jobs=1로 변경
)

random_search.fit(X, y)

# 최적의 모델 학습
best_model = random_search.best_estimator_
best_model.fit(X, y)

# 예측 확률
pred_proba = best_model.predict_proba(test)[:, 1]

# 제출 파일 생성
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('min.csv', index=False)

print(f"최적 모델 AUC: {random_search.best_score_:.4f}")
print("최종 모델 학습 완료. min.csv 파일 저장 완료.")
