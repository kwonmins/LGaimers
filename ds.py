import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline  # imblearn의 Pipeline 사용
from imblearn.over_sampling import SMOTE  # 불균형 데이터 처리

# 데이터 로드
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# 특성과 타겟 분리
X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# 범주형과 수치형 컬럼 분리
categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형", "배란 자극 여부", 
    "배란 유도 유형", "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부", 
    "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", 
    "여성 주 불임 원인", "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", 
    "불명확 불임 원인", "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", 
    "불임 원인 - 배란 장애", "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", 
    "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", 
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태", "배아 생성 주요 이유", 
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", 
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", 
    "DI 출산 횟수", "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이", 
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부", 
    "PGD 시술 여부", "PGS 시술 여부"
]

numeric_columns = [
    "임신 시도 또는 마지막 임신 경과 연수", "총 생성 배아 수", "미세주입된 난자 수", 
    "미세주입에서 생성된 배아 수", "이식된 배아 수", "미세주입 배아 이식 수", 
    "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수", 
    "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수", 
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "난자 채취 경과일", 
    "난자 해동 경과일", "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일"
]

# 전처리 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# SMOTE를 사용하여 불균형 데이터 처리
smote = SMOTE(random_state=42)

# 모델 파이프라인 (imblearn의 Pipeline 사용)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),  # SMOTE 적용
    ('classifier', ExtraTreesClassifier(random_state=42, class_weight='balanced'))  # 클래스 가중치 적용
])

# 하이퍼파라미터 튜닝
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# 교차 검증을 사용한 그리드 서치
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X, y)

# 최적의 모델로 예측
best_model = grid_search.best_estimator_
pred_proba = best_model.predict_proba(test)[:, 1]

# 결과 저장
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('./improved_submitds.csv', index=False)

# 최적의 ROC-AUC 점수 출력
print(f"Best ROC-AUC Score: {grid_search.best_score_:.4f}")