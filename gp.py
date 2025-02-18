import pandas as pd
import numpy as np
import os
import tempfile
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel

# ✅ Windows 환경에서 joblib 임시 폴더 설정 (메모리 최적화)
temp_dir = tempfile.gettempdir()
os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir

# ✅ 데이터 불러오기 및 전처리 함수
def load_and_preprocess():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train.drop(columns=['ID'], errors='ignore', inplace=True)
    test.drop(columns=['ID'], errors='ignore', inplace=True)

    X = train.drop(columns=['임신 성공 여부'], errors='ignore')
    y = train['임신 성공 여부']

    # ✅ Feature 이름 통일 (공백 → 언더바)
    X.columns = X.columns.str.replace(" ", "_")
    test.columns = test.columns.str.replace(" ", "_")

    # ✅ **수치형 컬럼 강제 변환 (오류 방지)**
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        test[col] = pd.to_numeric(test[col], errors='coerce')

    # ✅ **NaN 값 처리 (중앙값으로 채움)**
    X = X.fillna(X.median())
    test = test.fillna(test.median())

    # ✅ **Feature Engineering (연산 가능하도록 수치형 변환 완료 후 실행)**
    X['IVF_시술_비율'] = X['IVF_시술_횟수'] / (X['총_임신_횟수'] + 1)
    X['DI_시술_비율'] = X['DI_시술_횟수'] / (X['총_임신_횟수'] + 1)
    test['IVF_시술_비율'] = test['IVF_시술_횟수'] / (test['총_임신_횟수'] + 1)
    test['DI_시술_비율'] = test['DI_시술_횟수'] / (test['총_임신_횟수'] + 1)

    # ✅ Feature Selection 적용
    selector = SelectFromModel(LGBMClassifier(n_estimators=100, random_state=42))
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    X = selector.transform(X)

    # ✅ **Feature Selection 후 test 데이터에도 동일한 Feature 적용**
    selected_features = [f for f in selected_features if f in test.columns]
    test = test[selected_features]
    X = pd.DataFrame(X, columns=selected_features)
    test = pd.DataFrame(test, columns=selected_features)

    # ✅ **SMOTE 적용 (클래스 균형 조정)**
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X, y = smote.fit_resample(X, y)

    # ✅ **스케일링 적용**
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    test = scaler.transform(test)

    return X, y, test

# ✅ 데이터 로딩 및 전처리
X, y, test = load_and_preprocess()
print("✅ 데이터 로딩 및 전처리 완료!")

# ✅ **Bayesian Optimization을 통한 하이퍼파라미터 최적화**
def lgbm_eval(n_estimators, max_depth, learning_rate, subsample, colsample_bytree, lambda_l1, lambda_l2, num_leaves):
    model = LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        random_state=42,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        num_leaves=int(num_leaves),
        device='cpu',
        n_jobs=4  # ✅ 메모리 부족 방지
    )

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[early_stopping(15, verbose=False)]
        )

        y_pred = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

# ✅ Bayesian Optimization 설정
param_bounds = {
    'n_estimators': (200, 1000),
    'max_depth': (4, 12),
    'learning_rate': (0.005, 0.03),
    'subsample': (0.75, 1.0),
    'colsample_bytree': (0.75, 1.0),
    'lambda_l1': (0.01, 10),
    'lambda_l2': (0.01, 10),
    'num_leaves': (30, 150)
}

optimizer = BayesianOptimization(
    f=lgbm_eval,
    pbounds=param_bounds,
    random_state=42
)

optimizer.maximize(init_points=10, n_iter=50)

# ✅ 최적의 하이퍼파라미터 적용
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['num_leaves'] = int(best_params['num_leaves'])

# ✅ 모델 학습 (LightGBM + XGBoost + CatBoost)
best_lgb = LGBMClassifier(**best_params, boosting_type='gbdt', objective='binary', random_state=42, n_jobs=4)
best_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=4)
best_cat = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=6, random_state=42, verbose=0)

best_lgb.fit(X, y)
best_xgb.fit(X, y)
best_cat.fit(X, y)

# ✅ 앙상블 예측 (Stacking)
lgb_preds = best_lgb.predict_proba(test)[:, 1]
xgb_preds = best_xgb.predict_proba(test)[:, 1]
cat_preds = best_cat.predict_proba(test)[:, 1]

final_preds = (lgb_preds * 0.5) + (xgb_preds * 0.3) + (cat_preds * 0.2)

# ✅ 제출 파일 저장
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['probability'] = final_preds
sample_submission.to_csv('final_submission.csv', index=False)

print(f"🚀 최적 모델 AUC: {optimizer.max['target']:.4f}")
print("🎯 최종 모델 학습 완료. final_submission.csv 저장 완료.")
