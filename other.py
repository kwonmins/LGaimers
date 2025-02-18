import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# ✅ 데이터 로드
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# ✅ '임신 성공 여부' 컬럼 제거 (Feature 개수 맞추기)
y = train["임신 성공 여부"]
train.drop(columns=["임신 성공 여부"], inplace=True)

# ✅ Feature 개수 검증 (train과 test가 동일해야 함)
assert set(train.columns) == set(test.columns), "Feature 개수가 일치하지 않습니다."

# ✅ 문자열 데이터를 수치형으로 변환 (Label Encoding)
label_encoders = {}
for col in train.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    combined_data = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined_data.astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# ✅ NaN 값 처리 (수치형은 0으로, 범주형은 최빈값으로 채움)
num_imputer = SimpleImputer(strategy="constant", fill_value=0)
train.loc[:, train.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(train.select_dtypes(include=[np.number]))
test.loc[:, test.select_dtypes(include=[np.number]).columns] = num_imputer.transform(test.select_dtypes(include=[np.number]))

# ✅ 🔥 범주형 NaN 처리 (🚨 오류 해결)
cat_columns = train.select_dtypes(include=["object"]).columns
if len(cat_columns) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    train.loc[:, cat_columns] = cat_imputer.fit_transform(train[cat_columns])
    test.loc[:, cat_columns] = cat_imputer.transform(test[cat_columns])

# ✅ Feature Engineering (다항식 + 로그 변환 + 제곱근 변환)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
train_poly = poly.fit_transform(train)
test_poly = poly.transform(test)

train_poly = pd.DataFrame(train_poly)
test_poly = pd.DataFrame(test_poly)

train_log = np.log1p(train + 1)
test_log = np.log1p(test + 1)

train_sqrt = np.sqrt(train + 1)
test_sqrt = np.sqrt(test + 1)

# ✅ 기존 Feature + 새로운 Feature 합치기
train = pd.concat([train, train_poly, train_log, train_sqrt], axis=1)
test = pd.concat([test, test_poly, test_log, test_sqrt], axis=1)

# ✅ Feature Selection (불필요한 Feature 제거)
selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
selector.fit(train, y)

selected_features = train.columns[selector.feature_importances_ > 0.005]
train = train[selected_features]
test = test[selected_features]

# ✅ Feature 개수 검증
assert train.shape[1] == test.shape[1], "Feature 개수가 일치하지 않습니다."

# ✅ LightGBM 모델 설정
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 90,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 25,
    'lambda_l1': 2.0,
    'lambda_l2': 4.0,
    'seed': 42
}

# ✅ Train/Validation Split
X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=42, stratify=y)

# ✅ LightGBM 모델 학습
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

# ✅ 최종 예측
pred_proba = lgb_model.predict(test)

# ✅ 결과 저장
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('./improved_submission.csv', index=False)
