#%% =================================================================================
# 1. 필요한 라이브러리
# =================================================================================
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, make_scorer,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import joblib
import warnings

warnings.filterwarnings('ignore')


#%% =================================================================================
# 2. 기본 설정
# =================================================================================

RANDOM_STATE = 42
DATA_PATH = r"c:\semi_project1\screening\cad_data_categorized_before_smote.xlsx"
OUTPUT_EXCEL = "model_performance_comparison.xlsx"
SHAP_DIR = r"C:\semi_project1\screening\SHAP"

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False


#%% =================================================================================
# 3. 사용할 변수 직접 입력
#=================================================================================

model1_vars = [
     'Age_cat',
    'DM',
    'HTN',
    'Typical Chest Pain',
    'Atypical',
    'Dyspnea']

model2_vars = [
    'Age_cat',
    'DM',
    'HTN',
    'Typical Chest Pain',
    'Atypical',
    'Nonanginal',
    'Dyspnea',
    'BP_cat',
    'Diastolic Murmur',
    'Q Wave',
    'St Elevation',
    'St Depression',
    'Tinversion',
    'FBS_cat',
    'TG',
    'ESR',
    'Region RWMA',
    'VHD'
]

TARGET_COL = 'Target'


#%% =================================================================================
# 4. 데이터 로드 및 검증
# =================================================================================

print("\n" + "="*80)
print("전처리 완료 엑셀 로드")
print("="*80)

df_model_base = pd.read_excel(DATA_PATH)

print(f"전처리 완료 엑셀 로드: {DATA_PATH}")
print(f"shape: {df_model_base.shape}")

if TARGET_COL not in df_model_base.columns:
    raise ValueError(f"'{TARGET_COL}' 컬럼이 없습니다.")


def validate_features(df, features, target_col='Target', model_name='Model'):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"[{model_name}] 엑셀에 없는 변수: {missing}")

    duplicated = list(dict.fromkeys([f for f in features if features.count(f) > 1]))
    if duplicated:
        print(f"[{model_name}] 중복 변수 제거: {duplicated}")
        features = list(dict.fromkeys(features))

    final_features = [f for f in features if f != target_col]

    if not final_features:
        raise ValueError(f"[{model_name}] 사용할 feature가 없습니다.")

    return final_features


model1_vars = validate_features(df_model_base, model1_vars, TARGET_COL, 'Model1')
model2_vars = validate_features(df_model_base, model2_vars, TARGET_COL, 'Model2')

X_model1 = df_model_base[model1_vars].copy()
X_model2 = df_model_base[model2_vars].copy()
y = df_model_base[TARGET_COL].copy()

print("\n최종 Model 1 변수:")
print(model1_vars)
print(f"Model 1 변수 수: {len(model1_vars)}")

print("\n최종 Model 2 변수:")
print(model2_vars)
print(f"Model 2 변수 수: {len(model2_vars)}")

print(f"\nModel 1 입력 shape: {X_model1.shape}")
print(f"Model 2 입력 shape: {X_model2.shape}")

print("\n전체 클래스 분포:")
print(y.value_counts())

print("\n전체 클래스 비율:")
print((y.value_counts(normalize=True) * 100).round(2).astype(str) + "%")


#%% =================================================================================
# 5. train / test split
# =================================================================================

print("\n" + "="*80)
print("train / test split")
print("="*80)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X_model1, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_model2, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Model 1 train 클래스 분포:")
print(y1_train.value_counts())
print("\nModel 1 test 클래스 분포:")
print(y1_test.value_counts())

print("\nModel 2 train 클래스 분포:")
print(y2_train.value_counts())
print("\nModel 2 test 클래스 분포:")
print(y2_test.value_counts())

smote_check = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

print("\n[SMOTE 적용 확인 - Model 1 train]")
X1_res, y1_res = smote_check.fit_resample(X1_train, y1_train)
print("SMOTE 전:", y1_train.value_counts().to_dict())
print("SMOTE 후:", pd.Series(y1_res).value_counts().to_dict())

print("\n[SMOTE 적용 확인 - Model 2 train]")
X2_res, y2_res = smote_check.fit_resample(X2_train, y2_train)
print("SMOTE 전:", y2_train.value_counts().to_dict())
print("SMOTE 후:", pd.Series(y2_res).value_counts().to_dict())


#%% =================================================================================
# 6. 파이프라인 정의
# =================================================================================

print("\n" + "="*80)
print("파이프라인 정의")
print("="*80)


def create_smote_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))
    ])


def create_balanced_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=2000,
            solver='liblinear',
            class_weight='balanced'
        ))
    ])


def create_baseline_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))
    ])


def create_decision_tree_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=4
        ))
    ])


def create_decision_tree_smote_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ('clf', DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=4
        ))
    ])


def create_svm_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(
            probability=True,
            kernel='rbf',
            random_state=RANDOM_STATE
        ))
    ])


def create_svm_smote_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ('clf', SVC(
            probability=True,
            kernel='rbf',
            random_state=RANDOM_STATE
        ))
    ])


def create_xgboost_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        ))
    ])


def create_xgboost_smote_pipeline():
    return ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ('clf', XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        ))
    ])


#%% =================================================================================
# 7. hold-out 평가
# =================================================================================

print("\n" + "="*80)
print("hold-out 평가")
print("="*80)


def evaluate_holdout_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred).ravel()

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    precision = sklearn.metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = sklearn.metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = sklearn.metrics.f1_score(y_test, y_pred, zero_division=0)
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_proba)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n[{model_name}]")
    print("-"*60)
    print("Confusion Matrix")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(sklearn.metrics.classification_report(y_test, y_pred, digits=4))

    print(f"ROC-AUC     : {roc_auc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }


def get_all_model_specs(prefix):
    return [
        (f"{prefix}_Baseline", create_baseline_pipeline()),
        (f"{prefix}_ClassWeightBalanced", create_balanced_pipeline()),
        (f"{prefix}_SMOTE", create_smote_pipeline()),
        (f"{prefix}_DecisionTree_BeforeSMOTE", create_decision_tree_pipeline()),
        (f"{prefix}_DecisionTree_AfterSMOTE", create_decision_tree_smote_pipeline()),
        (f"{prefix}_SVM_BeforeSMOTE", create_svm_pipeline()),
        (f"{prefix}_SVM_AfterSMOTE", create_svm_smote_pipeline()),
        (f"{prefix}_XGBoost_BeforeSMOTE", create_xgboost_pipeline()),
        (f"{prefix}_XGBoost_AfterSMOTE", create_xgboost_smote_pipeline()),
    ]


holdout_results = []

for model_name, model_obj in get_all_model_specs("Model1"):
    holdout_results.append(
        evaluate_holdout_model(model_obj, X1_train, X1_test, y1_train, y1_test, model_name)
    )

for model_name, model_obj in get_all_model_specs("Model2"):
    holdout_results.append(
        evaluate_holdout_model(model_obj, X2_train, X2_test, y2_train, y2_test, model_name)
    )

holdout_results_df = pd.DataFrame(holdout_results)

print("\nHold-out 평가 결과 요약")
print("-"*80)
print(holdout_results_df)


#%% =================================================================================
# 8. 교차검증 평가(k-fold)
# =================================================================================

print("\n" + "="*80)
print("교차검증 평가(k-fold)")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def specificity_score_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'specificity': make_scorer(specificity_score_func)
}

def evaluate_cv(model, X, y, model_name):
    scores = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
    'Model': model_name,
    'Accuracy_mean': scores['test_accuracy'].mean(),
    'Precision_mean': scores['test_precision'].mean(),
    'Recall_mean': scores['test_recall'].mean(),
    'F1_mean': scores['test_f1'].mean(),
    'ROC_AUC_mean': scores['test_roc_auc'].mean(),
    'Specificity_mean': scores['test_specificity'].mean()
}

    print(f"\n[{model_name}]")
    for k, v in result.items():
        if k != 'Model':
            print(f"{k}: {v:.4f}")

    return result


cv_results = []

for model_name, model_obj in get_all_model_specs("Model1"):
    cv_results.append(evaluate_cv(model_obj, X_model1, y, f"{model_name}_CV"))

for model_name, model_obj in get_all_model_specs("Model2"):
    cv_results.append(evaluate_cv(model_obj, X_model2, y, f"{model_name}_CV"))

cv_results_df = pd.DataFrame(cv_results)

print("\n교차검증 결과 요약")
print("-"*80)
print(cv_results_df)


#%% =================================================================================
# 9. 결과 저장
# =================================================================================

print("\n" + "="*80)
print("결과 저장")
print("="*80)

holdout_summary_df = holdout_results_df.copy()

holdout_summary_df['Dataset'] = holdout_summary_df['Model'].apply(
    lambda x: 'Model1' if 'Model1' in x else 'Model2'
)

holdout_summary_df['Model_Type'] = holdout_summary_df['Model'].apply(
    lambda x: 'DecisionTree' if 'DecisionTree' in x
    else ('SVM' if 'SVM' in x
    else ('XGBoost' if 'XGBoost' in x
    else 'LogisticRegression'))
)

holdout_summary_df['Option'] = holdout_summary_df['Model'].apply(
    lambda x: 'ClassWeightBalanced' if 'ClassWeightBalanced' in x
    else ('BeforeSMOTE' if 'BeforeSMOTE' in x
    else ('AfterSMOTE' if 'AfterSMOTE' in x or ('SMOTE' in x and 'Baseline' not in x and 'ClassWeightBalanced' not in x)
    else 'Baseline'))
)

holdout_summary_df = holdout_summary_df[[
    'Dataset', 'Model_Type', 'Option',
    'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC',
    'Sensitivity', 'Specificity'
]].sort_values(
    by=['Dataset', 'Model_Type', 'F1'],
    ascending=[True, True, False]
).reset_index(drop=True)

cv_summary_df = cv_results_df.copy()

cv_summary_df['Dataset'] = cv_summary_df['Model'].apply(
    lambda x: 'Model1' if 'Model1' in x else 'Model2'
)

cv_summary_df['Model_Type'] = cv_summary_df['Model'].apply(
    lambda x: 'DecisionTree' if 'DecisionTree' in x
    else ('SVM' if 'SVM' in x
    else ('XGBoost' if 'XGBoost' in x
    else 'LogisticRegression'))
)

cv_summary_df['Option'] = cv_summary_df['Model'].apply(
    lambda x: 'ClassWeightBalanced' if 'ClassWeightBalanced' in x
    else ('BeforeSMOTE' if 'BeforeSMOTE' in x
    else ('AfterSMOTE' if 'AfterSMOTE' in x or ('SMOTE' in x and 'Baseline' not in x and 'ClassWeightBalanced' not in x)
    else 'Baseline'))
)

cv_summary_df = cv_summary_df[[
    'Dataset', 'Model_Type', 'Option',
    'Accuracy_mean', 'Precision_mean', 'Recall_mean', 'F1_mean', 'ROC_AUC_mean','Specificity_mean'
]].sort_values(
    by=['Dataset', 'Model_Type', 'F1_mean'],
    ascending=[True, True, False]
).reset_index(drop=True)

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    holdout_results_df.to_excel(writer, sheet_name='holdout_results', index=False)
    holdout_summary_df.to_excel(writer, sheet_name='holdout_summary', index=False)
    cv_results_df.to_excel(writer, sheet_name='cv_results', index=False)
    cv_summary_df.to_excel(writer, sheet_name='cv_summary', index=False)

print(f"저장 완료: {OUTPUT_EXCEL}")

#%% =================================================================================
#%% =================================================================================
# 10. 최종 모델 고정
#     Model 1 = LogisticRegression AfterSMOTE
#     Model 2 = LogisticRegression Baseline
# =================================================================================

print("\n" + "="*80)
print("최종 모델 고정")
print("="*80)

FINAL_MODEL1_NAME = "Model1_XGBoost_BeforeSMOTE"
FINAL_MODEL2_NAME = "Model2_XGBoost_BeforeSMOTE"

print(f"최종 선택 모델 - Model 1: {FINAL_MODEL1_NAME}")
print(f"최종 선택 모델 - Model 2: {FINAL_MODEL2_NAME}")


#%% =================================================================================
# 11. 모델 이름 -> 실제 파이프라인 매핑
# =================================================================================

print("\n" + "="*80)
print("파이프라인 매핑")
print("="*80)

def get_pipeline_by_name(model_name):
    if 'ClassWeightBalanced' in model_name:
        return create_balanced_pipeline()
    elif 'DecisionTree_AfterSMOTE' in model_name:
        return create_decision_tree_smote_pipeline()
    elif 'DecisionTree_BeforeSMOTE' in model_name:
        return create_decision_tree_pipeline()
    elif 'SVM_AfterSMOTE' in model_name:
        return create_svm_smote_pipeline()
    elif 'SVM_BeforeSMOTE' in model_name:
        return create_svm_pipeline()
    elif 'XGBoost_AfterSMOTE' in model_name:
        return create_xgboost_smote_pipeline()
    elif 'XGBoost_BeforeSMOTE' in model_name:
        return create_xgboost_pipeline()
    elif 'SMOTE' in model_name and 'Baseline' not in model_name:
        return create_smote_pipeline()
    else:
        return create_baseline_pipeline()


#%% =================================================================================
# 12. Model 1 / Model 2 CV 성능 + CV 기반 Youden's Index 계산
# =================================================================================

print("\n" + "="*80)
print("Model 1 / Model 2 CV 성능 + CV 기반 Youden's Index 계산")
print("="*80)

def calculate_cv_metrics_and_youden(final_model_name, X, y, dataset_name, model_type, option):
    final_model_cv = get_pipeline_by_name(final_model_name)

    # CV 평균 성능 추출
    cv_row = cv_summary_df[
        (cv_summary_df["Dataset"] == dataset_name) &
        (cv_summary_df["Model_Type"] == model_type) &
        (cv_summary_df["Option"] == option)
    ].iloc[0]

    # OOF(out-of-fold) 확률 예측
    y_oof_proba = cross_val_predict(
        final_model_cv,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # CV 기반 Youden threshold
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_oof_proba)
    youden_scores = tpr - fpr
    best_idx = np.argmax(youden_scores)
    youden_threshold = float(thresholds[best_idx])

    return cv_row, youden_threshold, youden_scores[best_idx]


# ---------------------------
# Model 1
# ---------------------------
model1_cv_row, model1_youden_threshold, model1_youden_score = calculate_cv_metrics_and_youden(
    final_model_name=FINAL_MODEL1_NAME,
    X=X_model1,
    y=y,
    dataset_name="Model1",
    model_type="XGBoost",
    option="BeforeSMOTE"
)

print("\n" + "-"*60)
print("[Model 1 CV 성능]")
print(f"Accuracy    : {model1_cv_row['Accuracy_mean']:.4f}")
print(f"Precision   : {model1_cv_row['Precision_mean']:.4f}")
print(f"Recall      : {model1_cv_row['Recall_mean']:.4f}")
print(f"F1          : {model1_cv_row['F1_mean']:.4f}")
print(f"ROC_AUC     : {model1_cv_row['ROC_AUC_mean']:.4f}")
print(f"Specificity : {model1_cv_row['Specificity_mean']:.4f}")

print("\n[Model 1 CV 기반 Youden's Index]")
print(f"Best threshold : {model1_youden_threshold:.4f}")
print(f"Youden score   : {model1_youden_score:.4f}")


# ---------------------------
# Model 2
# ---------------------------
model2_cv_row, model2_youden_threshold, model2_youden_score = calculate_cv_metrics_and_youden(
    final_model_name=FINAL_MODEL2_NAME,
    X=X_model2,
    y=y,
    dataset_name="Model2",
    model_type="XGBoost",
    option="BeforeSMOTE"
)

print("\n" + "-"*60)
print("[Model 2 CV 성능]")
print(f"Accuracy    : {model2_cv_row['Accuracy_mean']:.4f}")
print(f"Precision   : {model2_cv_row['Precision_mean']:.4f}")
print(f"Recall      : {model2_cv_row['Recall_mean']:.4f}")
print(f"F1          : {model2_cv_row['F1_mean']:.4f}")
print(f"ROC_AUC     : {model2_cv_row['ROC_AUC_mean']:.4f}")
print(f"Specificity : {model2_cv_row['Specificity_mean']:.4f}")

print("\n[Model 2 CV 기반 Youden's Index]")
print(f"Best threshold : {model2_youden_threshold:.4f}")
print(f"Youden score   : {model2_youden_score:.4f}")


#%% =================================================================================
# 13. 전체 데이터로 재학습 후 저장
# =================================================================================

print("\n" + "="*80)
print("최종 모델 저장")
print("="*80)

# ---------------------------
# Model 1 저장
# ---------------------------
final_model1_web = get_pipeline_by_name(FINAL_MODEL1_NAME)
final_model1_web.fit(X_model1, y)

joblib.dump(final_model1_web, 'final_model1_web.pkl')

web_config_model1 = {
    'model_name': FINAL_MODEL1_NAME,
    'model_vars': model1_vars,
    'youden_threshold': model1_youden_threshold,

    # CV 성능 저장
    'cv_accuracy': float(model1_cv_row['Accuracy_mean']),
    'cv_precision': float(model1_cv_row['Precision_mean']),
    'cv_recall': float(model1_cv_row['Recall_mean']),
    'cv_f1': float(model1_cv_row['F1_mean']),
    'cv_auc': float(model1_cv_row['ROC_AUC_mean']),
    'cv_specificity': float(model1_cv_row['Specificity_mean'])
}

joblib.dump(web_config_model1, 'web_config_model1.pkl')

print("\nModel 1 저장 완료:")
print(" - final_model1_web.pkl")
print(" - web_config_model1.pkl")


# ---------------------------
# Model 2 저장
# ---------------------------
final_model2_web = get_pipeline_by_name(FINAL_MODEL2_NAME)
final_model2_web.fit(X_model2, y)

joblib.dump(final_model2_web, 'final_model2_web.pkl')

web_config_model2 = {
    'model_name': FINAL_MODEL2_NAME,
    'model_vars': model2_vars,
    'youden_threshold': model2_youden_threshold,

    # CV 성능 저장
    'cv_accuracy': float(model2_cv_row['Accuracy_mean']),
    'cv_precision': float(model2_cv_row['Precision_mean']),
    'cv_recall': float(model2_cv_row['Recall_mean']),
    'cv_f1': float(model2_cv_row['F1_mean']),
    'cv_auc': float(model2_cv_row['ROC_AUC_mean']),
    'cv_specificity': float(model2_cv_row['Specificity_mean'])
}

joblib.dump(web_config_model2, 'web_config_model2.pkl')

print("\nModel 2 저장 완료:")
print(" - final_model2_web.pkl")
print(" - web_config_model2.pkl")
#%% =================================================================================
# 14. SHAP 분석 (XGBoost BeforeSMOTE 전용)
# =================================================================================

print("\n" + "="*80)
print("SHAP 분석")
print("="*80)

os.makedirs(SHAP_DIR, exist_ok=True)

def run_xgboost_shap_analysis(fitted_pipeline, X, feature_names, model_label, shap_dir):
    """
    fitted_pipeline: fit 완료된 ImbPipeline
    X: 원본 feature DataFrame
    feature_names: 변수명 리스트
    model_label: 저장 파일명에 붙일 라벨
    shap_dir: SHAP 결과 저장 폴더
    """

    # 1) 파이프라인에서 imputer만 적용
    X_imputed = fitted_pipeline.named_steps['imputer'].transform(X)

    # DataFrame으로 복원
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names, index=X.index)

    # 2) 최종 XGBoost 분류기 추출
    xgb_model = fitted_pipeline.named_steps['clf']

    # 3) SHAP Explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_imputed_df)

    # 이진분류에서 shap_values가 리스트로 나오는 경우 대비
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 4) summary plot (bar)
    plt.figure()
    shap.summary_plot(shap_values, X_imputed_df, plot_type='bar', show=False)
    plt.title(f"{model_label} SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{model_label}_shap_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5) summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_imputed_df, show=False)
    plt.title(f"{model_label} SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"{model_label}_shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6) mean absolute SHAP 값 저장
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='MeanAbsSHAP', ascending=False)

    shap_importance.to_excel(
        os.path.join(shap_dir, f"{model_label}_shap_importance.xlsx"),
        index=False
    )

    print(f"\n[{model_label}] SHAP 저장 완료")
    print(f"- {model_label}_shap_bar.png")
    print(f"- {model_label}_shap_summary.png")
    print(f"- {model_label}_shap_importance.xlsx")


# ---------------------------
# Model 1 SHAP
# ---------------------------
run_xgboost_shap_analysis(
    fitted_pipeline=final_model1_web,
    X=X_model1,
    feature_names=model1_vars,
    model_label="Model1_XGBoost_BeforeSMOTE",
    shap_dir=SHAP_DIR
)

# ---------------------------
# Model 2 SHAP
# ---------------------------
run_xgboost_shap_analysis(
    fitted_pipeline=final_model2_web,
    X=X_model2,
    feature_names=model2_vars,
    model_label="Model2_XGBoost_BeforeSMOTE",
    shap_dir=SHAP_DIR
)