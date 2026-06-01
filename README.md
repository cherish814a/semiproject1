# 심편한 💊
### 관상동맥질환(CAD) 위험도 사전 예측 AI 모델

---

## 📌 프로젝트 개요

**심편한**은 환자의 임상 데이터를 바탕으로 관상동맥질환(Coronary Artery Disease, CAD) 발병 위험도를 사전에 예측하는 AI 기반 스크리닝 보조 도구입니다.

현재 CAD 표준 확진 검사인 **관상동맥조영술(CAG)** 은 고침습적·고비용 시술로, 환자에게 신체적·경제적 부담을 초래합니다. 본 프로젝트는 기본 임상 데이터만으로 CAD 위험도를 1차 스크리닝하여 불필요한 고침습 검사를 최소화하고, 효율적인 진단 체계를 지원하는 것을 목표로 합니다.

---

## 🎯 목표

| 모델 | 대상 | 활용 |
|------|------|------|
| **Model 1** (B2C) | 환자 자가진단용 | 일상에서 쉽게 파악 가능한 기본 정보로 1차 위험도 스크리닝 |
| **Model 2** (B2B) | 의료진 정밀 진단용 | 병원 내원 후 임상·검사 데이터를 결합한 심층 예측 (CDSS 활용) |

---

## 📂 프로젝트 구조

```
├── Z-Alizadeh_sani_dataset.xlsx          # 원본 데이터셋 (303명, 56개 변수)
├── CAD_Data_Preprocessed_Modeling.xlsx   # 전처리 완료 데이터
├── CAD_Data_processing.py                # 데이터 전처리 및 통계 분석 파이프라인
├── cad_model_training_and_evaluation.py  # 모델 학습 및 평가
├── CAD_Risk_Prediction_Streamlit_App.py  # Streamlit 웹 애플리케이션
├── final_model1_web.pkl                  # 저장된 Model 1 (XGBoost)
├── final_model2_web.pkl                  # 저장된 Model 2 (XGBoost)
├── web_config_model1.pkl                 # Model 1 설정 (threshold, 성능지표)
└── web_config_model2.pkl                 # Model 2 설정 (threshold, 성능지표)
```

---

## 🗂️ 데이터

- **출처**: Z-Alizadeh Sani Dataset
- **샘플**: 총 303명
- **변수**: 56개 (환자 기본 정보, 증상, 심전도, 혈액검사 등)
- **타겟 변수**: `Cath` → `Target` (CAD=1, Normal=0)로 이진화

---

## ⚙️ 분석 파이프라인 (`CAD_Data_processing.py`)

```
데이터 로드
  ↓
전처리 (Y/N 인코딩 → 범주화 → One-Hot Encoding)
  ↓
VIF 기반 다중공선성 검사 및 변수 제거 (threshold=10)
  ↓
표준화 (Z-score, 연속형 변수)
  ↓
통계검정 (t-test / Chi-square)
  ↓
그룹별 변수 선정 (Model 1 / Model 2)
  ↓
위계적 로지스틱 회귀 (Type1: 전체 변수, Type2: 필터링 변수)
  ↓
결과 저장 (Excel + Forest Plot + 상관관계 히트맵)
```

**주요 전처리 내용**

- `Age`, `BMI`, `FBS`, `BP` → 임상 기준 기반 범주형 변환
- `BBB` → 원-핫 인코딩 (`BBB_LBBB`, `BBB_RBBB`)
- `VHD` → 서열 척도 (N=0, mild=1, Moderate=2, Severe=3)
- 결측치 없음 확인

---

## 🤖 모델 학습 및 평가 (`cad_model_training_and_evaluation.py`)

### 사용 변수

**Model 1 (자가진단용)**
`Age_cat`, `DM`, `HTN`, `Typical Chest Pain`, `Atypical`, `Dyspnea`

**Model 2 (정밀진단용)**
`Age_cat`, `DM`, `HTN`, `Typical Chest Pain`, `Atypical`, `Nonanginal`, `Dyspnea`, `BP_cat`, `Diastolic Murmur`, `Q Wave`, `St Elevation`, `St Depression`, `Tinversion`, `FBS_cat`, `TG`, `ESR`, `Region RWMA`, `VHD`

### 비교 알고리즘

- Logistic Regression (Baseline / Class-Weight Balanced / SMOTE)
- Decision Tree (Before/After SMOTE)
- SVM (Before/After SMOTE)
- **XGBoost** (Before/After SMOTE) ← **최종 선택**

### 평가 방법

- Hold-out (80:20 분할, stratify)
- 5-Fold Stratified Cross Validation
- 평가지표: Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity
- 임계값 최적화: **Youden's Index** 기반

### 최종 모델

| | 최종 모델 | 특징 |
|--|-----------|------|
| Model 1 | XGBoost (BeforeSMOTE) | 자가진단용 6개 변수 |
| Model 2 | XGBoost (BeforeSMOTE) | 정밀진단용 18개 변수 |

### XAI: SHAP 분석

- `shap.TreeExplainer` 기반 변수 기여도 시각화
- Summary Plot (Bar / Beeswarm)
- 변수별 Mean Absolute SHAP 저장

---

## 🖥️ Streamlit 웹 앱 (`CAD_Risk_Prediction_Streamlit_App.py`)

### 실행 방법

```bash
pip install streamlit pandas numpy scikit-learn xgboost shap imbalanced-learn joblib openpyxl
streamlit run CAD_Risk_Prediction_Streamlit_App.py
```

### 앱 구성 (3개 탭)

**Tab 1 — Model 1: 간편 위험도**
- 나이, 당뇨/고혈압 유무, 흉통/호흡곤란 증상 입력
- CAD 위험도 확률 바 (초록→노랑→빨간색 그라데이션)
- 위험인자별 건강 안내 메시지

**Tab 2 — Model 2: 정밀 예측**
- 기본 정보 + 병력/증상 + 심전도 검사 + 혈액검사 입력
- 예측 후 실시간 SHAP Waterfall Plot 및 변수 기여도 표
- 모델 성능 지표 표시 (CV 기준)

**Tab 3 — 모델 설명**
- 최종 모델명, CV 성능, 사용 변수 목록
- 나이·혈압·혈당 범주화 기준 안내

### 범주화 기준 (입력 변환)

| 변수 | 기준 |
|------|------|
| 나이 | 0: 0~44세 / 1: 45~59세 / 2: 60~74세 / 3: 75세~ |
| 수축기 혈압 | 0: ~119 / 1: 120~139 / 2: 140~ |
| 공복혈당 | 0: ~100 / 1: 101~125 / 2: 126~ |

---

## 🛠️ 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 분석 | `pandas`, `numpy` |
| 통계 검정 | `scipy`, `statsmodels` |
| 시각화 | `matplotlib`, `seaborn` |
| 머신러닝 | `scikit-learn`, `xgboost` |
| 불균형 처리 | `imbalanced-learn` (SMOTE) |
| XAI | `shap` |
| 웹 서비스 | `streamlit` |
| UI 디자인 | Figma |

---

## ⚠️ 면책 조항

본 서비스는 관상동맥질환(CAD) 위험도 예측을 위한 **보조 도구**입니다.  
실제 의학적 진단을 대체하지 않으며, 정확한 진단을 위해서는 반드시 전문 의료기관의 진료를 받으시기 바랍니다.