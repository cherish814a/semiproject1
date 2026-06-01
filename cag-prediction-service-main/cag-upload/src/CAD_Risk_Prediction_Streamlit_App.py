import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import base64
from xgboost import XGBClassifier

# =========================================================
# 0. 페이지 설정
# =========================================================
st.set_page_config(page_title="CAD 진단 보조 웹", layout="wide")

# =========================================================
# 1. 모델 및 설정 로드
# =========================================================
@st.cache_resource
def load_assets():
    model1 = joblib.load("final_model1_web.pkl")
    model2 = joblib.load("final_model2_web.pkl")

    # 권장 구조: 분리 저장
    try:
        config1 = joblib.load("web_config_model1.pkl")
        config2 = joblib.load("web_config_model2.pkl")
        return model1, model2, config1, config2, "separate"
    except Exception:
        # 예전 통합 구조도 최소한으로 대응
        config = joblib.load("web_config.pkl")
        return model1, model2, config, config, "combined"

model1, model2, config1, config2, config_mode = load_assets()

# SHAP 결과 저장 폴더
SHAP_DIR = Path(r"C:\semi_project1\screening\SHAP")

# =========================================================
# 2. 설정값 불러오기
# =========================================================
if config_mode == "separate":
    model1_vars = config1["model_vars"]
    model2_vars = config2["model_vars"]

    model1_threshold = config1.get("youden_threshold", 0.50)
    model2_threshold = config2.get("youden_threshold", 0.50)

    model1_name = config1.get("model_name", "Model1_LogisticRegression_AfterSMOTE")
    model2_name = config2.get("model_name", "Model2_LogisticRegression_AfterSMOTE")
else:
    model1_vars = config1["model1_vars"]
    model2_vars = config1["model2_vars"]

    model1_threshold = config1.get("model1_youden_threshold", 0.50)
    model2_threshold = config1.get("model2_youden_threshold", 0.50)

    model1_name = config1.get("model1_name", "Model1_LogisticRegression_AfterSMOTE")
    model2_name = config1.get("model2_name", "Model2_LogisticRegression_AfterSMOTE")

# =========================================================
# 3. CV 성능값만 가져오기
# =========================================================
if config_mode == "separate":
    model1_metrics = {
        "accuracy": config1.get("cv_accuracy", 0.0),
        "precision": config1.get("cv_precision", 0.0),
        "recall": config1.get("cv_recall", 0.0),
        "f1": config1.get("cv_f1", 0.0),
        "auc": config1.get("cv_auc", 0.0),
    }
    model2_metrics = {
        "accuracy": config2.get("cv_accuracy", 0.0),
        "precision": config2.get("cv_precision", 0.0),
        "recall": config2.get("cv_recall", 0.0),
        "f1": config2.get("cv_f1", 0.0),
        "auc": config2.get("cv_auc", 0.0),
    }
else:
    model1_metrics = {
        "accuracy": config1.get("model1_cv_accuracy", 0.0),
        "precision": config1.get("model1_cv_precision", 0.0),
        "recall": config1.get("model1_cv_recall", 0.0),
        "f1": config1.get("model1_cv_f1", 0.0),
        "auc": config1.get("model1_cv_auc", 0.0),
    }
    model2_metrics = {
        "accuracy": config1.get("model2_cv_accuracy", 0.0),
        "precision": config1.get("model2_cv_precision", 0.0),
        "recall": config1.get("model2_cv_recall", 0.0),
        "f1": config1.get("model2_cv_f1", 0.0),
        "auc": config1.get("model2_cv_auc", 0.0),
    }


# =========================================================
# 3-1. Model 2 실시간 SHAP용 background data
# =========================================================
DATA_PATH = r"c:\semi_project1\screening\cad_data_categorized_before_smote.xlsx"

@st.cache_data
def load_model2_background_data(selected_model2_vars):
    df = pd.read_excel(DATA_PATH)
    return df[selected_model2_vars].copy()

X2_background = load_model2_background_data(model2_vars)

# =========================================================
# 4. 변수명 표시용 매핑
# =========================================================
LABEL_MAP = {
    "Age_cat": "Age (나이)",
    "DM": "DM (당뇨)",
    "HTN": "HTN (고혈압)",
    "Typical Chest Pain": "Typical Chest Pain (전형적 흉통)",
    "Atypical": "Atypical (비전형적 흉통)",
    "Nonanginal": "Nonanginal (비협심증성 흉통)",
    "Dyspnea": "Dyspnea (호흡곤란)",
    "BP_cat": "BP (수축기 혈압)",
    "Diastolic Murmur": "Diastolic Murmur (이완기 심잡음)",
    "Q Wave": "Q Wave (Q파)",
    "St Elevation": "St Elevation (ST 상승)",
    "St Depression": "St Depression (ST 하강)",
    "Tinversion": "Tinversion (T파 역전)",
    "FBS_cat": "FBS (공복혈당)",
    "TG": "TG (중성지방)",
    "ESR": "ESR (적혈구침강속도)",
    "Region RWMA": "Region RWMA (국소 벽운동 이상 부위 수)",
    "VHD": "VHD (판막질환)"
}

YES_NO_VARS = {
    "DM", "HTN",
    "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea",
    "Diastolic Murmur", "Q Wave", "St Elevation", "St Depression",
    "Tinversion", "VHD"
}

# =========================================================
# 5. 공통 유틸 함수
# =========================================================
def display_name(var: str) -> str:
    return LABEL_MAP.get(var, var)

def yn_to_num(x: str) -> int:
    return 1 if x == "있다" else 0

def binary_risk_label(prob: float, threshold: float):
    if prob < threshold:
        return "저위험", "green"
    return "고위험", "red"

def make_input_df(var_list, values_dict):
    row = {}
    for v in var_list:
        row[v] = values_dict.get(v, 0)
    return pd.DataFrame([row])

def yes_no_widget(label: str, key: str) -> int:
    return yn_to_num(st.selectbox(label, ["없다", "있다"], key=key))

def yes_no_question_widget(label: str, key: str) -> int:
    return 1 if st.selectbox(label, ["아니오", "예"], key=key) == "예" else 0

def probability_to_color(prob: float) -> str:
    p = max(0.0, min(1.0, float(prob)))

    if p <= 0.5:
        start = np.array([34, 197, 94])
        end = np.array([250, 204, 21])
        t = p / 0.5
    else:
        start = np.array([250, 204, 21])
        end = np.array([220, 38, 38])
        t = (p - 0.5) / 0.5

    rgb = (start + (end - start) * t).astype(int)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

def render_probability_bar(prob: float, title: str = "CAD 위험도 확률"):
    percent = float(prob) * 100
    bar_color = probability_to_color(prob)

    st.write(f"### {title}")
    st.markdown(
        f"""
        <div style="margin-top: 8px; margin-bottom: 8px;">
            <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                {percent:.1f}%
            </div>
            <div style="
                width: 100%;
                background-color: #e5e7eb;
                border-radius: 9999px;
                height: 24px;
                overflow: hidden;
                border: 1px solid #d1d5db;
            ">
                <div style="
                    width: {percent:.1f}%;
                    background-color: {bar_color};
                    height: 100%;
                    border-radius: 9999px;
                    text-align: right;
                    color: white;
                    font-size: 12px;
                    line-height: 24px;
                    padding-right: 8px;
                    box-sizing: border-box;
                    font-weight: 700;
                ">
                    {percent:.1f}%
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 13px; color: #4b5563;">
                저위험에 가까울수록 초록색, 고위험에 가까울수록 빨간색
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_cv_metrics(metrics: dict):
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    row1_col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    row1_col2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    row1_col3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")

    row2_col1.metric("F1-score", f"{metrics['f1'] * 100:.1f}%")
    row2_col2.metric("ROC-AUC", f"{metrics['auc']:.3f}")
    row2_col3.empty()

def show_model2_result_metrics(metrics: dict):
    col1, col2 = st.columns(2)
    col1.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    col2.metric("ROC-AUC", f"{metrics['auc'] * 100:.1f}%")


def show_all_metrics(metrics: dict):
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    row1_col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    row1_col2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    row1_col3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")

    row2_col1.metric("F1-score", f"{metrics['f1'] * 100:.1f}%")
    row2_col2.metric("ROC-AUC", f"{metrics['auc'] * 100:.1f}%")
    row2_col3.empty()


def transform_for_xgboost_shap_from_pipeline(fitted_pipeline, X):
    """
    cad_model_training_and_evaluation.py의 XGBoost SHAP 방식에 맞춤
    - imputer만 적용
    - scaler, smote는 적용하지 않음
    """
    X_t = X.copy()

    if 'imputer' in fitted_pipeline.named_steps:
        X_t = fitted_pipeline.named_steps['imputer'].transform(X_t)

    X_t_df = pd.DataFrame(
        X_t,
        columns=X.columns,
        index=X.index
    )
    return X_t_df


def is_xgboost_pipeline(pipeline_obj):
    return isinstance(pipeline_obj.named_steps['clf'], XGBClassifier)


def show_realtime_model2_shap(model2_pipeline, X_background, X_input_one_row):
    st.markdown("### Model 2 SHAP 해석")

    if not is_xgboost_pipeline(model2_pipeline):
        st.info("현재 Model 2가 XGBoost가 아니어서 XGBoost SHAP를 표시할 수 없습니다.")
        return

    try:
        # 1) 학습 코드와 동일하게 imputer만 적용
        X_bg_trans_df = transform_for_xgboost_shap_from_pipeline(model2_pipeline, X_background)
        X_input_trans_df = transform_for_xgboost_shap_from_pipeline(model2_pipeline, X_input_one_row)

        # 2) 최종 XGBoost 분류기 추출
        xgb_model = model2_pipeline.named_steps['clf']

        # 3) SHAP Explainer
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_input_trans_df)

        # 이진분류에서 list로 반환되는 경우 대비
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 단일 환자 row를 1차원으로 정리
        single_shap = shap_values[0]
        single_input = X_input_trans_df.iloc[[0]]

        # 4) waterfall plot
        st.markdown("#### 현재 환자 기준 설명")

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

        explanation = shap.Explanation(
            values=single_shap,
            base_values=expected_value,
            data=single_input.iloc[0].values,
            feature_names=list(single_input.columns)
        )

        fig1 = plt.figure()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        # 5) 변수별 기여도 표
        st.markdown("#### 변수별 기여도")
        contrib_df = pd.DataFrame({
            "Feature": single_input.columns,
            "Input Value": X_input_one_row.iloc[0].values,
            "SHAP Value": single_shap
        })
        contrib_df["Abs SHAP"] = np.abs(contrib_df["SHAP Value"])
        contrib_df = contrib_df.sort_values("Abs SHAP", ascending=False).reset_index(drop=True)

        st.dataframe(
            contrib_df[["Feature", "Input Value", "SHAP Value"]].style.format({
                "Input Value": "{:.3f}",
                "SHAP Value": "{:.3f}"
            }),
            use_container_width=True
        )

        # 6) 상위 10개 bar plot
        st.markdown("#### 영향이 큰 변수 상위 10개")
        top10 = contrib_df.head(10).copy()

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top10["Feature"][::-1], top10["SHAP Value"][::-1])
        ax.set_xlabel("SHAP Value")
        ax.set_ylabel("Feature")
        ax.set_title("Top 10 Feature Contributions")
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

    except Exception as e:
        st.warning(f"실시간 XGBoost SHAP 계산 중 오류가 발생했습니다: {e}")


def bp_to_bp_cat(bp: float) -> int:
    if bp <= 119:
        return 0
    elif bp <= 139:
        return 1
    else:
        return 2


def fbs_to_fbs_cat(fbs: float) -> int:
    if fbs <= 100:
        return 0
    elif fbs <= 125:
        return 1
    else:
        return 2

# =========================================================
# 6. 나이 -> Age_cat 변환
# =========================================================
def age_to_age_cat(age: int) -> int:
    """
    - 0: young   = 0 ~ 44
    - 1: middle  = 45 ~ 59
    - 2: senior  = 60 ~ 74
    - 3: elderly = 75 ~ 100
    """
    if age < 45:
        return 0
    elif age < 60:
        return 1
    elif age < 75:
        return 2
    else:
        return 3

# =========================================================
# 7. Model 2용 입력 위젯 생성 함수
# =========================================================
def get_model2_widget_for_var(var, key_prefix="m2"):
    name = display_name(var)
    key = f"{key_prefix}_{var}"

    if var == "Age_cat":
        age = st.number_input(
            "Age (나이)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key=key
        )
        return age_to_age_cat(age)

    if var == "BP_cat":
        bp_value = st.number_input(
        "BP (수축기 혈압)",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0,
        key=key
    )

        bp_cat = bp_to_bp_cat(bp_value)
        return bp_cat


    if var == "FBS_cat":
        fbs_value = st.number_input(
        "FBS (공복혈당)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=1.0,
        key=key
    )

        fbs_cat = fbs_to_fbs_cat(fbs_value)
        return fbs_cat


    if var in YES_NO_VARS:
        if var in {"Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea"}:
            return yes_no_question_widget(name, key)
        return yes_no_widget(name, key)

    if var == "Region RWMA":
        return st.number_input(name, min_value=0.0, value=0.0, step=1.0, key=key)

    default_map = {
        "TG": 150.0,
        "ESR": 10.0,
    }

    return st.number_input(
        name,
        value=float(default_map.get(var, 0.0)),
        step=1.0,
        key=key
    )

# =========================================================
# 8. 헤더
# =========================================================
import os

logo_path = r"C:\semi_project1\screening\logo.png"

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
else:
    logo_base64 = None

if logo_base64:
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 28px;
            margin-bottom: 10px;
        ">
            <img src="data:image/png;base64,{logo_base64}" style="width: 220px; height: auto;" />
            <div>
                <div style="
                    font-size: 60px;
                    font-weight: 800;
                    line-height: 1.1;
                    margin: 0;
                    color: #2f3443;
                ">
                    심편한
                </div>
                <div style="
                    font-size: 24px;
                    color: #6b7280;
                    margin-top: 10px;
                    line-height: 1.3;
                ">
                    관상동맥질환(CAD) 위험 예측 보조도구
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("logo.png 파일을 찾을 수 없습니다.")
    st.markdown(
        """
        <div style="font-size: 60px; font-weight: 800; line-height: 1.1; color: #2f3443;">
            심편한
        </div>
        <div style="font-size: 24px; color: #6b7280; margin-top: 10px;">
            관상동맥질환(CAD) 위험 예측 보조도구
        </div>
        """,
        unsafe_allow_html=True
    )
tab1, tab2, tab3 = st.tabs([
    "Model 1: 간편 위험도",
    "Model 2: 정밀 예측",
    "모델 설명"
])

# =========================================================
# 9. Model 1 탭
# =========================================================
with tab1:
    left_margin, main_col, right_margin = st.columns([1, 6, 1])

    with main_col:
        st.header("Model 1 - 환자 자가진단용")

        st.write("### 기본 정보")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Age (나이)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                key="m1_age"
            )
        with c2:
            st.markdown("&nbsp;")

        st.write("### 병력 / 증상")

        c3, c4 = st.columns(2)
        with c3:
            dm = yes_no_widget("DM (당뇨)", "m1_dm")
            typical = yes_no_question_widget("운동하거나 스트레스를 받을 때 가슴이 쥐어짜는 듯 아프고, 쉬면 통증이 줄어드나요? (전형적 흉통)", "m1_typical")
            dyspnea = yes_no_question_widget("조금만 움직여도 숨이 차거나 답답한 느낌이 드나요? (호흡곤란)", "m1_dyspnea")
        with c4:
            htn = yes_no_widget("HTN (고혈압)", "m1_htn")
            atypical = yes_no_question_widget("가슴 통증이 있긴 하지만, 언제 아픈지나 양상이 일정하지 않아 심장 때문인지 애매한가요? (비전형적 흉통)", "m1_atypical")

        input_values1 = {
            "Age_cat": age_to_age_cat(age),
            "DM": dm,
            "HTN": htn,
            "Typical Chest Pain": typical,
            "Atypical": atypical,
            "Dyspnea": dyspnea
        }

        if st.button("Model 1 예측 실행", key="m1_predict"):
            X_input1 = make_input_df(model1_vars, input_values1)
            prob1 = model1.predict_proba(X_input1)[0, 1]

            label1, color1 = binary_risk_label(prob1, model1_threshold)

            st.markdown(f"## CAD 위험도: :{color1}[{label1}]")
            render_probability_bar(prob1, "CAD 위험도 확률")

            if label1 == "저위험":
                st.success("현재 입력 정보 기준 관상동맥질환 위험도는 낮은 편입니다. 다만 정확한 진단을 위해 병원 상담을 권장합니다.")
            else:
                st.error("현재 입력 정보 기준 관상동맥질환 위험도가 높게 나타납니다. 가능한 빠른 시일 내 병원 진료를 권장합니다.")


            st.write("### 분류 기준")
            st.write(f"- Youden's index 기반 threshold: {model1_threshold:.3f}")
            st.write(f"- 저위험: 예측확률 < {model1_threshold:.3f}")
            st.write(f"- 고위험: 예측확률 ≥ {model1_threshold:.3f}")

            st.write("### 위험요인별 안내")
            if dm == 1:
                st.markdown("#### 🍬 당뇨")
                st.warning(
                    "지속적으로 높은 혈당은 혈관 벽을 손상시키고, 혈관을 딱딱하게 만들어 동맥경화를 빠르게 진행시킵니다. "
                    "당뇨 환자는 CAD 발생 위험이 일반인보다 훨씬 높습니다.\n\n"
                    "👉 권장사항: 공복혈당 관리, 당분 섭취 조절, 꾸준한 운동과 약물치료 병행이 필수입니다."
                )

            if htn == 1:
                st.markdown("#### ❤️ 고혈압")
                st.warning(
                    "높은 혈압은 혈관에 지속적으로 강한 압력을 가해 혈관 내벽을 손상시키고, "
                    "그 부위에 콜레스테롤이 쌓여 혈관이 좁아지는 원인이 됩니다.\n\n"
                    "👉 권장사항: 저염식, 체중 감량, 규칙적인 운동, 필요 시 약물 복용으로 혈압을 관리하는 것이 중요합니다."
                )

            if typical == 1 or atypical == 1 or dyspnea == 1:
                st.markdown("#### 🫀 증상 관련")
                st.warning(
                    "흉통이나 호흡곤란은 CAD와 관련될 수 있는 중요한 증상입니다. "
                    "특히 운동 시 악화되거나 반복적으로 나타나면 진료가 필요합니다.\n\n"
                    "👉 권장사항: 증상이 반복되거나 심해지면 순환기내과 진료를 권장합니다."
                )

# =========================================================
# 10. Model 2 탭
# =========================================================
with tab2:
    left_margin, main_col, right_margin = st.columns([1, 6, 1])

    with main_col:
        st.header("Model 2 - 병원 정밀 예측용")

        input_values2 = {}

        basic_vars = [v for v in model2_vars if v in ["Age_cat"]]

        def render_group(title, vars_list):
            if not vars_list:
                return

            st.write(f"### {title}")
            cols = st.columns(2)

            for idx, var in enumerate(vars_list):
                with cols[idx % 2]:
                    input_values2[var] = get_model2_widget_for_var(var, key_prefix="m2")

        render_group("기본 정보", basic_vars)

        history_vars = [
            v for v in model2_vars
            if v in [
                "DM", "HTN", "Typical Chest Pain", "Atypical",
                "Nonanginal", "Dyspnea", "BP_cat", "Diastolic Murmur"

            ]
        ]

        exam_vars = [
            v for v in model2_vars
            if v in [
                "Q Wave", "St Elevation",
                "St Depression", "Tinversion", "Region RWMA", "VHD"
            ]
        ]

        lab_vars = [
            v for v in model2_vars
            if v in [
                "FBS_cat", "TG", "ESR"
            ]
        ]

        render_group("병력 / 증상", history_vars)
        render_group("검사", exam_vars)
        render_group("혈액검사", lab_vars)

        if st.button("Model 2 예측 실행", key="m2_predict"):
            X_input2 = make_input_df(model2_vars, input_values2)
            prob2 = model2.predict_proba(X_input2)[0, 1]

            label2, color2 = binary_risk_label(prob2, model2_threshold)

            st.markdown(f"## CAD 위험도: :{color2}[{label2}]")
            render_probability_bar(prob2, "CAD 위험도 확률")

            if label2 == "저위험":
                st.success("현재 입력 기준으로 저위험군으로 분류됩니다.")
            else:
                st.error("현재 입력 기준으로 고위험군으로 분류됩니다.")

            st.write("### 모델 성능 (CV 기준)")
            show_model2_result_metrics(model2_metrics)

            st.write("### 분류 기준")
            st.write(f"- Youden's index 기반 threshold: {model2_threshold:.3f}")
            st.write(f"- 저위험: 예측확률 < {model2_threshold:.3f}")
            st.write(f"- 고위험: 예측확률 ≥ {model2_threshold:.3f}")

            with st.expander("현재 환자 기준 SHAP 해석 보기", expanded=True):
                show_realtime_model2_shap(
                    model2_pipeline=model2,
                    X_background=X2_background,
                    X_input_one_row=X_input2
                )

# =========================================================
# 11. 설명 탭
# =========================================================
with tab3:
    left_margin, main_col, right_margin = st.columns([1, 6, 1])

    with main_col:
        st.header("모델 설명")
        st.write(f"**Model 1 최종 모델:** {model1_name}")
        st.write(f"**Model 2 최종 모델:** {model2_name}")

        st.write("### Model 1 성능 (CV 기준)")
        show_all_metrics(model1_metrics)

        st.write("### Model 2 성능 (CV 기준)")
        show_all_metrics(model2_metrics)

        st.write("### Model 1 변수")
        st.write([display_name(v) for v in model1_vars])

        st.write("### Model 2 변수")
        st.write([display_name(v) for v in model2_vars])

        st.write("### Age 범주화 기준")
        st.write("- 0: young = 0~44세")
        st.write("- 1: middle = 45~59세")
        st.write("- 2: senior = 60~74세")
        st.write("- 3: elderly = 75~100세")
        
        st.write("### BP 범주화 기준")
        st.write("- 0: 정상 = 0~119")
        st.write("- 1: 고혈압 전단계 = 120~139")
        st.write("- 2: 고혈압 = 140 이상")

        st.write("### FBS 범주화 기준")
        st.write("- 0: 정상 = 0~100")
        st.write("- 1: 전당뇨 = 101~125")
        st.write("- 2: 당뇨 = 126 이상")

        st.write("### 안내")
        st.info("본 서비스는 CAD 위험 예측 보조도구이며, 실제 진단을 대체하지 않습니다.")