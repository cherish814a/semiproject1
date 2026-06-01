import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 설정
# =============================================================================
CONFIG = {
    'CATEGORIZE_CONTINUOUS': True,
    'VISUALIZE_VARIABLES':   True,
    'VIF_THRESHOLD':         10.0,
    'DATA_PATH': r"D:\\CAD_260401\\Z-Alizadeh sani dataset.xlsx"
}

VARIABLE_NAME_KR = {
    'Age': '나이', 'Age_cat': '나이(범주)', 'Sex': '성별',
    'BMI': '체질량지수', 'BMI_cat': '체질량지수(범주)',
    'Weight': '체중', 'Length': '신장',
    'DM': '당뇨병', 'HTN': '고혈압', 'Current Smoker': '현재흡연',
    'EX-Smoker': '과거흡연', 'FH': '가족력', 'Obesity': '비만',
    'DLP': '이상지질혈증', 'CRF': '만성신부전', 'CVA': '뇌혈관질환',
    'CHF': '울혈성심부전', 'Airway disease': '기도질환',
    'Thyroid Disease': '갑상선질환',
    'Typical Chest Pain': '전형적흉통', 'Atypical': '비전형적흉통',
    'Nonanginal': '비협심증성통증', 'Dyspnea': '호흡곤란',
    'Exertional CP': '운동시흉통', 'LowTH Ang': '저역치협심증',
    'Function Class': '기능등급',
    'BP': '혈압', 'BP_cat': '혈압(범주)', 'PR': '맥박수',
    'Edema': '부종', 'Weak Peripheral Pulse': '약한말초맥',
    'Lung rales': '폐수포음', 'Systolic Murmur': '수축기잡음',
    'Diastolic Murmur': '이완기잡음',
    'Q Wave': 'Q파', 'St Elevation': 'ST상승', 'St Depression': 'ST하강',
    'Tinversion': 'T파역전', 'LVH': '좌심실비대',
    'Poor R Progression': 'R파진행불량',
    'BBB_LBBB': '좌각차단', 'BBB_RBBB': '우각차단',
    'FBS': '공복혈당', 'FBS_cat': '공복혈당(범주)',
    'CR': '크레아티닌', 'TG': '중성지방', 'LDL': '저밀도지단백',
    'HDL': '고밀도지단백', 'BUN': '혈중요소질소',
    'ESR': '적혈구침강속도', 'HB': '헤모글로빈',
    'K': '칼륨', 'Na': '나트륨', 'WBC': '백혈구',
    'Lymph': '림프구', 'Neut': '호중구', 'PLT': '혈소판',
    'EF-TTE': '좌심실박출률', 'Region RWMA': '국소벽운동이상',
    'VHD': '판막질환',
    'Target': '관상동맥질환여부', 'const': '상수항'
}

GROUP_DEFINITIONS = {
    'Demographics': ['Age_cat', 'Sex', 'BMI_cat'],
    'RiskFactors':  ['DM', 'HTN', 'Current Smoker', 'EX-Smoker', 'FH',
                     'Obesity', 'DLP', 'CRF', 'CVA', 'CHF',
                     'Airway disease', 'Thyroid Disease'],
    'Symptoms':     ['Typical Chest Pain', 'Atypical', 'Nonanginal', 'Dyspnea',
                     'Exertional CP', 'LowTH Ang', 'Function Class'],
    'PhysicalExam': ['BP_cat', 'PR', 'Edema', 'Weak Peripheral Pulse',
                     'Lung rales', 'Systolic Murmur', 'Diastolic Murmur'],
    'ECG':          ['Q Wave', 'St Elevation', 'St Depression', 'Tinversion',
                     'LVH', 'Poor R Progression'],
    'Laboratory':   ['FBS_cat', 'CR', 'TG', 'LDL', 'HDL', 'BUN',
                     'ESR', 'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT'],
    'Imaging':      ['EF-TTE', 'Region RWMA', 'VHD']
}

GROUP_NAME_KR = {
    'Demographics': '인구통계', 'RiskFactors': '위험인자',
    'Symptoms': '증상', 'PhysicalExam': '신체검사',
    'ECG': '심전도', 'Laboratory': '검사실검사', 'Imaging': '영상검사'
}

MODEL1_GROUPS = ['Demographics', 'RiskFactors', 'Symptoms']
MODEL1_EXCLUDE = {
    'RiskFactors': ['DLP', 'CRF', 'CHF'],
    'Symptoms':    ['Nonanginal', 'LowTH Ang', 'Function Class']
}
MODEL2_GROUPS = ['Demographics', 'RiskFactors', 'Symptoms',
                 'PhysicalExam', 'ECG', 'Laboratory', 'Imaging']

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 10,
                     'axes.unicode_minus': False})
sns.set_palette("husl")

# =============================================================================
# 2. 유틸리티
# =============================================================================
def kr(v):
    return VARIABLE_NAME_KR.get(v, v)


def var_type(df, var):
    nu = df[var].nunique(dropna=True)
    if nu == 2:
        return 'binary'
    elif nu <= 10 or var.endswith('_cat'):
        return 'ordinal'
    return 'continuous'


def calc_vif_series(X_df):
    """DataFrame의 각 컬럼에 대한 VIF를 Series로 반환"""
    vals = X_df.astype(float).values
    result = {}
    for i, col in enumerate(X_df.columns):
        try:
            result[col] = variance_inflation_factor(vals, i)
        except Exception:
            result[col] = np.nan
    return pd.Series(result)


def safe_write(writer, df, sheet_name):
    """비어있지 않은 DataFrame만 엑셀에 쓰기"""
    if df is not None and not df.empty:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# =============================================================================
# 3. 데이터 전처리
# =============================================================================
def load_and_preprocess_data():
    print("\n[1단계] 데이터 로드 및 전처리")

    df = pd.read_excel(CONFIG['DATA_PATH'], sheet_name='Sheet 1 - Table 1')
    df.columns = df.columns.str.strip()
    print(f"  로드 완료: {df.shape[0]}행 × {df.shape[1]}열")

    df['Target'] = df['Cath'].map({'Cad': 1, 'Normal': 0})
    df.drop('Cath', axis=1, inplace=True)
    n_cad, n_normal = int(df['Target'].sum()), int((df['Target'] == 0).sum())
    print(f"  Target 생성: CAD={n_cad}, Normal={n_normal}, "
          f"CAD비율={df['Target'].mean():.1%}")

    # 결측치 현황
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"\n  결측치 현황 ({len(missing)}개 변수):")
        for col, cnt in missing.items():
            print(f"    {col}: {cnt}개 ({cnt / len(df) * 100:.1f}%)")
    else:
        print("  결측치 없음")

    # Y/N 인코딩
    yn_vars = ['Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease',
               'CHF', 'DLP', 'Weak Peripheral Pulse', 'Lung rales',
               'Systolic Murmur', 'Diastolic Murmur', 'Dyspnea',
               'Atypical', 'Nonanginal', 'Exertional CP', 'LowTH Ang',
               'LVH', 'Poor R Progression']
    for v in yn_vars:
        if v in df.columns and {'Y', 'N'} & set(df[v].dropna().unique()):
            df[v] = df[v].map({'Y': 1, 'N': 0})

    df['Sex'] = df['Sex'].map({'Male': 1, 'Fmale': 0, 'Female': 0})
    df['VHD'] = df['VHD'].map({'N': 0, 'mild': 1, 'Moderate': 2, 'Severe': 3})

    if 'BBB' in df.columns:
        bbb_dum = pd.get_dummies(df['BBB'], prefix='BBB', drop_first=True, dtype=int)
        df = pd.concat([df, bbb_dum], axis=1)
        df.drop('BBB', axis=1, inplace=True)
        print(f"  BBB 더미변수 생성: {list(bbb_dum.columns)}")

    if CONFIG['CATEGORIZE_CONTINUOUS']:
        rules = {
            'Age': ([0, 45, 60, 75, 100],   [0, 1, 2, 3], 'Age_cat'),
            'BMI': ([0, 18.5, 25, 30, 100],  [0, 1, 2, 3], 'BMI_cat'),
            'FBS': ([0, 100, 126, 1000],      [0, 1, 2],    'FBS_cat'),
            'BP':  ([0, 120, 140, 300],       [0, 1, 2],    'BP_cat')
        }
        for src, (bins, labels, name) in rules.items():
            if src in df.columns:
                df[name] = pd.cut(df[src], bins=bins, labels=labels,
                                  right=False).astype('Int64')
                df.drop(src, axis=1, inplace=True)
        print("  임상 기준 범주화 완료 (Age/BMI/FBS/BP → _cat)")

    print(f"  전처리 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df

# =============================================================================
# 4. VIF 계산 및 제거
# =============================================================================
def compute_and_remove_vif(df):
    print("\n[2단계] VIF 계산 및 다중공선성 변수 제거 (전체 변수 적용)")

    feat_cols = [c for c in df.columns if c != 'Target']

    def classify(col):
        nu = df[col].nunique(dropna=True)
        if nu == 2: return 'binary'
        elif nu <= 10 or col.endswith('_cat'): return 'ordinal/categorical'
        return 'continuous'

    vtypes = {c: classify(c) for c in feat_cols}

    type_counts = pd.Series(list(vtypes.values())).value_counts()
    print(f"  전체 변수: {len(feat_cols)}개")
    for vt, cnt in type_counts.items():
        print(f"    {vt}: {cnt}개")
    print(f"  VIF 적용 대상: {len(feat_cols)}개 (전체)")

    df_comp = df[feat_cols].dropna()
    n_drop = len(df) - len(df_comp)
    msg = f"(결측치로 {n_drop}행 제외)" if n_drop else "(결측치 없음)"
    print(f"  VIF 계산용 완전 데이터: {len(df_comp)}행 {msg}")

    empty_vif = pd.DataFrame(columns=[
        'Variable', 'Variable_KR', 'Variable_Type',
        'VIF_before_removal', 'Flag_before',
        'VIF_after_removal', 'VIF_at_removal', 'Iteration', 'Removed'])
    empty_excl = pd.DataFrame(columns=[
        'Variable', 'Variable_KR', 'Variable_Type',
        'N_Unique', 'VIF_Applied', 'Reason'])

    if len(feat_cols) < 2:
        print("  VIF 계산을 위한 변수가 부족합니다.")
        return df, empty_vif, [], empty_excl

    # (A) 제거 전 VIF
    vif_before = calc_vif_series(df_comp[feat_cols])
    before_records = []
    for col in feat_cols:
        v = vif_before[col]
        flag = 'HIGH (≥10)' if not np.isnan(v) and v >= CONFIG['VIF_THRESHOLD'] else 'OK'
        before_records.append({
            'Variable': col, 'Variable_KR': kr(col),
            'Variable_Type': vtypes[col],
            'VIF_before_removal': round(v, 4), 'Flag_before': flag
        })
    vif_before_df = pd.DataFrame(before_records)

    high_count = (vif_before_df['Flag_before'] == 'HIGH (≥10)').sum()
    print(f"\n  VIF ≥ {CONFIG['VIF_THRESHOLD']} 변수 (제거 전): {high_count}개")
    print(f"\n  [제거 전 VIF 상위 10개]")
    for _, r in vif_before_df.sort_values('VIF_before_removal', ascending=False).head(10).iterrows():
        flag = " ← HIGH" if r['Flag_before'] == 'HIGH (≥10)' else ""
        print(f"    {r['Variable']:30s} ({r['Variable_Type']:20s})"
              f"  VIF = {r['VIF_before_removal']:8.2f}{flag}")

    # (B) 반복 제거
    remaining = feat_cols.copy()
    removed_log = []
    iteration = 0

    while len(remaining) >= 2:
        vifs = calc_vif_series(df_comp[remaining])
        max_vif = vifs.max()
        if max_vif < CONFIG['VIF_THRESHOLD']:
            break
        bad_var = vifs.idxmax()
        iteration += 1
        removed_log.append({
            'Variable': bad_var, 'Variable_KR': kr(bad_var),
            'Variable_Type': vtypes.get(bad_var, 'unknown'),
            'VIF_at_removal': round(vifs[bad_var], 4),
            'Iteration': iteration
        })
        remaining.remove(bad_var)
        print(f"  반복 {iteration:2d}: '{bad_var}'"
              f" ({vtypes.get(bad_var, '?')}) 제거"
              f" (VIF = {vifs[bad_var]:.2f})")

    # (C) 제거 후 VIF
    vif_after = calc_vif_series(df_comp[remaining])
    vif_after_df = pd.DataFrame([
        {'Variable': col, 'VIF_after_removal': round(vif_after[col], 4)}
        for col in remaining
    ])

    # (D) 결과 통합
    removed_vars = [r['Variable'] for r in removed_log]
    removed_df = pd.DataFrame(removed_log) if removed_log else pd.DataFrame(
        columns=['Variable', 'Variable_KR', 'Variable_Type', 'VIF_at_removal', 'Iteration'])

    vif_df = (vif_before_df
              .merge(vif_after_df, on='Variable', how='left')
              .merge(removed_df[['Variable', 'VIF_at_removal', 'Iteration']],
                     on='Variable', how='left'))
    vif_df['Removed'] = vif_df['Variable'].isin(removed_vars)
    vif_df = vif_df.sort_values('VIF_before_removal', ascending=False).reset_index(drop=True)

    df_vif = df.drop(columns=removed_vars, errors='ignore')

    # (E) 요약 출력
    if removed_log:
        rts = pd.Series([r['Variable_Type'] for r in removed_log]).value_counts()
        print(f"\n  VIF 제거 완료:")
        print(f"    총 제거: {len(removed_vars)}개 → {removed_vars}")
        print(f"    제거 타입별:")
        for vt, cnt in rts.items():
            print(f"      {vt}: {cnt}개")
        print(f"    잔존 변수: {len(remaining)}개")
    else:
        print(f"\n  VIF 제거 완료: 제거된 변수 없음 "
              f"(모든 VIF < {CONFIG['VIF_THRESHOLD']})")
        print(f"    잔존 변수: {len(remaining)}개")

    return df_vif, vif_df, removed_vars, empty_excl

# =============================================================================
# 5. 통계검정
# =============================================================================
def compute_descriptive_stats(df, var, target='Target'):
    g0, g1, total = (df[df[target] == 0][var].dropna(),
                     df[df[target] == 1][var].dropna(),
                     df[var].dropna())
    vt = var_type(df, var)

    if vt == 'binary':
        def fmt(s):
            n = int(s.sum())
            return f"{n} ({n / len(s) * 100:.1f}%)" if len(s) else "0 (0.0%)"
        return {'Stat_Type': 'n (%)', 'Total': fmt(total),
                'Normal_0': fmt(g0), 'CAD_1': fmt(g1)}
    elif vt == 'ordinal':
        def fmt(s):
            return (f"{s.median():.1f} [{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}]"
                    if len(s) else 'N/A')
        return {'Stat_Type': 'Median [IQR]', 'Total': fmt(total),
                'Normal_0': fmt(g0), 'CAD_1': fmt(g1)}
    else:
        def fmt(s):
            return f"{s.mean():.2f} ± {s.std():.2f}" if len(s) else 'N/A'
        return {'Stat_Type': 'Mean ± SD', 'Total': fmt(total),
                'Normal_0': fmt(g0), 'CAD_1': fmt(g1)}


def perform_statistical_test(df, var, target='Target'):
    if var not in df.columns:
        return {}
    vt = var_type(df, var)
    try:
        if vt == 'continuous':
            g0 = df[df[target] == 0][var].dropna()
            g1 = df[df[target] == 1][var].dropna()
            if len(g0) < 3 or len(g1) < 3:
                return {}

            _, p0 = shapiro(g0) if len(g0) <= 5000 else (0, 1.0)
            _, p1 = shapiro(g1) if len(g1) <= 5000 else (0, 1.0)
            normal = p0 >= 0.05 and p1 >= 0.05
            norm_str = (f"Normal (SW p={p0:.3f}/{p1:.3f})" if normal
                        else f"Non-normal (SW p={p0:.3f}/{p1:.3f})")

            if normal:
                _, lp = levene(g0, g1)
                eq = lp >= 0.05
                stat, pv = ttest_ind(g0, g1, equal_var=eq)
                test_name = "Student's t-test" if eq else "Welch's t-test"
                lev_str = f"Levene p={lp:.3f} ({'등분산' if eq else '이분산'})"
            else:
                stat, pv = mannwhitneyu(g0, g1, alternative='two-sided')
                test_name, lev_str = "Mann-Whitney U", "N/A (비모수)"

            psd = np.sqrt(((len(g0)-1)*g0.var() + (len(g1)-1)*g1.var()) /
                          (len(g0)+len(g1)-2))
            cd = (g1.mean()-g0.mean()) / psd if psd else 0

            return {'Test_Method': test_name, 'Normality': norm_str,
                    'Levene': lev_str, 'Statistic': round(stat, 4),
                    'P_value': round(pv, 6), 'Effect_Size': round(abs(cd), 4),
                    'Effect_Label': "Cohen's d", 'Significant': pv < 0.05}
        else:
            ct = pd.crosstab(df[var], df[target])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                return {}
            chi2, pv, dof, exp = chi2_contingency(ct)
            n = ct.sum().sum()
            cv = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
            low_exp = (exp < 5).sum() / exp.size * 100
            return {
                'Test_Method': 'Chi-square', 'Normality': 'N/A (범주형)',
                'Levene': 'N/A (범주형)', 'Statistic': round(chi2, 4),
                'P_value': round(pv, 6), 'Effect_Size': round(cv, 4),
                'Effect_Label': "Cramér's V", 'Significant': pv < 0.05,
                'Fisher_Note': ("Fisher's exact 권장 (기대빈도<5 셀 >20%)"
                                if low_exp > 20 else "")
            }
    except Exception as e:
        return {'Error': str(e)}


def build_statistical_table(df, variable_list, table_label='Table', target='Target'):
    n_tot, n_nor, n_cad = len(df), int((df[target]==0).sum()), int((df[target]==1).sum())
    rows = []
    for var in variable_list:
        if var not in df.columns:
            continue
        desc = compute_descriptive_stats(df, var, target)
        test = perform_statistical_test(df, var, target)
        rows.append({
            'Variable': var, 'Variable_KR': kr(var),
            'Variable_Type': var_type(df, var),
            'Stat_Type': desc.get('Stat_Type', ''),
            f'Total (N={n_tot})': desc.get('Total', ''),
            f'Normal (N={n_nor})': desc.get('Normal_0', ''),
            f'CAD (N={n_cad})': desc.get('CAD_1', ''),
            'Test_Method': test.get('Test_Method', ''),
            'Normality_Test': test.get('Normality', ''),
            'Levene_Test': test.get('Levene', ''),
            'Statistic': test.get('Statistic', ''),
            'P_value': test.get('P_value', ''),
            'Significant_p05': test.get('Significant', ''),
            'Effect_Size': test.get('Effect_Size', ''),
            'Effect_Label': test.get('Effect_Label', ''),
            'Fisher_Note': test.get('Fisher_Note', ''),
            'Table_Label': table_label
        })
    result = pd.DataFrame(rows)
    print(f"  [{table_label}] {len(result)}개 변수 통계검정 완료")
    return result

# =============================================================================
# 6. 그룹 정의 구성
# =============================================================================
def build_group_definitions(df_vif, removed_vars):
    bbb_cols = [c for c in df_vif.columns if c.startswith('BBB_')]
    grp_defs = {}
    for grp, vlist in GROUP_DEFINITIONS.items():
        extra = bbb_cols if grp == 'ECG' else []
        valid = [v for v in vlist + extra
                 if v in df_vif.columns and v not in removed_vars]
        if valid:
            grp_defs[grp] = valid

    print("\n  [전체 그룹 정의 - VIF 제거 반영]")
    for grp, vlist in grp_defs.items():
        print(f"    {grp}({GROUP_NAME_KR.get(grp, grp)}): "
              f"{len(vlist)}개 → {vlist}")
    return grp_defs


def build_model1_group_definitions(grp_defs):
    m1 = {}
    for grp in MODEL1_GROUPS:
        if grp not in grp_defs:
            continue
        excl = MODEL1_EXCLUDE.get(grp, [])
        valid = [v for v in grp_defs[grp] if v not in excl]
        if valid:
            m1[grp] = valid

    print("\n  [Model1 그룹 정의 - 자가진단 불가 변수 제외]")
    for grp, vlist in m1.items():
        excluded = [v for v in MODEL1_EXCLUDE.get(grp, []) if v in grp_defs.get(grp, [])]
        print(f"    {grp}({GROUP_NAME_KR.get(grp, grp)}): "
              f"{len(vlist)}개 유지"
              + (f", 제외: {excluded}" if excluded else ""))
        print(f"      → {vlist}")
    return m1

# =============================================================================
# 7. Type2 필터링
# =============================================================================
def build_type2_group_definitions(df_for_stats, grp_defs, model_groups):
    t2, rows = {}, []
    for grp in model_groups:
        for var in grp_defs.get(grp, []):
            if var not in df_for_stats.columns:
                continue
            test = perform_statistical_test(df_for_stats, var)
            pv = test.get('P_value', 1.0) if test else 1.0
            try:
                corr = (df_for_stats[var].corr(df_for_stats['Target'])
                        if pd.api.types.is_numeric_dtype(df_for_stats[var]) else 0.0)
            except Exception:
                corr = 0.0

            mp, mc = pv < 0.05, abs(corr) >= 0.3
            sel = mp or mc
            reason = []
            if mp: reason.append(f"p={pv:.4f}<0.05")
            if mc: reason.append(f"|r|={abs(corr):.3f}≥0.3")

            rows.append({
                'Group': grp, 'Group_KR': GROUP_NAME_KR.get(grp, grp),
                'Variable': var, 'Variable_KR': kr(var),
                'Variable_Type': var_type(df_for_stats, var),
                'Test_Method': test.get('Test_Method', '') if test else '',
                'P_value': round(pv, 6), 'Correlation': round(corr, 4),
                'Effect_Size': test.get('Effect_Size', '') if test else '',
                'Meets_p05': mp, 'Meets_corr03': mc,
                'Selected_Type2': sel,
                'Reason': " & ".join(reason) if reason else "Not selected"
            })
            if sel:
                t2.setdefault(grp, []).append(var)

    return t2, pd.DataFrame(rows)

# =============================================================================
# 8. 표준화
# =============================================================================
def apply_standardization(df):
    print("\n[3단계] 표준화 (Z-score)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {'Target', 'Age_cat', 'BMI_cat', 'FBS_cat', 'BP_cat'}
    exclude.update(c for c in num_cols if df[c].nunique(dropna=True) <= 2)
    to_scale = [c for c in num_cols if c not in exclude]
    if to_scale:
        df[to_scale] = StandardScaler().fit_transform(df[to_scale])
        print(f"  표준화 적용: {len(to_scale)}개 변수")
        print(f"  표준화 제외: {len(num_cols) - len(to_scale)}개 변수 "
              f"(이진/범주형/Target)")
    return df

# =============================================================================
# 9. 위계적 로지스틱 회귀
# =============================================================================
def fit_logistic_model(df, var_list):
    try:
        avail = [v for v in var_list if v in df.columns]
        if not avail:
            return None
        mask = df[avail].notna().all(axis=1) & df['Target'].notna()
        X, y = df[avail][mask].astype(float), df['Target'][mask].astype(int)
        if len(X) < 10 or y.nunique() < 2:
            return None
        return sm.Logit(y, sm.add_constant(X, has_constant='add')).fit(
            disp=0, maxiter=1000, method='bfgs')
    except Exception:
        return None


def run_hierarchical_modeling(df, group_order, grp_defs, type_lbl, model_lbl):
    results, cumvars = [], []
    for si, grp in enumerate(group_order):
        cumvars = cumvars + [v for v in grp_defs.get(grp, []) if v not in cumvars]
        if not cumvars:
            continue
        model = fit_logistic_model(df, cumvars)
        if model is None:
            continue
        inc = group_order[:si + 1]
        results.append({
            'Model_Label': model_lbl, 'Type': type_lbl,
            'Step': si + 1, 'Added_Group': grp,
            'Added_Group_KR': GROUP_NAME_KR.get(grp, grp),
            'Included_Groups': " + ".join(inc),
            'Included_Groups_KR': " + ".join(GROUP_NAME_KR.get(g, g) for g in inc),
            'Variables': cumvars.copy(), 'N_Variables': len(cumvars),
            'AIC': round(model.aic, 4), 'BIC': round(model.bic, 4),
            'Pseudo_R2': round(model.prsquared, 6),
            'Log_Likelihood': round(model.llf, 4),
            'N_obs': int(model.nobs), 'model_object': model
        })
    return results


def evaluate_all_group_orders(df, group_list, grp_defs, type_lbl, model_lbl,
                               max_full_perm=5):
    print(f"    [{model_lbl}-{type_lbl}] 그룹 순서 평가 (그룹 {len(group_list)}개)")

    if len(group_list) <= max_full_perm:
        perms = list(permutations(group_list))
        print(f"      전체 순열: {len(perms)}개")
    else:
        perms = [tuple(group_list), tuple(reversed(group_list))]
        rng, seen = np.random.default_rng(42), set(perms)
        for _ in range(5000):
            s = tuple(rng.permutation(group_list).tolist())
            if s not in seen:
                perms.append(s)
                seen.add(s)
            if len(perms) >= 22:
                break
        print(f"      샘플 순열: {len(perms)}개")

    all_steps, order_rows = [], []
    for pi, perm in enumerate(perms):
        olbl = f"Order_{pi + 1:03d}"
        steps = run_hierarchical_modeling(df, list(perm), grp_defs, type_lbl, model_lbl)
        for r in steps:
            r['Order_Label'] = olbl
            r['Group_Order_Str'] = " → ".join(perm)
            all_steps.append(r)
        if steps:
            fs = steps[-1]
            order_rows.append({
                'Model_Label': model_lbl, 'Type': type_lbl,
                'Order_Label': olbl, 'Group_Order': " → ".join(perm),
                'AIC': fs['AIC'], 'BIC': fs['BIC'],
                'Pseudo_R2': fs['Pseudo_R2'],
                'N_Variables': fs['N_Variables'], 'N_obs': fs['N_obs']
            })

    order_summary = pd.DataFrame(order_rows).sort_values('AIC')
    best_model = None
    if not order_summary.empty:
        best_lbl = order_summary.iloc[0]['Order_Label']
        best_steps = [r for r in all_steps if r['Order_Label'] == best_lbl]
        if best_steps:
            best_model = best_steps[-1]

    print(f"      완료: 단계별 결과 {len(all_steps)}개")
    return all_steps, best_model, order_summary


def extract_coefficients(best_result, type_lbl, model_lbl):
    if best_result is None:
        return pd.DataFrame()
    mo, ci = best_result['model_object'], best_result['model_object'].conf_int()
    rows = []
    for var in mo.params.index:
        if var == 'const':
            continue
        rows.append({
            'Model_Label': model_lbl, 'Type': type_lbl,
            'Variable': var, 'Variable_KR': kr(var),
            'Coefficient': round(mo.params[var], 6),
            'OR': round(np.exp(mo.params[var]), 4),
            'CI_Lower_95': round(np.exp(ci.loc[var, 0]), 4),
            'CI_Upper_95': round(np.exp(ci.loc[var, 1]), 4),
            'P_value': round(mo.pvalues[var], 6),
            'Significant': mo.pvalues[var] < 0.05
        })
    return pd.DataFrame(rows).sort_values('P_value')

# =============================================================================
# 10. 모델 비교
# =============================================================================
def _interpret_delta(metric, delta):
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return 'N/A'
    interp = {
        'AIC': ('Type2 Better (↓)' if delta < -2 else 'Type1 Better' if delta > 2 else 'Similar'),
        'BIC': ('Type2 Better (↓)' if delta < -2 else 'Type1 Better' if delta > 2 else 'Similar'),
        'Pseudo_R2': ('Type2 Better (↑)' if delta > 0.01 else 'Type1 Better' if delta < -0.01 else 'Similar'),
    }
    if metric in interp:
        return interp[metric]
    if metric == 'N_Variables':
        if delta < 0: return f'Type2: {abs(int(delta))}개 적음'
        if delta > 0: return f'Type2: {int(delta)}개 많음'
        return 'Same'
    return ''


def compare_type1_type2(t1_best, t2_best, model_lbl):
    if t1_best is None or t2_best is None:
        return pd.DataFrame()
    rows = []
    for m in ['AIC', 'BIC', 'Pseudo_R2', 'N_Variables']:
        v1, v2 = t1_best.get(m, np.nan), t2_best.get(m, np.nan)
        d = round(v2 - v1, 4) if not (np.isnan(v1) or np.isnan(v2)) else np.nan
        rows.append({
            'Model_Label': model_lbl, 'Metric': m,
            'Type1_All_Vars': round(v1, 4), 'Type2_Filtered': round(v2, 4),
            'Delta_T2_minus_T1': d, 'Interpretation': _interpret_delta(m, d)
        })
    return pd.DataFrame(rows)


def build_step_comparison_df(all_steps):
    if not all_steps:
        return pd.DataFrame()
    df_s = pd.DataFrame([{k: v for k, v in r.items() if k != 'model_object'}
                         for r in all_steps])
    rows = []
    keys = ['Model_Label', 'Type', 'Order_Label', 'Group_Order_Str',
            'Step', 'Added_Group', 'Added_Group_KR',
            'Included_Groups', 'Included_Groups_KR',
            'N_Variables', 'AIC', 'BIC', 'Pseudo_R2', 'Log_Likelihood', 'N_obs']

    for (ol, ml, tl), grp in df_s.groupby(['Order_Label', 'Model_Label', 'Type'], sort=False):
        grp = grp.sort_values('Step').reset_index(drop=True)
        for i, row in grp.iterrows():
            entry = {k: row[k] for k in keys}
            for metric in ['AIC', 'BIC', 'Pseudo_R2']:
                entry[f'Delta_{metric}'] = (
                    round(row[metric] - grp.loc[i-1, metric], 4 if metric != 'Pseudo_R2' else 6)
                    if i > 0 else np.nan)
            rows.append(entry)
    return pd.DataFrame(rows)

# =============================================================================
# 11. 메인 분석
# =============================================================================
def run_full_analysis(df_scaled, df_for_stats, grp_defs):
    print("\n[5단계] 위계적 로지스틱 회귀")
    m1_grp = build_model1_group_definitions(grp_defs)

    configs = {
        'Model1': ('환자 자가진단용', MODEL1_GROUPS, m1_grp),
        'Model2': ('병원 정밀진단용', MODEL2_GROUPS, grp_defs)
    }

    all_results, t2_details = {}, {}

    for mk, (name, model_groups, base_grp) in configs.items():
        print(f"\n  ═══ {mk} ({name}) ═══")

        t1_grp = {g: base_grp[g] for g in model_groups if g in base_grp and base_grp[g]}
        t2_grp, fdf = build_type2_group_definitions(df_for_stats, base_grp, model_groups)
        t2_details[mk] = fdf

        print(f"\n  [그룹별 변수 수: Type1 vs Type2]")
        for grp in model_groups:
            print(f"    {grp}({GROUP_NAME_KR.get(grp, grp)}): "
                  f"Type1={len(t1_grp.get(grp, []))}개, "
                  f"Type2={len(t2_grp.get(grp, []))}개")

        model_result = {}
        for tl, gd in [('Type1', t1_grp), ('Type2', t2_grp)]:
            active = [g for g in model_groups if g in gd and gd[g]]
            if not active:
                print(f"    [{tl}] 활성 그룹 없음 - 건너뜀")
                model_result[tl] = {
                    'step_results': [], 'best_full_model': None,
                    'order_summary': pd.DataFrame(),
                    'coeff_df': pd.DataFrame(),
                    'step_comparison_df': pd.DataFrame()
                }
                continue

            print(f"\n    [{tl}] 활성 그룹: {active}")
            for g in active:
                print(f"      {g}: {gd[g]}")

            steps, best, osum = evaluate_all_group_orders(
                df_scaled, active, gd, tl, mk)
            model_result[tl] = {
                'step_results': steps, 'best_full_model': best,
                'order_summary': osum,
                'coeff_df': extract_coefficients(best, tl, mk),
                'step_comparison_df': build_step_comparison_df(steps)
            }

        model_result['comparison_t1_vs_t2'] = compare_type1_type2(
            model_result['Type1']['best_full_model'],
            model_result['Type2']['best_full_model'], mk)
        model_result['name'] = name
        all_results[mk] = model_result

    return all_results, t2_details

# =============================================================================
# 12. 시각화
# =============================================================================
def plot_correlation_heatmap(df, phase_name, output_dir='correlation_heatmaps'):
    os.makedirs(output_dir, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) <= 1:
        return
    n = len(num_cols)
    fsz = (16, 14) if n <= 20 else (20, 18) if n <= 40 else (26, 22)
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=fsz)
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1,
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Heatmap ({phase_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, f"heatmap_{phase_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  히트맵 저장: {fname}")


def create_variable_plots(data, target_col, output_dir):
    if not CONFIG['VISUALIZE_VARIABLES']:
        return
    os.makedirs(output_dir, exist_ok=True)

    binning_map = {
        'ESR': lambda d: _auto_bins(d, 'ESR', 10, 'ESR_b'),
        'EF-TTE': lambda d: ([0, 30, 40, 50, 55, 65, 100],
                              ['<30\n(SevHF)', '30-39\n(ModHF)', '40-49\n(MildHF)',
                               '50-54\n(Border)', '55-64\n(Normal)', '65+\n(Hyper)'], 'EF_b'),
        'PR': lambda d: _auto_bins_pr(d),
    }

    created = 0
    for var in [c for c in data.columns if c != target_col]:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            if var in binning_map:
                bins, lbs, bcol = binning_map[var](data)
                tmp = data.copy()
                tmp[bcol] = pd.cut(tmp[var], bins=bins, labels=lbs,
                                   right=False, include_lowest=True)
                gs = (tmp.groupby(bcol)[target_col]
                      .agg(['count', 'sum', 'mean']).reset_index())
                gs = gs[gs['count'] > 0]
                xl = gs[bcol].astype(str)
            else:
                gs = (data.groupby(var)[target_col]
                      .agg(['count', 'sum', 'mean']).reset_index())
                xl = [str(x) for x in gs[var]]

            gs['pct'] = gs['mean'] * 100
            ax.bar(range(len(gs)), gs['pct'], color='lightcoral',
                   alpha=0.7, edgecolor='black')
            for i, (_, r) in enumerate(gs.iterrows()):
                ax.text(i, r['pct'] + 2, f"{r['pct']:.1f}%\n(n={int(r['sum'])})",
                        ha='center', fontsize=9, fontweight='bold')

            overall = data[target_col].mean() * 100
            ax.axhline(overall, color='red', linestyle='--', alpha=0.6,
                       label=f'Overall CAD Rate: {overall:.1f}%')
            ax.set_title(f'{var} - CAD Prevalence', fontweight='bold')
            ax.set_ylabel('CAD Rate (%)')
            ax.set_ylim(0, 110)
            ax.set_xticks(range(len(gs)))
            ax.set_xticklabels(xl, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{var}.png", dpi=300, bbox_inches='tight')
            plt.close()
            created += 1
        except Exception:
            plt.close()
    print(f"  변수별 그래프: {created}개")


def _auto_bins(data, var, step, bcol):
    mn, mx = int(data[var].min()), int(data[var].max())
    lo, hi = (mn // step) * step, ((mx // step) + 1) * step
    bins = list(range(lo, hi + step, step))
    lbs = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    return bins, lbs, bcol


def _auto_bins_pr(data):
    mn, mx = int(data['PR'].min()), int(data['PR'].max())
    lo, hi = (mn // 10) * 10, ((mx // 10) + 1) * 10
    bins = list(range(lo, hi + 10, 10))
    lbs = [f"{bins[i]}-{bins[i+1]-1}\n"
           f"({'Brady' if bins[i+1] <= 60 else 'Tachy' if bins[i] >= 100 else 'Normal'})"
           for i in range(len(bins) - 1)]
    return bins, lbs, 'PR_b'


def create_forest_plot(coeff_df, fname, title):
    if coeff_df.empty:
        return
    sig = coeff_df[coeff_df['Significant'] == True].copy()
    if sig.empty:
        print(f"  유의 변수 없음 - {fname} 생략")
        return

    types = sig['Type'].unique() if 'Type' in sig.columns else ['All']
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    fig, axes = plt.subplots(1, len(types),
                             figsize=(10 * len(types), max(6, len(sig) * 0.4 + 2)))
    if len(types) == 1:
        axes = [axes]

    for ai, tl in enumerate(types):
        ax = axes[ai]
        sub = sig[sig['Type'] == tl].sort_values('OR').reset_index(drop=True)
        yp = range(len(sub))
        ax.errorbar(sub['OR'], yp,
                    xerr=[sub['OR'] - sub['CI_Lower_95'],
                          sub['CI_Upper_95'] - sub['OR']],
                    fmt='o', color=colors[ai % 4],
                    ecolor='gray', capsize=4, markersize=8)
        ax.axvline(1, color='red', linestyle='--', alpha=0.8)
        ax.set_yticks(yp)
        ax.set_yticklabels([f"{r['Variable']} ({r['Variable_KR']})"
                            for _, r in sub.iterrows()], fontsize=9)
        ax.set_xlabel('Odds Ratio (95% CI)', fontweight='bold')
        ax.set_title(f'{title}\n{tl}', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(axis='x', alpha=0.3)
        for i, (_, r) in enumerate(sub.iterrows()):
            sm = "***" if r['P_value'] < 0.001 else "**" if r['P_value'] < 0.01 else "*"
            ax.text(r['OR'] * 1.05, i, f"{r['OR']:.2f}{sm}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Forest Plot 저장: {fname}")


def create_all_forest_plots(all_results):
    os.makedirs('forest_plots', exist_ok=True)

    for mk, md in all_results.items():
        dfs = [md[t]['coeff_df'] for t in ['Type1', 'Type2']
               if t in md and not md[t]['coeff_df'].empty]
        if dfs:
            create_forest_plot(pd.concat(dfs, ignore_index=True),
                               f'forest_plots/{mk}_T1vsT2.png',
                               f'{mk}: Type1 vs Type2')

    for tl in ['Type1', 'Type2']:
        dfs = [all_results[mk][tl]['coeff_df']
               for mk in ['Model1', 'Model2']
               if mk in all_results and tl in all_results[mk]
               and not all_results[mk][tl]['coeff_df'].empty]
        if dfs:
            create_forest_plot(pd.concat(dfs, ignore_index=True),
                               f'forest_plots/M1vsM2_{tl}.png',
                               f'Model1 vs Model2 ({tl})')

# =============================================================================
# 13. 결과 저장
# =============================================================================
def save_results(df_final, all_results, vif_df, vif_excluded_df,
                 table_all, table_m1, table_m2, t2_details,
                 output='cad_complete_analysis.xlsx'):
    print(f"\n[6단계] 결과 저장: {output}")
    df_final.to_csv('cad_data_final.csv', index=False, encoding='utf-8-sig')

    # Model1 제외 변수 기록
    exclude_reasons = {
        'DLP': 'DLP: 혈액검사(지질패널) 결과 필요',
        'CRF': 'CRF: 혈액검사(크레아티닌) 결과 필요',
        'CHF': 'CHF: 의사의 심부전 진단 필요',
        'Nonanginal': 'Nonanginal: 비협심증성 분류는 의학 지식 필요',
        'LowTH Ang': 'LowTH Ang: 저역치 개념은 의학적 판단 필요',
        'Function Class': 'Function Class: NYHA 등급은 의학적 평가 필요',
    }
    m1_excl_rows = [
        {'Group': grp, 'Group_KR': GROUP_NAME_KR.get(grp, grp),
         'Variable': var, 'Variable_KR': kr(var),
         'Reason': exclude_reasons.get(var, '자가진단 불가')}
        for grp, excl_vars in MODEL1_EXCLUDE.items()
        for var in excl_vars
    ]

    with pd.ExcelWriter(output, engine='openpyxl') as w:
        # 통계검정
        for tbl, sn in [(table_all, 'StatTest_ALL'),
                        (table_m1, 'StatTest_Model1'),
                        (table_m2, 'StatTest_Model2')]:
            safe_write(w, tbl, sn)

        safe_write(w, vif_df, 'VIF_All_Variables')
        safe_write(w, vif_excluded_df, 'VIF_Excluded')

        if m1_excl_rows:
            pd.DataFrame(m1_excl_rows).to_excel(w, sheet_name='Model1_Excluded_Vars', index=False)

        for mk, fdf in t2_details.items():
            safe_write(w, fdf, f'{mk}_Type2_Filtering')

        for mk, md in all_results.items():
            for tl in ['Type1', 'Type2']:
                if tl not in md:
                    continue
                td = md[tl]
                safe_write(w, td['step_comparison_df'], f'{mk}_{tl}_Steps')
                safe_write(w, td['order_summary'], f'{mk}_{tl}_OrderComp')
                safe_write(w, td['coeff_df'], f'{mk}_{tl}_Coeff')
            safe_write(w, md['comparison_t1_vs_t2'], f'{mk}_T1vsT2_Compare')

        # 전체 요약
        summary_rows = []
        for mk, md in all_results.items():
            for tl in ['Type1', 'Type2']:
                if tl not in md:
                    continue
                best = md[tl]['best_full_model']
                n_sig = (int((md[tl]['coeff_df']['Significant'] == True).sum())
                         if not md[tl]['coeff_df'].empty else 0)
                summary_rows.append({
                    'Model': mk, 'Type': tl, 'Purpose': md.get('name', ''),
                    'N_Variables': best['N_Variables'] if best else np.nan,
                    'AIC': best['AIC'] if best else np.nan,
                    'BIC': best['BIC'] if best else np.nan,
                    'Pseudo_R2': best['Pseudo_R2'] if best else np.nan,
                    'N_Sig_Vars': n_sig
                })
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(w, sheet_name='Models_Summary', index=False)

    print(f"  저장 완료: {output}")

# =============================================================================
# 14. 메인
# =============================================================================
def main():
    print("CAD 예측 모델 분석 파이프라인")
    print("=" * 60)
    print("""
분석 흐름:
  [1] 전처리   : 로드 → 인코딩 → 범주화 (결측치 원본 유지)
  [2] VIF      : 전체 변수(이진/범주형/연속형) 다중공선성 검사 → 자동 제거
                 결측치 있는 행은 listwise deletion으로 VIF 계산
  [3] 복사     : df_for_stats = VIF 제거 후, 표준화 전 원본
  [4] 표준화   : 연속형 변수 Z-score (결측치 셀은 NaN 유지)
  [5] 그룹 정의: VIF 제거 결과 반영
                 Model1 = 자가진단 가능 변수만 (MODEL1_EXCLUDE 적용)
                 Model2 = 전체 변수
  [6] 통계검정 : t-test / Chi-square (표준화 전 원본, dropna 적용)
  [7] 모델링   : Model1/Model2 × Type1/Type2 × 그룹순서 위계적 회귀
                 결측치 있는 행은 listwise deletion으로 모델 적합
  [8] 저장     : Excel + Forest Plot + 히트맵
""")

    # [1] 전처리
    df = load_and_preprocess_data()

    # [2] VIF
    df, vif_df, removed_vars, vif_excluded_df = compute_and_remove_vif(df)

    # [3] 통계검정용 원본
    df_for_stats = df.copy()

    plot_correlation_heatmap(df, 'After_VIF_Removal')
    create_variable_plots(df, 'Target', 'figures')

    try:
        df.to_excel('cad_data_after_vif.xlsx', index=False, engine='openpyxl')
        df.to_csv('cad_data_after_vif.csv', index=False, encoding='utf-8-sig')
        print("  VIF 제거 데이터 저장 완료")
    except Exception as e:
        print(f"  저장 실패: {e}")

    # [4] 표준화
    df = apply_standardization(df)

    # [5] 그룹 정의
    print("\n[4단계] 그룹 정의 구성")
    grp_defs = build_group_definitions(df, removed_vars)

    # [6] 통계검정
    print("\n[통계검정 테이블 생성]")
    all_vars = [c for c in df_for_stats.columns if c != 'Target']
    table_all = build_statistical_table(df_for_stats, all_vars, 'Table_ALL')

    m1_grp = build_model1_group_definitions(grp_defs)
    m1_vars = list(dict.fromkeys(v for g in MODEL1_GROUPS for v in m1_grp.get(g, [])))
    table_m1 = build_statistical_table(df_for_stats, m1_vars, 'Table_Model1')

    m2_vars = list(dict.fromkeys(v for g in MODEL2_GROUPS for v in grp_defs.get(g, [])))
    table_m2 = build_statistical_table(df_for_stats, m2_vars, 'Table_Model2')

    # [7] 모델링
    all_results, t2_details = run_full_analysis(df, df_for_stats, grp_defs)

    # [8] 시각화 + 저장
    create_all_forest_plots(all_results)
    save_results(df, all_results, vif_df, vif_excluded_df,
                 table_all, table_m1, table_m2, t2_details)

    # 최종 요약
    print("\n" + "=" * 60)
    print("분석 완료 요약")
    print("=" * 60)

    print("\n[Model1 자가진단용 - 포함 변수]")
    m1_final = build_model1_group_definitions(grp_defs)
    for grp, vlist in m1_final.items():
        print(f"  {grp}({GROUP_NAME_KR.get(grp, grp)}): {vlist}")

    print("\n[Model1 자가진단용 - 제외 변수]")
    for grp, excl in MODEL1_EXCLUDE.items():
        exist = [v for v in excl if v in grp_defs.get(grp, [])]
        if exist:
            print(f"  {grp}: {exist}")

    for mk, md in all_results.items():
        print(f"\n[{mk}] {md.get('name', '')}")
        for tl in ['Type1', 'Type2']:
            if tl not in md:
                continue
            best = md[tl]['best_full_model']
            if best:
                print(f"  {tl}: AIC={best['AIC']:.2f}, BIC={best['BIC']:.2f}, "
                      f"Pseudo_R²={best['Pseudo_R2']:.4f}, N_Vars={best['N_Variables']}")
        if not md['comparison_t1_vs_t2'].empty:
            print("  [Type1 vs Type2]")
            for _, r in md['comparison_t1_vs_t2'].iterrows():
                print(f"    {r['Metric']:12s}: T1={r['Type1_All_Vars']}, "
                      f"T2={r['Type2_Filtered']}, Δ={r['Delta_T2_minus_T1']} "
                      f"→ {r['Interpretation']}")

    print("\n생성 파일:")
    print("  cad_complete_analysis.xlsx")
    print("    ├ StatTest_ALL / Model1 / Model2  ← 통계검정")
    print("    ├ VIF_All_Variables               ← 전체 변수 VIF (이진/범주형 포함)")
    print("    ├ Model1_Excluded_Vars            ← Model1 제외 변수 근거")
    print("    ├ {M}_Type2_Filtering             ← Type2 필터링 상세")
    print("    ├ {M}_{T}_Steps                   ← 위계적 단계별 + Δ")
    print("    ├ {M}_{T}_OrderComp               ← 그룹 순서별 비교")
    print("    ├ {M}_{T}_Coeff                   ← OR / CI / p값")
    print("    ├ {M}_T1vsT2_Compare              ← AIC/BIC/R² 비교+Δ")
    print("    └ Models_Summary                  ← 전체 요약")
    print("  cad_data_after_vif.xlsx/csv         ← VIF 제거 후 원본")
    print("  cad_data_final.csv                  ← 표준화 완료 최종")
    print("  forest_plots/                       ← Forest Plot")
    print("  figures/                            ← 변수별 CAD 발병률")
    print("  correlation_heatmaps/               ← 상관관계 히트맵")


if __name__ == "__main__":
    main()
