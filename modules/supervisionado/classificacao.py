"""
DataForge EDU - Módulo Supervisionado: Classificação
KNN, Decision Tree, Random Forest, SVM, GB, XGBoost, LightGBM,
AdaBoost, Naive Bayes, Logistic Regression + teoria integrada
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib, os, io

from modules.icons import icon_html
from modules.utils import (
    inject_css, page_header, section_title, teoria_box, info_box,
    sucesso_box, aviso_box, erro_box, metricas_row, progresso_bar,
    badge, card, C_TEXT, C_TEXT_SEC, C_TEXT_MUTE, C_SURFACE, C_SURFACE2,
    C_BORDER, C_ACCENT, C_GREEN, C_AMBER, C_RED, C_TEAL, C_ACCENT2,
    C_BG, PALETTE, TEORIA, carregar_dataset_embutido,
    save_user_progress, add_pontos, save_historico_modelo
)
from modules.diagnostico import mostrar_diagnostico
from modules.explicacoes import explicar_classificacao, AJUDA_METRICAS

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False


# ══════════════════════════════════════════════════════
# CONFIGURAÇÃO DOS ALGORITMOS
# ══════════════════════════════════════════════════════
def get_algoritmos():
    algos = {
        "KNN": "K-Nearest Neighbors",
        "Decision Tree": "Árvore de Decisão",
        "Random Forest": "Random Forest",
        "Extra Trees": "Extra Trees",
        "Gradient Boosting": "Gradient Boosting",
        "AdaBoost": "AdaBoost",
        "SVM": "Support Vector Machine",
        "Logistic Regression": "Regressão Logística",
        "Naive Bayes (Gaussian)": "Naive Bayes Gaussiano",
        "Naive Bayes (Bernoulli)": "Naive Bayes Bernoulli",
        "SGD Classifier": "SGD Classifier",
        "LDA": "Linear Discriminant Analysis",
        "Bagging": "Bagging Classifier",
    }
    if XGB_OK:
        algos["XGBoost"] = "XGBoost"
    if LGB_OK:
        algos["LightGBM"] = "LightGBM"
    return algos


def _avancar_passo(n):
    """Callback dos botões «Continuar»: avança o passo ANTES do script correr (uma execução por clique, não duas)."""
    st.session_state.guiado_step = max(st.session_state.get("guiado_step", 1), n)



def build_model(algo: str, params: dict):
    if algo == "KNN":
        return KNeighborsClassifier(**params)
    elif algo == "Decision Tree":
        return DecisionTreeClassifier(**params, random_state=42)
    elif algo == "Random Forest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif algo == "Extra Trees":
        return ExtraTreesClassifier(**params, random_state=42, n_jobs=-1)
    elif algo == "Gradient Boosting":
        return GradientBoostingClassifier(**params, random_state=42)
    elif algo == "AdaBoost":
        return AdaBoostClassifier(**params, random_state=42)
    elif algo == "SVM":
        return SVC(**params, probability=True, random_state=42)
    elif algo == "Logistic Regression":
        return LogisticRegression(**params, random_state=42, max_iter=1000)
    elif algo == "Naive Bayes (Gaussian)":
        return GaussianNB()
    elif algo == "Naive Bayes (Bernoulli)":
        return BernoulliNB()
    elif algo == "SGD Classifier":
        return SGDClassifier(**params, random_state=42)
    elif algo == "LDA":
        return LinearDiscriminantAnalysis()
    elif algo == "Bagging":
        return BaggingClassifier(**params, random_state=42)
    elif algo == "XGBoost" and XGB_OK:
        return xgb.XGBClassifier(**params, random_state=42,
                                  eval_metric="logloss", verbosity=0)
    elif algo == "LightGBM" and LGB_OK:
        return lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    return None


# ══════════════════════════════════════════════════════
# HIPERPARÂMETROS POR ALGORITMO
# ══════════════════════════════════════════════════════
def render_hiperparametros(algo: str, prefix: str = "default") -> dict:
    params = {}
    c1, c2 = st.columns(2)
    kp = f"hp_{prefix}_{algo}".replace(" ", "_")

    if algo == "KNN":
        with c1:
            params["n_neighbors"] = st.slider("K (vizinhos)", 1, 30, 5, key=f"{kp}_k")
            params["metric"] = st.selectbox("Métrica de distância", ["euclidean","manhattan","minkowski"], key=f"{kp}_metric")
        with c2:
            params["weights"] = st.selectbox("Pesos", ["uniform","distance"], key=f"{kp}_weights")
            params["algorithm"] = st.selectbox("Algoritmo interno", ["auto","ball_tree","kd_tree","brute"], key=f"{kp}_algorithm")

    elif algo == "Decision Tree":
        with c1:
            md = st.slider("max_depth", 1, 30, 5, key=f"{kp}_max_depth")
            params["max_depth"] = md if md < 30 else None
            params["criterion"] = st.selectbox("Critério", ["gini","entropy","log_loss"], key=f"{kp}_criterion")
        with c2:
            params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, key=f"{kp}_mss")
            params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 10, 1, key=f"{kp}_msl")

    elif algo in ("Random Forest", "Extra Trees"):
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{kp}_n_estimators")
            md = st.slider("max_depth", 1, 30, 10, key=f"{kp}_max_depth")
            params["max_depth"] = md if md < 30 else None
        with c2:
            params["max_features"] = st.selectbox("max_features", ["sqrt","log2","None"], key=f"{kp}_max_features")
            if params["max_features"] == "None": params["max_features"] = None
            params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, key=f"{kp}_mss")

    elif algo == "Gradient Boosting":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{kp}_n_estimators")
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5], value=0.1, key=f"{kp}_lr")
        with c2:
            params["max_depth"] = st.slider("max_depth", 1, 10, 3, key=f"{kp}_max_depth")
            params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key=f"{kp}_subsample")

    elif algo == "XGBoost":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{kp}_n_estimators")
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3], value=0.1, key=f"{kp}_lr")
            params["max_depth"] = st.slider("max_depth", 1, 12, 6, key=f"{kp}_max_depth")
        with c2:
            params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key=f"{kp}_subsample")
            params["colsample_bytree"] = st.slider("colsample_bytree", 0.3, 1.0, 0.8, 0.05, key=f"{kp}_colsample")
            params["reg_alpha"] = st.select_slider("reg_alpha (L1)", options=[0, 0.01, 0.1, 1, 10], value=0, key=f"{kp}_reg_alpha")

    elif algo == "LightGBM":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10, key=f"{kp}_n_estimators")
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2], value=0.05, key=f"{kp}_lr")
            params["num_leaves"] = st.slider("num_leaves", 10, 200, 31, key=f"{kp}_num_leaves")
        with c2:
            params["min_child_samples"] = st.slider("min_child_samples", 5, 100, 20, key=f"{kp}_mcs")
            params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key=f"{kp}_subsample")

    elif algo == "AdaBoost":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 300, 50, 10, key=f"{kp}_n_estimators")
        with c2:
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0], value=1.0, key=f"{kp}_lr")

    elif algo == "SVM":
        with c1:
            params["C"] = st.select_slider("C (regularização)",
                options=[0.001, 0.01, 0.1, 1, 10, 100, 1000], value=1, key=f"{kp}_c")
            params["kernel"] = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"], key=f"{kp}_kernel")
        with c2:
            if params["kernel"] in ("rbf","poly","sigmoid"):
                params["gamma"] = st.selectbox("Gamma", ["scale","auto"], key=f"{kp}_gamma")
            if params["kernel"] == "poly":
                params["degree"] = st.slider("Grau (poly)", 2, 6, 3, key=f"{kp}_degree")

    elif algo == "Logistic Regression":
        with c1:
            params["C"] = st.select_slider("C (inverso da regularização)",
                options=[0.001, 0.01, 0.1, 1, 10, 100], value=1, key=f"{kp}_c")
            params["solver"] = st.selectbox("Solver", ["lbfgs","saga","liblinear"], key=f"{kp}_solver")
        with c2:
            params["penalty"] = st.selectbox("Penalização", ["l2","l1","elasticnet","None"], key=f"{kp}_penalty")
            if params["penalty"] == "None": params["penalty"] = None

    elif algo == "SGD Classifier":
        with c1:
            params["loss"] = st.selectbox("Loss", ["hinge","log_loss","modified_huber","perceptron"], key=f"{kp}_loss")
            params["alpha"] = st.select_slider("Alpha", options=[0.0001, 0.001, 0.01, 0.1], value=0.0001, key=f"{kp}_alpha")
        with c2:
            params["max_iter"] = st.slider("max_iter", 100, 2000, 1000, 100, key=f"{kp}_max_iter")
            params["penalty"] = st.selectbox("Penalização", ["l2","l1","elasticnet"], key=f"{kp}_penalty")

    elif algo == "Bagging":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 5, 100, 10, key=f"{kp}_n_estimators")
        with c2:
            params["max_samples"] = st.slider("max_samples", 0.1, 1.0, 1.0, 0.05, key=f"{kp}_max_samples")
            params["max_features"] = st.slider("max_features", 0.1, 1.0, 1.0, 0.05, key=f"{kp}_max_features")

    return params


# ══════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ══════════════════════════════════════════════════════
def _fig_style():
    return {"facecolor": "#161B27", "edgecolor": "none"}

def plot_confusion_matrix(y_test, y_pred, labels):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#161B27")
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="#9BA3B2", fontsize=10)
    ax.set_yticklabels(labels, color="#9BA3B2", fontsize=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#9BA3B2", fontsize=12)
    ax.set_xlabel("Previsto", color="#9BA3B2")
    ax.set_ylabel("Real", color="#9BA3B2")
    ax.set_facecolor("#161B27")
    ax.tick_params(colors="#9BA3B2")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3347")
    fig.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, classes):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#161B27")
    ax.set_facecolor("#161B27")
    if len(classes) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba, pos_label=classes[1] if hasattr(classes[1], '__len__') else 1)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=C_ACCENT, lw=2, label=f"AUC = {auc:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=classes)
        proba = model.predict_proba(X_test)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc = roc_auc_score(y_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc:.2f})", color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "--", color="#5C6478", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos", color="#9BA3B2")
    ax.set_ylabel("Taxa de Verdadeiros Positivos", color="#9BA3B2")
    ax.set_title("Curva ROC", color="#E8EBF0")
    ax.legend(facecolor="#1E2535", edgecolor="#2A3347", labelcolor="#9BA3B2", fontsize=9)
    ax.tick_params(colors="#9BA3B2")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3347")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, algo):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

    if importances is None:
        return None

    idx = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7, max(4, len(top_names) * 0.4)), facecolor="#161B27")
    ax.set_facecolor("#161B27")
    colors = [C_ACCENT if v == top_vals.max() else C_SURFACE2 for v in top_vals]
    bars = ax.barh(range(len(top_names)), top_vals, color=colors, edgecolor="#2A3347")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, color="#9BA3B2", fontsize=10)
    ax.set_xlabel("Importância", color="#9BA3B2")
    ax.set_title("Importância das Features (Top 15)", color="#E8EBF0")
    ax.tick_params(colors="#9BA3B2")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3347")
    for bar, val in zip(bars, top_vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#9BA3B2", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cv_scores(scores):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#161B27")
    ax.set_facecolor("#161B27")
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    bars = ax.bar(folds, scores, color=C_ACCENT, alpha=0.85, edgecolor="#2A3347")
    ax.axhline(scores.mean(), color=C_GREEN, linestyle="--", lw=1.5, label=f"Média: {scores.mean():.3f}")
    ax.set_ylim(max(0, scores.min() - 0.05), min(1.0, scores.max() + 0.05))
    ax.set_ylabel("Accuracy", color="#9BA3B2")
    ax.set_title("Validação Cruzada por Fold", color="#E8EBF0")
    ax.legend(facecolor="#1E2535", edgecolor="#2A3347", labelcolor="#9BA3B2")
    ax.tick_params(colors="#9BA3B2")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3347")
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", color="#9BA3B2", fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# RENDER PRINCIPAL
# ══════════════════════════════════════════════════════
def render_classificacao(username: str):
    page_header(
        "Classificação",
        "Aprendizagem supervisionada - prever categorias e classes"
    )

    tab_guiado, tab_livre = st.tabs(["  Modo Guiado", "  Modo Livre"])

    # ── MODO GUIADO ──────────────────────────────────
    with tab_guiado:
        _render_modo_guiado(username)

    # ── MODO LIVRE ───────────────────────────────────
    with tab_livre:
        _render_modo_livre(username)


# ══════════════════════════════════════════════════════
# MODO GUIADO
# ══════════════════════════════════════════════════════
def _render_modo_guiado(username: str):
    st.markdown(f"""<div style="background:rgba(79,142,247,.07);border:1px solid rgba(79,142,247,.2); border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.4rem;font-size:14px;color:{C_TEXT_SEC};"><strong style="color:{C_ACCENT};">Modo Guiado</strong> - O sistema orienta-te passo a passo. Aprende a teoria antes de praticar e recebe feedback automático sobre os teus resultados.</div>""", unsafe_allow_html=True)

    if "guiado_step" not in st.session_state:
        st.session_state.guiado_step = 1

    step = st.session_state.guiado_step

    # STEP 1 - Dataset
    _render_step_header(1, "Escolher Dataset", step)
    if step >= 1:
        with st.container():
            _step1_dataset(username)

    # STEP 2 - Algoritmo e teoria
    if step >= 2:
        _render_step_header(2, "Escolher Algoritmo", step)
        _step2_algoritmo(username)

    # STEP 3 - Pré-processamento
    if step >= 3:
        _render_step_header(3, "Pré-processamento", step)
        _step3_preprocessamento(username)

    # STEP 4 - Treino
    if step >= 4:
        _render_step_header(4, "Treinar Modelo", step)
        _step4_treino(username)


def _render_step_header(n: int, titulo: str, step_actual: int):
    if step_actual > n:
        cor, icon = C_GREEN, ""
    elif step_actual == n:
        cor, icon = C_ACCENT, str(n)
    else:
        cor, icon = C_TEXT_MUTE, str(n)

    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin:1.4rem 0 .8rem;"><div style="width:32px;height:32px;border-radius:50%;background:{cor}; display:flex;align-items:center;justify-content:center; font-size:13px;font-weight:700;color:#fff;flex-shrink:0;">{icon}</div><div style="font-size:16px;font-weight:700;color:{C_TEXT if step_actual >= n else C_TEXT_MUTE};">{titulo}</div></div>""", unsafe_allow_html=True)


def _step1_dataset(username: str):
    catalogo = [
        "Alunos em Angola (fictício) - Aprovação",
        "Iris - Classificação de flores",
        "Titanic - Sobrevivência",
        "Vinho - Qualidade",
        "Cancro - Diagnóstico",
        "Dígitos - Reconhecimento",
    ]
    fonte = st.radio("Fonte dos dados", ["Dataset embutido", "Upload CSV"], horizontal=True)
    df, desc, _ = None, "", ""

    if fonte == "Dataset embutido":
        escolha = st.selectbox("Escolhe o dataset", catalogo)
        df, desc, _ = carregar_dataset_embutido(escolha)
        if desc:
            info_box(f"**{escolha.split('-')[0].strip()}:** {desc}")

    else:
        ficheiro = st.file_uploader("Carrega o teu CSV", type=["csv"])
        if ficheiro:
            try:
                df = pd.read_csv(ficheiro)
                sucesso_box(f"CSV carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
                mostrar_diagnostico(df, expandido=False)
            except Exception as e:
                erro_box(f"Erro ao ler CSV: {e}")

    if df is not None:
        with st.expander("Pré-visualização dos dados"):
            st.dataframe(df.head(10), width='stretch')
            c1, c2, c3 = st.columns(3)
            c1.metric("Linhas", df.shape[0])
            c2.metric("Colunas", df.shape[1])
            c3.metric("Valores nulos", int(df.isnull().sum().sum()))

        st.session_state["clf_df"] = df
        st.button("Continuar para Algoritmo", key="step1_next", on_click=_avancar_passo, args=(2,))


def _step2_algoritmo(username: str):
    df = st.session_state.get("clf_df")
    if df is None:
        aviso_box("Volta ao passo 1 e carrega os dados primeiro.")
        return

    algos = get_algoritmos()
    chave = st.selectbox("Algoritmo", list(algos.keys()),
                         format_func=lambda k: f"{k}  -  {algos[k]}")

    # TEORIA INTEGRADA
    if chave in TEORIA:
        t = TEORIA[chave]
        with st.expander(f"Teoria: {t['nome']}", expanded=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"""<div class="teoria-box"><h4>Analogia</h4><p>{t['analogia']}</p></div><div class="teoria-box"><h4> Como funciona</h4><p>{t['como_funciona']}</p></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="teoria-box"><h4>Quando usar</h4><p>{t['quando_usar']}</p></div><div class="teoria-box"><h4>Cuidados</h4><p>{t['cuidados']}</p></div>""", unsafe_allow_html=True)

    st.session_state["clf_algo"] = chave
    st.button("Continuar para Pré-processamento", key="step2_next", on_click=_avancar_passo, args=(3,))


def _step3_preprocessamento(username: str):
    df = st.session_state.get("clf_df")
    if df is None:
        return

    cols = df.columns.tolist()
    target = st.selectbox("Coluna target (o que queremos prever)", cols, index=len(cols)-1)
    features = [c for c in cols if c != target]

    features_sel = st.multiselect("Features a usar", features, default=features)
    if not features_sel:
        aviso_box("Seleciona pelo menos uma feature.")
        return

    # Pré-processamento
    c1, c2, c3 = st.columns(3)
    with c1:
        nulos = st.selectbox("Tratar valores nulos", ["Remover linhas", "Preencher com média", "Preencher com mediana", "Preencher com moda"])
    with c2:
        scaler_opt = st.selectbox("Normalização", ["Nenhuma", "StandardScaler", "MinMaxScaler"])
    with c3:
        test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.5, 0.2, 0.05)

    # Diagnóstico de dados
    df_sub = df[features_sel + [target]].copy()
    nulos_total = df_sub.isnull().sum().sum()
    if nulos_total > 0:
        aviso_box(f"Detectados {nulos_total} valores nulos. Serão tratados conforme configuração.")

    cat_cols = df_sub[features_sel].select_dtypes(include="object").columns.tolist()
    if cat_cols:
        info_box(f"Colunas categóricas detectadas: **{', '.join(cat_cols)}** - serão codificadas automaticamente (LabelEncoder).")

    st.session_state["clf_target"] = target
    st.session_state["clf_features"] = features_sel
    st.session_state["clf_nulos"] = nulos
    st.session_state["clf_scaler"] = scaler_opt
    st.session_state["clf_test_size"] = test_size

    st.button("Continuar para Treino", key="step3_next", on_click=_avancar_passo, args=(4,))


def _step4_treino(username: str):
    df       = st.session_state.get("clf_df")
    algo     = st.session_state.get("clf_algo")
    target   = st.session_state.get("clf_target")
    features = st.session_state.get("clf_features", [])
    nulos    = st.session_state.get("clf_nulos", "Remover linhas")
    scaler_o = st.session_state.get("clf_scaler", "Nenhuma")
    ts       = st.session_state.get("clf_test_size", 0.2)

    if df is None or algo is None:
        aviso_box("Volta aos passos anteriores.")
        return

    section_title("HIPERPARÂMETROS")
    params = render_hiperparametros(algo, prefix="guiado")

    cv_k = st.slider("Folds de validação cruzada", 2, 10, 5, key="hp_guiado_cv_k")

    if st.button("Treinar Modelo", type="primary", key="btn_treinar_guiado"):
        with st.spinner("A treinar o modelo..."):
            try:
                resultado = _executar_treino(
                    df, features, target, algo, params,
                    nulos, scaler_o, ts, cv_k, username
                )
                if resultado:
                    _render_resultados(resultado, algo, username)
            except Exception as e:
                erro_box(f"Erro durante o treino: {str(e)}")
                st.exception(e)


# ══════════════════════════════════════════════════════
# MODO LIVRE
# ══════════════════════════════════════════════════════
def _render_modo_livre(username: str):
    st.markdown(f"""<div style="background:rgba(46,204,113,.07);border:1px solid rgba(46,204,113,.2); border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.4rem;font-size:14px;color:{C_TEXT_SEC};"><strong style="color:{C_GREEN};">Modo Livre</strong> - Controlo total. Configura tudo manualmente e experimenta sem restrições.</div>""", unsafe_allow_html=True)

    # Upload / Dataset
    col_fonte, _ = st.columns([2, 1])
    with col_fonte:
        fonte = st.radio("Fonte", ["Dataset embutido", "Upload CSV"], horizontal=True, key="livre_fonte")

    df, desc = None, ""
    if fonte == "Dataset embutido":
        catalogo_clf = [k for k in [
            "Alunos em Angola (fictício) - Aprovação",
            "Iris - Classificação de flores",
            "Titanic - Sobrevivência",
            "Vinho - Qualidade",
            "Cancro - Diagnóstico",
            "Dígitos - Reconhecimento",
        ]]
        escolha = st.selectbox("Dataset", catalogo_clf, key="livre_ds")
        df, desc, _ = carregar_dataset_embutido(escolha)
        if desc:
            info_box(desc)
    else:
        f = st.file_uploader("CSV", type=["csv"], key="livre_upload")
        if f:
            try:
                df = pd.read_csv(f)
                mostrar_diagnostico(df, expandido=False)
            except Exception as e:
                erro_box(str(e))

    if df is None:
        return

    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.selectbox("Target", cols, index=len(cols)-1, key="livre_target")
    with c2:
        features = st.multiselect("Features", [c for c in cols if c != target],
                                   default=[c for c in cols if c != target], key="livre_feat")
    with c3:
        test_size = st.slider("Teste %", 0.1, 0.5, 0.2, 0.05, key="livre_ts")

    c4, c5, c6 = st.columns(3)
    with c4:
        nulos = st.selectbox("Nulos", ["Remover linhas","Preencher com média","Preencher com mediana"], key="livre_nulos")
    with c5:
        scaler_o = st.selectbox("Scaler", ["Nenhuma","StandardScaler","MinMaxScaler"], key="livre_scaler")
    with c6:
        cv_k = st.slider("CV Folds", 2, 10, 5, key="livre_cv")

    algos = get_algoritmos()
    algo = st.selectbox("Algoritmo", list(algos.keys()),
                        format_func=lambda k: f"{k}  -  {algos[k]}", key="livre_algo")

    section_title("HIPERPARÂMETROS")
    params = render_hiperparametros(algo, prefix="livre")

    if not features:
        aviso_box("Seleciona pelo menos uma feature.")
        return

    if st.button("Treinar", type="primary", key="btn_livre"):
        with st.spinner("A treinar..."):
            try:
                resultado = _executar_treino(
                    df, features, target, algo, params,
                    nulos, scaler_o, test_size, cv_k, username
                )
                if resultado:
                    _render_resultados(resultado, algo, username)
            except Exception as e:
                erro_box(f"Erro: {str(e)}")
                st.exception(e)


# ══════════════════════════════════════════════════════
# LÓGICA DE TREINO
# ══════════════════════════════════════════════════════
def _executar_treino(df, features, target, algo, params,
                     nulos, scaler_o, test_size, cv_k, username):
    df = df[features + [target]].copy()

    # Nulos
    if nulos == "Remover linhas":
        df = df.dropna()
    elif nulos == "Preencher com média":
        df = df.fillna(df.mean(numeric_only=True))
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].fillna(df[c].mode()[0])
    elif nulos == "Preencher com mediana":
        df = df.fillna(df.median(numeric_only=True))
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].fillna(df[c].mode()[0])
    else:
        for c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

    # Encoding categóricas
    encoders = {}
    for c in df[features].select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    # Target
    le_target = None
    if df[target].dtype == object:
        le_target = LabelEncoder()
        df[target] = le_target.fit_transform(df[target].astype(str))

    X = df[features].to_numpy(dtype=float, na_value=np.nan)
    y = df[target].to_numpy()
    classes_orig = le_target.classes_ if le_target else np.unique(y).astype(str)

    # Scaler
    scaler = None
    if scaler_o == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler_o == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) < 20 else None
    )

    model = build_model(algo, params)
    if model is None:
        erro_box(f"Algoritmo {algo} não disponível.")
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=np.unique(y))
                auc = roc_auc_score(y_bin, model.predict_proba(X_test), multi_class="ovr", average="weighted")
        except Exception:
            pass

    # Validação cruzada
    cv = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    # Guardar histórico
    save_historico_modelo(username, {
        "tipo": "classificacao",
        "algoritmo": algo,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "dataset_shape": list(df.shape),
    })
    add_pontos(username, 10, f"Treinou {algo} - accuracy {acc:.1%}")
    save_user_progress(username, "supervisionado_classificacao", {
        "ultimo_algo": algo, "ultima_accuracy": round(acc, 4)
    })

    return {
        "model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "cv_scores": cv_scores, "classes": classes_orig,
        "feature_names": features, "scaler": scaler, "encoders": encoders,
        "le_target": le_target, "X_train": X_train, "y_train": y_train,
        "X": X, "y": y, "df_features": df[features],
        "setup": {"target": str(target), "nulos": nulos, "scaler": scaler_o, "test_size": test_size, "cv_k": cv_k, "n_linhas": int(len(df))},
    }


# ══════════════════════════════════════════════════════
# RESULTADOS
# ══════════════════════════════════════════════════════
def _descrever_para_tutor(r: dict, algo: str) -> tuple:
    """O que o aluno vê nos gráficos de classificação (matriz de confusão, ROC, importância, validação cruzada)."""
    from sklearn.metrics import confusion_matrix as _cmx
    c = r.get("setup") or {}
    cv = r["cv_scores"]
    cm = _cmx(r["y_test"], r["y_pred"])
    classes = [str(x) for x in r["classes"]]
    L = ["Página: Classificação. Algoritmo: " + str(algo) + ". "
         + (f"Alvo a prever: «{c['target']}». Features ({len(r['feature_names'])}): {', '.join(str(f) for f in r['feature_names'][:20])}. "
            f"{c['n_linhas']} linhas; {c['test_size']:.0%} para teste; nulos: {c['nulos']}; escala: {c['scaler']}; validação cruzada com {c['cv_k']} partes (folds)." if c else ""),
         f"Classes: {', '.join(classes[:12])}.",
         f"GRÁFICO «Validação Cruzada por Fold»: barras, uma por fold, com a accuracy em cada parte dos dados ({', '.join(f'{v:.3f}' for v in cv)}); "
         f"linha tracejada verde = média {cv.mean():.3f}. Barras parecidas = modelo estável.",
         f"ABA «Matriz de Confusão»: linhas = «Real», colunas = «Previsto»; cada célula conta casos; diagonal = acertos ({int(np.trace(cm))} de {int(cm.sum())}). "
         + (f"Classes por ordem: {', '.join(classes[:cm.shape[0]])}. Matriz: {cm.tolist()}." if cm.shape[0] <= 10 else ""),
         "ABA «Curva ROC»: eixo horizontal «Taxa de Falsos Positivos», vertical «Taxa de Verdadeiros Positivos»; a diagonal tracejada é o palpite ao acaso; "
         "quanto mais a curva sobe para o canto superior esquerdo, melhor (AUC mais perto de 1)." + (f" AUC = {r['auc']:.3f}." if r.get("auc") is not None else "")]
    m = r["model"]
    imp = getattr(m, "feature_importances_", None)
    if imp is None and hasattr(m, "coef_"):
        imp = np.abs(m.coef_[0]) if m.coef_.ndim > 1 else np.abs(m.coef_)
    if imp is not None:
        o = np.argsort(imp)[::-1][:5]
        L.append("ABA «Importância»: barras horizontais «Importância das Features (Top 15)»; as que mais pesaram: "
                 + ", ".join(f"{r['feature_names'][i]} ({float(imp[i]):.3f})" for i in o) + ".")
    L.append("ABA «Relatório»: tabela de texto com precision, recall, f1-score e número de casos (support) por classe.")
    return L, ("Vê 4-5 caixas de métricas, as barras da validação cruzada por fold e abas com matriz de confusão, curva ROC, importância das features e relatório.")


def _render_resultados(r: dict, algo: str, username: str):
    st.markdown("---")
    section_title("RESULTADOS DO TREINO")

    # KPIs
    kpis = [
        {"label": "Accuracy",  "valor": f"{r['acc']:.1%}",  "delta": None},
        {"label": "Precision", "valor": f"{r['prec']:.1%}", "delta": None},
        {"label": "Recall",    "valor": f"{r['rec']:.1%}",  "delta": None},
        {"label": "F1-Score",  "valor": f"{r['f1']:.1%}",   "delta": None},
    ]
    if r["auc"] is not None:
        kpis.append({"label": "ROC-AUC", "valor": f"{r['auc']:.3f}", "delta": None})

    cols = st.columns(len(kpis))
    for col, k in zip(cols, kpis):
        with col:
            st.metric(k["label"], k["valor"], help=AJUDA_METRICAS.get(k["label"]))

    # Diagnóstico automático + explicação em linguagem simples
    _diagnostico_automatico(r)
    explicar_classificacao(r)

    # Validação Cruzada
    section_title("VALIDAÇÃO CRUZADA")
    cv = r["cv_scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Média CV", f"{cv.mean():.3f}", help=AJUDA_METRICAS["Média CV"])
    c2.metric("Desvio Padrão", f"±{cv.std():.3f}", help=AJUDA_METRICAS["Desvio Padrão"])
    c3.metric("Min → Max", f"{cv.min():.3f} → {cv.max():.3f}")
    st.pyplot(plot_cv_scores(cv))

    # Visualizações
    section_title("VISUALIZAÇÕES")
    t1, t2, t3, t4 = st.tabs(["Matriz de Confusão", "Curva ROC", "Importância", "Relatório"])

    with t1:
        st.pyplot(plot_confusion_matrix(r["y_test"], r["y_pred"], r["classes"]))

    with t2:
        if hasattr(r["model"], "predict_proba"):
            st.pyplot(plot_roc_curve(r["model"], r["X_test"], r["y_test"], np.unique(r["y_test"])))
        else:
            info_box("Este algoritmo não suporta probabilidades - curva ROC não disponível.")

    with t3:
        fig_imp = plot_feature_importance(r["model"], r["feature_names"], algo)
        if fig_imp:
            st.pyplot(fig_imp)
        else:
            info_box("Este algoritmo não expõe importância de features directamente.")

    with t4:
        report = classification_report(r["y_test"], r["y_pred"],
                                       target_names=r["classes"], output_dict=False)
        st.code(report, language="text")

    try:    # o tutor de IA precisa de saber o que está nos gráficos
        from modules.tutor_ia import acrescentar_ecra
        _l, _e = _descrever_para_tutor(r, algo)
        acrescentar_ecra("Resultado de classificação", _l, _e)
    except Exception:
        pass

    # Previsão interactiva
    section_title("PREVISÃO INTERACTIVA")
    _previsao_interactiva(r)

    # Exportar modelo
    section_title("EXPORTAR MODELO")
    buf = io.BytesIO()
    joblib.dump({"model": r["model"], "scaler": r["scaler"],
                 "encoders": r["encoders"], "le_target": r["le_target"],
                 "features": r["feature_names"]}, buf)
    buf.seek(0)
    st.download_button(
        "Descarregar modelo (.pkl)",
        data=buf, file_name=f"{algo.replace(' ','_').lower()}_model.pkl",
        mime="application/octet-stream"
    )
    sucesso_box(f"Modelo treinado com sucesso! +10 pontos adicionados ao teu perfil. ")


def _diagnostico_automatico(r: dict):
    acc = r["acc"]
    cv_mean = r["cv_scores"].mean()
    cv_std = r["cv_scores"].std()

    mensagens = []

    if acc > 0.95 and cv_mean < acc - 0.05:
        mensagens.append(("Possível overfitting", f"A accuracy no teste ({acc:.1%}) é muito superior à validação cruzada ({cv_mean:.1%}). Tenta reduzir a complexidade do modelo.", "amber"))
    elif cv_mean < 0.6:
        mensagens.append(("Underfitting detectado", f"Accuracy CV baixa ({cv_mean:.1%}). O modelo pode ser demasiado simples ou os dados têm pouco sinal. Tenta um modelo mais complexo ou feature engineering.", "red"))
    elif cv_std > 0.1:
        mensagens.append(("Alta variância entre folds", f"Desvio padrão CV = {cv_std:.3f}. O modelo é instável. Tenta mais dados ou regularização.", "amber"))
    else:
        mensagens.append(("Modelo saudável", f"Accuracy {acc:.1%} com CV {cv_mean:.1%} ± {cv_std:.3f}. Bom equilíbrio bias-variância!", "green"))

    for titulo, msg, cor in mensagens:
        cor_map = {"green": C_GREEN, "amber": C_AMBER, "red": C_RED}
        bg_map  = {"green": "rgba(46,204,113,.1)", "amber": "rgba(243,156,18,.1)", "red": "rgba(231,76,60,.1)"}
        brd_map = {"green": "rgba(46,204,113,.3)", "amber": "rgba(243,156,18,.3)", "red": "rgba(231,76,60,.3)"}
        st.markdown(f"""<div style="background:{bg_map[cor]};border:1px solid {brd_map[cor]}; border-radius:10px;padding:.8rem 1rem;margin:.6rem 0;"><strong style="color:{cor_map[cor]};">{titulo}</strong><br><span style="font-size:13px;color:{C_TEXT_SEC};">{msg}</span></div>""", unsafe_allow_html=True)


def _previsao_interactiva(r: dict):
    info_box("Preenche os valores abaixo para obter uma previsão com o modelo treinado.")
    feat_names = r["feature_names"]
    df_feat = r["df_features"]

    nova = {}
    cols = st.columns(min(len(feat_names), 4))
    for i, feat in enumerate(feat_names):
        col = cols[i % len(cols)]
        with col:
            if df_feat[feat].dtype in [np.float64, np.int64, float, int]:
                mn = float(df_feat[feat].min())
                mx = float(df_feat[feat].max())
                med = float(df_feat[feat].median())
                nova[feat] = st.number_input(feat, value=med, min_value=mn, max_value=mx, key=f"pred_{feat}")
            else:
                opts = sorted(df_feat[feat].unique().tolist())
                nova[feat] = st.selectbox(feat, opts, key=f"pred_{feat}")

    if st.button("Prever", key="btn_prever"):
        try:
            row = pd.DataFrame([nova])
            for c, le in r["encoders"].items():
                if c in row.columns:
                    row[c] = le.transform(row[c].astype(str))
            X_novo = row[feat_names].to_numpy(dtype=float, na_value=np.nan)
            if r["scaler"]:
                X_novo = r["scaler"].transform(X_novo)
            pred = r["model"].predict(X_novo)[0]
            label = r["le_target"].inverse_transform([pred])[0] if r["le_target"] else str(pred)

            prob_str = ""
            if hasattr(r["model"], "predict_proba"):
                probs = r["model"].predict_proba(X_novo)[0]
                top_idx = np.argsort(probs)[::-1][:3]
                prob_str = " | ".join([
                    f"{r['classes'][i]}: {probs[i]:.1%}" for i in top_idx
                ])

            st.markdown(f"""<div style="background:rgba(79,142,247,.1);border:2px solid {C_ACCENT}; border-radius:12px;padding:1.2rem 1.4rem;text-align:center;margin-top:.8rem;"><div style="font-size:13px;color:{C_TEXT_MUTE};margin-bottom:4px;">Classe prevista</div><div style="font-size:28px;font-weight:800;color:{C_ACCENT};">{label}</div>{"<div style='font-size:12px;color:"+f"{C_TEXT_MUTE}"+";margin-top:6px;'>"+prob_str+"</div>" if prob_str else ""}</div>""", unsafe_allow_html=True)
        except Exception as e:
            erro_box(f"Erro na previsão: {e}")
