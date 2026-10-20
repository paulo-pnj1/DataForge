"""
DataForge EDU - Módulo Supervisionado: Regressão
Linear, Ridge, Lasso, ElasticNet, SVR, KNN, RF, GB + teoria
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, BayesianRidge, SGDRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
import joblib, io

from modules.icons import icon_html
from modules.utils import (
    page_header, section_title, teoria_box, info_box,
    sucesso_box, aviso_box, erro_box, C_TEXT, C_TEXT_SEC,
    C_TEXT_MUTE, C_SURFACE, C_BORDER, C_ACCENT, C_GREEN,
    C_AMBER, C_RED, C_TEAL, PALETTE, TEORIA,
    carregar_dataset_embutido, save_user_progress,
    add_pontos, save_historico_modelo
)
from modules.diagnostico import mostrar_diagnostico
from modules.explicacoes import explicar_regressao, AJUDA_METRICAS

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


TEORIA_REG = {
    "Linear Regression": TEORIA.get("Linear Regression", {}),
    "Ridge": TEORIA.get("Ridge", {}),
    "Lasso": TEORIA.get("Lasso", {}),
    "ElasticNet": {
        "nome": "ElasticNet",
        "analogia": "Combinação de Ridge e Lasso. Regularização mista L1+L2. O melhor dos dois mundos quando não sabes qual usar.",
        "como_funciona": "Penalização = alpha * (l1_ratio * |coef| + (1-l1_ratio) * coef²). l1_ratio=1 é Lasso puro, l1_ratio=0 é Ridge puro.",
        "quando_usar": "Muitas features, algumas correlacionadas. Quando não tens certeza entre Ridge e Lasso.",
        "cuidados": "Dois hiperparâmetros a tunear (alpha e l1_ratio).",
    },
    "SVR": {
        "nome": "Support Vector Regression",
        "analogia": "Como o SVM mas para regressão. Em vez de maximizar margem de separação, tenta encaixar a maior quantidade de pontos numa 'faixa' epsilon ao redor da linha.",
        "como_funciona": "Encontra o hiperplano que tem no máximo epsilon de desvio em relação a todos os pontos de treino, enquanto é o mais plano possível.",
        "quando_usar": "Datasets pequenos a médios. Alta dimensionalidade. Relações não-lineares com kernel rbf.",
        "cuidados": "Lento em datasets grandes. Requer normalização obrigatória.",
    },
    "Random Forest Regressor": {
        "nome": "Random Forest Regressor",
        "analogia": "Mesmo conceito da classificação. Ensemble de árvores de regressão. A previsão final é a média das previsões individuais.",
        "como_funciona": "Treina N árvores em subconjuntos aleatórios dos dados. Cada árvore prevê um valor; a média é o resultado final.",
        "quando_usar": "Uso geral. Robusto e eficaz. Boa alternativa a Gradient Boosting quando velocidade é importante.",
        "cuidados": "Pode fazer médias que subestimam extremos.",
    },
}


def get_algoritmos_reg():
    algos = {
        "Linear Regression": "Regressão Linear",
        "Ridge": "Ridge (L2)",
        "Lasso": "Lasso (L1)",
        "ElasticNet": "ElasticNet (L1+L2)",
        "Huber": "Huber Regression",
        "Bayesian Ridge": "Bayesian Ridge",
        "KNN Regressor": "KNN Regressor",
        "Decision Tree Regressor": "Árvore de Decisão",
        "Random Forest Regressor": "Random Forest",
        "Extra Trees Regressor": "Extra Trees",
        "Gradient Boosting Regressor": "Gradient Boosting",
        "AdaBoost Regressor": "AdaBoost",
        "SVR": "Support Vector Regression",
        "SGD Regressor": "SGD Regressor",
    }
    if XGB_OK:
        algos["XGBoost Regressor"] = "XGBoost"
    if LGB_OK:
        algos["LightGBM Regressor"] = "LightGBM"
    return algos


def build_model_reg(algo: str, params: dict):
    m = {
        "Linear Regression": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "Huber": HuberRegressor,
        "Bayesian Ridge": BayesianRidge,
        "KNN Regressor": KNeighborsRegressor,
        "Decision Tree Regressor": lambda **p: DecisionTreeRegressor(**p, random_state=42),
        "Random Forest Regressor": lambda **p: RandomForestRegressor(**p, random_state=42, n_jobs=-1),
        "Extra Trees Regressor": lambda **p: ExtraTreesRegressor(**p, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": lambda **p: GradientBoostingRegressor(**p, random_state=42),
        "AdaBoost Regressor": lambda **p: AdaBoostRegressor(**p, random_state=42),
        "SVR": SVR,
        "SGD Regressor": lambda **p: SGDRegressor(**p, random_state=42),
    }
    if XGB_OK:
        m["XGBoost Regressor"] = lambda **p: xgb.XGBRegressor(**p, random_state=42, verbosity=0)
    if LGB_OK:
        m["LightGBM Regressor"] = lambda **p: lgb.LGBMRegressor(**p, random_state=42, verbose=-1)

    factory = m.get(algo)
    if factory is None:
        return None
    try:
        return factory(**params)
    except Exception:
        return factory()


def render_hiperparametros_reg(algo: str) -> dict:
    params = {}
    c1, c2 = st.columns(2)

    if algo in ("Ridge", "Lasso"):
        with c1:
            params["alpha"] = st.select_slider("Alpha",
                options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], value=1.0)
    elif algo == "ElasticNet":
        with c1:
            params["alpha"] = st.select_slider("Alpha",
                options=[0.001, 0.01, 0.1, 1.0, 10.0], value=1.0)
        with c2:
            params["l1_ratio"] = st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.05)
    elif algo == "KNN Regressor":
        with c1:
            params["n_neighbors"] = st.slider("K", 1, 30, 5)
            params["weights"] = st.selectbox("Pesos", ["uniform", "distance"])
        with c2:
            params["metric"] = st.selectbox("Métrica", ["euclidean", "manhattan"])
    elif algo in ("Decision Tree Regressor",):
        with c1:
            md = st.slider("max_depth", 1, 30, 5)
            params["max_depth"] = md if md < 30 else None
        with c2:
            params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
    elif algo in ("Random Forest Regressor", "Extra Trees Regressor"):
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10)
            md = st.slider("max_depth", 1, 30, 10)
            params["max_depth"] = md if md < 30 else None
        with c2:
            params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
    elif algo == "Gradient Boosting Regressor":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10)
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
        with c2:
            params["max_depth"] = st.slider("max_depth", 1, 10, 3)
            params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    elif algo == "SVR":
        with c1:
            params["C"] = st.select_slider("C", options=[0.01, 0.1, 1, 10, 100], value=1)
            params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        with c2:
            params["epsilon"] = st.select_slider("Epsilon", options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)
    elif algo == "XGBoost Regressor":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10)
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
        with c2:
            params["max_depth"] = st.slider("max_depth", 1, 12, 6)
            params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    elif algo == "LightGBM Regressor":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, 10)
            params["learning_rate"] = st.select_slider("learning_rate",
                options=[0.001, 0.01, 0.05, 0.1, 0.2], value=0.05)
        with c2:
            params["num_leaves"] = st.slider("num_leaves", 10, 200, 31)

    return params


def _plot_real_vs_pred(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#161B27")
    ax.set_facecolor("#161B27")
    ax.scatter(y_test, y_pred, alpha=0.6, color=C_ACCENT, edgecolors="#2A3347", s=40)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "--", color=C_GREEN, lw=1.5, label="Perfeito")
    ax.set_xlabel("Valor Real", color="#9BA3B2")
    ax.set_ylabel("Valor Previsto", color="#9BA3B2")
    ax.set_title("Real vs Previsto", color="#E8EBF0")
    ax.legend(facecolor="#1E2535", edgecolor="#2A3347", labelcolor="#9BA3B2")
    ax.tick_params(colors="#9BA3B2")
    for spine in ax.spines.values(): spine.set_edgecolor("#2A3347")
    fig.tight_layout()
    return fig


def _plot_residuos(y_test, y_pred):
    residuos = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#161B27")
    for ax in axes: ax.set_facecolor("#161B27")

    axes[0].scatter(y_pred, residuos, alpha=0.6, color=C_TEAL, edgecolors="#2A3347", s=30)
    axes[0].axhline(0, color=C_AMBER, linestyle="--", lw=1.5)
    axes[0].set_xlabel("Valor Previsto", color="#9BA3B2")
    axes[0].set_ylabel("Resíduo", color="#9BA3B2")
    axes[0].set_title("Resíduos vs Previsto", color="#E8EBF0")

    axes[1].hist(residuos, bins=30, color=C_ACCENT, edgecolor="#2A3347", alpha=0.85)
    axes[1].set_xlabel("Resíduo", color="#9BA3B2")
    axes[1].set_ylabel("Frequência", color="#9BA3B2")
    axes[1].set_title("Distribuição dos Resíduos", color="#E8EBF0")

    for ax in axes:
        ax.tick_params(colors="#9BA3B2")
        for spine in ax.spines.values(): spine.set_edgecolor("#2A3347")
    fig.tight_layout()
    return fig


def _plot_cv_scores_reg(scores):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#161B27")
    ax.set_facecolor("#161B27")
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    bars = ax.bar(folds, scores, color=C_ACCENT, alpha=0.85, edgecolor="#2A3347")
    ax.axhline(scores.mean(), color=C_GREEN, linestyle="--", lw=1.5, label=f"Média R²: {scores.mean():.3f}")
    ax.set_ylabel("R²", color="#9BA3B2")
    ax.set_title("Validação Cruzada", color="#E8EBF0")
    ax.legend(facecolor="#1E2535", edgecolor="#2A3347", labelcolor="#9BA3B2")
    ax.tick_params(colors="#9BA3B2")
    for spine in ax.spines.values(): spine.set_edgecolor("#2A3347")
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", color="#9BA3B2", fontsize=9)
    fig.tight_layout()
    return fig


def render_regressao(username: str):
    page_header(
        "Regressão",
        "Aprendizagem supervisionada - prever valores contínuos",
        ""
    )

    algos = get_algoritmos_reg()

    # Dados
    catalogo_reg = [
        "Casas em Luanda (fictício) - Preços",
        "Banca de rua (fictício) - Vendas",
        "Boston Housing - Preços",
        "Diabetes - Progressão",
    ]
    fonte = st.radio("Fonte dos dados", ["Dataset embutido", "Upload CSV"], horizontal=True, key="reg_fonte")
    df = None

    if fonte == "Dataset embutido":
        escolha = st.selectbox("Dataset", catalogo_reg, key="reg_ds")
        df, desc, _ = carregar_dataset_embutido(escolha)
        if desc:
            info_box(desc)
    else:
        f = st.file_uploader("CSV", type=["csv"], key="reg_upload")
        if f:
            try:
                df = pd.read_csv(f)
                sucesso_box(f"{df.shape[0]} linhas × {df.shape[1]} colunas")
                mostrar_diagnostico(df, expandido=False)
            except Exception as e:
                erro_box(str(e))

    if df is None:
        return

    with st.expander("Pré-visualização"):
        st.dataframe(df.head(8), width='stretch')

    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.selectbox("Target (coluna a prever)", cols, index=len(cols)-1, key="reg_target")
    with c2:
        features = st.multiselect("Features", [c for c in cols if c != target],
                                   default=[c for c in cols if c != target], key="reg_feat")
    with c3:
        test_size = st.slider("Teste %", 0.1, 0.5, 0.2, 0.05, key="reg_ts")

    c4, c5, c6 = st.columns(3)
    with c4:
        nulos = st.selectbox("Nulos", ["Remover linhas", "Preencher com média", "Preencher com mediana"], key="reg_nulos")
    with c5:
        scaler_o = st.selectbox("Scaler", ["Nenhuma", "StandardScaler", "MinMaxScaler"], key="reg_scaler")
    with c6:
        cv_k = st.slider("CV Folds", 2, 10, 5, key="reg_cv")

    algo = st.selectbox("Algoritmo", list(algos.keys()),
                        format_func=lambda k: f"{k}  -  {algos[k]}", key="reg_algo")

    # Teoria
    t = TEORIA_REG.get(algo)
    if t and t.get("nome"):
        with st.expander(f"Teoria: {t['nome']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="teoria-box"><h4>Analogia</h4><p>{t.get("analogia","")}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="teoria-box"><h4> Como funciona</h4><p>{t.get("como_funciona","")}</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="teoria-box"><h4>Quando usar</h4><p>{t.get("quando_usar","")}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="teoria-box"><h4>Cuidados</h4><p>{t.get("cuidados","")}</p></div>', unsafe_allow_html=True)

    section_title("HIPERPARÂMETROS")
    params = render_hiperparametros_reg(algo)

    if not features:
        aviso_box("Seleciona pelo menos uma feature.")
        return

    if st.button("Treinar", type="primary", key="btn_reg"):
        with st.spinner("A treinar..."):
            try:
                _treinar_regressao(df, features, target, algo, params,
                                   nulos, scaler_o, test_size, cv_k, username)
            except Exception as e:
                erro_box(f"Erro: {str(e)}")
                st.exception(e)


def _treinar_regressao(df, features, target, algo, params,
                       nulos, scaler_o, test_size, cv_k, username):
    df = df[features + [target]].copy()

    if nulos == "Remover linhas":
        df = df.dropna()
    elif nulos == "Preencher com média":
        df = df.fillna(df.mean(numeric_only=True))
    else:
        df = df.fillna(df.median(numeric_only=True))

    encoders = {}
    for c in df[features].select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    X = df[features].to_numpy(dtype=float, na_value=np.nan)
    y = df[target].to_numpy(dtype=float)

    scaler = None
    if scaler_o == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler_o == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = build_model_reg(algo, params)
    if model is None:
        erro_box(f"Algoritmo {algo} não disponível.")
        return

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    cv = KFold(n_splits=cv_k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)

    # Guardar
    save_historico_modelo(username, {
        "tipo": "regressao", "algoritmo": algo,
        "r2": round(r2, 4), "rmse": round(rmse, 4),
        "dataset_shape": list(df.shape),
    })
    add_pontos(username, 10, f"Treinou regressão {algo} - R²={r2:.3f}")
    save_user_progress(username, "supervisionado_regressao", {"ultimo_algo": algo, "ultimo_r2": round(r2, 4)})

    # Resultados
    section_title("RESULTADOS")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r2:.4f}", help=AJUDA_METRICAS["R²"])
    c2.metric("RMSE", f"{rmse:.4f}", help=AJUDA_METRICAS["RMSE"])
    c3.metric("MAE", f"{mae:.4f}", help=AJUDA_METRICAS["MAE"])
    c4.metric("MSE", f"{mse:.4f}", help=AJUDA_METRICAS["MSE"])

    # Diagnóstico
    if r2 < 0:
        erro_box("R² negativo - o modelo é pior que uma linha horizontal (média). Verifica os dados ou tenta outro algoritmo.")
    elif r2 < 0.4:
        aviso_box(f"R² = {r2:.3f} é baixo. O modelo explica pouca variância. Considera feature engineering ou modelos mais complexos.")
    elif r2 > 0.95:
        aviso_box(f"R² = {r2:.3f} muito alto. Verifica se há data leakage (features que directamente codificam o target).")
    else:
        sucesso_box(f"R² = {r2:.3f} - bom resultado! O modelo explica {r2:.1%} da variância do target.")

    explicar_regressao(y_test, y_pred, r2, rmse, mae, cv_scores, target)

    section_title("VALIDAÇÃO CRUZADA")
    c1, c2 = st.columns(2)
    c1.metric("R² médio (CV)", f"{cv_scores.mean():.4f}")
    c2.metric("Desvio padrão", f"±{cv_scores.std():.4f}")
    st.pyplot(_plot_cv_scores_reg(cv_scores))

    section_title("VISUALIZAÇÕES")
    t1, t2 = st.tabs(["Real vs Previsto", "Resíduos"])
    with t1:
        st.pyplot(_plot_real_vs_pred(y_test, y_pred))
    with t2:
        st.pyplot(_plot_residuos(y_test, y_pred))

    # Exportar
    section_title("EXPORTAR")
    buf = io.BytesIO()
    joblib.dump({"model": model, "scaler": scaler, "encoders": encoders, "features": features}, buf)
    buf.seek(0)
    st.download_button("Descarregar modelo (.pkl)", data=buf,
                       file_name=f"{algo.replace(' ','_').lower()}_reg.pkl",
                       mime="application/octet-stream")
    sucesso_box("Modelo treinado! +10 pontos ")
