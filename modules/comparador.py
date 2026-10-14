"""
DataForge EDU - Comparador de Modelos
Treina até 3 algoritmos lado a lado e compara métricas
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER, C_SURFACE2,
    PALETTE, carregar_dataset_embutido, add_pontos
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

CLF_MODELS = {
    "KNN":                  lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":        lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":        lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression":  lambda: LogisticRegression(max_iter=500, random_state=42),
    "SVM (RBF)":            lambda: SVC(probability=True, random_state=42),
    "Naive Bayes":          lambda: GaussianNB(),
}
if HAS_XGB:
    CLF_MODELS["XGBoost"] = lambda: XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss")
if HAS_LGB:
    CLF_MODELS["LightGBM"] = lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

REG_MODELS = {
    "Linear Regression":    lambda: LinearRegression(),
    "Ridge":                lambda: Ridge(alpha=1.0),
    "KNN Regressor":        lambda: KNeighborsRegressor(n_neighbors=5),
    "Decision Tree":        lambda: DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":        lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVR":                  lambda: SVR(),
}
if HAS_XGB:
    REG_MODELS["XGBoost"] = lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
if HAS_LGB:
    REG_MODELS["LightGBM"] = lambda: LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)


def _bar_comparison(nomes, valores, titulo, metrica):
    cores = [PALETTE[i % len(PALETTE)] for i in range(len(nomes))]
    fig = go.Figure(go.Bar(
        x=nomes, y=valores, marker_color=cores,
        text=[f"{v:.4f}" for v in valores],
        textposition="auto",
        textfont=dict(color="#FFFFFF", size=13, family="Atkinson Hyperlegible")
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title=metrica, color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _radar_chart(nomes, metricas_dict):
    """Radar chart para comparação multidimensional"""
    cats = list(metricas_dict.keys())
    fig = go.Figure()
    for i, nome in enumerate(nomes):
        vals = [metricas_dict[c][i] for c in cats]
        vals.append(vals[0])  # fechar o polígono
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=nome,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fillcolor=f"rgba({','.join(str(int(PALETTE[i%len(PALETTE)].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.15)"
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color="#7A8BA8", gridcolor=C_BORDER),
            angularaxis=dict(color="#FFFFFF")
        ),
        paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
        legend=dict(font=dict(color="#FFFFFF")),
        title=dict(text="Comparação Multidimensional", font=dict(color="#FFFFFF", size=14))
    )
    return fig


def render_comparador(username: str):
    inject_css()
    page_header("Comparador de Modelos",
                "Treina até 3 algoritmos lado a lado e compara resultados", "")

    teoria_box("Porquê comparar?",
        "Não existe um algoritmo que seja sempre o melhor. É como escolher um carro: depende da estrada. "
        "Aqui pões até 3 algoritmos a fazer <b>a mesma prova</b> e vês qual se sai melhor com os teus dados.")

    tipo = st.radio("Tipo de problema", ["Classificação", "Regressão"], horizontal=True, key="cmp_tipo")

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        section_title("Dados")
        fonte = st.radio("Fonte", ["Embutido", "Upload"], horizontal=True, key="cmp_fonte")
        if fonte == "Embutido":
            if tipo == "Classificação":
                opcoes = ["Iris - Classificação de flores", "Vinho - Qualidade",
                          "Cancro - Diagnóstico", "Titanic - Sobrevivência"]
            else:
                opcoes = ["Casas em Luanda (fictício) - Preços", "Diabetes - Progressão", "Boston Housing - Preços"]
            nm = st.selectbox("Dataset", opcoes, key="cmp_ds")
            df, desc, _ = carregar_dataset_embutido(nm)
            if df is not None and desc:
                st.markdown(f'<div style="font-size:13px;color:#D0D8F0;">{desc}</div>', unsafe_allow_html=True)
        else:
            f = st.file_uploader("CSV", type=["csv"], key="cmp_up")
            df = pd.read_csv(f) if f else None

        if df is not None:
            cols = df.columns.tolist()
            target = st.selectbox("Target", cols, index=len(cols)-1, key="cmp_tgt")
            features = st.multiselect("Features", [c for c in cols if c != target],
                                      default=[c for c in cols if c != target][:8],
                                      key="cmp_feats")
            test_size = st.slider("% Teste", 10, 40, 20, key="cmp_ts") / 100
            cv_folds  = st.slider("Cross-validation (folds)", 3, 10, 5, key="cmp_cv")

            section_title("Algoritmos (escolhe até 3)")
            st.caption("Dica: escolhe algoritmos de famílias diferentes (ex.: KNN, Árvore e Random Forest) para ver diferenças reais.")
            models_dict = CLF_MODELS if tipo == "Classificação" else REG_MODELS
            algos = st.multiselect("Algoritmos", list(models_dict.keys()),
                                   default=list(models_dict.keys())[:3],
                                   max_selections=3, key="cmp_algos")

            if st.button("Comparar Modelos", width='stretch', key="cmp_run"):
                if not features or not algos:
                    aviso_box("Selecciona features e pelo menos 1 algoritmo.")
                else:
                    _run_comparison(df, features, target, algos, models_dict,
                                    tipo, test_size, cv_folds, username)

    with col_res:
        _show_comparison_result(tipo)


def _run_comparison(df, features, target, algos, models_dict, tipo, test_size, cv_folds, username):
    try:
        X = df[features].copy()
        y = df[target].copy()

        for col in X.select_dtypes(include=["object","category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))

        if tipo == "Classificação":
            if y.dtype == object or str(y.dtype) == "category":
                y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = pd.to_numeric(y, errors="coerce").fillna(y.median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        X_sc       = scaler.transform(X)

        resultados = []
        progress   = st.progress(0)

        for i, algo_name in enumerate(algos):
            with st.spinner(f"A treinar {algo_name}..."):
                t0 = time.perf_counter()
                m = models_dict[algo_name]()
                m.fit(X_train_sc, y_train)
                tempo = time.perf_counter() - t0
                y_pred = m.predict(X_test_sc)

                if tipo == "Classificação":
                    acc  = accuracy_score(y_test, y_pred)
                    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    cv   = cross_val_score(models_dict[algo_name](), X_sc, y,
                                          cv=cv_folds, scoring="accuracy").mean()
                    resultados.append({
                        "Algoritmo": algo_name,
                        "Accuracy":  acc,
                        "F1 Score":  f1,
                        "CV Accuracy": cv,
                        "Tempo (s)": tempo
                    })
                else:
                    r2   = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    cv   = cross_val_score(models_dict[algo_name](), X_sc, y,
                                          cv=cv_folds, scoring="r2").mean()
                    resultados.append({
                        "Algoritmo": algo_name,
                        "R²":  r2,
                        "RMSE": rmse,
                        "CV R²": cv,
                        "Tempo (s)": tempo
                    })
            progress.progress((i + 1) / len(algos))

        st.session_state.cmp_result = {"resultados": resultados, "tipo": tipo}
        add_pontos(username, 20, f"Comparação de {len(algos)} modelos")
        sucesso_box(f"Comparação concluída! {len(algos)} modelos treinados.")
    except Exception as e:
        erro_box(f"Erro: {e}")


def _show_comparison_result(tipo):
    res = st.session_state.get("cmp_result", {})
    if not res or res.get("tipo") != tipo:
        st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed #3A4560;border-radius:16px;margin-top:1rem;"><div style="font-size:40px;margin-bottom:1rem;"></div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Selecciona algoritmos e clica em <strong>Comparar Modelos</strong></div></div>""", unsafe_allow_html=True)
        return

    resultados = res["resultados"]
    df_res = pd.DataFrame(resultados).set_index("Algoritmo")
    nomes  = list(df_res.index)

    section_title("Tabela de Resultados")

    col_cfg = {
        "Accuracy": st.column_config.NumberColumn("Accuracy", help="Percentagem de acertos na prova (teste)."),
        "F1 Score": st.column_config.NumberColumn("F1 Score", help="Nota que junta Precision e Recall."),
        "CV Accuracy": st.column_config.NumberColumn("CV Accuracy", help="Média de várias provas diferentes. Mais fiável."),
        "R²": st.column_config.NumberColumn("R²", help="Quanto da variação dos dados o modelo explica (1 = perfeito)."),
        "RMSE": st.column_config.NumberColumn("RMSE", help="Erro típico, na unidade do alvo. Menor é melhor."),
        "CV R²": st.column_config.NumberColumn("CV R²", help="R² médio em várias provas. Mais fiável."),
        "Tempo (s)": st.column_config.NumberColumn("Tempo (s)", help="Tempo que demorou a treinar."),
    }
    st.dataframe(df_res.style.format("{:.4f}"), width='stretch', column_config=col_cfg)

    _veredicto(df_res, tipo)

    # Melhor modelo
    if tipo == "Classificação":
        melhor_idx = df_res["Accuracy"].idxmax()
        melhor_val = df_res.loc[melhor_idx, "Accuracy"]

        # Gráficos
        st.plotly_chart(_bar_comparison(nomes, df_res["Accuracy"].tolist(),
                                         "Accuracy - Conjunto de Teste", "Accuracy"),
                         width='stretch')
        st.plotly_chart(_bar_comparison(nomes, df_res["CV Accuracy"].tolist(),
                                         f"Accuracy - Cross-Validation", "CV Accuracy"),
                         width='stretch')

        if len(nomes) >= 2:
            metricas_radar = {
                "Accuracy":    df_res["Accuracy"].tolist(),
                "F1 Score":    df_res["F1 Score"].tolist(),
                "CV Accuracy": df_res["CV Accuracy"].tolist(),
            }
            st.plotly_chart(_radar_chart(nomes, metricas_radar), width='stretch')
    else:
        melhor_idx = df_res["R²"].idxmax()
        melhor_val = df_res.loc[melhor_idx, "R²"]

        st.plotly_chart(_bar_comparison(nomes, df_res["R²"].tolist(),
                                         "R² - Conjunto de Teste", "R²"),
                         width='stretch')
        st.plotly_chart(_bar_comparison(nomes, df_res["RMSE"].tolist(),
                                         "RMSE - quanto menor melhor", "RMSE"),
                         width='stretch')

    teoria_box("Como decidir?",
        "Olha primeiro para a coluna <b>CV</b>: é a média de várias provas, por isso é mais honesta do que uma só nota. "
        "Se dois modelos estiverem quase empatados, escolhe o <b>mais simples e mais rápido</b>: é mais fácil de explicar e de pôr a funcionar.")


def _veredicto(df_res, tipo):
    """Resumo em linguagem simples: quem ganhou, se houve empate e qual o mais rápido."""
    col_cv = "CV Accuracy" if tipo == "Classificação" else "CV R²"
    col_teste = "Accuracy" if tipo == "Classificação" else "R²"
    fmt = (lambda v: f"{v:.1%}") if tipo == "Classificação" else (lambda v: f"{v:.2f}")
    ordem = df_res.sort_values(col_cv, ascending=False)
    campeao, segundo = ordem.index[0], (ordem.index[1] if len(ordem) > 1 else None)
    rapido = df_res["Tempo (s)"].idxmin()
    dif = (ordem[col_cv].iloc[0] - ordem[col_cv].iloc[1]) if segundo else 1.0

    if segundo is None:
        msg = f"<b>{campeao}</b> teve {fmt(ordem[col_cv].iloc[0])} nas provas. Escolhe mais algoritmos para comparar."
    elif dif < 0.01:
        msg = (f"<b>Empate técnico.</b> {campeao} e {segundo} ficaram praticamente iguais ({fmt(ordem[col_cv].iloc[0])}). "
               f"Quando é assim, escolhe o mais simples ou o mais rápido (<b>{rapido}</b>).")
    else:
        msg = (f"<b>{campeao}</b> ganhou com {fmt(ordem[col_cv].iloc[0])} nas provas, à frente de {segundo} ({fmt(ordem[col_cv].iloc[1])}). "
               f"O mais rápido a treinar foi <b>{rapido}</b>.")
    sucesso_box(msg)

    if tipo == "Classificação" and (df_res[col_teste] >= 0.999).all():
        st.info("Todos tiveram 100% na prova. Isto quer dizer que os dados são **fáceis demais** para distinguir os algoritmos "
                "(acontece muito com o Iris). Experimenta o Vinho ou o Titanic para ver diferenças a sério.")
    gap = (df_res[col_teste] - df_res[col_cv]).max()
    if gap > 0.08:
        st.warning("Um dos modelos tem nota bem mais alta na prova do que nas provas repetidas. Pode ter sido sorte: confia mais na coluna CV.")
