"""
DataForge EDU - Detecção de Anomalias
Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_RED, C_GREEN, C_SURFACE, C_BORDER,
    PALETTE, carregar_dataset_embutido, add_pontos
)

ALGORITMOS = {
    "Isolation Forest":   "iforest",
    "Local Outlier Factor (LOF)": "lof",
    "One-Class SVM":      "ocsvm",
    "Elliptic Envelope":  "elliptic",
}


def render_anomalias(username: str):
    inject_css()
    page_header("Detecção de Anomalias",
                "Identifica pontos que se desviam do padrão normal", "")

    teoria_box("O que são anomalias?",
        "Anomalias (outliers) são observações que se desviam significativamente dos dados normais. "
        "Exemplos: fraude bancária, falhas em equipamentos, doenças raras, intrusões em redes. "
        "Os algoritmos aprendem o padrão 'normal' e marcam o que foge a esse padrão.")

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        section_title("Configuração")

        fonte = st.radio("Dados", ["Embutido", "Upload CSV"], horizontal=True, key="an_fonte")
        if fonte == "Embutido":
            opcoes = ["Iris - Classificação de flores", "Vinho - Qualidade",
                      "Diabetes - Progressão", "Blobs - Clustering básico"]
            nm = st.selectbox("Dataset", opcoes, key="an_ds")
            df, _, _ = carregar_dataset_embutido(nm)
        else:
            f = st.file_uploader("CSV", type=["csv"], key="an_up")
            df = pd.read_csv(f) if f else None

        if df is not None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = st.multiselect("Features", num_cols,
                                      default=num_cols[:min(4, len(num_cols))], key="an_feats")
            algo_label = st.selectbox("Algoritmo", list(ALGORITMOS.keys()), key="an_algo")
            algo = ALGORITMOS[algo_label]
            normalizar = st.checkbox("Normalizar", value=True, key="an_norm")

            contaminacao = st.slider("Contaminação esperada (%)",
                                     1, 30, 5, key="an_cont") / 100

            params = {"contamination": contaminacao}
            if algo == "iforest":
                params["n_estimators"] = st.slider("n_estimators", 50, 300, 100, 50, key="an_ne")
            if algo == "lof":
                params["n_neighbors"] = st.slider("n_neighbors", 5, 50, 20, key="an_nn")
            if algo == "ocsvm":
                params["nu"]     = contaminacao
                params["kernel"] = st.selectbox("Kernel", ["rbf","linear","poly"], key="an_ker")

            if st.button("Detectar Anomalias", width='stretch', key="an_run"):
                if not features:
                    aviso_box("Selecciona pelo menos 1 feature.")
                else:
                    X = df[features].dropna().to_numpy(dtype=float, na_value=0)
                    X_sc = StandardScaler().fit_transform(X) if normalizar else X
                    with st.spinner("A detectar..."):
                        try:
                            if algo == "iforest":
                                m = IsolationForest(contamination=contaminacao,
                                                    n_estimators=params.get("n_estimators",100),
                                                    random_state=42)
                            elif algo == "lof":
                                m = LocalOutlierFactor(n_neighbors=params.get("n_neighbors",20),
                                                       contamination=contaminacao)
                            elif algo == "ocsvm":
                                m = OneClassSVM(nu=contaminacao, kernel=params.get("kernel","rbf"))
                            elif algo == "elliptic":
                                m = EllipticEnvelope(contamination=contaminacao, random_state=42)

                            if algo == "lof":
                                preds = m.fit_predict(X_sc)
                            else:
                                preds = m.fit(X_sc).predict(X_sc)

                            # -1 = anomalia, 1 = normal
                            labels = np.where(preds == -1, "Anomalia", "Normal")
                            n_anom = int(np.sum(preds == -1))
                            n_norm = int(np.sum(preds == 1))

                            pca_v = PCA(n_components=2, random_state=42)
                            X_pca = pca_v.fit_transform(X_sc)

                            st.session_state.an_result = {
                                "labels": labels, "preds": preds,
                                "X_pca": X_pca, "n_anom": n_anom, "n_norm": n_norm,
                                "algo": algo_label, "df_idx": df[features].dropna().index
                            }
                            add_pontos(username, 8, f"Anomalias {algo_label}")
                            sucesso_box(f"{n_anom} anomalias detectadas ({n_anom/(n_anom+n_norm)*100:.1f}%)")
                        except Exception as e:
                            erro_box(f"Erro: {e}")

    with col_res:
        res = st.session_state.get("an_result", {})
        if res:
            labels  = res["labels"]
            X_pca   = res["X_pca"]
            n_anom  = res["n_anom"]
            n_norm  = res["n_norm"]
            total   = n_anom + n_norm

            c1, c2, c3 = st.columns(3)
            c1.metric("Normais",   n_norm, f"{n_norm/total*100:.1f}%")
            c2.metric("Anomalias", n_anom, f"{n_anom/total*100:.1f}%", delta_color="inverse")
            c3.metric("Total",     total)

            # Scatter
            fig = go.Figure()
            for lbl, cor, sz in [("Normal", C_ACCENT, 7), ("Anomalia", C_RED, 10)]:
                mask = labels == lbl
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0], y=X_pca[mask, 1],
                    mode="markers", name=lbl,
                    marker=dict(color=cor, size=sz, opacity=0.85,
                                line=dict(color="#FFFFFF" if lbl=="Anomalia" else "rgba(0,0,0,0)", width=1.5))
                ))
            fig.update_layout(
                title=dict(text=f"{res['algo']} - Anomalias (PCA 2D)", font=dict(color="#FFFFFF", size=15)),
                xaxis=dict(title="PC1", color="#FFFFFF", gridcolor=C_BORDER),
                yaxis=dict(title="PC2", color="#FFFFFF", gridcolor=C_BORDER),
                plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
            )
            st.plotly_chart(fig, width='stretch')

            # Pizza
            fig_pie = go.Figure(go.Pie(
                labels=["Normal", "Anomalia"],
                values=[n_norm, n_anom],
                marker=dict(colors=[C_ACCENT, C_RED]),
                textfont=dict(color="#FFFFFF", size=14),
                hole=0.4
            ))
            fig_pie.update_layout(
                paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
                legend=dict(font=dict(color="#FFFFFF")),
                title=dict(text="Proporção", font=dict(color="#FFFFFF"))
            )
            st.plotly_chart(fig_pie, width='stretch')

            teoria_box("Como interpretar",
                "Pontos <strong style='color:#FF6B6B;'>vermelhos</strong> são anomalias - afastam-se do padrão aprendido. "
                "Ajusta a <strong>contaminação</strong> conforme a percentagem real de outliers que esperas. "
                "Usa <strong>PCA 2D</strong> apenas para visualização - o modelo foi treinado no espaço original.")
        else:
            st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:40px;margin-bottom:1rem;"></div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura e clica em <strong>Detectar Anomalias</strong></div></div>""", unsafe_allow_html=True)
