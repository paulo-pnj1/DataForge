"""
DataForge EDU - Redução de Dimensionalidade
PCA, Kernel PCA, t-SNE, UMAP, ICA, Factor Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis
from sklearn.manifold import TSNE

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    info_box, aviso_box, sucesso_box, erro_box,
    C_ACCENT, C_GREEN, C_AMBER, C_SURFACE, C_BORDER, C_TEXT_MUTE, C_SURFACE2,
    PALETTE, carregar_dataset_embutido, add_pontos
)

ALGORITMOS = {
    "PCA - Principal Component Analysis":   "pca",
    "Kernel PCA":                           "kpca",
    "t-SNE":                                "tsne",
    "ICA - Independent Component Analysis": "ica",
    "Factor Analysis":                      "fa",
}
try:
    import umap
    ALGORITMOS["UMAP"] = "umap"
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def _scatter2d(X2, labels_raw, title, label_col="Classe"):
    if labels_raw is not None:
        labs = [str(l) for l in labels_raw]
        unique_labs = sorted(set(labs))
        fig = go.Figure()
        for i, lbl in enumerate(unique_labs):
            mask = [l == lbl for l in labs]
            fig.add_trace(go.Scatter(
                x=X2[mask, 0], y=X2[mask, 1],
                mode="markers", name=lbl,
                marker=dict(color=PALETTE[i % len(PALETTE)], size=7, opacity=0.8,
                            line=dict(color="#FFFFFF", width=0.4))
            ))
    else:
        fig = go.Figure(go.Scatter(
            x=X2[:, 0], y=X2[:, 1], mode="markers",
            marker=dict(color=C_ACCENT, size=6, opacity=0.7)
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(title="Componente 1", color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title="Componente 2", color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig


def _scatter3d(X3, labels_raw, title):
    labs = [str(l) for l in labels_raw] if labels_raw is not None else ["Dados"] * len(X3)
    unique_labs = sorted(set(labs))
    fig = go.Figure()
    for i, lbl in enumerate(unique_labs):
        mask = [l == lbl for l in labs]
        fig.add_trace(go.Scatter3d(
            x=X3[mask, 0], y=X3[mask, 1], z=X3[mask, 2],
            mode="markers", name=lbl,
            marker=dict(color=PALETTE[i % len(PALETTE)], size=4, opacity=0.8)
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=14)),
        paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
        legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig


def _variance_plot(pca_model):
    var_exp = pca_model.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)
    n = len(var_exp)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, n+1)), y=var_exp,
                         name="Variância por PC", marker_color=C_ACCENT,
                         text=[f"{v:.1f}%" for v in var_exp], textposition="auto",
                         textfont=dict(color="#FFFFFF")))
    fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=cum_var,
                             name="Variância Acumulada", mode="lines+markers",
                             line=dict(color=C_GREEN, width=2),
                             marker=dict(size=6, color=C_GREEN)))
    fig.update_layout(
        title=dict(text="Variância Explicada por Componente", font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(title="Componente Principal", color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title="Variância (%)", color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig


def _grupos_automaticos(X2):
    """Quantos grupos visíveis há na projecção (KMeans + silhouette). Só serve de pista para o tutor."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        X2 = np.asarray(X2)[:, :2]
        if len(X2) < 20:
            return ""
        melhor = None
        for k in (2, 3, 4):
            lab = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(X2)
            sc = silhouette_score(X2, lab, sample_size=min(1500, len(X2)), random_state=42)
            if melhor is None or sc > melhor[1]:
                melhor = (k, sc)
        k, sc = melhor
        forca = "separados com clareza" if sc >= 0.6 else "algo separados" if sc >= 0.4 else "pouco separados / misturados"
        return f"Análise automática: o mais natural são {k} grupos de pontos (silhouette {sc:.2f}: {forca})."
    except Exception:
        return ""


def _descrever_para_tutor(res: dict) -> tuple:
    """Descreve em texto o que o aluno VÊ (gráficos, eixos, features, números) para o tutor de IA.
    Só leva nomes de colunas do dataset e números agregados: nunca linhas de dados."""
    X = np.asarray(res["X_red"])
    algo, n_comp = res["algo"], res["n_comp"]
    feats = res.get("features") or []
    classes = res.get("classes") or []
    L = [f"Página: Redução de Dimensionalidade. Algoritmo escolhido: {algo}.",
         f"Dataset: {res.get('dataset', '?')} ({len(X)} linhas). Features usadas ({len(feats)}): {', '.join(feats)}. "
         f"Normalizado (StandardScaler): {'sim' if res.get('normalizar') else 'não'}.",
         f"Os {len(feats)} números de cada linha foram resumidos em {n_comp} números novos (Componentes), para caber num gráfico."]
    if res.get("label_col"):
        L.append(f"Cor dos pontos = coluna «{res['label_col']}» (classes: {', '.join(classes[:10])}).")
    else:
        L.append("Os pontos são todos da mesma cor, sem legenda: o aluno não escolheu coluna de cor/classe.")
    L.append(f"GRÁFICO 1 «{algo} - Projecção {n_comp}D»: gráfico de pontos. Cada ponto é UMA linha do dataset (ex.: uma flor). "
             "Eixo horizontal = «Componente 1», eixo vertical = «Componente 2». Estes eixos NÃO são colunas originais: "
             "são eixos novos criados pelo algoritmo. Pontos próximos = linhas parecidas; pontos afastados = diferentes.")
    for i in range(min(n_comp, 3)):
        L.append(f"Componente {i + 1} vai de {X[:, i].min():.2f} a {X[:, i].max():.2f}.")
    if n_comp == 2:
        g = _grupos_automaticos(X)
        if g:
            L.append(g)
    ratio = res.get("var_ratio")
    if ratio is not None:
        r = [float(v) * 100 for v in ratio]
        L.append("GRÁFICO 2 «Variância Explicada por Componente»: barras azuis = quanta da informação (variância) dos dados cada "
                 "componente guarda: " + ", ".join(f"Componente {i + 1} = {v:.1f}%" for i, v in enumerate(r))
                 + f". Linha verde = soma acumulada, chega a {sum(r):.1f}% (o que se perde ao resumir é {100 - sum(r):.1f}%).")
        load = res.get("loadings")
        if load is not None and feats:
            for i, linha in enumerate(np.asarray(load)[:3]):
                top = sorted(zip(feats, linha), key=lambda t: -abs(t[1]))[:2]
                L.append(f"Componente {i + 1} é construído sobretudo a partir de: "
                         + ", ".join(f"{f} ({w:+.2f})" for f, w in top) + ".")
        L.append(f"Caixa de informação por baixo: «Os {n_comp} componentes principais explicam {sum(r):.1f}% da variância total dos dados.»")
    estado = (f"Vê dois gráficos: (1) pontos em «Componente 1» x «Componente 2» ({algo}, {len(X)} pontos, "
              f"features: {', '.join(feats)[:120]}); (2) barras de variância por componente com linha acumulada.") if ratio is not None else \
             f"Vê um gráfico de pontos em «Componente 1» x «Componente 2» ({algo}, {len(X)} pontos, features: {', '.join(feats)[:120]})."
    return "\n".join(L)[:2900], estado[:580]


def render_reducao(username: str):
    inject_css()
    page_header("Redução de Dimensionalidade",
                "Visualiza dados de alta dimensão em 2D ou 3D", "")

    teoria_box("Porquê reduzir dimensionalidade?",
        "Datasets reais têm dezenas ou centenas de features. Redução de dimensionalidade "
        "permite <strong>visualizar</strong> os dados, <strong>remover ruído</strong>, "
        "acelerar o treino de modelos e descobrir a estrutura latente dos dados.")

    # ── CONFIGURAÇÃO ─────────────────────────────────
    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        section_title("Configuração")

        fonte = st.radio("Dados", ["Embutido", "Upload CSV"], horizontal=True, key="rd_fonte")
        if fonte == "Embutido":
            opcoes = ["Iris - Classificação de flores", "Vinho - Qualidade",
                      "Cancro - Diagnóstico", "Dígitos - Reconhecimento",
                      "Blobs - Clustering básico"]
            nm = st.selectbox("Dataset", opcoes, key="rd_ds")
            df, desc, _ = carregar_dataset_embutido(nm)
            if df is not None and desc:
                st.markdown(f'<div style="font-size:13px;color:var(--df-text2,#B8C4D8);">{desc}</div>', unsafe_allow_html=True)
        else:
            f = st.file_uploader("CSV", type=["csv"], key="rd_up")
            df = pd.read_csv(f) if f else None

        if df is not None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

            features = st.multiselect("Features", num_cols,
                                      default=num_cols[:min(8, len(num_cols))], key="rd_feats")
            label_col = st.selectbox("Coluna de cor/classe (opcional)",
                                     ["Nenhuma"] + cat_cols + num_cols, key="rd_lbl")

            algo_label = st.selectbox("Algoritmo", list(ALGORITMOS.keys()), key="rd_algo")
            algo = ALGORITMOS[algo_label]

            n_comp = st.select_slider("Componentes de saída", [2, 3], value=2, key="rd_nc")
            normalizar = st.checkbox("Normalizar (StandardScaler)", value=True, key="rd_norm")

            params = {}
            if algo == "kpca":
                params["kernel"] = st.selectbox("Kernel", ["rbf","poly","sigmoid","cosine"], key="rd_ker")
            if algo == "tsne":
                params["perplexity"] = st.slider("Perplexity", 5, 100, 30, key="rd_perp")
                params["n_iter"]     = st.slider("Iterações", 250, 2000, 1000, step=250, key="rd_iter")
            if algo == "umap" and HAS_UMAP:
                params["n_neighbors"] = st.slider("n_neighbors", 5, 100, 15, key="rd_nn")
                params["min_dist"]    = st.slider("min_dist", 0.0, 1.0, 0.1, key="rd_md")

            if st.button("Reduzir Dimensionalidade", width='stretch', key="rd_run"):
                if not features or len(features) < 2:
                    aviso_box("Selecciona pelo menos 2 features.")
                else:
                    X = df[features].dropna().to_numpy(dtype=float, na_value=0)
                    labels_raw = None
                    if label_col != "Nenhuma" and label_col in df.columns:
                        labels_raw = df.loc[df[features].notna().all(axis=1), label_col].to_numpy()
                        if labels_raw.dtype == object:
                            le = LabelEncoder()
                            labels_raw = le.fit_transform(labels_raw)

                    X_sc = StandardScaler().fit_transform(X) if normalizar else X

                    with st.spinner("A calcular..."):
                        try:
                            if algo == "pca":
                                model = PCA(n_components=n_comp, random_state=42)
                                X_red = model.fit_transform(X_sc)
                                st.session_state.rd_pca_model = model
                            elif algo == "kpca":
                                model = KernelPCA(n_components=n_comp, kernel=params.get("kernel","rbf"), random_state=42)
                                X_red = model.fit_transform(X_sc)
                            elif algo == "tsne":
                                model = TSNE(n_components=n_comp,
                                             perplexity=params.get("perplexity",30),
                                             n_iter=params.get("n_iter",1000),
                                             random_state=42)
                                X_red = model.fit_transform(X_sc)
                            elif algo == "ica":
                                model = FastICA(n_components=n_comp, random_state=42, max_iter=500)
                                X_red = model.fit_transform(X_sc)
                            elif algo == "fa":
                                model = FactorAnalysis(n_components=n_comp, random_state=42)
                                X_red = model.fit_transform(X_sc)
                            elif algo == "umap" and HAS_UMAP:
                                model = umap.UMAP(n_components=n_comp,
                                                  n_neighbors=params.get("n_neighbors",15),
                                                  min_dist=params.get("min_dist",0.1),
                                                  random_state=42)
                                X_red = model.fit_transform(X_sc)

                            _classes = []
                            if label_col != "Nenhuma" and label_col in df.columns:
                                _classes = sorted({str(v) for v in df.loc[df[features].notna().all(axis=1), label_col]})
                            st.session_state.rd_result = {
                                "X_red": X_red, "labels": labels_raw,
                                "algo": algo_label, "n_comp": n_comp,
                                "model": model if algo == "pca" else None,
                                # contexto para o tutor de IA (só nomes de colunas e números agregados)
                                "dataset": nm if fonte == "Embutido" else "CSV do aluno",
                                "features": list(features), "normalizar": bool(normalizar),
                                "label_col": label_col if label_col != "Nenhuma" else "", "classes": _classes,
                                "var_ratio": model.explained_variance_ratio_.tolist() if algo == "pca" else None,
                                "loadings": model.components_.tolist() if algo == "pca" else None,
                            }
                            add_pontos(username, 8, f"Redução {algo_label}")
                            sucesso_box("Redução concluída!")
                        except Exception as e:
                            erro_box(f"Erro: {e}")

    with col_res:
        res = st.session_state.get("rd_result", {})
        if res:
            X_red    = res["X_red"]
            labels   = res["labels"]
            algo_lbl = res["algo"]
            n_comp   = res["n_comp"]

            section_title(f"Resultado - {algo_lbl}")

            if n_comp == 2:
                st.plotly_chart(
                    _scatter2d(X_red, labels, f"{algo_lbl} - Projecção 2D"),
                    width='stretch'
                )
            else:
                st.plotly_chart(
                    _scatter3d(X_red, labels, f"{algo_lbl} - Projecção 3D"),
                    width='stretch'
                )

            if "rd_pca_model" in st.session_state:
                st.plotly_chart(
                    _variance_plot(st.session_state.rd_pca_model),
                    width='stretch'
                )
                var_total = st.session_state.rd_pca_model.explained_variance_ratio_.sum() * 100
                info_box(f"Os {n_comp} componentes principais explicam <strong>{var_total:.1f}%</strong> da variância total dos dados.")

            try:    # diz ao tutor de IA o que o aluno está a ver (gráficos, eixos, features, números)
                from modules.tutor_ia import registar
                _txt, _est = _descrever_para_tutor(res)
                registar(f"{algo_lbl} - Projecção {n_comp}D", _txt, _est)
            except Exception:
                pass

            teoria_box("Interpretar a visualização",
                "Pontos próximos no gráfico são similares no espaço original. "
                "Clusters visíveis sugerem estrutura natural nos dados. "
                "<strong>t-SNE e UMAP</strong> preservam estrutura local (boa para visualização). "
                "<strong>PCA</strong> preserva variância global (boa para pré-processamento).")
        else:
            st.markdown(f"""<div style="text-align:center;padding:5rem;color:var(--df-text3,#7A8BA8); border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:40px;margin-bottom:1rem;"></div><div style="font-size:16px;font-weight:700;color:var(--df-text,#FFFFFF);">Configura os parâmetros e clica em <strong>Reduzir Dimensionalidade</strong></div></div>""", unsafe_allow_html=True)
