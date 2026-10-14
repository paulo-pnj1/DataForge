"""
DataForge EDU - Clustering
KMeans, KMeans++, DBSCAN, HDBSCAN, Agglomerative, GMM, Birch, MeanShift, Spectral
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                              Birch, MeanShift, SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

from modules.utils import (
    inject_css, page_header, section_title, teoria_box, info_box,
    aviso_box, sucesso_box, erro_box, progresso_bar,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER, C_TEXT,
    C_TEXT_SEC, C_TEXT_MUTE, C_SURFACE2, PALETTE,
    carregar_dataset_embutido, add_pontos, save_historico_modelo, TEORIA
)

# ── ALGORITMOS ───────────────────────────────────────
ALGORITMOS = {
    "K-Means":              "kmeans",
    "K-Means++":            "kmeanspp",
    "DBSCAN":               "dbscan",
    "Agglomerative":        "agglomerative",
    "Gaussian Mixture (GMM)": "gmm",
    "Birch":                "birch",
    "Mean Shift":           "meanshift",
    "Spectral Clustering":  "spectral",
}

try:
    import hdbscan as hdbscan_lib
    ALGORITMOS["HDBSCAN"] = "hdbscan"
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def _build_model(algo: str, params: dict):
    if algo == "kmeans":
        return KMeans(n_clusters=params["n_clusters"], init="k-means++",
                      n_init=10, random_state=42)
    elif algo == "kmeanspp":
        return KMeans(n_clusters=params["n_clusters"], init="k-means++",
                      n_init=20, random_state=42)
    elif algo == "dbscan":
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    elif algo == "hdbscan":
        return hdbscan_lib.HDBSCAN(min_cluster_size=params.get("min_cluster_size", 5))
    elif algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=params["n_clusters"],
                                       linkage=params.get("linkage", "ward"))
    elif algo == "gmm":
        return GaussianMixture(n_components=params["n_clusters"],
                               covariance_type=params.get("cov_type", "full"),
                               random_state=42)
    elif algo == "birch":
        return Birch(n_clusters=params["n_clusters"],
                     threshold=params.get("threshold", 0.5))
    elif algo == "meanshift":
        return MeanShift()
    elif algo == "spectral":
        return SpectralClustering(n_clusters=params["n_clusters"],
                                  affinity=params.get("affinity", "rbf"),
                                  random_state=42)


def _get_labels(model, X, algo):
    if algo == "gmm":
        return model.predict(X)
    return model.labels_


def _elbow_plot(X_scaled, max_k=10):
    inertias, ks = [], range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ks), y=inertias, mode="lines+markers",
                             line=dict(color=C_ACCENT, width=3),
                             marker=dict(size=8, color=C_ACCENT)))
    fig.update_layout(
        title=dict(text="Método do Cotovelo - Escolha de K", font=dict(color="#FFFFFF", size=16)),
        xaxis=dict(title="Número de Clusters (K)", color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title="Inércia", color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _scatter_clusters(X_pca, labels, title="Clusters visualizados (PCA 2D)"):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cores = PALETTE[:max(n_clusters, 1)]
    fig = go.Figure()
    unique = sorted(set(labels))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        nome = f"Ruído" if lbl == -1 else f"Cluster {lbl}"
        cor = "#555555" if lbl == -1 else cores[i % len(cores)]
        fig.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1],
            mode="markers", name=nome,
            marker=dict(color=cor, size=7, opacity=0.8,
                        line=dict(color="#FFFFFF", width=0.5))
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(title="PC1", color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title="PC2", color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig


def render_clustering(username: str):
    inject_css()
    page_header("Clustering", "Agrupa dados sem rótulos em grupos naturais", "")

    # ── TABS ─────────────────────────────────────────
    tab_guiado, tab_livre = st.tabs(["  Modo Guiado  ", "  Modo Livre  "])

    # ════════════════════════════════════════════════
    # MODO GUIADO
    # ════════════════════════════════════════════════
    with tab_guiado:
        if "cl_step" not in st.session_state:
            st.session_state.cl_step = 1

        steps = ["Dataset", "Algoritmo", "Parâmetros", "Treinar", "Resultados"]
        pct = (st.session_state.cl_step - 1) / (len(steps) - 1)
        progresso_bar(pct, f"Passo {st.session_state.cl_step} de {len(steps)}: {steps[st.session_state.cl_step-1]}")

        # ── PASSO 1: DATASET ─────────────────────────
        if st.session_state.cl_step == 1:
            section_title("Passo 1 - Escolher Dataset")
            teoria_box("O que é Clustering?",
                "Clustering é aprendizagem <strong>não supervisionada</strong> - não há rótulos. "
                "O algoritmo descobre grupos (clusters) naturais nos dados. "
                "Usa-se para segmentação de clientes, detecção de padrões, compressão de dados.")

            fonte = st.radio("Fonte dos dados", ["Dataset embutido", "Upload CSV"], horizontal=True, key="cl_fonte")

            if fonte == "Dataset embutido":
                opcoes_cl = ["Meias-luas - Clustering", "Círculos - Clustering",
                             "Blobs - Clustering básico", "Iris - Classificação de flores",
                             "Vinho - Qualidade"]
                nome_ds = st.selectbox("Dataset", opcoes_cl, key="cl_ds_nome")
                df, desc, _ = carregar_dataset_embutido(nome_ds)
                if df is not None:
                    st.markdown(f'<div style="font-size:14px;color:#D0D8F0;padding:.6rem 0;">{desc}</div>',
                                unsafe_allow_html=True)
                    st.dataframe(df.head(5), width='stretch')
                    st.session_state.cl_df = df
            else:
                f = st.file_uploader("Ficheiro CSV", type=["csv"], key="cl_upload")
                if f:
                    df = pd.read_csv(f)
                    st.dataframe(df.head(5), width='stretch')
                    st.session_state.cl_df = df

            if st.button("Próximo →", key="cl_s1", width='stretch'):
                if "cl_df" not in st.session_state:
                    aviso_box("Selecciona ou carrega um dataset primeiro.")
                else:
                    st.session_state.cl_step = 2
                    st.rerun()

        # ── PASSO 2: ALGORITMO ───────────────────────
        elif st.session_state.cl_step == 2:
            section_title("Passo 2 - Escolher Algoritmo")
            algo_label = st.selectbox("Algoritmo de Clustering", list(ALGORITMOS.keys()), key="cl_algo_sel")
            st.session_state.cl_algo = ALGORITMOS[algo_label]
            st.session_state.cl_algo_label = algo_label

            algo_key = ALGORITMOS[algo_label]
            info = {
                "kmeans":       ("K-Means", "Divide em K grupos minimizando a inércia intracluster. Rápido e eficiente para clusters esféricos.", C_ACCENT),
                "kmeanspp":     ("K-Means++", "K-Means com inicialização inteligente - centróides espaçados para convergência mais rápida.", C_ACCENT),
                "dbscan":       ("DBSCAN", "Encontra clusters por densidade. Detecta outliers automaticamente. Não precisa definir K.", C_GREEN),
                "hdbscan":      ("HDBSCAN", "Versão hierárquica do DBSCAN. Mais robusto a variações de densidade.", C_GREEN),
                "agglomerative":("Agglomerative", "Clustering hierárquico bottom-up. Começa com N clusters e vai fundindo.", C_AMBER),
                "gmm":          ("Gaussian Mixture", "Assume que os dados seguem distribuições gaussianas sobrepostas. Soft assignment.", C_AMBER),
                "birch":        ("Birch", "Eficiente para datasets grandes. Usa estrutura de árvore CF para agrupar.", C_RED),
                "meanshift":    ("Mean Shift", "Encontra picos de densidade. Não precisa de K. Lento em datasets grandes.", C_RED),
                "spectral":     ("Spectral Clustering", "Usa eigenvalores do grafo de similaridade. Excelente para clusters não-convexos.", C_ACCENT),
            }
            nome_i, desc_i, cor_i = info.get(algo_key, (algo_label, "", C_ACCENT))
            st.markdown(f"""<div style="background:{C_SURFACE};border:2px solid {cor_i};border-left:5px solid {cor_i}; border-radius:12px;padding:1.2rem 1.4rem;margin:1rem 0;"><div style="font-size:17px;font-weight:800;color:#FFFFFF;margin-bottom:6px;">{nome_i}</div><div style="font-size:15px;color:#D0D8F0;">{desc_i}</div></div>""", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Anterior", key="cl_s2b", width='stretch'):
                    st.session_state.cl_step = 1; st.rerun()
            with col2:
                if st.button("Próximo →", key="cl_s2", width='stretch'):
                    st.session_state.cl_step = 3; st.rerun()

        # ── PASSO 3: PARÂMETROS ──────────────────────
        elif st.session_state.cl_step == 3:
            section_title("Passo 3 - Configurar Parâmetros")
            algo = st.session_state.get("cl_algo", "kmeans")
            df   = st.session_state.get("cl_df", pd.DataFrame())

            # Features
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = st.multiselect("Features para clustering", num_cols,
                                      default=num_cols[:min(4, len(num_cols))], key="cl_feats")
            normalizar = st.checkbox("Normalizar dados (StandardScaler)", value=True, key="cl_norm")
            params = {}

            needs_k = algo in ["kmeans", "kmeanspp", "agglomerative", "gmm", "birch", "spectral"]
            if needs_k:
                params["n_clusters"] = st.slider("Número de clusters (K)", 2, 12, 3, key="cl_k")
                if algo in ["kmeans", "kmeanspp"]:
                    st.plotly_chart(
                        _elbow_plot(StandardScaler().fit_transform(df[features].dropna()) if features else np.array([[0]])),
                        width='stretch'
                    )
            if algo == "dbscan":
                params["eps"] = st.slider("eps (raio da vizinhança)", 0.1, 5.0, 0.5, 0.1, key="cl_eps")
                params["min_samples"] = st.slider("min_samples", 2, 20, 5, key="cl_ms")
            if algo == "hdbscan":
                params["min_cluster_size"] = st.slider("min_cluster_size", 2, 50, 5, key="cl_mcs")
            if algo == "agglomerative":
                params["linkage"] = st.selectbox("Linkage", ["ward","complete","average","single"], key="cl_link")
            if algo == "gmm":
                params["cov_type"] = st.selectbox("Covariance type", ["full","tied","diag","spherical"], key="cl_cov")
            if algo == "birch":
                params["threshold"] = st.slider("Threshold", 0.1, 2.0, 0.5, 0.1, key="cl_thr")
            if algo == "spectral":
                params["affinity"] = st.selectbox("Affinity", ["rbf","nearest_neighbors"], key="cl_aff")

            st.session_state.cl_features = features
            st.session_state.cl_params   = params

            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Anterior", key="cl_s3b", width='stretch'):
                    st.session_state.cl_step = 2; st.rerun()
            with col2:
                if st.button("Treinar Agora →", key="cl_s3", width='stretch'):
                    if not features:
                        aviso_box("Selecciona pelo menos 2 features.")
                    else:
                        st.session_state.cl_step = 4; st.rerun()

        # ── PASSO 4: TREINAR ─────────────────────────
        elif st.session_state.cl_step == 4:
            section_title("Passo 4 - A Treinar...")
            df       = st.session_state.get("cl_df", pd.DataFrame())
            features = st.session_state.get("cl_features", [])
            params   = st.session_state.get("cl_params", {})
            algo     = st.session_state.get("cl_algo", "kmeans")
            normalizar = st.session_state.get("cl_norm", True)

            X = df[features].dropna().to_numpy(dtype=float, na_value=0)
            if normalizar:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            with st.spinner("A calcular clusters..."):
                try:
                    model = _build_model(algo, params)
                    model.fit(X_scaled)
                    labels = _get_labels(model, X_scaled, algo)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_ruido    = int(np.sum(labels == -1))

                    sil_score = None
                    if n_clusters >= 2 and len(set(labels)) > 1:
                        try:
                            sil_score = silhouette_score(X_scaled, labels)
                        except Exception:
                            pass
                    ch_score = None
                    db_score = None
                    if n_clusters >= 2:
                        try:
                            ch_score = calinski_harabasz_score(X_scaled, labels)
                            db_score = davies_bouldin_score(X_scaled, labels)
                        except Exception:
                            pass

                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X_scaled)

                    st.session_state.cl_result = {
                        "labels": labels, "X_pca": X_pca, "X_scaled": X_scaled,
                        "n_clusters": n_clusters, "n_ruido": n_ruido,
                        "sil": sil_score, "ch": ch_score, "db": db_score,
                        "model": model, "features": features
                    }
                    sucesso_box(f"Clustering concluído! {n_clusters} clusters encontrados.")
                    add_pontos(username, 10, f"Clustering com {st.session_state.cl_algo_label}")
                    st.session_state.cl_step = 5
                    st.rerun()
                except Exception as e:
                    erro_box(f"Erro: {e}")

        # ── PASSO 5: RESULTADOS ──────────────────────
        elif st.session_state.cl_step == 5:
            section_title("Passo 5 - Resultados")
            res = st.session_state.get("cl_result", {})
            if not res:
                aviso_box("Executa o treino primeiro.")
            else:
                labels    = res["labels"]
                X_pca     = res["X_pca"]
                n_clusters= res["n_clusters"]
                n_ruido   = res["n_ruido"]

                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Clusters", n_clusters)
                c2.metric("Pontos de ruído", n_ruido)
                c3.metric("Silhouette", f"{res['sil']:.3f}" if res['sil'] is not None else "N/A")
                c4.metric("Davies-Bouldin", f"{res['db']:.3f}" if res['db'] is not None else "N/A")

                st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

                # Scatter PCA
                st.plotly_chart(
                    _scatter_clusters(X_pca, labels,
                                      f"Clusters - {st.session_state.get('cl_algo_label','')} (PCA 2D)"),
                    width='stretch'
                )

                # Distribuição dos clusters
                unique, counts = np.unique(labels, return_counts=True)
                fig_dist = go.Figure(go.Bar(
                    x=[f"Cluster {l}" if l != -1 else "Ruído" for l in unique],
                    y=counts,
                    marker_color=[C_ACCENT if l != -1 else "#555" for l in unique],
                    text=counts, textposition="auto",
                    textfont=dict(color="#FFFFFF", size=13)
                ))
                fig_dist.update_layout(
                    title=dict(text="Dimensão dos Clusters", font=dict(color="#FFFFFF", size=15)),
                    xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                    font=dict(color="#FFFFFF")
                )
                st.plotly_chart(fig_dist, width='stretch')

                # Explicação das métricas
                teoria_box("Interpretar os resultados",
                    "<strong>Silhouette Score</strong> (-1 a 1): quanto mais próximo de 1, melhor a separação dos clusters. "
                    "Acima de 0.5 é bom, acima de 0.7 é excelente.<br>"
                    "<strong>Davies-Bouldin</strong>: quanto menor, melhor. Mede a compacidade e separação dos clusters.<br>"
                    "<strong>Pontos de ruído</strong> (DBSCAN/HDBSCAN): pontos que não pertencem a nenhum cluster - possivelmente outliers.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("← Recomeçar", width='stretch'):
                        st.session_state.cl_step = 1; st.rerun()
                with col2:
                    if st.button("Experimentar outro algoritmo →", width='stretch'):
                        st.session_state.cl_step = 2; st.rerun()

    # ════════════════════════════════════════════════
    # MODO LIVRE
    # ════════════════════════════════════════════════
    with tab_livre:
        st.markdown(f'<div style="font-size:17px;font-weight:800;color:#FFFFFF;margin-bottom:1rem;">Modo Livre - Controlo Total</div>', unsafe_allow_html=True)

        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            fonte_l = st.radio("Dados", ["Embutido", "Upload"], horizontal=True, key="cl_l_fonte")
            if fonte_l == "Embutido":
                opcoes_l = ["Blobs - Clustering básico", "Meias-luas - Clustering",
                            "Círculos - Clustering", "Iris - Classificação de flores"]
                nm = st.selectbox("Dataset", opcoes_l, key="cl_l_ds")
                df_l, _, _ = carregar_dataset_embutido(nm)
            else:
                f_l = st.file_uploader("CSV", type=["csv"], key="cl_l_up")
                df_l = pd.read_csv(f_l) if f_l else None

            if df_l is not None:
                num_l = df_l.select_dtypes(include=[np.number]).columns.tolist()
                feats_l = st.multiselect("Features", num_l, default=num_l[:min(4,len(num_l))], key="cl_l_feats")
                algo_l_label = st.selectbox("Algoritmo", list(ALGORITMOS.keys()), key="cl_l_algo")
                algo_l = ALGORITMOS[algo_l_label]
                norm_l = st.checkbox("Normalizar", value=True, key="cl_l_norm")
                params_l = {}
                if algo_l in ["kmeans","kmeanspp","agglomerative","gmm","birch","spectral"]:
                    params_l["n_clusters"] = st.slider("K", 2, 15, 3, key="cl_l_k")
                if algo_l == "dbscan":
                    params_l["eps"] = st.slider("eps", 0.1, 5.0, 0.5, 0.1, key="cl_l_eps")
                    params_l["min_samples"] = st.slider("min_samples", 2, 20, 5, key="cl_l_ms")
                if algo_l == "hdbscan":
                    params_l["min_cluster_size"] = st.slider("min_cluster_size", 2, 30, 5, key="cl_l_mcs")
                if algo_l == "agglomerative":
                    params_l["linkage"] = st.selectbox("Linkage", ["ward","complete","average","single"], key="cl_l_lnk")

                if st.button("Agrupar", width='stretch', key="cl_l_run"):
                    if not feats_l:
                        aviso_box("Selecciona features.")
                    else:
                        X_l = df_l[feats_l].dropna().to_numpy(dtype=float, na_value=0)
                        X_l_sc = StandardScaler().fit_transform(X_l) if norm_l else X_l
                        try:
                            m_l = _build_model(algo_l, params_l)
                            m_l.fit(X_l_sc)
                            lbl_l = _get_labels(m_l, X_l_sc, algo_l)
                            pca_l = PCA(n_components=2, random_state=42)
                            Xp_l  = pca_l.fit_transform(X_l_sc)
                            nc_l  = len(set(lbl_l)) - (1 if -1 in lbl_l else 0)
                            try:
                                sil_l = silhouette_score(X_l_sc, lbl_l) if nc_l >= 2 else None
                            except Exception:
                                sil_l = None
                            st.session_state.cl_livre_res = {"lbl": lbl_l, "Xp": Xp_l, "nc": nc_l, "sil": sil_l, "algo": algo_l_label}
                            add_pontos(username, 5, f"Clustering livre {algo_l_label}")
                        except Exception as e:
                            erro_box(str(e))

        with col_res:
            res_l = st.session_state.get("cl_livre_res", {})
            if res_l:
                c1, c2 = st.columns(2)
                c1.metric("Clusters", res_l["nc"])
                c2.metric("Silhouette", f"{res_l['sil']:.3f}" if res_l['sil'] else "N/A")
                st.plotly_chart(
                    _scatter_clusters(res_l["Xp"], res_l["lbl"], f"{res_l['algo']} - PCA 2D"),
                    width='stretch'
                )
            else:
                st.markdown(f"""<div style="text-align:center;padding:4rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:40px;margin-bottom:1rem;"></div><div style="font-size:16px;font-weight:700;">Configura e clica em <strong style="color:#FFFFFF;">Agrupar</strong></div></div>""", unsafe_allow_html=True)
