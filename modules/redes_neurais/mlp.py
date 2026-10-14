"""
DataForge EDU - MLP / Redes Neurais
MLP Classifier e Regressor (scikit-learn) + visualizações educativas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, r2_score, mean_squared_error, mean_absolute_error)
import pickle, io

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box, progresso_bar,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_TEXT, C_TEXT_SEC, C_SURFACE2, PALETTE,
    carregar_dataset_embutido, add_pontos, save_historico_modelo
)


def _confusion_heatmap(cm, classes):
    fig = go.Figure(go.Heatmap(
        z=cm, x=classes, y=classes,
        colorscale=[[0, C_SURFACE], [1, C_ACCENT]],
        text=cm.astype(str), texttemplate="%{text}",
        textfont=dict(color="#FFFFFF", size=14),
        showscale=False
    ))
    fig.update_layout(
        title=dict(text="Matriz de Confusão", font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(title="Previsto", color="#FFFFFF"),
        yaxis=dict(title="Real", color="#FFFFFF", autorange="reversed"),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _loss_curve(loss_curve_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_curve_, mode="lines",
        line=dict(color=C_ACCENT, width=2.5),
        name="Loss de treino"
    ))
    fig.update_layout(
        title=dict(text="Curva de Perda (Loss) durante o Treino", font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(title="Iteração", color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title="Loss", color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _network_diagram(hidden_layers, n_inputs, n_outputs):
    """Diagrama visual da arquitectura da rede"""
    layers = [n_inputs] + list(hidden_layers) + [n_outputs]
    n_layers = len(layers)
    max_nodes = max(layers)

    node_x, node_y, node_text = [], [], []
    edge_x, edge_y = [], []

    node_positions = []
    for l_idx, n_nodes in enumerate(layers):
        x = l_idx / (n_layers - 1)
        positions = []
        for n_idx in range(n_nodes):
            y = (n_idx + 0.5) / n_nodes
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"L{l_idx}N{n_idx}")
            positions.append((x, y))
        node_positions.append(positions)

    # Arestas entre camadas adjacentes (limitar para não sobrecarregar)
    for l_idx in range(len(node_positions) - 1):
        src_pos = node_positions[l_idx][:min(8, len(node_positions[l_idx]))]
        dst_pos = node_positions[l_idx+1][:min(8, len(node_positions[l_idx+1]))]
        for sx, sy in src_pos:
            for dx, dy in dst_pos:
                edge_x += [sx, dx, None]
                edge_y += [sy, dy, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(107,163,255,0.2)", width=1),
        hoverinfo="none", showlegend=False
    ))

    colors = [C_ACCENT] * layers[0] + \
             sum([[C_GREEN]*n for n in hidden_layers], []) + \
             [C_AMBER] * layers[-1]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=18, color=colors,
                    line=dict(color="#FFFFFF", width=1.5)),
        hoverinfo="none", showlegend=False
    ))

    # Labels das camadas
    labels_x = [l_idx / (n_layers - 1) for l_idx in range(n_layers)]
    labels_text = ["Input"] + [f"Hidden {i+1}\n({n})" for i, n in enumerate(hidden_layers)] + ["Output"]
    for lx, lt in zip(labels_x, labels_text):
        fig.add_annotation(x=lx, y=-0.08, text=lt.replace("\n", "<br>"),
                           showarrow=False, font=dict(color="#B8C4D8", size=11),
                           xanchor="center")

    fig.update_layout(
        title=dict(text="Arquitectura da Rede Neural", font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.1]),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), height=350,
        margin=dict(t=50, b=50, l=20, r=20)
    )
    return fig


def render_mlp(username: str):
    inject_css()
    page_header("MLP - Redes Neurais Multicamada",
                "Aprende padrões complexos através de camadas de neurónios", "")

    teoria_box("O que é uma Rede Neural MLP?",
        "Um <strong>Multilayer Perceptron</strong> é composto por: "
        "uma <strong>camada de entrada</strong> (uma por feature), "
        "<strong>camadas ocultas</strong> (onde a aprendizagem acontece) e "
        "uma <strong>camada de saída</strong> (uma por classe ou 1 para regressão). "
        "Cada neurónio aplica uma função de activação não-linear (ReLU, tanh, sigmoid).")

    tab_clf, tab_reg = st.tabs(["  Classificação  ", "  Regressão  "])

    # ════════════════════════════════════════════════
    # CLASSIFICAÇÃO
    # ════════════════════════════════════════════════
    with tab_clf:
        col_cfg, col_res = st.columns([1, 2])

        with col_cfg:
            section_title("Dados")
            fonte = st.radio("Fonte", ["Embutido", "Upload"], horizontal=True, key="mlp_c_fonte")
            if fonte == "Embutido":
                opcoes = ["Iris - Classificação de flores", "Vinho - Qualidade",
                          "Cancro - Diagnóstico", "Dígitos - Reconhecimento"]
                nm = st.selectbox("Dataset", opcoes, key="mlp_c_ds")
                df, desc, _ = carregar_dataset_embutido(nm)
                if df is not None and desc:
                    st.markdown(f'<div style="font-size:13px;color:#D0D8F0;">{desc}</div>', unsafe_allow_html=True)
            else:
                f = st.file_uploader("CSV", type=["csv"], key="mlp_c_up")
                df = pd.read_csv(f) if f else None

            if df is not None:
                cols = df.columns.tolist()
                target = st.selectbox("Target (coluna alvo)", cols, index=len(cols)-1, key="mlp_c_tgt")
                features = st.multiselect("Features", [c for c in cols if c != target],
                                          default=[c for c in cols if c != target][:min(8, len(cols))],
                                          key="mlp_c_feats")

                section_title("Arquitectura")
                n_layers = st.slider("Nº camadas ocultas", 1, 5, 2, key="mlp_c_nl")
                hidden = []
                for i in range(n_layers):
                    n = st.slider(f"Neurónios - Camada {i+1}", 8, 256, [64, 32, 16, 8, 4][i], key=f"mlp_c_h{i}")
                    hidden.append(n)

                section_title("Hiperparâmetros")
                activation = st.selectbox("Activação", ["relu","tanh","logistic"], key="mlp_c_act")
                solver     = st.selectbox("Optimizador", ["adam","sgd","lbfgs"], key="mlp_c_sol")
                lr         = st.select_slider("Learning rate", [0.0001,0.001,0.01,0.1], value=0.001, key="mlp_c_lr")
                max_iter   = st.slider("Épocas máximas", 50, 500, 200, 50, key="mlp_c_ep")
                test_size  = st.slider("% Teste", 10, 40, 20, key="mlp_c_ts") / 100

                # Diagrama da rede
                if features:
                    n_in  = len(features)
                    try:
                        n_out = df[target].nunique()
                    except Exception:
                        n_out = 1
                    st.plotly_chart(_network_diagram(tuple(hidden), min(n_in, 8), min(n_out, 4)),
                                    width='stretch')

                if st.button("Treinar Rede Neural", width='stretch', key="mlp_c_run"):
                    if not features:
                        aviso_box("Selecciona features.")
                    else:
                        _train_mlp_clf(df, features, target, hidden, activation,
                                       solver, lr, max_iter, test_size, username)

        with col_res:
            _show_mlp_clf_result()

    # ════════════════════════════════════════════════
    # REGRESSÃO
    # ════════════════════════════════════════════════
    with tab_reg:
        col_cfg2, col_res2 = st.columns([1, 2])

        with col_cfg2:
            section_title("Dados")
            fonte2 = st.radio("Fonte", ["Embutido", "Upload"], horizontal=True, key="mlp_r_fonte")
            if fonte2 == "Embutido":
                opcoes2 = ["Diabetes - Progressão", "Boston Housing - Preços"]
                nm2 = st.selectbox("Dataset", opcoes2, key="mlp_r_ds")
                df2, desc2, _ = carregar_dataset_embutido(nm2)
                if df2 is not None and desc2:
                    st.markdown(f'<div style="font-size:13px;color:#D0D8F0;">{desc2}</div>', unsafe_allow_html=True)
            else:
                f2 = st.file_uploader("CSV", type=["csv"], key="mlp_r_up")
                df2 = pd.read_csv(f2) if f2 else None

            if df2 is not None:
                cols2   = df2.columns.tolist()
                target2 = st.selectbox("Target", cols2, index=len(cols2)-1, key="mlp_r_tgt")
                feats2  = st.multiselect("Features", [c for c in cols2 if c != target2],
                                          default=[c for c in cols2 if c != target2][:8],
                                          key="mlp_r_feats")

                section_title("Arquitectura")
                nl2    = st.slider("Camadas ocultas", 1, 4, 2, key="mlp_r_nl")
                hid2   = []
                for i in range(nl2):
                    n = st.slider(f"Neurónios - Camada {i+1}", 8, 256, [64,32,16,8][i], key=f"mlp_r_h{i}")
                    hid2.append(n)

                act2   = st.selectbox("Activação", ["relu","tanh","logistic"], key="mlp_r_act")
                sol2   = st.selectbox("Optimizador", ["adam","sgd","lbfgs"], key="mlp_r_sol")
                lr2    = st.select_slider("Learning rate", [0.0001,0.001,0.01,0.1], value=0.001, key="mlp_r_lr")
                ep2    = st.slider("Épocas", 50, 500, 200, 50, key="mlp_r_ep")
                ts2    = st.slider("% Teste", 10, 40, 20, key="mlp_r_ts") / 100

                if st.button("Treinar Rede Neural", width='stretch', key="mlp_r_run"):
                    if not feats2:
                        aviso_box("Selecciona features.")
                    else:
                        _train_mlp_reg(df2, feats2, target2, hid2, act2, sol2, lr2, ep2, ts2, username)

        with col_res2:
            _show_mlp_reg_result()


# ── HELPERS ──────────────────────────────────────────
def _train_mlp_clf(df, features, target, hidden, activation, solver, lr, max_iter, test_size, username):
    try:
        X = df[features].copy()
        y = df[target].copy()

        # Encoding
        for col in X.select_dtypes(include=["object","category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        if y.dtype == object or str(y.dtype) == "category":
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))
            classes = le_y.classes_
        else:
            classes = [str(c) for c in sorted(y.unique())]

        X = X.fillna(X.median(numeric_only=True))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        with st.spinner("A treinar a rede neural..."):
            model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden),
                activation=activation,
                solver=solver,
                learning_rate_init=lr,
                max_iter=max_iter,
                random_state=42,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.1
            )
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm   = confusion_matrix(y_test, y_pred)
            rep  = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

            st.session_state.mlp_clf_res = {
                "acc": acc, "f1": f1, "cm": cm, "classes": classes,
                "loss_curve": model.loss_curve_,
                "n_iter": model.n_iter_,
                "report": rep, "model": model, "scaler": scaler
            }
            add_pontos(username, 15, "MLP Classificação")
            save_historico_modelo(username, {"tipo": "classificacao", "algoritmo": "MLP",
                                             "accuracy": acc, "f1": f1})
            sucesso_box(f"Treino concluído em {model.n_iter_} épocas! Accuracy: {acc:.1%}")
    except Exception as e:
        erro_box(f"Erro: {e}")


def _show_mlp_clf_result():
    res = st.session_state.get("mlp_clf_res", {})
    if not res:
        st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed #3A4560;border-radius:16px;margin-top:1rem;"><div style="font-size:40px;margin-bottom:1rem;"></div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura a rede e clica em <strong>Treinar</strong></div></div>""", unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",  f"{res['acc']:.1%}")
    c2.metric("F1 Score",  f"{res['f1']:.1%}")
    c3.metric("Épocas",    res['n_iter'])

    st.plotly_chart(_loss_curve(res["loss_curve"]), width='stretch')

    classes = [str(c) for c in res["classes"]]
    cm = res["cm"]
    if cm.shape[0] <= 10:
        st.plotly_chart(_confusion_heatmap(cm, classes[:cm.shape[0]]), width='stretch')

    # Download modelo
    buf = io.BytesIO()
    pickle.dump({"model": res["model"], "scaler": res["scaler"]}, buf)
    st.download_button("Descarregar Modelo (.pkl)", buf.getvalue(),
                       "mlp_classificacao.pkl", width='stretch')

    teoria_box("A curva de perda (loss)",
        "A loss desce ao longo das épocas enquanto a rede aprende. "
        "Se parar de descer cedo, experimenta <strong>mais épocas</strong> ou <strong>learning rate menor</strong>. "
        "Se oscilar muito, experimenta <strong>learning rate menor</strong> ou <strong>batch size maior</strong>.")


def _train_mlp_reg(df, features, target, hidden, activation, solver, lr, max_iter, test_size, username):
    try:
        X = df[features].copy()
        y = df[target].copy()
        for col in X.select_dtypes(include=["object","category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))
        y = pd.to_numeric(y, errors="coerce").fillna(y.median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        with st.spinner("A treinar..."):
            model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden),
                activation=activation,
                solver=solver,
                learning_rate_init=lr,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True
            )
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            r2   = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)

            st.session_state.mlp_reg_res = {
                "r2": r2, "rmse": rmse, "mae": mae,
                "y_test": y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test, "y_pred": y_pred,
                "loss_curve": model.loss_curve_,
                "n_iter": model.n_iter_,
                "model": model, "scaler": scaler
            }
            add_pontos(username, 15, "MLP Regressão")
            save_historico_modelo(username, {"tipo": "regressao", "algoritmo": "MLP",
                                             "r2": r2, "rmse": rmse})
            sucesso_box(f"Treino concluído! R²: {r2:.4f}")
    except Exception as e:
        erro_box(f"Erro: {e}")


def _show_mlp_reg_result():
    res = st.session_state.get("mlp_reg_res", {})
    if not res:
        st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed #3A4560;border-radius:16px;margin-top:1rem;"><div style="font-size:40px;margin-bottom:1rem;"></div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura e clica em <strong>Treinar</strong></div></div>""", unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",   f"{res['r2']:.4f}")
    c2.metric("RMSE", f"{res['rmse']:.4f}")
    c3.metric("MAE",  f"{res['mae']:.4f}")
    c4.metric("Épocas", res['n_iter'])

    st.plotly_chart(_loss_curve(res["loss_curve"]), width='stretch')

    # Real vs Previsto
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=res["y_test"], y=res["y_pred"], mode="markers",
        marker=dict(color=C_ACCENT, size=6, opacity=0.7,
                    line=dict(color="#FFFFFF", width=0.3)),
        name="Previsões"
    ))
    mn = min(res["y_test"].min(), res["y_pred"].min())
    mx = max(res["y_test"].max(), res["y_pred"].max())
    fig_rv.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                line=dict(color=C_RED, dash="dash", width=2),
                                name="Perfeito"))
    fig_rv.update_layout(
        title=dict(text="Real vs Previsto", font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(title="Real", color="#FFFFFF", gridcolor="#3A4560"),
        yaxis=dict(title="Previsto", color="#FFFFFF", gridcolor="#3A4560"),
        plot_bgcolor="#161B27", paper_bgcolor="#161B27",
        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
    )
    st.plotly_chart(fig_rv, width='stretch')

    buf = io.BytesIO()
    pickle.dump({"model": res["model"], "scaler": res["scaler"]}, buf)
    st.download_button("Descarregar Modelo (.pkl)", buf.getvalue(),
                       "mlp_regressao.pkl", width='stretch')
