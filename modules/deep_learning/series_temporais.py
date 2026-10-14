"""
DataForge EDU - Séries Temporais
LSTM, GRU, ARIMA (statsmodels), previsão interactiva
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_SURFACE2, add_pontos
)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _DummyModule:
        Module = object

    nn = _DummyModule()

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ── DATASETS EMBUTIDOS ────────────────────────────────
def _gerar_serie_sinusoidal(n=200, noise=0.1):
    t = np.linspace(0, 4 * np.pi, n)
    y = np.sin(t) + np.sin(2 * t) * 0.5 + np.random.normal(0, noise, n)
    return pd.Series(y, name="valor")

def _gerar_serie_tendencia(n=200, noise=0.2):
    t  = np.arange(n)
    y  = 0.05 * t + np.sin(t * 0.3) * 2 + np.random.normal(0, noise, n)
    return pd.Series(y, name="valor")

def _gerar_serie_vendas(n=365):
    t       = np.arange(n)
    tendencia = 0.02 * t
    sazonalidade = 5 * np.sin(2 * np.pi * t / 7)
    ruido   = np.random.normal(0, 1.5, n)
    y       = 20 + tendencia + sazonalidade + ruido
    datas   = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(y.clip(0), index=datas, name="vendas")

DATASETS_SERIES = {
    "Onda Sinusoidal (simples)":    ("sinusoidal", "200 pontos com padrão senoidal. Ideal para aprender LSTM básico."),
    "Tendência + Sazonalidade":     ("tendencia",  "200 pontos com tendência crescente e sazonalidade. Mais realista."),
    "Vendas Diárias (365 dias)":    ("vendas",     "1 ano de vendas diárias simuladas com sazonalidade semanal."),
}


def _criar_sequencias(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def _plot_serie(serie, title="Série Temporal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=serie.values if hasattr(serie, 'values') else serie,
        x=serie.index.astype(str) if hasattr(serie.index, 'astype') else list(range(len(serie))),
        mode="lines", line=dict(color=C_ACCENT, width=2), name="Valores"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _plot_previsao(y_real, y_pred, n_train, title="Real vs Previsto"):
    fig = go.Figure()
    x_all   = list(range(len(y_real)))
    x_train = x_all[:n_train]
    x_test  = x_all[n_train:]
    fig.add_trace(go.Scatter(
        x=x_train, y=y_real[:n_train], mode="lines",
        line=dict(color=C_ACCENT, width=2), name="Treino"
    ))
    fig.add_trace(go.Scatter(
        x=x_test, y=y_real[n_train:], mode="lines",
        line=dict(color=C_GREEN, width=2), name="Real (teste)"
    ))
    fig.add_trace(go.Scatter(
        x=x_test, y=y_pred, mode="lines",
        line=dict(color=C_RED, width=2, dash="dash"), name="Previsto"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig


def render_series_temporais(username: str):
    inject_css()
    page_header("Séries Temporais",
                "Prevê valores futuros com LSTM, GRU e ARIMA", "")

    teoria_box("O que são séries temporais?",
        "Uma série temporal é uma sequência de valores ordenados no tempo. "
        "Exemplos: preço do petróleo, temperatura diária, vendas mensais, taxa de câmbio. "
        "<strong>LSTM e GRU</strong> são redes neurais especializadas em capturar padrões temporais. "
        "<strong>ARIMA</strong> é o método clássico estatístico para previsão.")

    tab_dl, tab_arima = st.tabs(["  LSTM / GRU  ", "  ARIMA (Clássico)  "])

    # ════════════════════════════════════════════════
    # LSTM / GRU
    # ════════════════════════════════════════════════
    with tab_dl:
        if not HAS_TORCH:
            aviso_box("PyTorch necessário. Adiciona <code>torch</code> ao requirements.txt")
            return

        col_cfg, col_res = st.columns([1, 2])

        with col_cfg:
            section_title("Dados")
            fonte = st.radio("Fonte", ["Dataset embutido", "Upload CSV"], horizontal=True, key="st_fonte")

            if fonte == "Dataset embutido":
                ds_nome = st.selectbox("Dataset", list(DATASETS_SERIES.keys()), key="st_ds")
                ds_key, ds_desc = DATASETS_SERIES[ds_nome]
                if ds_key == "sinusoidal":
                    serie = _gerar_serie_sinusoidal()
                elif ds_key == "tendencia":
                    serie = _gerar_serie_tendencia()
                else:
                    serie = _gerar_serie_vendas()
                st.markdown(f'<div style="font-size:13px;color:#D0D8F0;font-weight:600;">{ds_desc}</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(_plot_serie(serie, ds_nome), width='stretch')
                valores = serie.to_numpy(dtype=float)
            else:
                f = st.file_uploader("CSV com coluna numérica", type=["csv"], key="st_up")
                if f:
                    df_st = pd.read_csv(f)
                    col_v = st.selectbox("Coluna de valores", df_st.select_dtypes(include=[np.number]).columns.tolist(),
                                         key="st_col")
                    valores = df_st[col_v].dropna().to_numpy(dtype=float)
                    st.plotly_chart(_plot_serie(pd.Series(valores), col_v), width='stretch')
                else:
                    valores = None

            if valores is not None:
                section_title("Arquitectura")
                modelo_tipo = st.selectbox("Modelo", ["LSTM", "GRU"], key="st_modelo")
                seq_len     = st.slider("Comprimento da sequência", 5, 50, 20, key="st_seq")
                hidden_sz   = st.slider("Neurónios ocultos", 16, 256, 64, key="st_hid")
                n_layers    = st.slider("Camadas", 1, 3, 2, key="st_nl")
                dropout     = st.slider("Dropout", 0.0, 0.5, 0.2, 0.1, key="st_drop")
                epochs      = st.slider("Épocas", 10, 200, 50, key="st_ep")
                lr          = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001, key="st_lr")
                test_pct    = st.slider("% Teste", 10, 40, 20, key="st_ts") / 100
                n_futuro    = st.slider("Previsão futura (passos)", 5, 50, 10, key="st_fut")

                if st.button("Treinar e Prever", width='stretch', key="st_run"):
                    _train_lstm_gru(valores, modelo_tipo, seq_len, hidden_sz,
                                    n_layers, dropout, epochs, lr, test_pct, n_futuro, username)

        with col_res:
            res = st.session_state.get("st_result", {})
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE",  f"{res['rmse']:.4f}")
                c2.metric("MAE",   f"{res['mae']:.4f}")
                c3.metric("Épocas", res["epochs"])

                st.plotly_chart(res["fig_pred"], width='stretch')
                st.plotly_chart(res["fig_loss"], width='stretch')

                # Previsão futura
                if res.get("y_futuro") is not None:
                    fig_fut = go.Figure()
                    n_hist = min(50, len(res["y_real"]))
                    fig_fut.add_trace(go.Scatter(
                        x=list(range(n_hist)),
                        y=res["y_real"][-n_hist:],
                        mode="lines", line=dict(color=C_ACCENT, width=2), name="Histórico"
                    ))
                    fig_fut.add_trace(go.Scatter(
                        x=list(range(n_hist, n_hist + len(res["y_futuro"]))),
                        y=res["y_futuro"],
                        mode="lines+markers",
                        line=dict(color=C_GREEN, width=2, dash="dot"),
                        marker=dict(size=6, color=C_GREEN),
                        name="Previsão futura"
                    ))
                    fig_fut.update_layout(
                        title=dict(text="Previsão para o Futuro", font=dict(color="#FFFFFF", size=15)),
                        xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                        yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
                    )
                    st.plotly_chart(fig_fut, width='stretch')

                teoria_box("RMSE vs MAE",
                    "<strong>RMSE</strong> (Root Mean Squared Error): penaliza mais os erros grandes. "
                    "Bom quando grandes desvios são inaceitáveis. "
                    "<strong>MAE</strong> (Mean Absolute Error): erro médio absoluto. "
                    "Mais intuitivo - 'em média, erro X unidades'.")
            else:
                st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#128200;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura e clica em <strong>Treinar e Prever</strong></div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # ARIMA
    # ════════════════════════════════════════════════
    with tab_arima:
        if not HAS_STATSMODELS:
            aviso_box("statsmodels necessário. Adiciona ao requirements.txt")
            return

        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            section_title("Dados ARIMA")
            ds_a = st.selectbox("Dataset", list(DATASETS_SERIES.keys()), key="ar_ds")
            ds_key_a, _ = DATASETS_SERIES[ds_a]
            if ds_key_a == "sinusoidal":
                serie_a = _gerar_serie_sinusoidal()
            elif ds_key_a == "tendencia":
                serie_a = _gerar_serie_tendencia()
            else:
                serie_a = _gerar_serie_vendas()
            valores_a = serie_a.to_numpy(dtype=float)

            section_title("Parâmetros ARIMA(p,d,q)")
            p = st.slider("p - AutoRegressivo", 0, 5, 2, key="ar_p")
            d = st.slider("d - Integração (diferenciação)", 0, 2, 1, key="ar_d")
            q = st.slider("q - Média Móvel", 0, 5, 2, key="ar_q")
            n_prev = st.slider("Passos a prever", 5, 50, 20, key="ar_np")
            test_a = st.slider("% Teste", 10, 30, 20, key="ar_ts") / 100

            st.markdown(f"""<div style="background:{C_SURFACE2};border:2px solid {C_BORDER};border-radius:10px; padding:.8rem 1rem;font-size:13px;color:#D0D8F0;font-weight:600;"><strong style="color:#FFFFFF;">ARIMA({p},{d},{q})</strong><br>p={p}: usa {p} valores passados<br>d={d}: diferencia {d} vez(es) para estacionaridade<br>q={q}: usa {q} erros passados</div>""", unsafe_allow_html=True)

            if st.button("Ajustar ARIMA", width='stretch', key="ar_run"):
                _train_arima(valores_a, p, d, q, n_prev, test_a, username)

        with col_a2:
            res_a = st.session_state.get("arima_result", {})
            if res_a:
                c1, c2 = st.columns(2)
                c1.metric("RMSE Teste", f"{res_a['rmse']:.4f}")
                c2.metric("MAE Teste",  f"{res_a['mae']:.4f}")
                st.plotly_chart(res_a["fig"], width='stretch')
                teoria_box("Quando usar ARIMA?",
                    "ARIMA é excelente para séries estacionárias ou que ficam estacionárias após diferenciação. "
                    "<strong>p</strong>: componente autorregressiva (correlação com valores anteriores). "
                    "<strong>d</strong>: número de diferenciações para remover tendência. "
                    "<strong>q</strong>: componente de média móvel dos erros. "
                    "Para séries com sazonalidade forte, usa SARIMA.")
            else:
                st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#128202;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura os parâmetros e clica em <strong>Ajustar ARIMA</strong></div></div>""", unsafe_allow_html=True)


def _train_lstm_gru(valores, modelo_tipo, seq_len, hidden_sz, n_layers,
                    dropout, epochs, lr, test_pct, n_futuro, username):
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(valores.reshape(-1, 1)).flatten()
        X, y   = _criar_sequencias(scaled, seq_len)

        n_test  = max(1, int(len(X) * test_pct))
        n_train = len(X) - n_test

        X_train = torch.tensor(X[:n_train], dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y[:n_train], dtype=torch.float32).unsqueeze(-1)
        X_test  = torch.tensor(X[n_train:], dtype=torch.float32).unsqueeze(-1)
        y_test  = torch.tensor(y[n_train:], dtype=torch.float32).unsqueeze(-1)

        if modelo_tipo == "LSTM":
            model = LSTMModel(1, hidden_sz, n_layers, dropout)
        else:
            model = GRUModel(1, hidden_sz, n_layers, dropout)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses   = []
        progress = st.progress(0)
        status   = st.empty()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out  = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                progress.progress((epoch + 1) / epochs)
                status.markdown(
                    f'<div style="color:#FFFFFF;font-weight:700;">Época {epoch+1}/{epochs} - Loss: {loss.item():.6f}</div>',
                    unsafe_allow_html=True
                )

        model.eval()
        with torch.no_grad():
            y_pred_sc = model(X_test).numpy().flatten()
        y_pred = scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
        y_real_test = scaler.inverse_transform(y[n_train:].reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_real_test, y_pred))
        mae  = mean_absolute_error(y_real_test, y_pred)

        y_real_all = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        fig_pred   = _plot_previsao(y_real_all, y_pred, n_train, f"{modelo_tipo} - Real vs Previsto")

        # Loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=losses, mode="lines", line=dict(color=C_ACCENT, width=2), name="Loss"
        ))
        fig_loss.update_layout(
            title=dict(text="Curva de Loss", font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(title="Época", color="#FFFFFF", gridcolor=C_BORDER),
            yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
            plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
            font=dict(color="#FFFFFF")
        )

        # Previsão futura
        last_seq = scaled[-seq_len:].copy()
        y_futuro_sc = []
        model.eval()
        with torch.no_grad():
            for _ in range(n_futuro):
                inp  = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                pred = model(inp).item()
                y_futuro_sc.append(pred)
                last_seq = np.append(last_seq[1:], pred)
        y_futuro = scaler.inverse_transform(np.array(y_futuro_sc).reshape(-1, 1)).flatten()

        st.session_state.st_result = {
            "rmse": rmse, "mae": mae, "epochs": epochs,
            "fig_pred": fig_pred, "fig_loss": fig_loss,
            "y_real": y_real_all, "y_futuro": y_futuro
        }
        add_pontos(username, 20, f"Séries Temporais {modelo_tipo}")
        sucesso_box(f"{modelo_tipo} treinado! RMSE: {rmse:.4f} · MAE: {mae:.4f}")
    except Exception as e:
        erro_box(f"Erro: {e}")


def _train_arima(valores, p, d, q, n_prev, test_pct, username):
    try:
        n_test  = max(1, int(len(valores) * test_pct))
        n_train = len(valores) - n_test
        train   = valores[:n_train]
        test    = valores[n_train:]

        with st.spinner(f"A ajustar ARIMA({p},{d},{q})..."):
            model = ARIMA(train, order=(p, d, q))
            fit   = model.fit()

        y_pred = fit.forecast(steps=n_test)
        rmse   = np.sqrt(mean_squared_error(test, y_pred))
        mae    = mean_absolute_error(test, y_pred)

        # Previsão futura extra
        y_futuro = fit.forecast(steps=n_prev + n_test)[n_test:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(n_train)), y=train,
            mode="lines", line=dict(color=C_ACCENT, width=2), name="Treino"
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_train, n_train + n_test)), y=test,
            mode="lines", line=dict(color=C_GREEN, width=2), name="Real (teste)"
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_train, n_train + n_test)), y=y_pred,
            mode="lines", line=dict(color=C_RED, width=2, dash="dash"), name="ARIMA previsto"
        ))
        fig.add_trace(go.Scatter(
            x=list(range(n_train + n_test, n_train + n_test + n_prev)), y=y_futuro,
            mode="lines+markers",
            line=dict(color=C_AMBER, width=2, dash="dot"),
            marker=dict(size=5, color=C_AMBER),
            name="Futuro previsto"
        ))
        fig.update_layout(
            title=dict(text=f"ARIMA({p},{d},{q}) - Previsão", font=dict(color="#FFFFFF", size=15)),
            xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
            yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
            plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
            font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
        )

        st.session_state.arima_result = {"rmse": rmse, "mae": mae, "fig": fig}
        add_pontos(username, 15, f"ARIMA({p},{d},{q})")
        sucesso_box(f"ARIMA ajustado! RMSE: {rmse:.4f} · MAE: {mae:.4f}")
    except Exception as e:
        erro_box(f"Erro ARIMA: {e}")