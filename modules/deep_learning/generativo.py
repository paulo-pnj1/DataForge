"""
DataForge EDU - Modelos Generativos
Autoencoder, VAE (Variational Autoencoder), GAN simples
PyTorch - versão educativa com visualizações claras
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_SURFACE2, PALETTE, add_pontos
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _DummyModule:
        Module = object

    nn = _DummyModule()


# ════════════════════════════════════════════════════
# AUTOENCODER
# ════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ════════════════════════════════════════════════════
# VAE
# ════════════════════════════════════════════════════
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()
        )
        self.fc_mu      = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder    = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h   = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon = F.mse_loss(recon_x, x, reduction='sum')
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


# ════════════════════════════════════════════════════
# GAN
# ════════════════════════════════════════════════════
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════
# DADOS SINTÉTICOS
# ════════════════════════════════════════════════════
def _gerar_dados_2d(tipo="gaussiano", n=500):
    if tipo == "gaussiano":
        X = np.random.randn(n, 2).astype(np.float32)
    elif tipo == "espiral":
        t = np.linspace(0, 4 * np.pi, n)
        X = np.column_stack([t * np.cos(t), t * np.sin(t)]).astype(np.float32)
        X += np.random.randn(*X.shape).astype(np.float32) * 0.3
        X = (X - X.mean(0)) / X.std(0)
    elif tipo == "mistura_gaussianas":
        c1 = np.random.randn(n // 2, 2) + np.array([2, 2])
        c2 = np.random.randn(n // 2, 2) + np.array([-2, -2])
        X  = np.vstack([c1, c2]).astype(np.float32)
    else:
        X = np.random.randn(n, 2).astype(np.float32)
    return X


def _scatter_2d(X, title, color=None, colorscale="Viridis"):
    fig = go.Figure(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(
            color=color if color is not None else C_ACCENT,
            colorscale=colorscale if color is not None else None,
            size=5, opacity=0.7,
            line=dict(color="#FFFFFF", width=0.3)
        )
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER, zeroline=False),
        yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER, zeroline=False),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), height=350
    )
    return fig


def render_generativo(username: str):
    inject_css()
    page_header("Modelos Generativos",
                "Autoencoder, VAE e GAN - aprende a gerar dados", "")

    if not HAS_TORCH:
        aviso_box("PyTorch necessário. Adiciona <code>torch</code> ao requirements.txt")
        return

    teoria_box("O que são modelos generativos?",
        "Modelos generativos <strong>aprendem a distribuição dos dados</strong> e podem gerar novos exemplos. "
        "<strong>Autoencoder</strong>: comprime e reconstrói dados - útil para redução de ruído e anomalias. "
        "<strong>VAE</strong>: aprende um espaço latente contínuo e amostrado - gera variações suaves. "
        "<strong>GAN</strong>: dois adversários (gerador vs discriminador) - produz amostras muito realistas.")

    tab_ae, tab_vae, tab_gan = st.tabs(["  Autoencoder  ", "  VAE  ", "  GAN  "])

    # ════════════════════════════════════════════════
    # AUTOENCODER
    # ════════════════════════════════════════════════
    with tab_ae:
        col1, col2 = st.columns([1, 2])
        with col1:
            section_title("Configuração")
            tipo_dados = st.selectbox("Dados", ["gaussiano", "espiral", "mistura_gaussianas"], key="ae_dados")
            n_samples  = st.slider("Nº amostras", 200, 2000, 500, 100, key="ae_n")
            latent_dim = st.slider("Dimensão latente", 1, 8, 2, key="ae_ld")
            hidden_str = st.selectbox("Camadas ocultas", ["[64]", "[64, 32]", "[128, 64]", "[256, 128, 64]"],
                                      key="ae_hid")
            hidden_dims = list(map(int, hidden_str.strip("[]").split(",")))
            epochs_ae  = st.slider("Épocas", 20, 500, 100, key="ae_ep")
            lr_ae      = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001, key="ae_lr")

            if st.button("Treinar Autoencoder", width='stretch', key="ae_run"):
                X = _gerar_dados_2d(tipo_dados, n_samples)
                _train_autoencoder(X, latent_dim, hidden_dims, epochs_ae, lr_ae, username)

        with col2:
            res = st.session_state.get("ae_result", {})
            if res:
                c1, c2 = st.columns(2)
                c1.metric("Loss Final", f"{res['loss_final']:.6f}")
                c2.metric("Épocas", res["epochs"])

                fig_loss = go.Figure(go.Scatter(
                    y=res["losses"], mode="lines",
                    line=dict(color=C_ACCENT, width=2)
                ))
                fig_loss.update_layout(
                    title=dict(text="Loss de Reconstrução", font=dict(color="#FFFFFF", size=14)),
                    xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                    font=dict(color="#FFFFFF"), height=250
                )
                st.plotly_chart(fig_loss, width='stretch')

                col_orig, col_rec = st.columns(2)
                with col_orig:
                    st.plotly_chart(_scatter_2d(res["X_orig"][:, :2], "Dados Originais"),
                                    width='stretch')
                with col_rec:
                    st.plotly_chart(_scatter_2d(res["X_rec"][:, :2], "Reconstruídos"),
                                    width='stretch')

                if res["latent_dim"] >= 2:
                    st.plotly_chart(_scatter_2d(res["Z"][:, :2], "Espaço Latente (2D)"),
                                    width='stretch')

                teoria_box("O espaço latente",
                    "O <strong>espaço latente</strong> é a representação comprimida dos dados. "
                    "Se o autoencoder aprendeu bem, pontos próximos no espaço latente "
                    "são semanticamente similares nos dados originais. "
                    "Com <strong>latente = 2</strong> conseguimos visualizar directamente.")
            else:
                st.markdown(f"""<div style="text-align:center;padding:4rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#127968;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Clica em <strong>Treinar Autoencoder</strong></div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # VAE
    # ════════════════════════════════════════════════
    with tab_vae:
        col1v, col2v = st.columns([1, 2])
        with col1v:
            section_title("VAE - Variational Autoencoder")
            tipo_v    = st.selectbox("Dados", ["gaussiano", "espiral", "mistura_gaussianas"], key="vae_dados")
            n_v       = st.slider("Amostras", 200, 2000, 500, key="vae_n")
            ld_v      = st.slider("Dimensão latente", 1, 8, 2, key="vae_ld")
            hid_v     = st.slider("Neurónios ocultos", 32, 256, 128, key="vae_hid")
            ep_v      = st.slider("Épocas", 20, 300, 100, key="vae_ep")
            lr_v      = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001, key="vae_lr")
            n_gen_v   = st.slider("Amostras a gerar", 50, 500, 200, key="vae_ngen")

            if st.button("Treinar VAE", width='stretch', key="vae_run"):
                X_v = _gerar_dados_2d(tipo_v, n_v)
                _train_vae(X_v, ld_v, hid_v, ep_v, lr_v, n_gen_v, username)

        with col2v:
            res_v = st.session_state.get("vae_result", {})
            if res_v:
                c1, c2 = st.columns(2)
                c1.metric("Loss Final", f"{res_v['loss_final']:.2f}")
                c2.metric("Épocas", res_v["epochs"])

                fig_lv = go.Figure(go.Scatter(
                    y=res_v["losses"], mode="lines",
                    line=dict(color=C_ACCENT, width=2)
                ))
                fig_lv.update_layout(
                    title=dict(text="Loss VAE (Reconstrução + KL)", font=dict(color="#FFFFFF", size=14)),
                    xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                    font=dict(color="#FFFFFF"), height=250
                )
                st.plotly_chart(fig_lv, width='stretch')

                col_ov, col_gv = st.columns(2)
                with col_ov:
                    st.plotly_chart(_scatter_2d(res_v["X_orig"][:, :2], "Dados Reais"),
                                    width='stretch')
                with col_gv:
                    st.plotly_chart(_scatter_2d(res_v["X_gen"][:, :2], "Gerados pelo VAE"),
                                    width='stretch')

                if res_v["ld"] >= 2:
                    st.plotly_chart(_scatter_2d(res_v["Z"][:, :2], "Espaço Latente VAE"),
                                    width='stretch')
            else:
                st.markdown(f"""<div style="text-align:center;padding:4rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#127775;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Clica em <strong>Treinar VAE</strong></div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # GAN
    # ════════════════════════════════════════════════
    with tab_gan:
        col1g, col2g = st.columns([1, 2])
        with col1g:
            section_title("GAN - Generative Adversarial Network")
            tipo_g   = st.selectbox("Dados reais", ["gaussiano", "espiral", "mistura_gaussianas"], key="gan_dados")
            n_g      = st.slider("Amostras reais", 200, 1000, 500, key="gan_n")
            noise_g  = st.slider("Dimensão do ruído (z)", 4, 64, 16, key="gan_noise")
            hid_g    = st.slider("Neurónios ocultos", 32, 256, 128, key="gan_hid")
            ep_g     = st.slider("Épocas", 50, 500, 200, key="gan_ep")
            lr_g     = st.select_slider("Learning Rate", [0.0001, 0.0002, 0.001], value=0.0002, key="gan_lr")
            n_gen_g  = st.slider("Amostras a gerar", 100, 1000, 300, key="gan_ngen")

            st.markdown(f"""<div style="background:{C_SURFACE2};border:2px solid {C_BORDER};border-radius:10px; padding:.8rem 1rem;font-size:13px;color:#D0D8F0;font-weight:600;margin-top:1rem;">O GAN treina dois modelos:<br><strong style="color:{C_GREEN};">Gerador</strong>: cria dados falsos<br><strong style="color:{C_RED};">Discriminador</strong>: distingue real de falso<br>Ambos competem até o gerador enganar o discriminador.</div>""", unsafe_allow_html=True)

            if st.button("Treinar GAN", width='stretch', key="gan_run"):
                X_g = _gerar_dados_2d(tipo_g, n_g)
                _train_gan(X_g, noise_g, hid_g, ep_g, lr_g, n_gen_g, username)

        with col2g:
            res_g = st.session_state.get("gan_result", {})
            if res_g:
                c1, c2, c3 = st.columns(3)
                c1.metric("Loss G Final", f"{res_g['g_losses'][-1]:.4f}")
                c2.metric("Loss D Final", f"{res_g['d_losses'][-1]:.4f}")
                c3.metric("Épocas", res_g["epochs"])

                # Loss G vs D
                fig_gl = go.Figure()
                ep_list = list(range(1, len(res_g["g_losses"]) + 1))
                fig_gl.add_trace(go.Scatter(x=ep_list, y=res_g["g_losses"], mode="lines",
                                            name="Gerador", line=dict(color=C_GREEN, width=2)))
                fig_gl.add_trace(go.Scatter(x=ep_list, y=res_g["d_losses"], mode="lines",
                                            name="Discriminador", line=dict(color=C_RED, width=2)))
                fig_gl.update_layout(
                    title=dict(text="Loss: Gerador vs Discriminador", font=dict(color="#FFFFFF", size=14)),
                    xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                    plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                    font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF")),
                    height=280
                )
                st.plotly_chart(fig_gl, width='stretch')

                col_rg, col_gg = st.columns(2)
                with col_rg:
                    st.plotly_chart(_scatter_2d(res_g["X_real"][:, :2], "Dados Reais"),
                                    width='stretch')
                with col_gg:
                    st.plotly_chart(_scatter_2d(res_g["X_fake"][:, :2], "Gerados pelo GAN"),
                                    width='stretch')

                teoria_box("Loss do GAN",
                    "No treino ideal, as losses do <strong>Gerador</strong> e <strong>Discriminador</strong> "
                    "convergem para valores próximos (equilíbrio de Nash). "
                    "Se a loss do Gerador sobe muito, o discriminador está a ganhar - experimenta "
                    "reduzir o learning rate ou aumentar a dimensão do ruído.")
            else:
                st.markdown(f"""<div style="text-align:center;padding:4rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#127917;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Clica em <strong>Treinar GAN</strong></div></div>""", unsafe_allow_html=True)


# ── HELPERS DE TREINO ─────────────────────────────────
def _train_autoencoder(X, latent_dim, hidden_dims, epochs, lr, username):
    try:
        X_t   = torch.tensor(X, dtype=torch.float32)
        model = Autoencoder(X.shape[1], latent_dim, hidden_dims)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        prog  = st.progress(0)
        stat  = st.empty()

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            recon, z = model(X_t)
            loss = F.mse_loss(recon, X_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (ep + 1) % max(1, epochs // 10) == 0:
                prog.progress((ep + 1) / epochs)
                stat.markdown(f'<div style="color:#FFFFFF;font-weight:700;">Época {ep+1}/{epochs} - Loss: {loss.item():.6f}</div>', unsafe_allow_html=True)

        model.eval()
        with torch.no_grad():
            X_rec, Z = model(X_t)

        st.session_state.ae_result = {
            "X_orig": X, "X_rec": X_rec.numpy(), "Z": Z.numpy(),
            "losses": losses, "loss_final": losses[-1],
            "epochs": epochs, "latent_dim": latent_dim
        }
        add_pontos(username, 15, "Autoencoder")
        sucesso_box(f"Autoencoder treinado! Loss final: {losses[-1]:.6f}")
    except Exception as e:
        erro_box(f"Erro: {e}")


def _train_vae(X, latent_dim, hidden_dim, epochs, lr, n_gen, username):
    try:
        X_t   = torch.tensor(X, dtype=torch.float32)
        model = VAE(X.shape[1], latent_dim, hidden_dim)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        prog  = st.progress(0)
        stat  = st.empty()

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            recon, mu, logvar = model(X_t)
            loss = vae_loss(recon, X_t, mu, logvar)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (ep + 1) % max(1, epochs // 10) == 0:
                prog.progress((ep + 1) / epochs)
                stat.markdown(f'<div style="color:#FFFFFF;font-weight:700;">Época {ep+1}/{epochs} - Loss: {loss.item():.2f}</div>', unsafe_allow_html=True)

        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(X_t)
            Z     = model.reparameterize(mu, logvar).numpy()
            z_gen = torch.randn(n_gen, latent_dim)
            X_gen = model.decode(z_gen).numpy()

        st.session_state.vae_result = {
            "X_orig": X, "X_gen": X_gen, "Z": Z,
            "losses": losses, "loss_final": losses[-1],
            "epochs": epochs, "ld": latent_dim
        }
        add_pontos(username, 20, "VAE")
        sucesso_box(f"VAE treinado! {n_gen} amostras geradas.")
    except Exception as e:
        erro_box(f"Erro VAE: {e}")


def _train_gan(X, noise_dim, hidden_dim, epochs, lr, n_gen, username):
    try:
        X_t   = torch.tensor(X, dtype=torch.float32)
        data_dim = X.shape[1]
        G  = Generator(noise_dim, data_dim, hidden_dim)
        D  = Discriminator(data_dim, hidden_dim)
        opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        g_losses, d_losses = [], []
        prog = st.progress(0)
        stat = st.empty()
        batch = min(64, len(X))

        for ep in range(epochs):
            # Treino Discriminador
            D.train(); G.eval()
            idx   = torch.randperm(len(X_t))[:batch]
            real  = X_t[idx]
            z     = torch.randn(batch, noise_dim)
            fake  = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_D = criterion(d_real, torch.ones(batch, 1)) + \
                     criterion(d_fake, torch.zeros(batch, 1))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Treino Gerador
            G.train(); D.eval()
            z    = torch.randn(batch, noise_dim)
            fake = G(z)
            loss_G = criterion(D(fake), torch.ones(batch, 1))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

            if (ep + 1) % max(1, epochs // 10) == 0:
                prog.progress((ep + 1) / epochs)
                stat.markdown(
                    f'<div style="color:#FFFFFF;font-weight:700;">Época {ep+1}/{epochs} - '
                    f'G: {loss_G.item():.4f} | D: {loss_D.item():.4f}</div>',
                    unsafe_allow_html=True
                )

        G.eval()
        with torch.no_grad():
            z_final = torch.randn(n_gen, noise_dim)
            X_fake  = G(z_final).numpy()

        st.session_state.gan_result = {
            "X_real": X, "X_fake": X_fake,
            "g_losses": g_losses, "d_losses": d_losses,
            "epochs": epochs
        }
        add_pontos(username, 25, "GAN")
        sucesso_box(f"GAN treinado! {n_gen} amostras geradas.")
    except Exception as e:
        erro_box(f"Erro GAN: {e}")