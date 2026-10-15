"""
DataForge EDU - Deep Q-Network (DQN)
CartPole-v1 com PyTorch + visualizacao educativa
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import deque
import random

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_SURFACE2, add_pontos
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Marcador para a definição das classes abaixo não falhar sem PyTorch
    # (render_dqn mostra um aviso amigável e nunca as usa).
    nn = type("nn", (), {"Module": object})

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def _train_dqn_cartpole(episodes, hidden_str, lr, gamma, epsilon_start,
                         epsilon_decay, epsilon_min, batch_size, buffer_size,
                         target_update, username):
    try:
        env = gym.make("CartPole-v1")
        state_dim  = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dims = tuple(map(int, hidden_str.strip("[]").split(",")))

        policy_net = DQNNetwork(state_dim, action_dim, hidden_dims)
        target_net = DQNNetwork(state_dim, action_dim, hidden_dims)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        buffer    = ReplayBuffer(buffer_size)
        epsilon   = epsilon_start

        rewards_hist, eps_hist, loss_hist = [], [], []
        progress = st.progress(0)
        status   = st.empty()
        best_reward = -np.inf

        for ep in range(episodes):
            state, _ = env.reset()
            total_r  = 0.0
            done     = False
            ep_loss  = []

            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_vals = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = int(q_vals.argmax().item())

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, float(done))
                state    = next_state
                total_r += reward

                if len(buffer) >= batch_size:
                    states_b, actions_b, rewards_b, next_b, dones_b = buffer.sample(batch_size)
                    states_t     = torch.tensor(states_b)
                    actions_t    = torch.tensor(actions_b).unsqueeze(1)
                    rewards_t    = torch.tensor(rewards_b)
                    next_t       = torch.tensor(next_b)
                    dones_t      = torch.tensor(dones_b)

                    q_curr = policy_net(states_t).gather(1, actions_t).squeeze()
                    with torch.no_grad():
                        q_next = target_net(next_t).max(1)[0]
                    q_target = rewards_t + gamma * q_next * (1 - dones_t)

                    loss = F.mse_loss(q_curr, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_loss.append(loss.item())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards_hist.append(total_r)
            eps_hist.append(epsilon)
            loss_hist.append(np.mean(ep_loss) if ep_loss else 0)

            if total_r > best_reward:
                best_reward = total_r

            if (ep + 1) % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if (ep + 1) % max(1, episodes // 20) == 0 or ep == episodes - 1:
                progress.progress((ep + 1) / episodes)
                avg_r = np.mean(rewards_hist[-50:]) if len(rewards_hist) >= 50 else np.mean(rewards_hist)
                status.markdown(
                    f'<div style="color:var(--df-text,#FFFFFF);font-weight:700;">Ep {ep+1}/{episodes} - '
                    f'Reward: {total_r:.0f} | Media(50): {avg_r:.1f} | epsilon: {epsilon:.3f}</div>',
                    unsafe_allow_html=True
                )

        env.close()
        avg_last = np.mean(rewards_hist[-50:]) if len(rewards_hist) >= 50 else np.mean(rewards_hist)
        st.session_state.dqn_result = {
            "rewards": rewards_hist, "eps_hist": eps_hist,
            "loss_hist": loss_hist, "best_reward": best_reward,
            "avg_last_50": avg_last, "episodes": episodes
        }
        add_pontos(username, 30, "DQN CartPole")
        if avg_last >= 195:
            sucesso_box(f"CartPole resolvido! Media(50): {avg_last:.1f} >= 195 (criterio de sucesso)")
        else:
            sucesso_box(f"DQN treinado! Media(50): {avg_last:.1f} | Melhor episodio: {best_reward:.0f}")
    except Exception as e:
        erro_box(f"Erro DQN: {e}")


def render_dqn(username: str):
    inject_css()
    page_header("DQN - Deep Q-Network",
                "Rede neural que aprende a jogar CartPole", "")

    if not HAS_TORCH:
        aviso_box("PyTorch necessario. Adiciona <code>torch</code> ao requirements.txt")
        return
    if not HAS_GYM:
        aviso_box("Gymnasium necessario. Adiciona <code>gymnasium</code> ao requirements.txt")
        return

    teoria_box("DQN vs Q-Learning Tabular",
        "O <strong>Q-Learning tabular</strong> guarda um valor por (estado, accao) numa tabela. "
        "Funciona bem para espacos pequenos como GridWorld. "
        "O <strong>DQN</strong> usa uma <strong>rede neural</strong> para aproximar a funcao Q, "
        "permitindo lidar com espacos de estado contínuos e de alta dimensao como imagens. "
        "Duas inovacoes chave: <strong>Replay Buffer</strong> (quebra correlacao entre amostras) "
        "e <strong>Target Network</strong> (estabiliza o treino).")

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        section_title("Ambiente: CartPole-v1")
        st.markdown(f"""
        <div style="background:{C_SURFACE2};border:2px solid {C_BORDER};border-radius:10px;
        padding:.8rem 1rem;font-size:13px;color:var(--df-text2,#B8C4D8);font-weight:600;margin-bottom:1rem;">
        <strong style="color:var(--df-text,#FFFFFF);">CartPole</strong>: equilibra um poste numa plataforma movel.<br>
        Estado: 4 valores (posicao, velocidade, angulo, velocidade angular)<br>
        Accoes: 2 (esquerda, direita)<br>
        Recompensa: +1 por cada passo que o poste se mantiver em pe<br>
        <strong>Resolvido</strong> quando media(50 eps) >= 195
        </div>
        """, unsafe_allow_html=True)

        section_title("Arquitectura DQN")
        hidden_str  = st.selectbox("Camadas ocultas", ["[64]","[128]","[128, 64]","[256, 128]","[128, 128]"],
                                   index=2, key="dqn_hid")
        batch_size  = st.select_slider("Batch size", [32, 64, 128], value=64, key="dqn_bs")
        buffer_size = st.select_slider("Replay Buffer", [1000, 5000, 10000], value=10000, key="dqn_buf")
        target_upd  = st.slider("Target update (cada N eps)", 5, 50, 10, key="dqn_tu")

        section_title("Hiperparametros")
        episodes      = st.slider("Episodios", 100, 1000, 300, 50, key="dqn_ep")
        lr            = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001, key="dqn_lr")
        gamma         = st.slider("Gamma (desconto)", 0.9, 0.999, 0.99, 0.001, key="dqn_gamma")
        eps_start     = st.slider("Epsilon inicial", 0.5, 1.0, 1.0, 0.1, key="dqn_eps")
        eps_decay     = st.slider("Epsilon decay", 0.990, 0.9999, 0.995, 0.001, key="dqn_epd", format="%.4f")
        eps_min       = st.slider("Epsilon minimo", 0.01, 0.1, 0.01, 0.01, key="dqn_epm")

        if st.button("Treinar DQN", use_container_width=True, key="dqn_run"):
            _train_dqn_cartpole(episodes, hidden_str, lr, gamma, eps_start,
                                eps_decay, eps_min, batch_size, buffer_size,
                                target_upd, username)

    with col_res:
        res = st.session_state.get("dqn_result", {})
        if res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Melhor Reward",   f"{res['best_reward']:.0f}")
            c2.metric("Media (ult 50)",  f"{res['avg_last_50']:.1f}")
            c3.metric("Episodios",       res["episodes"])

            if res["avg_last_50"] >= 195:
                sucesso_box("CartPole RESOLVIDO! Media >= 195")

            # Reward por episodio
            window = max(1, len(res["rewards"]) // 20)
            smooth = np.convolve(res["rewards"], np.ones(window)/window, mode='valid').tolist()
            ep_all = list(range(len(res["rewards"])))
            ep_sm  = list(range(window - 1, len(res["rewards"])))

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=ep_all, y=res["rewards"], mode="lines",
                line=dict(color=C_ACCENT, width=1, opacity=0.4), name="Reward"))
            fig_r.add_trace(go.Scatter(x=ep_sm, y=smooth, mode="lines",
                line=dict(color=C_GREEN, width=2.5), name=f"Media({window})"))
            fig_r.add_hline(y=195, line_color=C_AMBER, line_dash="dash",
                            annotation_text="Meta: 195", annotation_font_color="#FFFFFF")
            fig_r.update_layout(
                title=dict(text="Reward por Episodio", font=dict(color="#FFFFFF", size=15)),
                xaxis=dict(title="Episodio", color="#FFFFFF", gridcolor=C_BORDER),
                yaxis=dict(title="Reward", color="#FFFFFF", gridcolor=C_BORDER),
                plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
            )
            st.plotly_chart(fig_r, use_container_width=True)

            # Loss e Epsilon
            fig_le = go.Figure()
            ep_list = list(range(len(res["loss_hist"])))
            fig_le.add_trace(go.Scatter(x=ep_list, y=res["loss_hist"], mode="lines",
                line=dict(color=C_RED, width=1.5), name="Loss", yaxis="y"))
            fig_le.add_trace(go.Scatter(x=ep_list, y=res["eps_hist"], mode="lines",
                line=dict(color=C_AMBER, width=2), name="Epsilon", yaxis="y2"))
            fig_le.update_layout(
                title=dict(text="Loss e Epsilon ao longo do treino", font=dict(color="#FFFFFF", size=14)),
                xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
                yaxis=dict(title="Loss", color=C_RED, gridcolor=C_BORDER),
                yaxis2=dict(title="Epsilon", color=C_AMBER, overlaying="y", side="right"),
                plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
            )
            st.plotly_chart(fig_le, use_container_width=True)

            teoria_box("Replay Buffer e Target Network",
                "O <strong>Replay Buffer</strong> guarda as ultimas experiencias e amostras mini-batches aleatorios, "
                "quebrando a correlacao temporal que desestabilizaria o treino. "
                "A <strong>Target Network</strong> e uma copia 'congelada' da rede principal "
                "actualizada periodicamente - evita que o alvo mude a cada passo, tornando o treino mais estavel.")
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:5rem;color:var(--df-text3,#7A8BA8);
            border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;">
                <div style="font-size:48px;margin-bottom:1rem;">\U0001f3ae</div>
                <div style="font-size:16px;font-weight:700;color:var(--df-text,#FFFFFF);">
                    Configura e clica em <strong>Treinar DQN</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)