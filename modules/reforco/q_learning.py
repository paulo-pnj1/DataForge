"""
DataForge EDU - Q-Learning Tabular
GridWorld customizavel + visualizacao do processo de aprendizagem
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_SURFACE2, PALETTE, add_pontos
)


class GridWorld:
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    ACTION_NAMES = ["UP","DN","LT","RT"]
    ACTION_ARROWS = ["\u2191","\u2193","\u2190","\u2192"]

    def __init__(self, n=5, obstacles=None, traps=None, goal=None, start=None):
        self.n         = n
        self.goal      = goal  if goal  else (n-1, n-1)
        self.start     = start if start else (0, 0)
        self.obstacles = set(obstacles) if obstacles else set()
        self.traps     = set(traps)     if traps     else set()
        self.reset()

    def reset(self):
        self.pos = self.start
        return self._state()

    def _state(self):
        return self.pos[0] * self.n + self.pos[1]

    def n_states(self):
        return self.n * self.n

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if nr < 0 or nr >= self.n or nc < 0 or nc >= self.n:
            nr, nc = self.pos
        if (nr, nc) in self.obstacles:
            nr, nc = self.pos
        self.pos = (nr, nc)
        if self.pos == self.goal:
            return self._state(), 10.0, True
        if self.pos in self.traps:
            return self._state(), -5.0, True
        return self._state(), -0.1, False

    def render_grid(self, Q=None, path=None):
        n = self.n
        z = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                pos = (r, c)
                if pos == self.goal:        z[r, c] = 4
                elif pos in self.traps:     z[r, c] = 3
                elif pos in self.obstacles: z[r, c] = 2
                elif pos == self.start:     z[r, c] = 1
                else:                       z[r, c] = 0

        if path:
            for pr, pc in path:
                if (pr, pc) not in self.obstacles and (pr, pc) != self.goal \
                   and (pr, pc) not in self.traps and (pr, pc) != self.start:
                    z[pr, pc] = 0.5

        text_grid = [["" for _ in range(n)] for _ in range(n)]
        if Q is not None:
            for r in range(n):
                for c in range(n):
                    pos = (r, c)
                    if pos in self.obstacles:
                        text_grid[r][c] = "\u25a0"
                    elif pos == self.goal:
                        text_grid[r][c] = "\u2605"
                    elif pos in self.traps:
                        text_grid[r][c] = "\u2715"
                    else:
                        s = r * n + c
                        best_a = int(np.argmax(Q[s]))
                        text_grid[r][c] = self.ACTION_ARROWS[best_a]

        colorscale = [
            [0.0,  "#0F1117"],
            [0.12, "#2A3A5C"],
            [0.25, "#A78BFA"],
            [0.5,  "#3A4560"],
            [0.75, "#1E2535"],
            [0.9,  "#FF6B6B"],
            [1.0,  "#4ADE80"],
        ]

        fig = go.Figure(go.Heatmap(
            z=z, colorscale=colorscale, showscale=False,
            text=text_grid, texttemplate="%{text}",
            textfont=dict(size=22, color="#FFFFFF"),
            hoverinfo="none", xgap=3, ygap=3,
        ))
        fig.update_layout(
            height=420,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
            plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
            margin=dict(t=40, b=10, l=10, r=10),
            title=dict(text="GridWorld - Politica Aprendida", font=dict(color="#FFFFFF", size=14))
        )
        return fig


def q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
    Q = np.zeros((env.n_states(), 4))
    rewards_per_ep, steps_per_ep, epsilon_hist = [], [], []

    for ep in range(episodes):
        state     = env.reset()
        total_rew = 0.0
        steps     = 0
        done      = False

        while not done and steps < 200:
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = env.step(action)
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) * (1 - done) - Q[state, action]
            )
            state      = next_state
            total_rew += reward
            steps     += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_ep.append(total_rew)
        steps_per_ep.append(steps)
        epsilon_hist.append(epsilon)

    return Q, rewards_per_ep, steps_per_ep, epsilon_hist


def _get_best_path(env, Q, max_steps=100):
    env.reset()
    path    = [env.start]
    visited = {env.start}
    done    = False
    steps   = 0
    while not done and steps < max_steps:
        s = env.pos[0] * env.n + env.pos[1]
        a = int(np.argmax(Q[s]))
        env.step(a)
        if env.pos in visited:
            break
        visited.add(env.pos)
        path.append(env.pos)
        if env.pos == env.goal or env.pos in env.traps:
            done = True
        steps += 1
    return path


def _plot_training_curves(rewards, steps_hist, eps_hist):
    window = max(1, len(rewards) // 20)
    def smooth(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='valid').tolist()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Recompensa por Episodio", "Recompensa Suavizada",
                        "Passos por Episodio", "Epsilon (Exploracao)"],
        vertical_spacing=0.18)

    ep     = list(range(len(rewards)))
    ep_sm  = list(range(window - 1, len(rewards)))

    fig.add_trace(go.Scatter(x=ep, y=rewards, mode="lines",
        line=dict(color=C_ACCENT, width=1), name="Recompensa"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep_sm, y=smooth(rewards, window), mode="lines",
        line=dict(color=C_GREEN, width=2), name="Suavizada"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=steps_hist, mode="lines",
        line=dict(color=C_AMBER, width=1.5), name="Passos"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ep, y=eps_hist, mode="lines",
        line=dict(color=C_RED, width=2), name="Epsilon"), row=2, col=2)

    fig.update_layout(
        height=480, plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), showlegend=False,
        title=dict(text="Curvas de Treino Q-Learning", font=dict(color="#FFFFFF", size=15))
    )
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(color="#FFFFFF", gridcolor=C_BORDER, row=r, col=c)
            fig.update_yaxes(color="#FFFFFF", gridcolor=C_BORDER, row=r, col=c)
    for ann in fig.layout.annotations:
        ann.font.color = "#FFFFFF"
    return fig


def _plot_q_heatmap(Q, n):
    V = Q.max(axis=1).reshape(n, n)
    fig = go.Figure(go.Heatmap(
        z=V,
        colorscale=[[0, C_SURFACE], [0.5, C_AMBER], [1, C_GREEN]],
        text=[[f"{V[r,c]:.2f}" for c in range(n)] for r in range(n)],
        texttemplate="%{text}", textfont=dict(color="#FFFFFF", size=11),
        showscale=True, colorbar=dict(tickfont=dict(color="#FFFFFF"))
    ))
    fig.update_layout(
        title=dict(text="Valor Maximo Q por Estado (V*)", font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), height=350
    )
    return fig


def render_q_learning(username: str):
    inject_css()
    page_header("Q-Learning", "O agente aprende a navegar por tentativa e erro", "")

    teoria_box("Como funciona o Q-Learning?",
        "O agente esta num <strong>ambiente</strong> e toma <strong>accoes</strong>. "
        "Recebe <strong>recompensas</strong> e aprende a maximiza-las atraves da tabela Q. "
        "Com <strong>epsilon-greedy</strong>, o agente explora aleatoriamente no inicio "
        "e vai ficando mais ganancioso (greedy) conforme aprende.")

    tab_treino, tab_demo = st.tabs(["  Treinar Agente  ", "  Demo Passo a Passo  "])

    with tab_treino:
        col_cfg, col_res = st.columns([1, 2])

        with col_cfg:
            section_title("Ambiente GridWorld")
            n_grid = st.slider("Tamanho da grelha (N x N)", 4, 10, 5, key="ql_n")

            st.markdown(f'<div style="font-size:13px;color:#D0D8F0;font-weight:700;margin-bottom:.3rem;">Obstaculos (linha,col separados por espaco):</div>', unsafe_allow_html=True)
            obs_input  = st.text_input("Ex: 1,1 2,2 3,1", value="1,1 2,2 3,1", key="ql_obs")
            trap_input = st.text_input("Armadilhas Ex: 1,3 3,3", value="1,3", key="ql_trap")

            def parse_cells(s, n):
                cells = set()
                for part in s.strip().split():
                    try:
                        r, c = part.split(",")
                        r, c = int(r.strip()), int(c.strip())
                        if 0 <= r < n and 0 <= c < n:
                            cells.add((r, c))
                    except Exception:
                        pass
                return cells

            obstacles = parse_cells(obs_input,  n_grid)
            traps     = parse_cells(trap_input, n_grid)
            obstacles.discard((0, 0)); obstacles.discard((n_grid-1, n_grid-1))
            traps.discard((0, 0));     traps.discard((n_grid-1, n_grid-1))

            section_title("Hiperparametros Q-Learning")
            episodes      = st.slider("Episodios", 100, 5000, 1000, 100, key="ql_ep")
            alpha         = st.slider("alpha (learning rate)", 0.01, 1.0, 0.1, 0.01, key="ql_alpha")
            gamma         = st.slider("gamma (desconto futuro)", 0.5, 0.999, 0.95, 0.01, key="ql_gamma")
            epsilon_start = st.slider("epsilon inicial", 0.1, 1.0, 1.0, 0.1, key="ql_eps")
            epsilon_decay = st.slider("epsilon decay", 0.990, 0.9999, 0.995, 0.001, key="ql_epd", format="%.4f")
            epsilon_min   = st.slider("epsilon minimo", 0.01, 0.2, 0.01, 0.01, key="ql_epm")

            st.markdown(f"""
            <div style="background:{C_SURFACE2};border:2px solid {C_BORDER};border-radius:10px;
            padding:.8rem 1rem;font-size:13px;color:#D0D8F0;font-weight:600;margin-top:.5rem;">
            <strong style="color:#FFFFFF;">Legenda:</strong><br>
            \u25cf Inicio (0,0) &nbsp;|\u25cf Meta ({n_grid-1},{n_grid-1})<br>
            \u25a0 Obstaculo &nbsp;| \u2715 Armadilha
            </div>
            """, unsafe_allow_html=True)

            if st.button("Treinar Agente", use_container_width=True, key="ql_run"):
                env = GridWorld(n_grid, obstacles, traps)
                with st.spinner(f"A treinar por {episodes} episodios..."):
                    Q, rewards, steps_h, eps_h = q_learning(
                        env, episodes, alpha, gamma,
                        epsilon_start, epsilon_decay, epsilon_min
                    )
                    path = _get_best_path(GridWorld(n_grid, obstacles, traps), Q)
                    reached = len(path) > 0 and path[-1] == (n_grid-1, n_grid-1)

                st.session_state.ql_result = {
                    "Q": Q, "rewards": rewards, "steps": steps_h,
                    "eps_hist": eps_h, "path": path,
                    "env_params": (n_grid, list(obstacles), list(traps)),
                    "reached_goal": reached
                }
                add_pontos(username, 20, "Q-Learning GridWorld")
                if reached:
                    sucesso_box(f"Agente treinou com sucesso! Caminho optimo: {len(path)} passos.")
                else:
                    aviso_box("Treino concluido mas o agente nao convergiu. Experimenta mais episodios.")

        with col_res:
            res = st.session_state.get("ql_result", {})
            if res:
                n_r, obs_r, trap_r = res["env_params"]
                env_r = GridWorld(n_r, set(map(tuple, obs_r)), set(map(tuple, trap_r)))
                Q, path = res["Q"], res["path"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Episodios",       len(res["rewards"]))
                c2.metric("Recomp. media",   f"{np.mean(res['rewards'][-100:]):.2f}")
                c3.metric("Passos (optimo)", len(path) if res["reached_goal"] else "N/A")

                st.plotly_chart(env_r.render_grid(Q, path), use_container_width=True)
                st.plotly_chart(_plot_q_heatmap(Q, n_r), use_container_width=True)
                st.plotly_chart(_plot_training_curves(res["rewards"], res["steps"], res["eps_hist"]),
                                use_container_width=True)

                teoria_box("Interpretar os resultados",
                    "As <strong>setas</strong> mostram a accao optima em cada estado. "
                    "O <strong>heatmap V*</strong> mostra quanto o agente valoriza cada estado "
                    "- estados perto da meta tem valor mais alto. "
                    "A <strong>recompensa</strong> deve aumentar e os <strong>passos</strong> diminuir ao longo do treino.")
            else:
                st.markdown(f"""
                <div style="text-align:center;padding:5rem;color:#7A8BA8;
                border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;">
                    <div style="font-size:48px;margin-bottom:1rem;">\U0001f916</div>
                    <div style="font-size:16px;font-weight:700;color:#FFFFFF;">
                        Configura o ambiente e clica em <strong>Treinar Agente</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_demo:
        section_title("Demo Passo a Passo")
        res = st.session_state.get("ql_result", {})
        if res and res.get("path"):
            path = res["path"]
            n_r, obs_r, trap_r = res["env_params"]
            env_demo = GridWorld(n_r, set(map(tuple, obs_r)), set(map(tuple, trap_r)))

            st.markdown(f"""
            <div style="background:{C_SURFACE};border:2px solid {C_BORDER};border-radius:12px;
            padding:1rem;margin-bottom:1rem;">
                <div style="font-size:15px;font-weight:800;color:#FFFFFF;margin-bottom:.5rem;">
                    Caminho encontrado: {len(path)} passos
                </div>
                <div style="font-size:13px;color:#D0D8F0;font-weight:600;">
                    {" \u2192 ".join([f"({r},{c})" for r, c in path])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Animar Caminho", use_container_width=True, key="ql_anim"):
                placeholder = st.empty()
                for i in range(len(path) + 1):
                    partial = path[:i]
                    fig = env_demo.render_grid(res["Q"], partial)
                    pos_txt = str(path[i-1]) if i > 0 else str(path[0])
                    fig.update_layout(title=dict(
                        text=f"Passo {i}/{len(path)} - Posicao: {pos_txt}",
                        font=dict(color="#FFFFFF", size=14)
                    ))
                    placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.45)
                sucesso_box("Animacao concluida!")
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem;color:#7A8BA8;
            border:2px dashed {C_BORDER};border-radius:16px;">
                <div style="font-size:16px;font-weight:700;color:#FFFFFF;">
                    Treina o agente primeiro para ver a animacao
                </div>
            </div>
            """, unsafe_allow_html=True)