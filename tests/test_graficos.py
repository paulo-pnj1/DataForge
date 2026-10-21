"""Gráficos: o DQN rebentava depois de treinar por causa de line=dict(opacity=...), que o Plotly rejeita."""
import ast
import glob
import os

import numpy as np
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_graficos_do_dqn_desenham_com_os_resultados_de_um_treino():
    from modules.reforco.dqn import _figuras_dqn
    rng = np.random.default_rng(0)
    n = 300
    res = {"rewards": (20 + rng.random(n) * 100).tolist(), "eps_hist": np.linspace(1, .05, n).tolist(),
           "loss_hist": rng.random(n).tolist(), "best_reward": 375, "avg_last_50": 88.5, "episodes": n}
    fig_r, fig_le = _figuras_dqn(res)
    assert len(fig_r.data) == 2 and len(fig_le.data) == 2
    assert fig_r.data[0].opacity == 0.4                    # a opacidade está no traço
    fig_r.to_json(); fig_le.to_json()                      # é o que o Streamlit envia ao navegador


def test_graficos_do_dqn_com_poucos_episodios():
    from modules.reforco.dqn import _figuras_dqn
    res = {"rewards": [10.0, 12.0], "eps_hist": [1.0, .9], "loss_hist": [0.0, .1], "best_reward": 12, "avg_last_50": 11, "episodes": 2}
    _figuras_dqn(res)


def _props(*classes):
    out = set()
    for c in classes:
        out |= set(c._valid_props)
    return out


def test_nenhum_grafico_usa_propriedades_invalidas_em_line_ou_marker():
    """Varredura estática: line=dict(...) e marker=dict(...) só podem ter propriedades que o Plotly aceita."""
    linhas = _props(go.scatter.Line, go.scatter3d.Line, go.scattergl.Line, go.scatterpolar.Line)
    marcadores = _props(go.scatter.Marker, go.scatter3d.Marker, go.scattergl.Marker, go.bar.Marker,
                        go.histogram.Marker, go.scatterpolar.Marker, go.pie.Marker)
    erros = []
    ficheiros = glob.glob(os.path.join(RAIZ, "modules", "**", "*.py"), recursive=True) + [os.path.join(RAIZ, "app.py")]
    for f in ficheiros:
        arvore = ast.parse(open(f, encoding="utf8").read())
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Call):
                continue
            for kw in no.keywords:
                if kw.arg in ("line", "marker") and isinstance(kw.value, ast.Call) and getattr(kw.value.func, "id", "") == "dict":
                    validas = linhas if kw.arg == "line" else marcadores
                    erros += [f"{os.path.relpath(f, RAIZ)}:{kw.value.lineno} {kw.arg}={k.arg}"
                              for k in kw.value.keywords if k.arg and k.arg not in validas]
    assert not erros, erros


@pytest.fixture
def pasta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _botao(at, key):
    return [b for b in at.button if b.key == key][0]


def test_nenhum_ficheiro_usa_use_container_width():
    """O Streamlit retirou use_container_width: o formato novo é width="stretch"."""
    usos = []
    for f in glob.glob(os.path.join(RAIZ, "modules", "**", "*.py"), recursive=True) + [os.path.join(RAIZ, "app.py")]:
        if "use_container_width" in open(f, encoding="utf8").read():
            usos.append(os.path.relpath(f, RAIZ))
    assert not usos, usos


def test_q_learning_treina_e_desenha_os_graficos_e_a_animacao(pasta):
    at = AppTest.from_string(f"""
import sys; sys.path.insert(0, {RAIZ!r})
import streamlit as st
st.session_state.setdefault("tema", "claro")
from modules.reforco.q_learning import render_q_learning
render_q_learning("aluno")
""", default_timeout=180).run()
    assert not at.exception
    _botao(at, "ql_run").click().run()
    assert not at.exception and len(at.get("plotly_chart")) >= 1
    _botao(at, "ql_anim").click().run()
    assert not at.exception


def test_laboratorio_compara_modelos_e_desenha_resultados(pasta):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=240)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = "aluno"
    at.session_state["name"] = "Aluno"
    at.session_state["pagina"] = "laboratorio"
    at.run()
    assert not at.exception
    _botao(at, "lab_demo_usar").click().run()
    assert not at.exception
    _botao(at, "lab_tr_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) >= 1
