"""Tema escuro/claro: paleta, contraste, cores em HTML vs gráficos e adaptação dos gráficos. `pytest -q`."""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)


def _lum(h: str) -> float:
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contraste(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_cor_e_hex_em_python_e_variavel_css_em_html():
    from modules.utils import C_ACCENT, C_TEXT_MUTE, TEMAS
    assert C_ACCENT == TEMAS["escuro"]["accent"]                       # gráficos recebem o hex
    assert C_ACCENT.lstrip("#") == TEMAS["escuro"]["accent"].lstrip("#")
    assert f"{C_ACCENT}" == f"var(--df-accent, {TEMAS['escuro']['accent']})"   # HTML recebe a variável
    assert f"{C_TEXT_MUTE}".startswith("var(--df-text-mute,")


def test_os_dois_temas_definem_as_mesmas_cores():
    from modules.utils import TEMAS
    assert set(TEMAS["escuro"]) == set(TEMAS["claro"])


def test_tema_claro_tem_contraste_legivel():
    from modules.utils import TEMAS
    P = TEMAS["claro"]
    assert _contraste(P["text"], P["bg"]) >= 7
    for cor in ("text_sec", "text_mute", "text2", "text3", "accent", "accent2", "green", "amber", "red", "teal"):
        assert _contraste(P[cor], P["bg"]) >= 4.5, cor
        assert _contraste(P[cor], P["surface"]) >= 4.2, cor            # também sobre os cards
    assert _contraste(P["on_accent"], P["accent"]) >= 4.5              # texto dos botões primários


def test_css_vars_acompanham_o_tema(monkeypatch):
    from modules import utils
    monkeypatch.setattr(utils, "tema_actual", lambda: "claro")
    assert "--df-bg: #FFFFFF;" in utils._css_vars()
    monkeypatch.setattr(utils, "tema_actual", lambda: "escuro")
    assert "--df-bg: #0D1117;" in utils._css_vars()


def _figura_escura():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(facecolor="#161B27")
    ax.set_facecolor("#161B27")
    ax.plot([0, 1, 2], [0, 1, 4], color="#58A6FF", label="modelo")
    ax.set_title("Título", color="#FFFFFF")
    ax.set_xlabel("x", color="#9BA3B2")
    ax.tick_params(colors="#9BA3B2")
    ax.text(1, 1, "nota", color="#FFFFFF")
    ax.legend(facecolor="#1E2535", edgecolor="#2A3347", labelcolor="#9BA3B2")
    return fig, ax


def test_figura_matplotlib_no_tema_claro_fica_legivel():
    from matplotlib.colors import to_hex
    from modules.utils import TEMAS, tema_figura_matplotlib
    P = TEMAS["claro"]
    fig, ax = _figura_escura()
    tema_figura_matplotlib(fig, "claro")
    assert to_hex(fig.get_facecolor()).upper() == P["surface"]
    assert to_hex(ax.get_facecolor()).upper() == P["surface"]
    assert to_hex(ax.title.get_color()).upper() == P["text"]
    assert to_hex(ax.texts[0].get_color()).upper() == P["text"]
    assert to_hex(ax.get_legend().get_texts()[0].get_color()).upper() == P["text"]
    assert to_hex(ax.lines[0].get_color()).upper() == "#58A6FF"        # as cores dos dados não mudam


def test_figura_matplotlib_no_tema_escuro_nao_muda_nada():
    from matplotlib.colors import to_hex
    from modules.utils import tema_figura_matplotlib
    fig, ax = _figura_escura()
    tema_figura_matplotlib(fig, "escuro")
    assert to_hex(fig.get_facecolor()).upper() == "#161B27"
    assert to_hex(ax.title.get_color()).upper() == "#FFFFFF"


def test_mapa_de_calor_mantem_o_texto_das_celulas():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_hex
    from modules.utils import tema_figura_matplotlib
    fig, ax = plt.subplots(facecolor="#161B27")
    ax.imshow(np.array([[9, 1], [1, 9]]), cmap="Blues")
    t = ax.text(0, 0, "9", color="white")
    tema_figura_matplotlib(fig, "claro")
    assert to_hex(t.get_color()).upper() == "#FFFFFF"                  # branco sobre célula azul escura


def test_figura_plotly_no_tema_claro():
    go = pytest.importorskip("plotly.graph_objects")
    from modules.utils import TEMAS, tema_figura_plotly
    P = TEMAS["claro"]
    fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
    fig.update_layout(paper_bgcolor="#161B27", plot_bgcolor="#161B27", font=dict(color="#FFFFFF"),
                      xaxis=dict(color="#FFFFFF"))
    tema_figura_plotly(fig, "claro")
    assert fig.layout.paper_bgcolor == P["surface"]
    assert fig.layout.font.color == P["text"]
    assert fig.layout.xaxis.color == P["text_sec"]
