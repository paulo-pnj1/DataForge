"""Botões convertidos de st.rerun() para on_click: cada clique deve mudar de página/passo e correr o script só uma vez."""
import os

import pytest
from streamlit.testing.v1 import AppTest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def pasta_temporaria(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in ("SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(v, raising=False)


def _app(pagina, **extra):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = "aluno"
    at.session_state["name"] = "Aluno"
    at.session_state["pagina"] = pagina
    for k, v in extra.items():
        at.session_state[k] = v
    return at.run()


def _botao(at, key):
    return [b for b in at.button if b.key == key][0]


def test_escola_abrir_e_voltar():
    at = _app("escola")
    assert not at.exception
    abrir = [b for b in at.button if (b.key or "").startswith("esc_abrir_")][0]
    id_ = abrir.key.replace("esc_abrir_", "")
    abrir.click().run()
    assert not at.exception and at.session_state["escola_aberta"] == id_
    _botao(at, "esc_voltar").click().run()
    assert not at.exception and at.session_state["escola_aberta"] is None


def test_escola_seguinte_e_anterior():
    at = _app("escola")
    primeiro = [b for b in at.button if (b.key or "").startswith("esc_abrir_")][0]
    id1 = primeiro.key.replace("esc_abrir_", "")
    primeiro.click().run()
    _botao(at, "esc_seg").click().run()
    assert not at.exception and at.session_state["escola_aberta"] != id1
    _botao(at, "esc_ant").click().run()
    assert at.session_state["escola_aberta"] == id1


def test_classificacao_continuar_avanca_o_passo():
    at = _app("classificacao")
    assert not at.exception
    antes = at.session_state["guiado_step"]
    _botao(at, "step1_next").click().run()
    assert not at.exception and at.session_state["guiado_step"] == max(antes, 2)
    _botao(at, "step2_next").click().run()
    assert not at.exception and at.session_state["guiado_step"] == 3


def _numerico():
    from modules.utils import carregar_dataset_embutido
    df, _, _ = carregar_dataset_embutido("Iris - Classificação de flores")
    return df.select_dtypes("number")


def test_clustering_passos():
    at = _app("clustering", cl_step=2, cl_df=_numerico(), cl_nome="Iris")
    assert not at.exception
    _botao(at, "cl_s2").click().run()
    assert not at.exception and at.session_state["cl_step"] == 3
    _botao(at, "cl_s3b").click().run()
    assert not at.exception and at.session_state["cl_step"] == 2
    _botao(at, "cl_s2b").click().run()
    assert not at.exception and at.session_state["cl_step"] == 1


def test_projeto_voltar_e_novo():
    from modules.projeto import PROJETOS
    from modules.utils import carregar_dataset_embutido
    ds, tipo, ctx = list(PROJETOS.values())[0]
    df, _, _ = carregar_dataset_embutido(ds)
    at = _app("projeto", proj={"dataset": ds, "tipo": tipo, "contexto": ctx, "alvo": df.columns[-1], "n": len(df), "passo": 1})
    assert not at.exception
    _botao(at, "proj_b1").click().run()
    assert not at.exception and at.session_state["proj"]["passo"] == 0


def test_o_menu_lateral_corre_o_script_uma_so_vez():
    at = _app("dashboard")
    _botao(at, "nav_escola").click().run()
    assert not at.exception and at.session_state["pagina"] == "escola"
