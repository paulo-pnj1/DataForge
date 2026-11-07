"""Bloco «Guardar modelo» comum a todas as páginas de treino.

Cada página só diz COMO construir (objecto, metadados) a partir do seu resultado; aqui fica o resto: nome, botão, erros e o
atalho para testar o modelo com dados novos na página «Testar Modelos».
"""
import logging

import streamlit as st

from modules.ml_engine import model_persistence as mp
from modules.ml_engine.ciclo_ml import ErroCiclo
from modules.ml_engine.modelo_legado import ErroLegado
from modules.utils import ir_para

log = logging.getLogger("dataforge.ml")


def bloco_guardar(username: str, chave: str, nome_sugerido: str, construir):
    """`construir(nome)` devolve (objecto, metadados). `chave` tem de mudar quando o resultado muda (ex.: f"clf_{id(r)}")."""
    st.caption("Guarda este modelo na tua conta para o testares com dados novos mais tarde, sem treinar outra vez. "
               "Fica só visível para ti e é verificado com uma assinatura antes de ser aberto.")
    nome = st.text_input("Nome do modelo", value=nome_sugerido, max_chars=80, key=f"gd_nome_{chave}")
    if st.button("Guardar modelo", type="primary", key=f"gd_go_{chave}"):
        try:
            objecto, meta = construir(nome)
            st.session_state[f"gd_ok_{chave}"] = mp.guardar_modelo(username, objecto, meta)
        except (mp.ErroPersistencia, ErroLegado, ErroCiclo) as e:
            st.error(str(e))
        except Exception:
            log.exception("Falha ao guardar o modelo (%s)", chave)
            st.error("Não consegui guardar o modelo.")
    info = st.session_state.get(f"gd_ok_{chave}")
    if info:
        st.success(f"Guardado como «{info['nome']}» (versão {info['versao']}).")
        if st.button("Testar este modelo com dados novos", key=f"gd_testar_{chave}"):
            st.session_state["tm_mod"] = info["id"]
            ir_para("testar_modelos")
