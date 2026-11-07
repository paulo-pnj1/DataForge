"""Bloco «Guardar / Descarregar modelo» comum a todas as páginas de treino.

Cada página só diz COMO construir (objecto, metadados) a partir do seu resultado; aqui fica o resto: nome, guardar na conta,
descarregar o ZIP (qualquer algoritmo) e o atalho para testar o modelo com dados novos na página «Testar Modelos».
"""
import logging
import re

import streamlit as st

from modules.ml_engine import model_persistence as mp
from modules.ml_engine.ciclo_ml import ErroCiclo
from modules.ml_engine.modelo_legado import ErroLegado
from modules.ml_engine.modelos_extra import exportar_modelo
from modules.utils import ir_para

log = logging.getLogger("dataforge.ml")


def _ficheiro(nome: str) -> str:
    return (re.sub(r"[^\w\-]+", "_", str(nome), flags=re.UNICODE).strip("_") or "modelo")[:60] + ".zip"


def bloco_guardar(username: str, chave: str, nome_sugerido: str, construir):
    """`construir(nome)` devolve (objecto, metadados). `chave` tem de mudar quando o resultado muda (ex.: f"clf_{id(r)}")."""
    st.caption("Guarda este modelo na tua conta para o testares com dados novos mais tarde, sem treinar outra vez, ou descarrega-o para usar "
               "noutro computador. Na conta fica só visível para ti e é verificado com uma assinatura antes de ser aberto.")
    nome = st.text_input("Nome do modelo", value=nome_sugerido, max_chars=80, key=f"gd_nome_{chave}")
    c1, c2 = st.columns(2)
    if c1.button("Guardar modelo", type="primary", key=f"gd_go_{chave}", width="stretch"):
        try:
            objecto, meta = construir(nome)
            st.session_state[f"gd_ok_{chave}"] = mp.guardar_modelo(username, objecto, meta)
        except (mp.ErroPersistencia, ErroLegado, ErroCiclo) as e:
            st.error(str(e))
        except Exception:
            log.exception("Falha ao guardar o modelo (%s)", chave)
            st.error("Não consegui guardar o modelo.")
    if c2.button("Preparar ficheiro para descarregar", key=f"gd_prep_{chave}", width="stretch"):
        try:
            objecto, meta = construir(nome)
            st.session_state[f"gd_zip_{chave}"] = (exportar_modelo(objecto, meta), nome)
        except (ErroLegado, ErroCiclo) as e:
            st.error(str(e))
        except Exception:
            log.exception("Falha ao preparar o modelo para descarregar (%s)", chave)
            st.error("Não consegui preparar o ficheiro.")
    pronto = st.session_state.get(f"gd_zip_{chave}")
    if pronto:
        st.download_button("Descarregar o modelo (.zip: modelo, metadados e código de exemplo)", pronto[0], file_name=_ficheiro(pronto[1]),
                           mime="application/zip", key=f"gd_dl_{chave}")
    info = st.session_state.get(f"gd_ok_{chave}")
    if info:
        st.success(f"Guardado como «{info['nome']}» (versão {info['versao']}).")
        if st.button("Testar este modelo com dados novos", key=f"gd_testar_{chave}"):
            st.session_state["tm_mod"] = info["id"]
            ir_para("testar_modelos")
