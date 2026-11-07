"""Bloco «Guardar este modelo», igual em todas as páginas onde se treina um modelo.

Cada página só tem de chamar `bloco_guardar(...)` com o modelo que acabou de treinar e os passos de preparação
que usou (scaler, encoders...). O resto — empacotar, assinar, guardar por utilizador — é feito aqui.
"""
import streamlit as st

from modules.ml_engine import model_persistence as mp, pacote_modelo as pm


def _ir_testar():
    st.session_state.pagina = "testar_modelo"
    if "guiado_step" in st.session_state:
        st.session_state.guiado_step = 1


def bloco_guardar(username: str, estimador, *, chave: str, tarefa: str, features, alvo: str, algoritmo: str,
                  X_ref, origem: str, nome_padrao: str = "", scaler=None, encoders=None, le_target=None,
                  colunas_dummy=None, metricas=None, parametros=None, n_treino=None, n_teste=None, aberto: bool = True):
    """Desenha o bloco e guarda quando a pessoa carrega no botão."""
    motivo = pm.pode_guardar(estimador, tarefa)
    with st.expander("💾 Guardar este modelo para usar mais tarde", expanded=aberto):
        if motivo:
            st.info(motivo)
            return
        st.caption("Guardado, podes testá-lo com dados novos (um caso ou um ficheiro) na página «Testar Modelo». "
                   "Só tu o vês e fica protegido por uma assinatura.")
        nome = st.text_input("Nome do modelo", value=nome_padrao or f"{algoritmo} – {alvo}", key=f"{chave}_gd_nome", max_chars=80)
        if st.button("Guardar modelo", type="primary", key=f"{chave}_gd_btn"):
            try:
                pacote, meta = pm.construir(
                    estimador, tarefa=tarefa, features=features, alvo=alvo, algoritmo=algoritmo, nome=nome, origem=origem,
                    X_ref=X_ref, scaler=scaler, encoders=encoders, le_target=le_target, colunas_dummy=colunas_dummy,
                    metricas=pm.metricas_simples(metricas), parametros=parametros, n_treino=n_treino, n_teste=n_teste)
                info = mp.guardar_modelo(username, pacote, meta)
                st.session_state[f"{chave}_gd_ok"] = f"Guardado como «{info['nome']}» (versão {info['versao']})."
            except (pm.ErroEmpacotar, mp.ErroPersistencia) as e:
                st.session_state.pop(f"{chave}_gd_ok", None)
                st.error(str(e))
            except Exception:
                st.session_state.pop(f"{chave}_gd_ok", None)
                st.error("Não consegui guardar este modelo.")
        if st.session_state.get(f"{chave}_gd_ok"):
            st.success(st.session_state[f"{chave}_gd_ok"])
            st.button("Testar com dados novos →", key=f"{chave}_gd_ir", on_click=_ir_testar)
