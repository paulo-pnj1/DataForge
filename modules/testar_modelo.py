"""Testar Modelo: escolhe QUALQUER modelo que guardaste e experimenta-o com dados novos.

Funciona com tudo o que se guarda no DataForge: classificação, regressão, clustering, anomalias e séries temporais
(treinados em qualquer página). Um caso de cada vez, vários casos numa tabela, ou um ficheiro inteiro.
O modelo de exemplo «Previsor de Alunos (KNN)» continua disponível na lista.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from modules.ml_engine import data_loader, model_persistence as mp, prediction_service as ps
from modules.ml_engine.pacote_modelo import NOMES_TAREFA
from modules.utils import page_header

EXEMPLO_ID = "__exemplo_alunos__"
NOMES_TIPO = {**NOMES_TAREFA, "serie": "Série temporal"}


def _ir(pagina: str):
    st.session_state.pagina = pagina
    if "guiado_step" in st.session_state:
        st.session_state.guiado_step = 1


@st.cache_resource(show_spinner=False, max_entries=8)
def _abrir(username: str, modelo_id: str):
    return mp.carregar_modelo(username, modelo_id)


def _exemplo_existe() -> bool:
    try:
        from modules import previsor_alunos as pa
        return os.path.exists(pa.CAMINHO_MODELO)
    except Exception:
        return False


def _resumo(meta: dict):
    tipo = meta.get("tipo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tipo", NOMES_TIPO.get(tipo, str(tipo)))
    c2.metric("Algoritmo", str(meta.get("algoritmo", "–")))
    met = meta.get("metricas_teste") or {}
    for k in ("accuracy", "acc", "r2", "R2", "f1"):
        if isinstance(met.get(k), (int, float)):
            c3.metric({"accuracy": "Exactidão", "acc": "Exactidão", "r2": "R²", "R2": "R²", "f1": "F1"}[k], f"{met[k]:.3f}")
            break
    else:
        c3.metric("Criado em", str(meta.get("criado_em", "–"))[:16])
    if tipo in ("classificacao", "regressao"):
        st.caption(f"Prevê **{meta.get('alvo')}** a partir de {len(meta.get('features') or [])} variáveis: "
                   + ", ".join(map(str, (meta.get("features") or [])[:12])) + ("…" if len(meta.get("features") or []) > 12 else "") + ".")
    elif tipo in ("agrupamento", "anomalias"):
        st.caption(f"Usa {len(meta.get('features') or [])} variáveis: " + ", ".join(map(str, (meta.get("features") or [])[:12])) + ".")


def _passo(lo, hi):
    rng = (hi - lo) if lo is not None and hi is not None else 0
    return 1.0 if rng >= 20 else 0.1 if rng >= 2 else 0.01


def _campos(meta: dict, mid: str, pref: str) -> dict:
    """Um campo por variável do modelo (lista de opções para categorias, número para o resto)."""
    valores = {}
    cols = st.columns(2)
    cats = meta.get("cat_cols") or []
    datas = meta.get("colunas_data") or []
    for k, f in enumerate(meta.get("features") or []):
        col = cols[k % 2]
        chave = f"{pref}_{mid}_{f}"
        if f in cats:
            op = sorted(map(str, (meta.get("categorias_conhecidas") or {}).get(f, [])))
            valores[f] = col.selectbox(str(f), op, key=chave) if op else col.text_input(str(f), key=chave)
        elif f in datas:
            valores[f] = str(col.date_input(str(f), key=chave))
        else:
            iv = (meta.get("intervalos_numericos") or {}).get(f)
            ok = bool(iv and len(iv) >= 2 and iv[0] is not None and iv[1] is not None)
            lo, hi = (float(iv[0]), float(iv[1])) if ok else (None, None)
            padrao = round((lo + hi) / 2, 4) if ok else 0.0
            valores[f] = col.number_input(str(f), value=float(padrao), step=_passo(lo, hi), format="%g", key=chave,
                                          help=f"Visto no treino: {lo:g} a {hi:g}" if ok else None)
    return valores


def _mostrar_um(meta: dict, previsao, detalhe: dict, rel: dict):
    tipo = meta["tipo"]
    if tipo == "classificacao":
        st.metric(f"Previsão de {meta.get('alvo')}", str(previsao))
        if detalhe:
            tab = pd.DataFrame({"Classe": [k[5:] for k in detalhe], "Probabilidade": list(detalhe.values())})
            st.bar_chart(tab.set_index("Classe"), height=180)
            st.dataframe(tab.sort_values("Probabilidade", ascending=False), hide_index=True, width="stretch")
    elif tipo == "regressao":
        st.metric(f"Previsão de {meta.get('alvo')}", f"{float(previsao):.4g}")
    elif tipo == "agrupamento":
        st.metric("Este caso pertence a", str(previsao))
        st.caption("Os grupos são números sem nome: compara este caso com os outros do mesmo grupo para perceber o que os une.")
    elif tipo == "anomalias":
        st.metric("Veredicto", str(previsao))
        if previsao == "Anomalia":
            st.warning("Este caso é muito diferente do que o modelo viu no treino.")
        else:
            st.success("Este caso parece normal para o modelo.")
    for a in rel.get("avisos", []):
        st.warning(a)


def _modo_caso(modelo, meta, mid, pref):
    valores = _campos(meta, mid, pref)
    if st.button("Prever", type="primary", key=f"{pref}_{mid}_go"):
        try:
            p, det, rel = ps.prever_um(modelo, meta, valores)
        except ps.ErroPrevisao as e:
            st.error(str(e))
            return
        except Exception:
            st.error("O modelo não conseguiu prever este caso. Verifica os valores ou treina o modelo outra vez.")
            return
        _mostrar_um(meta, p, det, rel)


def _tabela_modelo(meta: dict) -> pd.DataFrame:
    """Três linhas de exemplo (centro do intervalo visto no treino; primeira categoria) para a pessoa editar."""
    linha = {}
    for f in meta.get("features") or []:
        if f in (meta.get("cat_cols") or []):
            op = sorted(map(str, (meta.get("categorias_conhecidas") or {}).get(f, [])))
            linha[f] = op[0] if op else ""
        elif f in (meta.get("colunas_data") or []):
            linha[f] = str(pd.Timestamp.today().date())
        else:
            iv = (meta.get("intervalos_numericos") or {}).get(f)
            linha[f] = round((iv[0] + iv[1]) / 2, 4) if iv and None not in iv[:2] else 0.0
    return pd.DataFrame([dict(linha) for _ in range(3)])


def _mostrar_tabela(saida: pd.DataFrame, rel: dict, pref: str, mid: str):
    for a in rel.get("avisos", []):
        st.warning(a)
    st.dataframe(saida.head(500), width="stretch")
    st.download_button("Descarregar previsões (CSV)", saida.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv",
                       key=f"{pref}_{mid}_dl")


def _modo_varios(modelo, meta, mid, pref):
    st.caption("Edita os valores, acrescenta linhas (+ no fim da tabela) e carrega em «Prever».")
    base = _tabela_modelo(meta)
    novo = st.data_editor(base, num_rows="dynamic", key=f"{pref}_{mid}_ed", width="stretch")
    if st.button("Prever todas as linhas", type="primary", key=f"{pref}_{mid}_go_v"):
        try:
            saida, rel = ps.prever_tabela(modelo, meta, pd.DataFrame(novo))
        except ps.ErroPrevisao as e:
            st.error(str(e))
            return
        except Exception:
            st.error("O modelo não conseguiu prever estas linhas. Verifica se não há células vazias.")
            return
        _mostrar_tabela(saida, rel, pref, mid)


def _modo_ficheiro(modelo, meta, mid, pref):
    st.caption("O ficheiro precisa de ter as colunas: " + ", ".join(map(str, meta.get("features") or [])) + ".")
    f = st.file_uploader("Ficheiro com casos novos (CSV ou Excel)", type=["csv", "xlsx", "xlsm"], key=f"{pref}_{mid}_file")
    if f is None:
        return
    try:
        novo = data_loader.carregar_ficheiro(f.name, f.getvalue())
        saida, rel = ps.prever_tabela(modelo, meta, novo)
    except (data_loader.ErroDados, ps.ErroPrevisao) as e:
        st.error(str(e))
        return
    except Exception:
        st.error("Não consegui prever com este ficheiro.")
        return
    _mostrar_tabela(saida, rel, pref, mid)


def _modo_serie(modelo, meta, mid, pref):
    h = int(st.number_input("Períodos a prever", 1, 365, int(meta.get("horizonte", 12) or 12), key=f"{pref}_{mid}_h"))
    try:
        out = ps.prever_futuro(modelo, meta, h)
    except ps.ErroPrevisao as e:
        st.error(str(e))
        return
    except Exception:
        st.error("O modelo não conseguiu prever. Treina-o outra vez.")
        return
    st.caption(f"Previsões a partir de {meta.get('ultima_data')}, o fim dos dados com que o modelo foi treinado.")
    st.dataframe(out, width="stretch", hide_index=True)
    st.download_button("Descarregar (CSV)", out.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv", key=f"{pref}_{mid}_dl_s")


def render_testar_modelo(username: str, pref: str = "tm", com_cabecalho: bool = True):
    if com_cabecalho:
        page_header("Testar Modelo", "Escolhe um modelo que guardaste e experimenta-o com dados novos.")
    try:
        modelos = mp.listar_modelos(username)
    except mp.ErroPersistencia as e:
        st.error(str(e))
        return
    exemplo = _exemplo_existe()
    if not modelos and not exemplo:
        st.info("Ainda não guardaste nenhum modelo. Treina um em qualquer página (Classificação, Regressão, MLP, Clustering, "
                "Laboratório…) e carrega em «Guardar modelo». Depois volta aqui para o testar.")
        c1, c2 = st.columns(2)
        c1.button("Ir para Classificação", key=f"{pref}_ir_clf", on_click=_ir, args=("classificacao",))
        c2.button("Ir para o Laboratório de ML", key=f"{pref}_ir_lab", on_click=_ir, args=("laboratorio",))
        return

    nomes = {m["id"]: f"{m['nome']} · v{m.get('versao', 1)} · {NOMES_TIPO.get(m.get('tipo'), m.get('tipo'))}" for m in modelos}
    if exemplo:
        nomes[EXEMPLO_ID] = "Exemplo do projecto: Previsor de Alunos (KNN)"
    mid = st.selectbox("Modelo a testar", list(nomes), format_func=nomes.get, key=f"{pref}_modelo")

    if mid == EXEMPLO_ID:
        from modules.previsor_alunos import render_previsor_alunos
        st.divider()
        render_previsor_alunos(username, com_cabecalho=False)
        return

    meta0 = next(m for m in modelos if m["id"] == mid)
    _resumo(meta0)
    try:
        modelo, meta = _abrir(username, mid)
    except mp.ErroPersistencia as e:
        st.error(str(e))
        return
    except Exception:
        st.error("Não consegui abrir este modelo. Pode ter sido guardado com outra versão das bibliotecas: treina-o e guarda-o outra vez.")
        return
    if meta.get("aviso_versoes"):
        st.warning(meta["aviso_versoes"])
    st.divider()
    if meta.get("tipo") == "serie":
        _modo_serie(modelo, meta, mid, pref)
        return
    modo = st.radio("Como queres testar?", ["Um caso", "Vários casos (tabela)", "Um ficheiro"], horizontal=True, key=f"{pref}_modo")
    if modo == "Um caso":
        _modo_caso(modelo, meta, mid, pref)
    elif modo == "Vários casos (tabela)":
        _modo_varios(modelo, meta, mid, pref)
    else:
        _modo_ficheiro(modelo, meta, mid, pref)
