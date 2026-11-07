"""Testar Modelos: escolhe QUALQUER modelo que guardaste e experimenta-o com dados novos (um caso, um ficheiro, ou o futuro de uma série).

Funciona com os modelos guardados em qualquer página: Classificação, Regressão, Comparador, Laboratório de ML,
Avaliar e Optimizar e «O Meu Primeiro Modelo».
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ml_engine import ciclo_ml as cl, model_evaluation as ev, model_persistence as mp, prediction_service as ps
from modules.ml_engine.data_loader import ErroDados, carregar_ficheiro
from modules.utils import ir_para, page_header, section_title


def _vazio():
    st.info("Ainda não guardaste nenhum modelo. Treina um em qualquer página e carrega em «Guardar modelo»: ele aparece aqui.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Treinar uma classificação", key="tm_v_clf", width="stretch"):
        ir_para("classificacao")
    if c2.button("Treinar uma regressão", key="tm_v_reg", width="stretch"):
        ir_para("regressao")
    if c3.button("Laboratório de ML", key="tm_v_lab", width="stretch"):
        ir_para("laboratorio")


def _cartao(meta: dict):
    tipo = meta.get("tipo")
    m = meta.get("metricas_teste") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algoritmo", meta.get("algoritmo", "—"))
    c2.metric("O que prevê", str(meta.get("alvo", "—")))
    if tipo == "classificacao":
        c3.metric("F1 / Accuracy (teste)", " / ".join(f"{m[k]:.1%}" for k in ("f1", "accuracy") if isinstance(m.get(k), (int, float))) or "—")
    elif tipo == "regressao":
        c3.metric("R² (teste)", f"{m['r2']:.3f}" if isinstance(m.get("r2"), (int, float)) else "—")
    else:
        c3.metric("Tipo", "Série temporal")
    c4.metric("Guardado em", str(meta.get("criado_em", "—")))


def _o_que_precisa(meta: dict):
    iv = meta.get("intervalos_numericos") or {}
    cats = meta.get("categorias_conhecidas") or {}
    datas = set(meta.get("colunas_data") or [])
    linhas = []
    for f in meta.get("features") or []:
        if f in datas:
            linhas.append({"Coluna": f, "Tipo": "data", "Valores vistos no treino": "—"})
        elif f in cats:
            linhas.append({"Coluna": f, "Tipo": "categoria", "Valores vistos no treino": ", ".join(map(str, cats[f][:8])) + ("…" if len(cats[f]) > 8 else "")})
        else:
            v = iv.get(f)
            linhas.append({"Coluna": f, "Tipo": "número", "Valores vistos no treino": f"{v[0]:.4g} a {v[1]:.4g}" if v and None not in v[:2] else "—"})
    if linhas:
        st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)


def _formulario(meta: dict, chave: str) -> dict:
    valores, cols = {}, st.columns(2)
    for k, f in enumerate(meta["features"]):
        c = cols[k % 2]
        if f in meta.get("cat_cols", []):
            op = sorted(meta.get("categorias_conhecidas", {}).get(f, []))
            valores[f] = c.selectbox(f, op, key=f"{chave}_{f}") if op else c.text_input(f, key=f"{chave}_{f}")
        elif f in meta.get("colunas_data", []):
            valores[f] = str(c.date_input(f, key=f"{chave}_{f}"))
        else:
            iv = (meta.get("intervalos_numericos") or {}).get(f)
            ajuda = f"Visto no treino: {iv[0]:.4g} a {iv[1]:.4g}" if iv and None not in iv[:2] else None
            valores[f] = c.number_input(f, value=float(iv[2]) if iv and len(iv) > 2 and iv[2] is not None else 0.0, key=f"{chave}_{f}", help=ajuda)
    return valores


def _serie(modelo, meta: dict, p: str):
    h = int(st.number_input("Períodos a prever", 1, 365, int(meta.get("horizonte", 12)), key=f"{p}_h"))
    try:
        out = ps.prever_futuro(modelo, meta, h)
    except ps.ErroPrevisao as e:
        st.error(str(e))
        return
    except Exception:
        st.error("O modelo não conseguiu prever. Treina-o outra vez.")
        return
    st.caption(f"Previsões a partir de {meta['ultima_data']}, o fim dos dados com que o modelo foi treinado.")
    st.dataframe(out, width="stretch", hide_index=True)
    st.download_button("Descarregar (CSV)", out.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv", key=f"{p}_dl_s")


def _um_caso(modelo, meta: dict, p: str, mid: str):
    valores = _formulario(meta, f"{p}_{mid}")
    if not st.button("Prever", type="primary", key=f"{p}_go"):
        return
    try:
        previsao, det, rel = ps.prever_um(modelo, meta, valores)
    except ps.ErroPrevisao as e:
        st.error(str(e))
        return
    st.metric(f"Previsão de {meta['alvo']}", f"{previsao:.4g}" if isinstance(previsao, (int, float, np.floating)) else str(previsao))
    if det:
        tab = pd.DataFrame({"Classe": [k[5:] for k in det], "Probabilidade": list(det.values())})
        fig = go.Figure(go.Bar(x=tab["Classe"], y=tab["Probabilidade"]))
        fig.update_layout(title="Probabilidade de cada classe", yaxis_range=[0, 1], height=280, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    for a in rel["avisos"]:
        st.warning(a)


def _ficheiro(modelo, meta: dict, p: str):
    st.download_button("Descarregar um CSV de exemplo com as colunas obrigatórias", cl.csv_modelo(meta), file_name="exemplo_entrada.csv", mime="text/csv", key=f"{p}_modelo")
    f = st.file_uploader("Ficheiro com os casos novos (CSV ou Excel)", type=["csv", "xlsx", "xlsm"], key=f"{p}_file")
    if f is None:
        return
    try:
        novo = carregar_ficheiro(f.name, f.getvalue())
        saida, rel = cl.prever_ficheiro(modelo, meta, novo)
    except (ErroDados, cl.ErroCiclo) as e:
        st.error(str(e))
        return
    for a in rel["avisos"]:
        st.warning(a)
    nome = f"previsao_{meta['alvo']}"
    st.success(f"{len(saida)} previsões feitas.")
    st.dataframe(saida.head(200), width="stretch")
    if meta["tipo"] == "classificacao":
        vc = saida[nome].astype(str).value_counts()
        fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist()))
    else:
        fig = go.Figure(go.Histogram(x=saida[nome], nbinsx=30))
    fig.update_layout(title="Distribuição das previsões", height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.download_button("Descarregar as previsões (CSV)", saida.to_csv(index=False).encode("utf-8"), file_name="previsoes.csv", mime="text/csv", key=f"{p}_dl")
    if meta["alvo"] in novo.columns:                  # o ficheiro já traz a resposta certa: compara
        try:
            m = cl.metricas_com_valores_reais(meta, saida[nome].tolist(), novo[meta["alvo"]])
            st.markdown(f"**O ficheiro tem a coluna «{meta['alvo']}»: comparei as previsões com os valores reais**")
            for a in m["avisos"]:
                st.warning(a)
            st.dataframe(pd.DataFrame([{"Métrica": ev.rotulo(meta["tipo"], k), "Nestes dados": v} for k, v in m["metricas"].items()]).style.format({"Nestes dados": "{:.4f}"}),
                         width="stretch", hide_index=True)
        except cl.ErroCiclo as e:
            st.caption(str(e))


def painel_previsao(username: str, p: str = "tm"):
    """Escolher um modelo guardado e testá-lo. Usado nesta página e no separador «Previsões» do Laboratório de ML."""
    try:
        modelos = mp.listar_modelos(username)
    except mp.ErroPersistencia as e:
        st.error(str(e))
        return
    if not modelos:
        _vazio()
        return
    nomes = {m["id"]: f"{m['nome']} · v{m['versao']} · {m['algoritmo']} ({m['criado_em']})" for m in modelos}
    mid = st.selectbox("Escolhe o modelo", list(nomes), format_func=nomes.get, key=f"{p}_mod")
    try:
        modelo, meta = mp.carregar_modelo(username, mid)
    except mp.ErroPersistencia as e:
        st.error(str(e))
        return
    except Exception:
        st.error("Não consegui abrir este modelo.")
        return
    if meta.get("aviso_versoes"):
        st.warning(meta["aviso_versoes"])
    _cartao(meta)
    if meta.get("tipo") == "serie":
        _serie(modelo, meta, p)
        return
    with st.expander("O que este modelo precisa para prever"):
        _o_que_precisa(meta)
    modo = st.radio("Como queres testar?", ["Um caso", "Um ficheiro"], horizontal=True, key=f"{p}_modo")
    if modo == "Um caso":
        _um_caso(modelo, meta, p, mid)
    else:
        _ficheiro(modelo, meta, p)


def render_testar_modelos(username: str):
    page_header("Testar Modelos", "Experimenta qualquer modelo guardado com dados novos: um caso à mão ou um ficheiro inteiro.")
    st.caption("Os dados que usas aqui não entram no treino: servem só para ver o que o modelo responde.")
    section_title("O TEU MODELO")
    painel_previsao(username, "tm")
