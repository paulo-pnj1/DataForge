"""Testar Modelos: escolhe QUALQUER modelo que guardaste e experimenta-o com dados novos (um caso, um ficheiro, ou o futuro de uma série).

Funciona com os modelos guardados em qualquer página: Classificação, Regressão, Comparador, Laboratório de ML,
Avaliar e Optimizar e «O Meu Primeiro Modelo».
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ml_engine import ciclo_ml as cl, model_evaluation as ev, model_persistence as mp, prediction_service as ps
from modules.ml_engine import importar_legado as il
from modules.ml_engine import modelos_extra as mx
from modules.ml_engine.modelo_legado import ErroLegado
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
    c2.metric("O que prevê", str(meta.get("alvo") or {"cluster": "o grupo", "anomalia": "se é anomalia", "reducao": "as componentes", "reforco_q": "a melhor acção",
                                                     "reforco_dqn": "a melhor acção"}.get(tipo, "—")))
    if tipo == "classificacao":
        c3.metric("F1 / Accuracy (teste)", " / ".join(f"{m[k]:.1%}" for k in ("f1", "accuracy") if isinstance(m.get(k), (int, float))) or "—")
    elif tipo == "regressao":
        c3.metric("R² (teste)", f"{m['r2']:.3f}" if isinstance(m.get("r2"), (int, float)) else "—")
    elif tipo == "cluster":
        c3.metric("Grupos", str(m.get("grupos", "—")))
    elif tipo == "anomalia":
        c3.metric("Anomalias no treino", str(m.get("anomalias", "—")))
    elif tipo == "reducao":
        c3.metric("Componentes", str(m.get("componentes", "—")))
    elif tipo == "reforco_q":
        c3.metric("Chega ao objectivo", "sim" if m.get("chegou_ao_objectivo") else "não")
    elif tipo == "reforco_dqn":
        c3.metric("Média dos últimos 50", f"{m['media_ultimos_50']:.1f}" if isinstance(m.get("media_ultimos_50"), (int, float)) else "—")
    else:
        from modules.ml_engine.pacote_generico import TIPOS as _TIPOS_DL
        c3.metric("Tipo", _TIPOS_DL.get(tipo, "Série temporal"))
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


def _grafico_resultado(meta: dict, saida: pd.DataFrame):
    tipo = meta["tipo"]
    if tipo == "cluster":
        vc = saida["grupo"].astype(str).value_counts()
        fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist()))
        fig.update_layout(title="Casos em cada grupo")
    elif tipo == "anomalia":
        vc = saida["resultado"].value_counts()
        fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist()))
        fig.update_layout(title="Normais e anomalias")
    elif tipo == "reducao" and "componente_2" in saida.columns:
        fig = go.Figure(go.Scatter(x=saida["componente_1"], y=saida["componente_2"], mode="markers"))
        fig.update_layout(title="Casos novos nas duas primeiras componentes", xaxis_title="componente 1", yaxis_title="componente 2")
    elif tipo in mx.TIPOS_REFORCO:
        vc = saida["melhor_accao"].value_counts()
        fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist()))
        fig.update_layout(title="Melhor acção em cada caso")
    else:
        return
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def _resultado_tabela(modelo, meta: dict, df: pd.DataFrame, p: str, so_resultado: bool):
    try:
        saida, rel = mx.aplicar_tabela(modelo, meta, df)
    except cl.ErroCiclo as e:
        st.error(str(e))
        return None
    for a in rel["avisos"]:
        st.warning(a)
    for a in meta.get("avisos") or []:
        st.caption(a)
    if so_resultado:
        st.dataframe(saida[[c for c in saida.columns if c not in df.columns]], width="stretch", hide_index=True)
    else:
        st.dataframe(saida.head(200), width="stretch")
    return saida


def _nao_supervisionado(modelo, meta: dict, p: str, mid: str):
    modo = st.radio("Como queres testar?", ["Um caso", "Um ficheiro"], horizontal=True, key=f"{p}_modo")
    if modo == "Um caso":
        valores = _formulario(meta, f"{p}_{mid}")
        if st.button("Aplicar o modelo", type="primary", key=f"{p}_go"):
            _resultado_tabela(modelo, meta, pd.DataFrame([valores]), p, True)
        return
    st.download_button("Descarregar um CSV de exemplo com as colunas obrigatórias", cl.csv_modelo(meta), file_name="exemplo_entrada.csv", mime="text/csv", key=f"{p}_modelo")
    f = st.file_uploader("Ficheiro com os casos novos (CSV ou Excel)", type=["csv", "xlsx", "xlsm"], key=f"{p}_file")
    if f is None:
        return
    try:
        novo = carregar_ficheiro(f.name, f.getvalue())
    except ErroDados as e:
        st.error(str(e))
        return
    saida = _resultado_tabela(modelo, meta, novo, p, False)
    if saida is not None:
        _grafico_resultado(meta, saida)
        st.download_button("Descarregar o resultado (CSV)", saida.to_csv(index=False).encode("utf-8"), file_name="resultado.csv", mime="text/csv", key=f"{p}_dl")


def _grelha_q(modelo, caminho: list):
    n = modelo.n
    V = modelo.Q.max(axis=1).reshape(n, n)
    fig = go.Figure(go.Heatmap(z=V, colorscale="Viridis", colorbar=dict(title="V*")))
    fig.add_trace(go.Scatter(x=[c for _, c in caminho], y=[r for r, _ in caminho], mode="lines+markers", name="Caminho", line=dict(width=3)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Valor de cada casa e caminho do agente", height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def _reforco_q(modelo, meta: dict, p: str):
    modo = st.radio("Como queres testar?", ["Uma casa da grelha", "Um ficheiro"], horizontal=True, key=f"{p}_modo")
    if modo == "Uma casa da grelha":
        n = modelo.n
        c1, c2 = st.columns(2)
        linha = int(c1.number_input("Linha", 0, n - 1, 0, key=f"{p}_lin"))
        coluna = int(c2.number_input("Coluna", 0, n - 1, 0, key=f"{p}_col"))
        if (linha, coluna) in modelo.obstaculos:
            st.warning("Essa casa é um obstáculo: o agente não pode estar lá.")
            return
        r = modelo.melhor_accao(linha, coluna)
        caminho = modelo.caminho((linha, coluna))
        st.metric("Melhor acção nesta casa", r["accao"])
        st.dataframe(pd.DataFrame({"Acção": list(r["valores_q"]), "Valor Q": list(r["valores_q"].values())}), width="stretch", hide_index=True)
        chega = caminho[-1] == modelo.objectivo
        st.write(f"Seguindo sempre a melhor acção, o agente faz {len(caminho) - 1} passos e " + ("chega ao objectivo." if chega else "não chega ao objectivo."))
        _grelha_q(modelo, caminho)
        return
    st.download_button("Descarregar um CSV de exemplo (colunas linha e coluna)", pd.DataFrame([{"linha": 0, "coluna": 0}]).to_csv(index=False).encode("utf-8"),
                       file_name="exemplo_entrada.csv", mime="text/csv", key=f"{p}_modelo")
    f = st.file_uploader("Ficheiro com as casas (CSV ou Excel)", type=["csv", "xlsx", "xlsm"], key=f"{p}_file")
    if f is None:
        return
    try:
        novo = carregar_ficheiro(f.name, f.getvalue())
    except ErroDados as e:
        st.error(str(e))
        return
    saida = _resultado_tabela(modelo, meta, novo, p, False)
    if saida is not None:
        _grafico_resultado(meta, saida)
        st.download_button("Descarregar o resultado (CSV)", saida.to_csv(index=False).encode("utf-8"), file_name="resultado.csv", mime="text/csv", key=f"{p}_dl")


def _reforco_dqn(modelo, meta: dict, p: str):
    modo = st.radio("Como queres testar?", ["Um estado", "Um ficheiro"], horizontal=True, key=f"{p}_modo")
    if modo == "Um estado":
        st.caption("O estado do CartPole tem 4 números: posição do carrinho, a sua velocidade, o ângulo do poste e a velocidade angular.")
        cols = st.columns(2)
        estado = {c: float(cols[i % 2].number_input(c, value=0.0, step=0.05, format="%.3f", key=f"{p}_{c}")) for i, c in enumerate(modelo.nomes_estado)}
        if st.button("Que acção escolhe a rede?", type="primary", key=f"{p}_go"):
            _resultado_tabela(modelo, meta, pd.DataFrame([estado]), p, True)
        return
    st.download_button("Descarregar um CSV de exemplo com as colunas do estado", pd.DataFrame([{c: 0.0 for c in modelo.nomes_estado}]).to_csv(index=False).encode("utf-8"),
                       file_name="exemplo_entrada.csv", mime="text/csv", key=f"{p}_modelo")
    f = st.file_uploader("Ficheiro com os estados (CSV ou Excel)", type=["csv", "xlsx", "xlsm"], key=f"{p}_file")
    if f is None:
        return
    try:
        novo = carregar_ficheiro(f.name, f.getvalue())
    except ErroDados as e:
        st.error(str(e))
        return
    saida = _resultado_tabela(modelo, meta, novo, p, False)
    if saida is not None:
        _grafico_resultado(meta, saida)
        st.download_button("Descarregar o resultado (CSV)", saida.to_csv(index=False).encode("utf-8"), file_name="resultado.csv", mime="text/csv", key=f"{p}_dl")


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
    from modules.ml_engine.pacote_generico import TIPOS as _TIPOS_DL
    if meta.get("tipo") in _TIPOS_DL:       # texto, imagens, geradores e séries com rede neuronal
        from modules.testar_modelos_dl import testar_generico
        testar_generico(modelo, meta, p, mid)
        return
    if meta.get("tipo") == "serie":
        _serie(modelo, meta, p)
        return
    if meta.get("tipo") in mx.TIPOS_NAO_SUPERVISIONADOS:
        with st.expander("O que este modelo precisa"):
            _o_que_precisa(meta)
        _nao_supervisionado(modelo, meta, p, mid)
        return
    if meta.get("tipo") == "reforco_q":
        _reforco_q(modelo, meta, p)
        return
    if meta.get("tipo") == "reforco_dqn":
        _reforco_dqn(modelo, meta, p)
        return
    with st.expander("O que este modelo precisa para prever"):
        _o_que_precisa(meta)
    modo = st.radio("Como queres testar?", ["Um caso", "Um ficheiro"], horizontal=True, key=f"{p}_modo")
    if modo == "Um caso":
        _um_caso(modelo, meta, p, mid)
    else:
        _ficheiro(modelo, meta, p)


def _importar(username: str):
    with st.expander("Tens um modelo descarregado de antes (.pkl)? Importa-o aqui"):
        st.caption("Serve para os ficheiros .pkl descarregados das páginas Classificação, Regressão e Redes Neurais antes de existir o botão «Guardar modelo». "
                   "Por segurança só se reconstroem objectos do scikit-learn: se o ficheiro tiver outra coisa, não é aberto. Importa só ficheiros que descarregaste desta plataforma.")
        f = st.file_uploader("Ficheiro .pkl", type=["pkl", "joblib"], key="tm_imp_file")
        if f is None:
            return
        try:
            info = il.inspeccionar(f.getvalue())
        except ErroLegado as e:
            st.error(str(e))
            return
        st.success(f"Modelo reconhecido: {info['algoritmo']} ({'classificação' if info['tipo'] == 'classificacao' else 'regressão'}).")
        nome = st.text_input("Nome para guardar", value=f"{info['algoritmo']} importado", max_chars=80, key="tm_imp_nome")
        alvo = st.text_input("O que prevê este modelo? (nome da coluna)", key="tm_imp_alvo")
        feats = None
        if not info["features"]:
            txt = st.text_input(f"Colunas do treino, pela ordem e separadas por vírgulas ({info['n_features']} no total)", key="tm_imp_feats")
            feats = [x.strip() for x in txt.split(",") if x.strip()]
            st.caption("Este ficheiro não trazia os nomes das colunas. Só funciona se todas eram números (as redes neurais antigas não guardavam a codificação do texto).")
        if st.button("Importar e guardar", type="primary", key="tm_imp_go"):
            try:
                objecto, meta = il.construir(info, nome, alvo, feats)
                guardado = mp.guardar_modelo(username, objecto, meta)
                st.session_state["tm_mod"] = guardado["id"]
                st.success(f"Importado como «{guardado['nome']}» (versão {guardado['versao']}). Já aparece na lista de modelos acima.")
            except (ErroLegado, mp.ErroPersistencia) as e:
                st.error(str(e))


def render_testar_modelos(username: str):
    page_header("Testar Modelos", "Experimenta qualquer modelo guardado com dados novos: um caso à mão ou um ficheiro inteiro.")
    st.caption("Os dados que usas aqui não entram no treino: servem só para ver o que o modelo responde.")
    section_title("O TEU MODELO")
    painel_previsao(username, "tm")
    _importar(username)
