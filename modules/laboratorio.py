"""Laboratório de ML: biblioteca de algoritmos, treino e comparação, AutoML, séries temporais, previsões e modelos guardados.

Toda a lógica está em `modules/ml_engine` (testável sem Streamlit); aqui só há interface.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ml_engine import auto_ml, data_loader, datasets_demo as demo, forecasting as fc, model_evaluation as ev, model_persistence as mp
from modules.ml_engine import model_registry as reg, prediction_service as ps
from modules.ml_engine.data_validation import sugerir_tipo_problema, validar_dados
from modules.ml_engine.model_comparison import comparar_modelos, recomendar
from modules.ml_engine.preprocessing import ErroDados, preparar_dados, sugerir_features
from modules.utils import page_header, section_title

MAX_MODELOS_COMPARACAO = 12
NOMES_TIPO = {"regressao": "Regressão (prever um número)", "classificacao": "Classificação (prever uma categoria)"}


# ── Dados ──────────────────────────────────────────────────────────────────────
def _limpar_resultados():
    for k in ("lab_comp", "lab_comp_chave", "lab_auto", "lab_serie_comp", "lab_serie_chave"):
        st.session_state.pop(k, None)


def _bloco_exemplos():
    """Datasets fictícios para experimentar sem ficheiro próprio."""
    with st.expander("Não tens dados? Usa um dataset de exemplo (fictício)"):
        st.caption("Dados gerados por código, sem pessoas nem empresas reais. Têm padrões reais para os modelos "
                   "aprenderem, ruído e alguns valores em falta.")
        ds = demo.listar()
        i = st.selectbox("Dataset", [d.id for d in ds], key="lab_demo_id",
                         format_func=lambda k: f"{demo.NOMES_TIPO[demo.DATASETS[k].tipo]} · {demo.DATASETS[k].nome}")
        d = demo.DATASETS[i]
        st.write(d.descricao)
        st.caption(f"Variável a prever: «{d.alvo}»" + (" (separador «Séries temporais»)" if d.tipo == "serie"
                                                         else " (separadores «Treinar e comparar» ou «AutoML»)"))
        n = int(st.number_input("Número de linhas", 30, 20000, d.n_padrao, key=f"lab_demo_n_{i}"))
        c1, c2 = st.columns(2)
        if c1.button("Usar este dataset", key="lab_demo_usar", type="primary"):
            df = demo.gerar(i, n)
            st.session_state.update(lab_id=f"demo:{i}:{n}", lab_df=df, lab_amostrado=None,
                                    lab_up_n=st.session_state.get("lab_up_n", 0) + 1)
            _limpar_resultados()
            st.rerun()
        c2.download_button("Descarregar em CSV", demo.para_csv(demo.gerar(i, n)), file_name=f"{i}.csv",
                           mime="text/csv", key="lab_demo_csv")


def _fonte_dados():
    """Carrega CSV/Excel (ou um dataset de exemplo) para a sessão. Devolve (df, identificador) ou (None, None)."""
    st.markdown("#### 1. Os teus dados")
    ficheiro = st.file_uploader("Ficheiro CSV ou Excel (.xlsx)", type=["csv", "xlsx", "xlsm", "txt"],
                                key=f"lab_upload_{st.session_state.get('lab_up_n', 0)}",
                                help="Máximo 50 MB. A primeira linha tem de ter os nomes das colunas.")
    _bloco_exemplos()
    if ficheiro is None:
        ident_atual = st.session_state.get("lab_id") or ""
        if ident_atual.startswith("demo:") and st.session_state.get("lab_df") is not None:
            df = st.session_state["lab_df"]
            _, id_demo, _ = ident_atual.split(":")
            st.success(f"A usar o dataset de exemplo «{demo.DATASETS[id_demo].nome}» (fictício). "
                       f"Variável a prever: «{demo.DATASETS[id_demo].alvo}».")
            st.caption(f"{len(df)} linhas × {df.shape[1]} colunas")
            with st.expander("Ver as primeiras linhas"):
                st.dataframe(df.head(20), use_container_width=True)
            return df, ident_atual
        st.info("Carrega um ficheiro ou escolhe um dataset de exemplo para começar.")
        return None, None
    conteudo = ficheiro.getvalue()
    folha = None
    if data_loader.extensao(ficheiro.name) in ("xlsx", "xlsm"):
        try:
            folhas = data_loader.folhas_excel(ficheiro.name, conteudo)
        except Exception:
            folhas = []
        if len(folhas) > 1:
            folha = st.selectbox("Folha do Excel", folhas, key="lab_folha")
    ident = hashlib.sha256(conteudo).hexdigest()[:16] + str(folha)
    if st.session_state.get("lab_id") != ident:
        try:
            df = data_loader.carregar_ficheiro(ficheiro.name, conteudo, folha)
        except data_loader.ErroDados as e:
            st.error(str(e))
            return None, None
        except Exception:
            st.error("Não consegui ler este ficheiro. Confirma que é um CSV ou Excel válido.")
            return None, None
        df, amostrado = data_loader.amostrar(df)
        st.session_state.update(lab_id=ident, lab_df=df, lab_amostrado=amostrado)
        _limpar_resultados()
    df = st.session_state["lab_df"]
    if st.session_state.get("lab_amostrado"):
        st.warning(st.session_state["lab_amostrado"])
    st.caption(f"{len(df)} linhas × {df.shape[1]} colunas")
    with st.expander("Ver as primeiras linhas"):
        st.dataframe(df.head(20), use_container_width=True)
    return df, ident


def _mostrar_validacao(rel):
    for e in rel.erros:
        st.error(e)
    for a in rel.avisos:
        st.warning(a)


def _escolhas_problema(df, prefixo: str):
    """Alvo, tipo, variáveis, percentagem de teste e semente. Devolve dict ou None."""
    c1, c2 = st.columns(2)
    alvo = c1.selectbox("O que queres prever? (variável-alvo)", list(df.columns), index=df.shape[1] - 1, key=f"{prefixo}_alvo",
                        help="Por omissão, a última coluna (é onde costuma estar a resposta).")
    sug, motivo = sugerir_tipo_problema(df, alvo)
    tipo = c2.radio("Tipo de problema", list(NOMES_TIPO), index=0 if sug == "regressao" else 1,
                    format_func=NOMES_TIPO.get, key=f"{prefixo}_tipo__{alvo}", help=f"Sugestão: {motivo}")
    st.caption(f"Sugestão automática: {NOMES_TIPO[sug].split(' (')[0]}. {motivo} Podes mudar.")
    padrao = sugerir_features(df, alvo)
    feats = st.multiselect("Variáveis usadas para prever", [c for c in df.columns if c != alvo], default=padrao,
                           key=f"{prefixo}_feats__{alvo}", help="Os identificadores e as colunas vazias ou constantes já vêm de fora.")
    c3, c4 = st.columns(2)
    teste = c3.slider("Percentagem para teste", 10, 40, 20, 5, key=f"{prefixo}_teste") / 100
    seed = int(c4.number_input("Semente aleatória", 0, 10_000, 42, key=f"{prefixo}_seed", help="Com a mesma semente os resultados repetem-se."))
    if not feats:
        st.error("Escolhe pelo menos uma variável para prever.")
        return None
    rel = validar_dados(df[feats + [alvo]], alvo, tipo)
    _mostrar_validacao(rel)
    if not rel.ok:
        return None
    return dict(alvo=alvo, tipo=tipo, feats=feats, teste=teste, seed=seed)


def _preparar(df, e):
    try:
        return preparar_dados(df, e["alvo"], e["tipo"], e["feats"], e["teste"], e["seed"])
    except ErroDados as ex:
        st.error(str(ex))
        return None


def _resumo_preparacao(dados):
    st.success(f"{dados.n_treino} exemplos para treinar e {dados.n_teste} para testar. O teste só serve para a nota final.")
    for n in dados.notas:
        st.caption("" + n)
    for a in dados.avisos:
        st.warning(a)


# ── Gráficos ───────────────────────────────────────────────────────────────────
def _fig_matriz(cm, classes):
    fig = go.Figure(go.Heatmap(z=cm, x=[str(c) for c in classes], y=[str(c) for c in classes], colorscale="Blues",
                               text=cm, texttemplate="%{text}", showscale=False))
    fig.update_layout(title="Matriz de confusão", xaxis_title="Previsto", yaxis_title="Real", yaxis_autorange="reversed",
                      height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _fig_curva(x, y, titulo, xt, yt, diag=False):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name=titulo))
    if diag:
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig.update_layout(title=titulo, xaxis_title=xt, yaxis_title=yt, height=340, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _graficos_modelo(comp, r):
    d = comp.dados
    y_pred = r.predicoes.get("y_pred")
    if y_pred is None:
        return
    if comp.tipo == "regressao":
        y = np.asarray(d.y_test, dtype=float)
        fig = go.Figure(go.Scatter(x=y, y=y_pred, mode="markers", name="teste", opacity=.7))
        lo, hi = float(min(y.min(), np.min(y_pred))), float(max(y.max(), np.max(y_pred)))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"))
        fig.update_layout(title="Real × previsto (no teste)", xaxis_title="Real", yaxis_title="Previsto", height=380, margin=dict(l=10, r=10, t=50, b=10))
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig, use_container_width=True)
        res = y - np.asarray(y_pred, dtype=float)
        f2 = go.Figure(go.Histogram(x=res, nbinsx=30))
        f2.update_layout(title="Erros (real − previsto)", height=380, margin=dict(l=10, r=10, t=50, b=10))
        c2.plotly_chart(f2, use_container_width=True)
        return
    cm = ev.matriz_confusao(d.y_test, y_pred, len(d.classes))
    c1, c2 = st.columns(2)
    c1.plotly_chart(_fig_matriz(cm, d.classes), use_container_width=True)
    curvas = ev.curvas_binarias(d.y_test, r.predicoes.get("score")) if len(d.classes) == 2 else {}
    if curvas:
        c2.plotly_chart(_fig_curva(*curvas["roc"], "Curva ROC", "Falsos positivos", "Verdadeiros positivos", True), use_container_width=True)
        st.plotly_chart(_fig_curva(curvas["pr"][1], curvas["pr"][0], "Curva Precision-Recall", "Recall", "Precision"), use_container_width=True)
    else:
        c2.caption("As curvas ROC e Precision-Recall só se desenham com 2 classes e um modelo que dê probabilidades.")


# ── Resultados partilhados (comparação e AutoML) ────────────────────────────────
def _mostrar_comparacao(comp, rec, chave: str, username: str, origem: str):
    for a in comp.avisos:
        st.warning(a)
    if comp.interrompido:
        st.warning("A comparação parou antes de avaliar todos os modelos (limite de tempo ou de modelos). Os não avaliados aparecem como «Ignorado».")
    criterios = ev.CRITERIOS[comp.tipo]
    ordem = st.selectbox("Ordenar por", criterios + ["tempo"], format_func=lambda k: "Tempo de treino" if k == "tempo" else ev.rotulo(comp.tipo, k),
                         key=f"{chave}_ordem")
    tab = comp.tabela_visivel(ordem)
    fmt = {c: "{:.4f}" for c in tab.columns if c not in ("Modelo", "Estado")}
    st.dataframe(tab.style.format(fmt, na_rep="—"), use_container_width=True, hide_index=True)
    st.caption("Notas do teste (dados que o modelo nunca viu). A coluna (CV) é a validação cruzada só com os dados de treino. "
               "«Referência» é um modelo sem inteligência (média ou classe mais frequente): um modelo útil tem de a superar.")
    if rec and rec.modelo_id:
        st.success(f"**Recomendação: {rec.nome}** (critério: {rec.criterio}, medido em {rec.base})")
        st.write(rec.justificacao)
        for a in rec.avisos:
            st.warning(a)
        if rec.alternativas:
            st.caption("Alternativas próximas: " + "; ".join(f"{n} ({v})" for n, v in rec.alternativas))
    elif rec:
        st.warning(rec.justificacao or "Não há modelo para recomendar.")
    validos = [r for r in comp.validos() if r.estado == "ok"]
    if validos:
        nomes = {r.id: r.nome for r in validos}
        escolhido = st.selectbox("Ver detalhes de", list(nomes), format_func=nomes.get, key=f"{chave}_det",
                                 index=list(nomes).index(rec.modelo_id) if rec and rec.modelo_id in nomes else 0)
        r = comp.por_id(escolhido)
        with st.expander("Gráficos e métricas", expanded=True):
            _graficos_modelo(comp, r)
            if comp.tipo == "regressao" and r.metricas.get("residuos"):
                st.caption("Resíduos: " + ", ".join(f"{k} = {v:.4g}" for k, v in r.metricas["residuos"].items()))
        for a in r.avisos:
            st.warning(a)
        _bloco_guardar(comp, r, chave, username, origem)
    for r in comp.falhas():
        st.caption(f"{r.nome}: {r.motivo}")


def _bloco_guardar(comp, r, chave, username, origem):
    with st.expander("Guardar este modelo"):
        nome = st.text_input("Nome", value=f"{r.nome} – {comp.dados.alvo}", key=f"{chave}_nome", max_chars=80)
        if st.button("Guardar modelo", key=f"{chave}_guardar"):
            try:
                info = mp.guardar_modelo(username, r.pipeline, mp.meta_tabular(comp.dados, r, nome, origem))
                st.success(f"Guardado como «{info['nome']}» (versão {info['versao']}). Vê-o no separador «Modelos guardados».")
            except mp.ErroPersistencia as e:
                st.error(str(e))
            except Exception:
                st.error("Não consegui guardar o modelo.")


# ── Separadores ────────────────────────────────────────────────────────────────
def _tab_biblioteca():
    st.markdown("Todos os algoritmos disponíveis. Os que precisam de uma biblioteca que não está instalada aparecem como indisponíveis, com o motivo.")
    problema = st.radio("Problema", ["regressao", "classificacao", "serie"], horizontal=True, key="lab_bib_prob",
                        format_func=lambda k: reg.NOMES_PROBLEMA[k])
    cats = ["Todas"] + reg.categorias(problema)
    cat = st.selectbox("Categoria", cats, key="lab_bib_cat")
    tab = reg.catalogo(problema)
    if cat != "Todas":
        tab = tab[tab["Categoria"] == cat]
    pesquisa = st.text_input("Procurar", key="lab_bib_q")
    if pesquisa:
        tab = tab[tab["Algoritmo"].str.contains(pesquisa, case=False, na=False) | tab["Descrição"].str.contains(pesquisa, case=False, na=False)]
    st.dataframe(tab.drop(columns="id"), use_container_width=True, hide_index=True)
    st.caption(f"{len(tab)} algoritmos. Disponíveis: {int((tab['Disponível'] == 'Sim').sum())}.")


def _controlo_param(nome, spec, chave):
    if spec["tipo"] == "choice":
        return st.selectbox(nome, spec["opcoes"], index=spec["opcoes"].index(spec["padrao"]), key=chave, help=spec.get("ajuda") or None)
    if spec["tipo"] == "int":
        return int(st.number_input(nome, spec["min"], spec["max"], int(spec["padrao"]), 1, key=chave, help=spec.get("ajuda") or None))
    return float(st.number_input(nome, spec["min"], spec["max"], float(spec["padrao"]), format="%g", key=chave, help=spec.get("ajuda") or None))


def _tab_treinar(df, ident, username):
    e = _escolhas_problema(df, "lab_tr")
    if not e:
        return
    st.markdown("#### 3. Algoritmos")
    infos = [m for m in reg.listar(e["tipo"]) if m.tipo_modelo == "tabular"]
    disp = [m for m in infos if m.disponivel()]
    indisp = [m for m in infos if not m.disponivel()]
    padrao = [m.id for m in disp if m.id in auto_ml.PRIORIDADE[e["tipo"]][:3]]
    escolha = st.multiselect("Escolhe os algoritmos a comparar", [m.id for m in disp], default=padrao, key=f"lab_tr_algos_{e['tipo']}",
                             format_func=lambda i: reg.obter(i).nome)
    if indisp:
        st.caption("Indisponíveis neste ambiente: " + "; ".join(f"{m.nome} (precisa de «{m.dependencia}»)" for m in indisp))
    params = {}
    with st.expander("Ajustar parâmetros (opcional)"):
        for i in escolha:
            info = reg.obter(i)
            if info.parametros:
                st.markdown(f"**{info.nome}**")
                cols = st.columns(min(3, len(info.parametros)))
                params[i] = {k: _controlo_param(k, s, f"lab_p_{i}_{k}") for (k, s), c in zip(info.parametros.items(), cols * 10)}
    c1, c2, c3 = st.columns(3)
    cv = c1.selectbox("Validação cruzada", [0, 3, 5], index=1, key="lab_tr_cv", format_func=lambda v: "Sem" if v == 0 else f"{v} blocos")
    max_m = int(c2.number_input("Máximo de modelos", 1, MAX_MODELOS_COMPARACAO, 6, key="lab_tr_max"))
    orc = int(c3.number_input("Tempo máximo (s)", 10, 900, 180, 10, key="lab_tr_orc"))
    if len(escolha) > max_m:
        st.warning(f"Escolheste {len(escolha)} algoritmos mas o limite é {max_m}: os últimos ficam de fora.")
    chave = hashlib.sha256(json.dumps([ident, e, escolha, params, cv, max_m], sort_keys=True, default=str).encode()).hexdigest()
    if st.button("Treinar e comparar", type="primary", disabled=not escolha, key="lab_tr_go"):
        dados = _preparar(df, e)
        if dados is None:
            return
        if st.session_state.get("lab_comp_chave") == chave:
            st.info("Estes são os mesmos dados e as mesmas escolhas de há pouco: reutilizei o resultado em vez de treinar de novo.")
        else:
            barra = st.progress(0.0, text="A começar…")
            comp = comparar_modelos(escolha, dados, params, cv, e["seed"], max_m, orc,
                                    progresso=lambda i, n, nome: barra.progress(i / max(n, 1), text=f"A treinar {nome} ({i + 1}/{n})"))
            barra.empty()
            comp_rec = recomendar(comp)
            st.session_state.update(lab_comp=(comp, comp_rec), lab_comp_chave=chave)
    guardado = st.session_state.get("lab_comp")
    if guardado:
        comp, rec = guardado
        st.markdown("#### 4. Resultados")
        _resumo_preparacao(comp.dados)
        _mostrar_comparacao(comp, rec, "lab_tr", username, "comparação")


def _tab_automl(df, ident, username):
    st.markdown("O AutoML escolhe os algoritmos compatíveis com os teus dados, compara-os com validação cruzada **só nos dados de treino** e "
                "recomenda o mais simples entre os que ficam equivalentes ao melhor. Não afina parâmetros nem usa o teste para escolher.")
    e = _escolhas_problema(df, "lab_au")
    if not e:
        return
    c1, c2, c3 = st.columns(3)
    max_m = int(c1.number_input("Máximo de modelos", 2, MAX_MODELOS_COMPARACAO, 6, key="lab_au_max"))
    orc = int(c2.number_input("Tempo máximo (s)", 20, 900, 120, 10, key="lab_au_orc"))
    cv = int(c3.selectbox("Blocos de validação cruzada", [3, 5], key="lab_au_cv"))
    if st.button("Executar AutoML", type="primary", key="lab_au_go"):
        barra = st.progress(0.0, text="A preparar…")
        try:
            res = auto_ml.executar_automl(df, e["alvo"], e["tipo"], e["feats"], max_m, orc, cv, e["teste"], e["seed"],
                                          progresso=lambda i, n, nome: barra.progress(min(i / max(n, 1), 1.0), text=f"{nome} ({i + 1}/{n})"))
            st.session_state["lab_auto"] = res
        except ErroDados as ex:
            st.error(str(ex))
        barra.empty()
    res = st.session_state.get("lab_auto")
    if not res:
        return
    st.markdown("#### O que o AutoML fez")
    for p in res.passos:
        st.markdown("- " + p)
    if res.puladas:
        with st.expander(f"{len(res.puladas)} algoritmos não usados"):
            for n, m in res.puladas:
                st.caption(f"{n}: {m}")
    if len(res.triagem):
        with st.expander("Triagem rápida (amostra dos dados de treino)"):
            st.dataframe(res.triagem, use_container_width=True, hide_index=True)
    for a in res.avisos:
        st.warning(a)
    _resumo_preparacao(res.dados)
    _mostrar_comparacao(res.comparacao, res.recomendacao, "lab_au", username, "AutoML")
    st.caption(f"Tempo total: {res.tempo_total:.1f} s. «Melhor» significa melhor entre os modelos avaliados, não o melhor algoritmo possível.")


def _tab_series(df, ident, username):
    col_data = fc.detectar_coluna_data(df)
    if col_data is None:
        st.info("Não encontrei nenhuma coluna de datas nos teus dados. As séries temporais precisam de uma coluna com a data de cada valor.")
    c1, c2 = st.columns(2)
    cols = list(df.columns)
    cd = c1.selectbox("Coluna de data", cols, index=cols.index(col_data) if col_data in cols else 0, key="lab_se_data")
    num = [c for c in cols if c != cd and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        st.error("Não há nenhuma coluna numérica para prever.")
        return
    cv_ = c2.selectbox("Valor a prever", num, key="lab_se_valor")
    c3, c4, c5 = st.columns(3)
    dup = c3.selectbox("Datas repetidas", ["media", "soma", "ultimo"], key="lab_se_dup",
                       format_func={"media": "Fazer a média", "soma": "Somar", "ultimo": "Ficar com o último"}.get)
    horizonte = int(c4.number_input("Horizonte (períodos a prever)", 1, 365, 12, key="lab_se_h"))
    n_splits = int(c5.selectbox("Blocos da validação progressiva", [2, 3, 4], index=1, key="lab_se_n"))
    try:
        serie = fc.preparar_serie(df, cd, cv_, duplicados=dup)
    except (ErroDados, ValueError) as ex:
        st.error(str(ex))
        return
    for n in serie.notas:
        st.caption("" + n)
    for a in serie.avisos:
        st.warning(a)
    st.caption(f"{serie.n} pontos, frequência {serie.freq or 'irregular'}, período sazonal {serie.periodo}.")
    st.plotly_chart(go.Figure(go.Scatter(x=serie.datas, y=serie.valores, mode="lines")).update_layout(
        title="A série", height=300, margin=dict(l=10, r=10, t=50, b=10)), use_container_width=True)
    infos = [m for m in reg.listar("serie")]
    disp = [m for m in infos if m.disponivel() and not m.id.startswith("serie_naive") and m.id != "serie_snaive"]
    escolha = st.multiselect("Modelos", [m.id for m in disp], default=[m.id for m in disp if m.id in ("serie_regressao", "serie_rf")],
                             format_func=lambda i: reg.obter(i).nome, key="lab_se_algos")
    ind = [m for m in infos if not m.disponivel()]
    if ind:
        st.caption("Indisponíveis neste ambiente: " + "; ".join(f"{m.nome} (precisa de «{m.dependencia}»)" for m in ind))
    chave = hashlib.sha256(json.dumps([ident, cd, cv_, dup, horizonte, n_splits, escolha], default=str).encode()).hexdigest()
    if st.button("Comparar modelos de séries", type="primary", disabled=not escolha, key="lab_se_go"):
        if st.session_state.get("lab_serie_chave") != chave:
            barra = st.progress(0.0, text="A começar…")
            try:
                comp = fc.comparar_series(serie, escolha, horizonte, n_splits, max_modelos=MAX_MODELOS_COMPARACAO, orcamento_s=240,
                                          progresso=lambda i, n, nome: barra.progress(i / max(n, 1), text=f"{nome} ({i + 1}/{n})"))
            except ErroDados as ex:
                barra.empty()
                st.error(str(ex))
                return
            barra.empty()
            st.session_state.update(lab_serie_comp=(comp, fc.recomendar_serie(comp)), lab_serie_chave=chave)
    g = st.session_state.get("lab_serie_comp")
    if not g:
        return
    comp, rec = g
    d = comp.divisao
    st.info(f"Divisão cronológica: treino até ao ponto {d.fim_treino}, validação até {d.fim_validacao}, teste nos últimos {comp.horizonte} pontos. "
            "Nenhum modelo vê o futuro durante a avaliação.")
    for a in comp.avisos:
        st.warning(a)
    tab = comp.tabela().drop(columns=["_id", "_ok"])
    st.dataframe(tab.style.format({c: "{:.4f}" for c in tab.columns if c not in ("Modelo", "Estado")}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    if rec and rec.modelo_id:
        st.success(f"**Recomendação: {rec.nome}** (critério: {rec.criterio})")
        st.write(rec.justificacao)
        for a in rec.avisos:
            st.warning(a)
    elif rec:
        st.warning(rec.justificacao)
    validos = [r for r in comp.validos() if not r.referencia]
    if validos:
        nomes = {r.id: r.nome for r in validos}
        sel = st.selectbox("Ver previsão de", list(nomes), format_func=nomes.get, key="lab_se_ver",
                           index=list(nomes).index(rec.modelo_id) if rec and rec.modelo_id in nomes else 0)
        pv = comp.previsoes_dataframe(sel)
        fig = go.Figure()
        for fase, cor in (("historico", None), ("teste", None), ("futuro", None)):
            p = pv[pv["fase"] == fase]
            if fase == "historico":
                fig.add_trace(go.Scatter(x=pv["data"][pv["valor_real"].notna()], y=pv["valor_real"].dropna(), name="Real", mode="lines"))
            else:
                fig.add_trace(go.Scatter(x=p["data"], y=p["previsao"], name=f"Previsão ({fase})", mode="lines"))
        fut = pv[pv["fase"] == "futuro"]
        if fut["inferior"].notna().any():
            fig.add_trace(go.Scatter(x=list(fut["data"]) + list(fut["data"])[::-1], y=list(fut["superior"]) + list(fut["inferior"])[::-1],
                                     fill="toself", opacity=.2, name="Intervalo (modelo)", line=dict(width=0)))
        else:
            st.caption("Este modelo só dá previsões pontuais: não inventamos intervalos de confiança.")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Descarregar previsões (CSV)", pv.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv", key="lab_se_dl")
        with st.expander("Guardar este modelo"):
            nome = st.text_input("Nome", value=f"{nomes[sel]} – {comp.serie.col_valor}", key="lab_se_nome", max_chars=80)
            if st.button("Guardar modelo", key="lab_se_guardar"):
                try:
                    r = comp.por_id(sel)
                    info = mp.guardar_modelo(username, r.modelo_final, mp.meta_serie(comp.serie, r, nome, comp.horizonte, "séries"))
                    st.success(f"Guardado como «{info['nome']}» (versão {info['versao']}).")
                except mp.ErroPersistencia as ex:
                    st.error(str(ex))
                except Exception:
                    st.error("Não consegui guardar o modelo.")


def _tab_previsoes(username):
    try:
        modelos = mp.listar_modelos(username)
    except mp.ErroPersistencia as e:
        st.error(str(e))
        return
    if not modelos:
        st.info("Ainda não guardaste nenhum modelo. Treina um e carrega em «Guardar modelo».")
        return
    nomes = {m["id"]: f"{m['nome']} (v{m['versao']}, {m['criado_em']})" for m in modelos}
    mid = st.selectbox("Modelo", list(nomes), format_func=nomes.get, key="lab_pr_mod")
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
    if meta["tipo"] == "serie":
        h = int(st.number_input("Períodos a prever", 1, 365, int(meta.get("horizonte", 12)), key="lab_pr_h"))
        try:
            out = ps.prever_futuro(modelo, meta, h)
        except ps.ErroPrevisao as e:
            st.error(str(e))
            return
        except Exception:
            st.error("O modelo não conseguiu prever. Treina-o outra vez.")
            return
        st.caption(f"Previsões a partir de {meta['ultima_data']}, o fim dos dados com que o modelo foi treinado.")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Descarregar (CSV)", out.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv", key="lab_pr_dl_s")
        return
    modo = st.radio("Como queres prever?", ["Um caso", "Um ficheiro"], horizontal=True, key="lab_pr_modo")
    if modo == "Um caso":
        valores = {}
        cols = st.columns(2)
        for k, f in enumerate(meta["features"]):
            if f in meta.get("cat_cols", []):
                op = sorted(meta.get("categorias_conhecidas", {}).get(f, []))
                valores[f] = cols[k % 2].selectbox(f, op, key=f"lab_pr_{mid}_{f}") if op else cols[k % 2].text_input(f, key=f"lab_pr_{mid}_{f}")
            elif f in meta.get("colunas_data", []):
                valores[f] = str(cols[k % 2].date_input(f, key=f"lab_pr_{mid}_{f}"))
            else:
                iv = (meta.get("intervalos_numericos") or {}).get(f)
                valores[f] = cols[k % 2].number_input(f, value=float(iv[0]) if iv and iv[0] is not None else 0.0, key=f"lab_pr_{mid}_{f}",
                                                      help=f"Visto no treino: {iv[0]:.4g} a {iv[1]:.4g}" if iv and None not in iv else None)
        if st.button("Prever", type="primary", key="lab_pr_go"):
            try:
                p, det, rel = ps.prever_um(modelo, meta, valores)
                st.metric(f"Previsão de {meta['alvo']}", f"{p:.4g}" if isinstance(p, (int, float, np.floating)) else str(p))
                if det:
                    st.dataframe(pd.DataFrame({"Classe": [k[5:] for k in det], "Probabilidade": list(det.values())}), hide_index=True)
                for a in rel["avisos"]:
                    st.warning(a)
            except ps.ErroPrevisao as e:
                st.error(str(e))
    else:
        f = st.file_uploader("Ficheiro com os casos novos (CSV ou Excel)", type=["csv", "xlsx", "xlsm"], key="lab_pr_file")
        if f is not None:
            try:
                novo = data_loader.carregar_ficheiro(f.name, f.getvalue())
                rel = ps.verificar_entrada(novo, meta)
                for a in rel["avisos"]:
                    st.warning(a)
                saida, _ = ps.prever_tabela(modelo, meta, novo)
            except (data_loader.ErroDados, ps.ErroPrevisao) as e:
                st.error(str(e))
                return
            st.dataframe(saida.head(200), use_container_width=True)
            st.download_button("Descarregar previsões (CSV)", saida.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv", key="lab_pr_dl")


def _tab_guardados(username):
    try:
        modelos = mp.listar_modelos(username)
    except mp.ErroPersistencia as e:
        st.error(str(e))
        return
    st.caption("Os modelos ficam no servidor, só visíveis para ti, e são verificados com uma assinatura antes de serem abertos. "
               "Em alojamentos com disco temporário (como o Streamlit Community Cloud) podem desaparecer quando a aplicação reinicia.")
    if not modelos:
        st.info("Ainda não tens modelos guardados.")
        return
    tab = pd.DataFrame([{"Nome": m["nome"], "Versão": m["versao"], "Tipo": m["tipo"], "Algoritmo": m["algoritmo"], "Alvo": m.get("alvo"),
                         "Criado em": m["criado_em"], "Tamanho (KB)": round(m["tamanho_bytes"] / 1024, 1)} for m in modelos])
    st.dataframe(tab, use_container_width=True, hide_index=True)
    nomes = {m["id"]: f"{m['nome']} (v{m['versao']})" for m in modelos}
    mid = st.selectbox("Modelo", list(nomes), format_func=nomes.get, key="lab_gd_mod")
    m = next(x for x in modelos if x["id"] == mid)
    with st.expander("Detalhes (metadados)"):
        st.json({k: v for k, v in m.items() if k not in ("categorias_conhecidas",)})
    c1, c2 = st.columns(2)
    novo = c1.text_input("Novo nome", value=m["nome"], key="lab_gd_nome", max_chars=80)
    if c1.button("Renomear", key="lab_gd_ren"):
        try:
            mp.renomear_modelo(username, mid, novo)
            st.rerun()
        except mp.ErroPersistencia as e:
            st.error(str(e))
    confirmar = c2.checkbox("Confirmo que quero apagar", key=f"lab_gd_conf_{mid}")
    if c2.button("Apagar", disabled=not confirmar, key="lab_gd_del"):
        mp.apagar_modelo(username, mid)
        st.rerun()


# ── Entrada ────────────────────────────────────────────────────────────────────
def render_laboratorio(username: str):
    page_header("Laboratório de ML", "Treina, compara e guarda modelos com os teus próprios dados.")
    st.caption("Tudo o que o Laboratório mostra são resultados reais calculados nos teus dados. Nada é inventado: se um modelo falha ou não está instalado, vês o motivo.")
    t = st.tabs(["Biblioteca", "Treinar e comparar", "AutoML", "Séries temporais", "Previsões", "Modelos guardados"])
    with t[0]:
        _tab_biblioteca()
    df = ident = None
    with t[1]:
        df, ident = _fonte_dados()
        if df is not None:
            _tab_treinar(df, ident, username)
    df = st.session_state.get("lab_df")
    ident = st.session_state.get("lab_id")
    with t[2]:
        if df is None:
            st.info("Carrega um ficheiro (ou escolhe um dataset de exemplo) no separador «Treinar e comparar».")
        else:
            _tab_automl(df, ident, username)
    with t[3]:
        if df is None:
            st.info("Carrega um ficheiro (ou escolhe um dataset de exemplo) no separador «Treinar e comparar».")
        else:
            _tab_series(df, ident, username)
    with t[4]:
        _tab_previsoes(username)
    with t[5]:
        _tab_guardados(username)
    try:
        from modules.tutor_ia import SUGESTOES_TREINO, registar_painel
        comp = st.session_state.get("lab_comp")
        if comp:
            c, rec = comp
            registar_painel("laboratorio", "Laboratório de ML: comparação",
                            "Tabela de comparação:\n" + c.tabela_visivel().to_string(index=False)[:3000]
                            + ("\nRecomendação: " + rec.nome + ". " + rec.justificacao if rec and rec.modelo_id else ""),
                            SUGESTOES_TREINO, padrao_primeiro=True)
    except Exception:
        pass
