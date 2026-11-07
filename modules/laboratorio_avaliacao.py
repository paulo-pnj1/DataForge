"""Laboratório de Avaliação e Optimização de Modelos: o caminho completo depois do treino, em 8 etapas.

Toda a lógica está em `modules/ml_engine/ciclo_ml.py` (testável sem Streamlit) e reutiliza o motor do Laboratório de ML
(`preparar_dados`, `treinar_modelo`, `comparar_modelos`, `model_persistence`, `prediction_service`). Aqui só há interface.
"""
import json
import logging
import types

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import laboratorio as lab
from modules.laboratorio_avaliacao_conteudo import ETAPAS, LICOES
from modules.ml_engine import auto_ml, ciclo_ml as cl, model_evaluation as ev, model_persistence as mp
from modules.ml_engine import model_registry as reg, prediction_service as ps
from modules.ml_engine.data_loader import ErroDados, carregar_ficheiro
from modules.ml_engine.model_comparison import comparar_modelos, recomendar
from modules.ml_engine.model_training import treinar_modelo
from modules.ml_engine.preprocessing import preparar_dados
from modules.utils import ir_para, page_header, section_title

log = logging.getLogger("dataforge.ml")
FONTES = ["Um modelo que treinei no Laboratório de ML", "O modelo do «O Meu Primeiro Modelo»", "Treinar um modelo aqui"]
PERCENTAGEM = ("accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy")


# ══════════════════════════════════════════════════════════════════════════════
# Estado: o modelo que está a ser avaliado
# ══════════════════════════════════════════════════════════════════════════════
def abrir_no_laboratorio(dados, resultado, origem: str, comp=None):
    """Chamado por outras páginas (Laboratório de ML, O Meu Primeiro Modelo) para continuar o percurso aqui."""
    definir_modelo(dados, resultado, origem, comp)
    ir_para("avaliacao_modelos")


def definir_modelo(dados, resultado, origem: str, comp=None, otimizacao=None):
    st.session_state["av_modelo"] = {"dados": dados, "resultado": resultado, "origem": origem, "comp": comp,
                                     "otimizacao": otimizacao, "nome": f"{resultado.nome} – {dados.alvo}"}
    for k in ("av_curva", "av_comp", "av_otim", "av_prev"):
        st.session_state.pop(k, None)


def _meta(av) -> dict:
    """Metadados do modelo actual (guardados ou não): servem para prever, gerar código e monitorizar."""
    chave = (id(av["resultado"]), av["nome"])
    if av.get("_meta_chave") != chave:
        av["meta"] = cl.meta_ciclo(av["dados"], av["resultado"], av["nome"], av["origem"], av.get("otimizacao"))
        av["_meta_chave"] = chave
    return av["meta"]


def _marcar(etapa: str):
    st.session_state.setdefault("av_feitas", set()).add(etapa)


def _fmt(chave: str, v) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if chave in PERCENTAGEM else f"{v:.4g}"


# ── escolher o modelo ─────────────────────────────────────────────────────────
def _modelos_do_laboratorio() -> list:
    """(rótulo, comparação, resultado) dos modelos treinados no Laboratório de ML nesta sessão."""
    achados = []
    g = st.session_state.get("lab_comp")
    if g:
        achados.append(("Treinar e comparar", g[0]))
    a = st.session_state.get("lab_auto")
    if a is not None and getattr(a, "comparacao", None) is not None:
        achados.append(("AutoML", a.comparacao))
    out = []
    for origem, comp in achados:
        for r in comp.validos():
            out.append((f"{r.nome} ({origem})", comp, r))
    return out


def _do_primeiro_modelo():
    """Volta a treinar o Random Forest do «Primeiro Modelo» pelo motor do Laboratório (com Pipeline e sem fuga de dados)."""
    from modules.primeiro_modelo import PROBLEMAS
    from modules.utils import carregar_dataset_embutido
    res = st.session_state.get("pm_res")
    if not res:
        return None
    cfg = PROBLEMAS[res["nome"]]
    df, _, _ = carregar_dataset_embutido(cfg["dataset"])
    df = df.dropna().copy()
    if cfg.get("rotulos"):
        df[cfg["alvo"]] = df[cfg["alvo"]].map(cfg["rotulos"])
    tipo = "classificacao" if cfg["tipo"] == "clf" else "regressao"
    dados = preparar_dados(df, cfg["alvo"], tipo)
    r = treinar_modelo("rf_clf" if tipo == "classificacao" else "rf_reg", dados, {"n_estimators": 120}, 42, 3)
    return dados, r


def _passo_modelo():
    av = st.session_state.get("av_modelo")
    with st.expander("0. Que modelo queres avaliar?" if av else "0. Escolhe o modelo a avaliar", expanded=av is None):
        fontes = FONTES
        fonte = st.radio("De onde vem o modelo?", fontes, key="av_fonte", horizontal=False)
        if fonte == fontes[0]:
            modelos = _modelos_do_laboratorio()
            if not modelos:
                st.info("Ainda não treinaste nada no Laboratório de ML nesta sessão. Treina e compara lá, ou escolhe outra opção.")
                if st.button("Ir para o Laboratório de ML", key="av_ir_lab"):
                    ir_para("laboratorio")
            else:
                i = st.selectbox("Modelo", range(len(modelos)), format_func=lambda k: modelos[k][0], key="av_sel_lab")
                if st.button("Avaliar este modelo", type="primary", key="av_usar_lab"):
                    _, comp, r = modelos[i]
                    definir_modelo(comp.dados, r, "laboratório", comp)
                    st.rerun()
        elif fonte == fontes[1]:
            if not st.session_state.get("pm_res"):
                st.info("Ainda não treinaste o teu primeiro modelo. Faz isso primeiro, e volta aqui.")
                if st.button("Ir para «O Meu Primeiro Modelo»", key="av_ir_pm"):
                    ir_para("primeiro_modelo")
            elif st.button("Avaliar o modelo do Primeiro Modelo", type="primary", key="av_usar_pm"):
                with st.spinner("A preparar o modelo…"):
                    try:
                        feito = _do_primeiro_modelo()
                    except Exception as e:
                        feito = None
                        st.error(f"Não consegui preparar este modelo. (Detalhe: {e})")
                if feito:
                    definir_modelo(feito[0], feito[1], "primeiro modelo")
                    st.rerun()
        else:
            df, _ident = lab._fonte_dados()
            if df is not None:
                e = lab._escolhas_problema(df, "av")
                if e:
                    infos = [m for m in reg.listar(e["tipo"]) if m.tipo_modelo == "tabular" and m.disponivel()]
                    padrao = [m.id for m in infos if m.id in auto_ml.PRIORIDADE[e["tipo"]][:1]] or [infos[0].id]
                    ids = [m.id for m in infos]
                    mid = st.selectbox("Algoritmo", ids, index=ids.index(padrao[0]), format_func=lambda k: reg.obter(k).nome, key=f"av_alg_{e['tipo']}")
                    if st.button("Treinar e avaliar", type="primary", key="av_treinar"):
                        dados = lab._preparar(df, e)
                        if dados is not None:
                            with st.spinner("A treinar…"):
                                r = treinar_modelo(mid, dados, None, e["seed"], 3)
                            if r.ok:
                                definir_modelo(dados, r, "treinado no laboratório de avaliação")
                                st.rerun()
                            else:
                                st.error(r.motivo)
    return st.session_state.get("av_modelo")


# ══════════════════════════════════════════════════════════════════════════════
# Peças comuns: lição de cada etapa e desafio
# ══════════════════════════════════════════════════════════════════════════════
def _licao(chave: str):
    L = LICOES[chave]
    section_title(L["titulo"].upper())
    with st.expander("Aprender: o que é, porquê, como e como interpretar", expanded=False):
        st.markdown("**O que é.** " + L["o_que"])
        st.markdown("**Porque é importante.** " + L["porque"])
        st.markdown("**Como funciona.** " + L["como"])
        st.markdown("**Como interpretar.** " + L["interpretar"])
        st.markdown("**Exemplo prático.** " + L["exemplo"])
        d = L["desafio"]
        st.markdown("**Desafio:** " + d["pergunta"])
        esc = st.radio("A tua resposta", d["opcoes"], index=None, key=f"av_q_{chave}", label_visibility="collapsed")
        if st.button("Verificar resposta", key=f"av_qv_{chave}", disabled=esc is None):
            if d["opcoes"].index(esc) == d["certa"]:
                st.success("Certo! " + d["explicacao"])
            else:
                st.warning("Ainda não. " + d["explicacao"])


def _layout(fig, altura=340):
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 1 — Avaliar
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_avaliar(av):
    d, r = av["dados"], av["resultado"]
    for a in d.avisos + list(r.avisos):
        st.warning(a)
    st.caption(f"{r.nome} · {d.n_treino} exemplos de treino · {d.n_teste} de teste (dados que o modelo nunca viu).")
    cartoes = cl.metricas_cartoes(d.tipo)
    cols = st.columns(len(cartoes))
    for (k, rot), c in zip(cartoes, cols):
        c.metric(rot, _fmt(k, r.metricas.get(k)))
        c.caption(cl.interpretar_metrica(d.tipo, k, r.metricas.get(k), d))
    with st.expander("O que significa cada métrica"):
        for k, _ in cartoes + ([("roc_auc", "")] if d.tipo == "classificacao" else []):
            if k in ev.AJUDA:
                st.markdown("- " + ev.AJUDA[k])
    if d.tipo == "classificacao":
        fig = go.Figure(go.Bar(x=[rot for _, rot in cartoes], y=[r.metricas.get(k) for k, _ in cartoes]))
        st.plotly_chart(_layout(fig.update_layout(title="Métricas no teste", yaxis_range=[0, 1])), width="stretch")
        st.markdown("**Relatório de classificação** (por classe)")
        rep = cl.relatorio_classificacao(d, r)
        st.dataframe(rep.style.format({c: "{:.3f}" for c in ("Precision", "Recall", "F1-score")}, na_rep="—"), width="stretch", hide_index=True)
    elif r.metricas.get("residuos"):
        st.caption("Resíduos (real − previsto): " + ", ".join(f"{k} = {v:.4g}" for k, v in r.metricas["residuos"].items()))
    lab._graficos_modelo(types.SimpleNamespace(dados=d, tipo=d.tipo), r)
    _marcar("avaliar")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 2 — Generalização
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_generalizar(av):
    d, r = av["dados"], av["resultado"]
    if "ref" not in av:
        av["ref"] = cl.obter_referencia(d)
        av["treino"] = cl.avaliar_treino(r, d)
    cartoes = cl.metricas_cartoes(d.tipo)
    linhas = []
    for k, rot in cartoes:
        cv = r.cv.get(k) if r.cv else None
        linhas.append({"Métrica": rot, "Treino": av["treino"].get(k), "Teste": r.metricas.get(k),
                       "Validação cruzada (treino)": cv[0] if cv else None, "Referência simples": av["ref"].metricas.get(k) if av["ref"].ok else None})
    tab = pd.DataFrame(linhas)
    st.dataframe(tab.style.format({c: "{:.4f}" for c in tab.columns if c != "Métrica"}, na_rep="—"), width="stretch", hide_index=True)
    p = cl.principal(d.tipo)
    rot_p = ev.rotulo(d.tipo, p)
    fig = go.Figure()
    for nome, vals in (("Treino", [av["treino"].get(p)]), ("Teste", [r.metricas.get(p)]),
                       ("Validação cruzada", [r.cv[p][0] if r.cv and p in r.cv else None]),
                       ("Referência", [av["ref"].metricas.get(p) if av["ref"].ok else None])):
        if vals[0] is not None:
            fig.add_trace(go.Bar(name=nome, x=[rot_p], y=vals))
    st.plotly_chart(_layout(fig.update_layout(title=f"{rot_p}: treino × teste", barmode="group")), width="stretch")
    g = cl.diagnosticar_generalizacao(r, d, av["ref"])
    if g.estado == "sobreajuste" or g.estado == "subajuste":
        st.warning(f"**{g.titulo}.** {g.explicacao}")
    elif g.estado == "bom":
        st.success(f"**{g.titulo}.** {g.explicacao}")
    else:
        st.info(f"**{g.titulo}.** {g.explicacao}")
    for a in g.avisos:
        st.caption(a)
    if g.sugestoes:
        st.markdown("**O que podes tentar:**")
        for s in g.sugestoes:
            st.markdown("- " + s)
    st.caption("Isto é um indício a partir de uma só divisão treino/teste, não um diagnóstico definitivo.")
    st.markdown("#### Curva de aprendizagem")
    st.caption("Mostra se mais exemplos ajudariam. Calcula-se só com os dados de treino (o teste não entra).")
    if st.button("Calcular a curva de aprendizagem", key="av_curva_go"):
        with st.spinner("A calcular…"):
            try:
                st.session_state["av_curva"] = cl.curva_aprendizagem(r, d)
            except cl.ErroCiclo as e:
                st.session_state.pop("av_curva", None)
                st.error(str(e))
    c = st.session_state.get("av_curva")
    if c is not None:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=c["exemplos"], y=c["treino"], mode="lines+markers", name="Treino"))
        f2.add_trace(go.Scatter(x=c["exemplos"], y=c["validacao"], mode="lines+markers", name="Validação cruzada",
                                error_y=dict(type="data", array=c["validacao_desvio"])))
        st.plotly_chart(_layout(f2.update_layout(title="Curva de aprendizagem", xaxis_title="Exemplos de treino", yaxis_title=rot_p)), width="stretch")
        st.caption("Se as duas linhas estão longe uma da outra, o modelo decora (mais dados ajudam). "
                   "Se as duas estão juntas e baixas, o modelo é simples demais.")
    _marcar("generalizar")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 3 — Comparar algoritmos
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_comparar(av):
    d = av["dados"]
    st.caption("Todos os algoritmos usam a mesma divisão treino/teste e os mesmos blocos de validação cruzada que o modelo actual.")
    infos = [m for m in reg.listar(d.tipo) if m.tipo_modelo == "tabular" and m.disponivel()]
    ids = [m.id for m in infos]
    padrao = [i for i in auto_ml.PRIORIDADE[d.tipo][:4] if i in ids]
    escolha = st.multiselect("Algoritmos a comparar", ids, default=padrao, format_func=lambda k: reg.obter(k).nome, key=f"av_cmp_algos_{d.tipo}")
    c1, c2, c3 = st.columns(3)
    cv = int(c1.selectbox("Validação cruzada", [3, 5], key="av_cmp_cv", format_func=lambda v: f"{v} blocos"))
    max_m = int(c2.number_input("Máximo de modelos", 1, lab.MAX_MODELOS_COMPARACAO, 6, key="av_cmp_max"))
    orc = int(c3.number_input("Tempo máximo (s)", 10, 900, 180, 10, key="av_cmp_orc"))
    if st.button("Comparar algoritmos", type="primary", disabled=not escolha, key="av_cmp_go"):
        barra = st.progress(0.0, text="A começar…")
        comp = comparar_modelos(escolha, d, None, cv, d.random_state, max_m, orc,
                                progresso=lambda i, n, nome: barra.progress(min(i / max(n, 1), 1.0), text=f"A treinar {nome} ({i + 1}/{n})"))
        barra.empty()
        st.session_state["av_comp"] = comp
    comp = st.session_state.get("av_comp")
    if comp is None:
        return
    for a in comp.avisos:
        st.warning(a)
    criterios = ev.CRITERIOS[d.tipo]
    ordem = st.selectbox("Escolher o melhor segundo", criterios + ["tempo"], key="av_cmp_ordem",
                         format_func=lambda k: "Tempo de treino" if k == "tempo" else ev.rotulo(d.tipo, k))
    tab = comp.tabela_visivel(ordem)
    st.dataframe(tab.style.format({c: "{:.4f}" for c in tab.columns if c not in ("Modelo", "Estado")}, na_rep="—"), width="stretch", hide_index=True)
    st.caption("Notas do teste. A coluna (CV) é a validação cruzada só com o treino. «Referência» é um palpite sem inteligência: um modelo útil tem de a superar.")
    if ordem != "tempo":
        rec = recomendar(comp, ordem)
        if rec.modelo_id:
            st.success(f"**Melhor segundo {rec.criterio}: {rec.nome}** (medido em {rec.base}).")
            st.write(rec.justificacao)
            for a in rec.avisos:
                st.warning(a)
    mm = cl.melhor_por_metrica(comp)
    st.markdown("**Cada critério pode eleger um modelo diferente**")
    st.dataframe(mm.style.format({"Valor no teste": "{:.4f}"}, na_rep="—"), width="stretch", hide_index=True)
    if cl.divergem(mm):
        st.info("Os critérios não concordam: escolhe o que melhor reflecte o que importa no teu problema (por exemplo, recall se não podes falhar casos).")
    metrica = ordem if ordem != "tempo" else cl.principal(d.tipo)
    g = cl.dados_grafico_comparacao(comp, metrica)
    if len(g):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Teste", x=g["modelo"], y=g["teste"]))
        if g["cv"].notna().any():
            fig.add_trace(go.Bar(name="Validação cruzada", x=g["modelo"], y=g["cv"], error_y=dict(type="data", array=g["cv_desvio"].fillna(0))))
        st.plotly_chart(_layout(fig.update_layout(title=ev.rotulo(d.tipo, metrica), barmode="group")), width="stretch")
        st.plotly_chart(_layout(go.Figure(go.Bar(x=g["modelo"], y=g["tempo"])).update_layout(title="Tempo de treino (s)")), width="stretch")
        nomes = {r.id: r.nome for r in comp.validos()}
        sel = st.selectbox("Continuar o percurso com", list(nomes), format_func=nomes.get, key="av_cmp_sel")
        if st.button("Usar este modelo no laboratório", key="av_cmp_usar"):
            definir_modelo(d, comp.por_id(sel), "comparação", comp)
            st.session_state["av_comp"] = comp
            st.rerun()
    for r in comp.falhas():
        st.caption(f"{r.nome}: {r.motivo}")
    _marcar("comparar")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 4 — Optimizar hiperparâmetros
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_optimizar(av):
    d, r = av["dados"], av["resultado"]
    infos = [m for m in reg.listar(d.tipo) if m.tipo_modelo == "tabular" and m.disponivel() and m.parametros]
    if not infos:
        st.info("Não há algoritmos com parâmetros para optimizar.")
        return
    ids = [m.id for m in infos]
    por_omissao = ids.index(r.id) if r.id in ids else 0
    mid = st.selectbox("Algoritmo a optimizar", ids, index=por_omissao, format_func=lambda k: reg.obter(k).nome, key="av_op_alg")
    info = reg.obter(mid)
    base = r.params if mid == r.id else None
    st.caption("Em cada parâmetro escreve os valores a testar, separados por vírgulas. Deixa vazio para não o variar.")
    sugestao = cl.espaco_padrao(info)
    texto = {}
    for nome, s in info.parametros.items():
        ajuda = (s.get("ajuda") or "") + (f" Intervalo: {s['min']} a {s['max']}." if s["tipo"] != "choice" else "")
        if s["tipo"] == "choice":
            texto[nome] = st.multiselect(nome, s["opcoes"], default=[v for v in sugestao[nome][:2]], key=f"av_op_{mid}_{nome}", help=ajuda or None)
        else:
            texto[nome] = st.text_input(nome, value=", ".join(str(v) for v in sugestao[nome]), key=f"av_op_{mid}_{nome}", help=ajuda or None)
    espaco, erros = cl.validar_espaco(info, texto)
    for e in erros:
        st.error(e)
    c1, c2, c3 = st.columns(3)
    metodo = c1.radio("Método", ["grid", "random"], key="av_op_metodo",
                      format_func=lambda k: "Grelha (GridSearchCV)" if k == "grid" else "Aleatória (RandomizedSearchCV)")
    criterio = c2.selectbox("Critério a optimizar", list(cl.SCORING[d.tipo]), key=f"av_op_crit_{d.tipo}", format_func=lambda k: ev.rotulo(d.tipo, k))
    folds = int(c3.selectbox("Validação cruzada", [3, 5], key="av_op_cv", format_func=lambda v: f"{v} blocos"))
    combos = cl.contar_combinacoes(espaco)
    n_iter = combos
    if metodo == "random" and combos:
        n_iter = int(st.number_input("Combinações a testar", 1, max(combos, 1), min(10, combos), key="av_op_niter"))
    if combos:
        st.info(f"{combos} combinações possíveis. Vão testar-se {n_iter}, com {folds} blocos: {n_iter * folds} treinos (limite de {cl.LIMITE_COMBINACOES} combinações).")
    if st.button("Executar a optimização", type="primary", disabled=not espaco or bool(erros), key="av_op_go"):
        with st.spinner("A procurar os melhores hiperparâmetros…"):
            try:
                st.session_state["av_otim"] = cl.otimizar(mid, d, espaco, metodo, criterio, n_iter, folds, base)
            except cl.ErroCiclo as e:
                st.session_state.pop("av_otim", None)
                st.error(str(e))
    o = st.session_state.get("av_otim")
    if o is None:
        return
    st.markdown("#### Antes e depois")
    linhas = []
    for k, rot in cl.metricas_cartoes(d.tipo):
        linhas.append({"Métrica (teste)": rot, "Original": o.original.metricas.get(k), "Optimizado": o.otimizado.metricas.get(k)})
    linhas.append({"Métrica (teste)": f"{ev.rotulo(d.tipo, o.criterio)} (validação cruzada)", "Original": o.cv_original, "Optimizado": o.cv_otimizado})
    tab = pd.DataFrame(linhas)
    st.dataframe(tab.style.format({"Original": "{:.4f}", "Optimizado": "{:.4f}"}, na_rep="—"), width="stretch", hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original", x=tab["Métrica (teste)"], y=tab["Original"]))
    fig.add_trace(go.Bar(name="Optimizado", x=tab["Métrica (teste)"], y=tab["Optimizado"]))
    st.plotly_chart(_layout(fig.update_layout(title="Original × optimizado", barmode="group")), width="stretch")
    st.markdown("**Parâmetros:** original " + json.dumps(o.params_originais, ensure_ascii=False) + " → optimizado " + json.dumps(o.melhores_params, ensure_ascii=False))
    for linha in o.conclusao:
        st.write(linha)
    for a in o.avisos:
        st.warning(a)
    st.caption(f"{o.metodo == 'grid' and 'Grelha' or 'Aleatória'}: {o.n_testadas} combinações em {o.tempo:.1f} s. A procura só viu os dados de treino; o teste foi usado uma vez, no fim, para esta comparação.")
    with st.expander("Melhores combinações testadas"):
        st.dataframe(o.tabela, width="stretch", hide_index=True)
    if st.button("Usar o modelo optimizado no laboratório", key="av_op_usar"):
        definir_modelo(d, o.otimizado, "optimização", av.get("comp"), otimizacao=o)
        st.session_state["av_otim"] = o
        st.rerun()
    _marcar("optimizar")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 5 — Guardar e exportar
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_guardar(av, username):
    d, r = av["dados"], av["resultado"]
    av["nome"] = st.text_input("Nome do modelo", value=av["nome"], max_chars=80, key="av_g_nome")
    meta = av.get("salvo") or _meta(av)
    st.dataframe(pd.DataFrame([(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)) for k, v in cl.metadados_legiveis(meta)],
                              columns=["Informação", "Valor"]), width="stretch", hide_index=True)
    st.caption("O pré-processamento (imputação, escala, one-hot) está dentro do Pipeline e viaja com o modelo; a ordem das características e as classes ficam nos metadados.")
    if st.button("Guardar o modelo na minha conta", type="primary", key="av_g_guardar"):
        try:
            info = mp.guardar_modelo(username, r.pipeline, cl.meta_ciclo(d, r, av["nome"], av["origem"], av.get("otimizacao")))
            av["salvo"] = info
            st.success(f"Guardado como «{info['nome']}» (versão {info['versao']}).")
            _marcar("guardar")
        except mp.ErroPersistencia as e:
            st.error(str(e))
        except Exception:
            st.error("Não consegui guardar o modelo.")
    st.markdown("#### Descarregar")
    meta = av.get("salvo") or _meta(av)
    try:
        c1, c2, c3 = st.columns(3)
        c1.download_button("Modelo (.joblib)", cl.exportar_joblib(r.pipeline), file_name="modelo.joblib", mime="application/octet-stream", key="av_dl_joblib")
        c2.download_button("Metadados (.json)", json.dumps(meta, ensure_ascii=False, indent=1, default=str).encode("utf-8"),
                           file_name="metadados.json", mime="application/json", key="av_dl_json")
        c3.download_button("Pacote completo (.zip)", cl.exportar_pacote(r.pipeline, meta), file_name="modelo_dataforge.zip", mime="application/zip", key="av_dl_zip")
    except Exception:
        st.error("Não consegui preparar os ficheiros para descarregar.")
    st.warning("Abrir um ficheiro joblib executa código. Só abras ficheiros que tu mesmo exportaste ou de quem confias. "
               "Na plataforma, os modelos guardados são verificados com uma assinatura antes de abrir.")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 6 — Prediction Playground
# ══════════════════════════════════════════════════════════════════════════════
def _formulario_caso(meta: dict, chave: str) -> dict:
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
            ajuda = f"Visto no treino: {iv[0]:.4g} a {iv[1]:.4g}" if iv and None not in iv else None
            valores[f] = c.number_input(f, value=float(iv[2]) if iv and iv[2] is not None else 0.0, key=f"{chave}_{f}", help=ajuda)
    return valores


def _etapa_prever(av, username):
    r = av["resultado"]
    fonte = st.radio("Que modelo usar?", ["O modelo que estou a avaliar", "Um modelo guardado"], key="av_p_fonte", horizontal=True)
    if fonte == "O modelo que estou a avaliar":
        modelo, meta = r.pipeline, _meta(av)
    else:
        try:
            guardados = [m for m in mp.listar_modelos(username) if m.get("tipo") in ("regressao", "classificacao")]
        except mp.ErroPersistencia as e:
            st.error(str(e))
            return
        if not guardados:
            st.info("Ainda não guardaste nenhum modelo de classificação ou regressão (etapa 5).")
            return
        nomes = {m["id"]: f"{m['nome']} (v{m['versao']}, {m['criado_em']})" for m in guardados}
        mid = st.selectbox("Modelo guardado", list(nomes), format_func=nomes.get, key="av_p_guardado")
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
    modo = st.radio("Como queres prever?", ["Um caso", "Um ficheiro CSV"], key="av_p_modo", horizontal=True)
    if modo == "Um caso":
        valores = _formulario_caso(meta, "av_pc")
        if st.button("Prever", type="primary", key="av_p_go"):
            try:
                p, det, rel = ps.prever_um(modelo, meta, valores)
            except ps.ErroPrevisao as e:
                st.error(str(e))
                return
            st.metric(f"Previsão de {meta['alvo']}", f"{p:.4g}" if isinstance(p, (int, float, np.floating)) else str(p))
            if det:
                tab = pd.DataFrame({"Classe": [k[5:] for k in det], "Probabilidade": list(det.values())})
                st.plotly_chart(_layout(go.Figure(go.Bar(x=tab["Classe"], y=tab["Probabilidade"])).update_layout(title="Probabilidade de cada classe", yaxis_range=[0, 1]), 280), width="stretch")
            for a in rel["avisos"]:
                st.warning(a)
            _marcar("prever")
    else:
        st.download_button("Descarregar um CSV de exemplo com as colunas obrigatórias", cl.csv_modelo(meta), file_name="exemplo_entrada.csv", mime="text/csv", key="av_p_modelo")
        f = st.file_uploader("CSV com os casos novos", type=["csv", "xlsx"], key="av_p_file")
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
        alvo = f"previsao_{meta['alvo']}"
        st.success(f"{len(saida)} previsões feitas.")
        st.dataframe(saida.head(200), width="stretch")
        if meta["tipo"] == "classificacao":
            vc = saida[alvo].astype(str).value_counts()
            fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist()))
        else:
            fig = go.Figure(go.Histogram(x=saida[alvo], nbinsx=30))
        st.plotly_chart(_layout(fig.update_layout(title="Distribuição das previsões"), 300), width="stretch")
        st.download_button("Descarregar as previsões (CSV)", saida.to_csv(index=False).encode("utf-8"), file_name="previsoes.csv", mime="text/csv", key="av_p_dl")
        _marcar("prever")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 7 — Preparar a implementação
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_implementar(av):
    meta = _meta(av)
    st.info("Isto é educativo: gera código que podes correr e adaptar. A plataforma não coloca o teu modelo em produção.")
    tipos = {"uso": "Usar em Python", "fastapi": "API REST (FastAPI)", "streamlit": "Aplicação Web (Streamlit)", "cliente": "Integrar noutra aplicação (cliente HTTP)"}
    t = st.radio("O que queres ver?", list(tipos), format_func=tipos.get, key="av_i_tipo")
    try:
        codigo = cl.gerar_codigo(meta, t)
    except cl.ErroCiclo as e:
        st.error(str(e))
        return
    ficheiro = {"uso": "usar_modelo.py", "fastapi": "api.py", "streamlit": "app_modelo.py", "cliente": "cliente.py"}[t]
    st.code(codigo, language="python")
    st.download_button(f"Descarregar {ficheiro}", codigo.encode("utf-8"), file_name=ficheiro, mime="text/x-python", key=f"av_i_dl_{t}")
    if t == "fastapi":
        st.caption("Para o executar: guarda o modelo.joblib na mesma pasta e corre «uvicorn api:app --reload». A documentação aparece em /docs.")
    elif t == "streamlit":
        st.caption("Para o executar: «streamlit run app_modelo.py». Para publicar de graça podes usar o Streamlit Community Cloud.")
    elif t == "cliente":
        st.caption("O cliente chama a API do exemplo anterior. Qualquer linguagem que faça pedidos HTTP serve.")
    st.markdown("**Antes de pores isto em produção:** autenticação, limites de uso, registos, testes automáticos, versões fixas das bibliotecas e um plano para monitorizar o modelo (etapa 8).")
    _marcar("implementar")


# ══════════════════════════════════════════════════════════════════════════════
# Etapa 8 — Monitorização
# ══════════════════════════════════════════════════════════════════════════════
def _etapa_monitorizar(av):
    meta = _meta(av)
    st.markdown("**O que se acompanha depois de implementar:**")
    for t in ("Distribuição dos novos dados: continuam parecidos com os do treino?",
              "Deriva de dados (data drift): alguma variável mudou muito?",
              "Distribuição das previsões: o modelo passou a prever de outra forma?",
              "Métricas ao longo do tempo: só se calculam quando se conhece a resposta real.",
              "Quando reavaliar ou treinar de novo: quando a deriva é forte ou as métricas pioram."):
        st.markdown("- " + t)
    st.caption("A monitorização contínua depende de ir recebendo dados novos e, para as métricas, dos valores reais correspondentes. "
               "Aqui podes simular um ponto de controlo com um ficheiro de dados novos.")
    f = st.file_uploader("Dados novos (CSV ou Excel), com as mesmas colunas do treino", type=["csv", "xlsx"], key="av_m_file")
    if f is None:
        return
    try:
        novo = carregar_ficheiro(f.name, f.getvalue())
        saida, rel = cl.prever_ficheiro(av["resultado"].pipeline, meta, novo)
        previsoes = saida[f"previsao_{meta['alvo']}"].tolist()
        res = cl.analisar_deriva(meta, novo, previsoes)
    except (ErroDados, cl.ErroCiclo) as e:
        st.error(str(e))
        return
    for a in rel["avisos"] + res["avisos"]:
        st.warning(a)
    tab = res["variaveis"]
    if len(tab):
        st.markdown("#### Deriva por variável (PSI)")
        st.dataframe(tab.style.format({"PSI": "{:.3f}", "Média no treino": "{:.4g}", "Média nova": "{:.4g}"}, na_rep="—"), width="stretch", hide_index=True)
        graf = tab.dropna(subset=["PSI"])
        if len(graf):
            fig = go.Figure(go.Bar(x=graf["Variável"], y=graf["PSI"]))
            fig.add_hline(y=0.10, line_dash="dash")
            fig.add_hline(y=0.25, line_dash="dash")
            st.plotly_chart(_layout(fig.update_layout(title="PSI (0,10 = atenção · 0,25 = deriva forte)")), width="stretch")
    pv = res["previsoes"]
    if pv:
        st.markdown("#### Distribuição das previsões")
        st.write(f"PSI das previsões: {pv['psi']:.3f} ({pv['nivel']}).")
        if "antes" in pv:
            cls = list(pv["antes"])
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="No teste do treino", x=cls, y=[pv["antes"][c] for c in cls]))
            fig2.add_trace(go.Bar(name="Nos dados novos", x=cls, y=[pv["agora"][c] for c in cls]))
            st.plotly_chart(_layout(fig2.update_layout(barmode="group", title="Proporção de cada classe prevista")), width="stretch")
    st.markdown("#### Métricas com valores reais (opcional)")
    candidatas = [c for c in novo.columns if c == meta["alvo"]]
    if candidatas:
        try:
            m = cl.metricas_com_valores_reais(meta, previsoes, novo[candidatas[0]])
            for a in m["avisos"]:
                st.warning(a)
            st.caption(f"Calculadas em {m['n']} casos novos com valor real.")
            antes = meta.get("metricas_teste") or {}
            st.dataframe(pd.DataFrame([{"Métrica": ev.rotulo(meta["tipo"], k), "No teste (treino)": antes.get(k), "Nos dados novos": v}
                                       for k, v in m["metricas"].items()]).style.format({"No teste (treino)": "{:.4f}", "Nos dados novos": "{:.4f}"}, na_rep="—"),
                         width="stretch", hide_index=True)
        except cl.ErroCiclo as e:
            st.warning(str(e))
    else:
        st.caption(f"Para medir as métricas, inclui no ficheiro uma coluna «{meta['alvo']}» com o valor real de cada caso.")
    fortes = [] if not len(tab) else tab[tab["Nível"] == "forte"]["Variável"].tolist()
    if fortes:
        st.warning("Deriva forte em: " + ", ".join(fortes) + ". Convém investigar e, se as métricas com valores reais piorarem, reavaliar ou treinar de novo.")
    elif len(tab):
        st.success("Nenhuma variável com deriva forte neste ponto de controlo. Continua a acompanhar com dados novos.")
    st.caption("O PSI é uma regra prática, não um teste estatístico: a deriva pede atenção, mas só as métricas com valores reais mostram se o modelo piorou.")
    _marcar("monitorizar")


# ══════════════════════════════════════════════════════════════════════════════
# Entrada
# ══════════════════════════════════════════════════════════════════════════════
def _tutor(av):
    """Diz ao Tutor IA o que está no ecrã, só com números reais calculados."""
    try:
        from modules.tutor_ia import SUGESTOES_TREINO, registar_painel
        d, r = av["dados"], av["resultado"]
        ctx = {"modelo": r.nome, "problema": d.tipo, "alvo": d.alvo, "n_treino": d.n_treino, "n_teste": d.n_teste}
        for k, v in r.metricas.items():
            if isinstance(v, (int, float)):
                ctx[f"teste_{k}"] = v
        for k, v in (r.metricas_treino or {}).items():
            ctx[f"treino_{k}"] = v
        for k, v in (r.cv or {}).items():
            ctx[f"cv_{k}"] = v[0]
        if av.get("ref") is not None and av["ref"].ok:
            ctx.update({f"referencia_{k}": v for k, v in av["ref"].metricas.items() if isinstance(v, (int, float))})
        o = st.session_state.get("av_otim")
        if o is not None:
            ctx["otimizacao_criterio"] = o.criterio
            ctx["otimizacao_melhores_parametros"] = json.dumps(o.melhores_params, ensure_ascii=False)
        registar_painel("avaliacao", "Laboratório de Avaliação", cl.contexto_tutor(ctx), SUGESTOES_TREINO, padrao_primeiro=True)
    except Exception:
        pass


def render_avaliacao(username: str):
    page_header("Laboratório de Avaliação e Optimização", "Do modelo treinado ao modelo pronto a usar: avalia, optimiza, guarda, prevê e monitoriza.")
    st.caption("Tudo o que vês aqui são resultados reais calculados nos teus dados. Nada é inventado.")
    av = _passo_modelo()
    if av is None:
        return
    feitas = st.session_state.setdefault("av_feitas", set())
    n_feitas = len([k for k, _ in ETAPAS if k in feitas])
    st.progress(n_feitas / len(ETAPAS), text=f"Percurso: {n_feitas} de {len(ETAPAS)} etapas feitas")
    st.caption(" · ".join(("✓ " if k in feitas else "") + nome for k, nome in ETAPAS))
    st.markdown(f"**Modelo em avaliação:** {av['nome']} ({av['origem']})")
    tabs = st.tabs([f"{i + 1} · {nome}" for i, (_, nome) in enumerate(ETAPAS)])
    corpos = {
        "avaliar": lambda: _etapa_avaliar(av), "generalizar": lambda: _etapa_generalizar(av), "comparar": lambda: _etapa_comparar(av),
        "optimizar": lambda: _etapa_optimizar(av), "guardar": lambda: _etapa_guardar(av, username), "prever": lambda: _etapa_prever(av, username),
        "implementar": lambda: _etapa_implementar(av), "monitorizar": lambda: _etapa_monitorizar(av),
    }
    for (chave, _), tab in zip(ETAPAS, tabs):
        with tab:
            _licao(chave)
            try:
                corpos[chave]()
            except Exception:
                log.exception("Falha na etapa %s do laboratório de avaliação", chave)
                st.error("Algo correu mal nesta etapa. Tenta outra vez; se continuar, volta a escolher o modelo no passo 0.")
    _tutor(av)
