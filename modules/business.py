"""
DataForge Business - Ecrãs do painel Empresa.

Páginas: Início, Dados, Análises, Previsões, Relatórios, Equipa e Perfil.
A lógica (modelos, diagnóstico, PDF) está em business_core.py; aqui só há interface.
Nada aqui fala de aluno, turma, lições ou pontos: o painel Business é só para a empresa.
"""
import datetime
import io

import pandas as pd
import streamlit as st

from modules import business_core as bc
from modules import db, ia
from modules.i18n import idioma_actual, tr
from modules.utils import (aviso_box, badge, card, erro_box, info_box, ir_para, page_header, section_title,
                           sucesso_box, tema_figura_plotly)

PAPEIS = {"gestor": "Gestor", "analista": "Analista", "membro": "Membro"}
MAX_UPLOAD_MB = 50


# ══════════════════════════════════════════════════════
# CONTEXTO: EMPRESA ACTIVA E PERMISSÕES
# ══════════════════════════════════════════════════════
def empresas_do(username: str) -> list:
    return [m for m in db.membros_do_utilizador(username) if m["tipo"] == "empresa"]


def empresa_activa(username: str):
    """Empresa em que a pessoa está a trabalhar (a escolhida no menu lateral, ou a primeira)."""
    lista = empresas_do(username)
    if not lista:
        return None
    escolhida = st.session_state.get("org_activa")
    for m in lista:
        if m["org_id"] == escolhida:
            return m
    st.session_state["org_activa"] = lista[0]["org_id"]
    return lista[0]


def papel_rotulo(username: str):
    m = empresa_activa(username)
    return tr(PAPEIS.get(m["papel"], "Membro")) if m else None


def pode(m: dict, accao: str) -> bool:
    """gestor: tudo · analista: carregar dados e treinar · membro: só consultar e prever."""
    papel = (m or {}).get("papel")
    if accao in ("carregar", "treinar"):
        return papel in ("gestor", "analista")
    if accao == "equipa":
        return papel == "gestor"
    return papel in PAPEIS


def render_seletor_empresa(username: str):
    """Menu lateral: só aparece se a pessoa pertencer a mais do que uma empresa."""
    lista = empresas_do(username)
    if len(lista) < 2:
        return
    ids = [m["org_id"] for m in lista]
    nomes = {m["org_id"]: m["nome"] for m in lista}
    actual = st.session_state.get("org_activa", ids[0])
    st.selectbox("Empresa", ids, index=ids.index(actual) if actual in ids else 0,
                 format_func=lambda i: nomes[i], key="org_activa")


def _sem_empresa():
    page_header("DataForge Business", "Machine Learning nos dados da tua empresa")
    st.info("Ainda não pertences a nenhuma empresa activa. Pede o código de convite ao gestor da tua empresa e junta-te no teu Perfil.")


def render_sem_empresa(username: str):
    """Conta de empresa sem nenhuma empresa activa (foi removida, por exemplo): só pode juntar-se com um código."""
    page_header("DataForge Business", "Machine Learning nos dados da tua empresa")
    st.info("A tua conta não pertence a nenhuma empresa activa. Escreve o código de convite que o gestor da tua empresa te deu.")
    cod = st.text_input("Código de convite da empresa", key="bn_sememp_cod").strip().upper()
    if st.button("Juntar-me", type="primary", key="bn_sememp_btn") and cod:
        from modules.admin import adicionar_membro_por_codigo
        ok, msg = adicionar_membro_por_codigo(username, cod, "membro")
        if ok:
            st.session_state.pop("_membros_cache", None)
            st.rerun()
        else:
            st.error(msg)


def _dt_curta(iso) -> str:
    return str(iso or "")[:10]


def _tabela_aviso(res: dict):
    for a in res.get("avisos") or []:
        aviso_box(bc.texto_aviso(a))


def _carregar_tabela(dataset_id: str):
    """Conjunto de dados guardado -> DataFrame (com cache na sessão, que é pesado)."""
    cache = st.session_state.setdefault("_bn_cache_dados", {})
    if dataset_id not in cache:
        row = db.bn_get_dataset(dataset_id)
        if not row:
            return None
        if len(cache) >= 3:
            cache.pop(next(iter(cache)))
        cache[dataset_id] = bc.texto_para_tabela(row["conteudo"])
    return cache[dataset_id]


def _esquecer_cache(dataset_id: str = None):
    cache = st.session_state.get("_bn_cache_dados", {})
    if dataset_id:
        cache.pop(dataset_id, None)
    else:
        cache.clear()


def _guardar_conjunto(m: dict, username: str, nome: str, df: pd.DataFrame, limpar: bool) -> str:
    removidas = 0
    if limpar:
        df, removidas = bc.limpar_dados(df)
    df_final, texto, reduzido = bc.reduzir_para_guardar(df)
    did = db.bn_criar_dataset(m["org_id"], nome, texto, len(df_final), df_final.shape[1], username, reduzido)
    if removidas:
        st.toast(bc.T("{} linhas repetidas removidas.", removidas))
    if reduzido:
        aviso_box(bc.T("A tabela era grande: guardei uma amostra de {} linhas.", len(df_final)))
    return did


def _mostrar_diagnostico(df: pd.DataFrame):
    icones = {"ok": sucesso_box, "info": info_box, "aviso": aviso_box, "erro": erro_box}
    for nivel, modelo, args in bc.diagnostico(df):
        icones.get(nivel, info_box)(bc.T(modelo, *args))


def _assistente_ia(m: dict, df: pd.DataFrame, did: str):
    """Pergunta em linguagem normal -> a IA sugere problema, coluna a prever e colunas a ignorar (validado contra os dados)."""
    with st.expander(tr("Assistente com IA (opcional)")):
        if not ia.configurada():
            st.caption(tr("A IA está desligada. Quem administra a aplicação pode ligá-la com a chave «ia_api_key» nos segredos."))
            return
        st.caption(tr("Descreve o que queres saber e a IA sugere o problema e as colunas. Só são enviados os nomes das colunas e as categorias, nunca as linhas dos dados."))
        msg = st.session_state.pop("bn_ia_msg", None)
        if msg:
            sucesso_box(msg)
        if st.button("Testar a ligação à IA", key="bn_ia_testar"):
            with st.spinner(tr("A testar...")):
                try:
                    ia.testar(m["org_id"])
                except ia.IAErro as e:
                    erro_box(tr(str(e)))
                else:
                    sucesso_box(bc.T("A ligação à IA funciona ({} · modelo {}).", ia.provedor(), ia.modelo()))
        pergunta = st.text_area("O que queres saber?", key="bn_ia_pergunta", max_chars=500,
                                placeholder=tr("ex.: quero descobrir que clientes vão cancelar o serviço"))
        st.caption(bc.T("Pedidos à IA que restam hoje: {}", ia.pedidos_restantes(m["org_id"])))
        if st.button("Sugerir com IA", key="bn_ia_sugerir"):
            with st.spinner(tr("A pensar...")):
                try:
                    plano = ia.sugerir_analise(pergunta, df, m["org_id"], idioma_actual())
                except ia.IAErro as e:
                    erro_box(tr(str(e)))
                    return
            prob = plano["problema"]
            # estes valores têm de ser postos antes de os campos existirem neste ecrã (por isso o assistente vem primeiro)
            st.session_state["bn_problema"] = prob
            st.session_state["bn_an_problema"] = prob
            if plano["alvo"]:
                st.session_state[f"bn_alvo_{prob}_{did}"] = plano["alvo"]
            st.session_state[f"bn_excl_{prob}_{did}"] = plano["ignorar"]
            st.session_state["bn_ia_msg"] = bc.T("Sugestão aplicada: {}. {}", tr(bc.PROBLEMAS[prob]["titulo"]), plano["porque"])
            st.rerun()


def _fig(fig):
    fig = tema_figura_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════
# INÍCIO
# ══════════════════════════════════════════════════════
def render_inicio(username: str, name: str):
    m = empresa_activa(username)
    if not m:
        _sem_empresa()
        return
    org = m["org_id"]
    page_header(m["nome"], "DataForge Business · Machine Learning nos dados da tua empresa")
    dados = db.bn_datasets(org)
    analises = db.bn_analises(org)
    equipa = db.list_membros(org_id=org)
    modelos = [a for a in analises if a.get("modelo_guardado")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conjuntos de dados", len(dados))
    c2.metric("Análises feitas", len(analises))
    c3.metric("Modelos prontos", len(modelos))
    c4.metric("Pessoas na equipa", len(equipa))

    section_title("Começar em 4 passos")
    passos = [
        ("1", "Carregar os dados", "Um ficheiro CSV ou Excel com o histórico da empresa.", bool(dados), "business_dados", "Ir para Dados"),
        ("2", "Escolher o problema", "Vendas, procura, clientes que saem, fraude ou segmentos.", bool(analises), "business_analises", "Ir para Análises"),
        ("3", "Usar o modelo", "Prever para dados novos e descarregar a lista.", bool(modelos), "business_previsoes", "Ir para Previsões"),
        ("4", "Partilhar o relatório", "Um PDF de 1 página para a gestão.", bool(analises), "business_relatorios", "Ir para Relatórios"),
    ]
    cols = st.columns(4)
    for col, (n, titulo, desc, feito, destino, botao) in zip(cols, passos):
        with col:
            marca = badge(tr("Feito"), "green") if feito else badge(tr("Por fazer"), "amber")
            card(f"<b>{n}. {tr(titulo)}</b><br><span style='font-size:13px;opacity:.85'>{tr(desc)}</span><br><br>{marca}")
            if st.button(botao, key=f"bn_passo_{n}", use_container_width=True):
                ir_para(destino)

    section_title("Que problema queres resolver?")
    cols = st.columns(len(bc.PROBLEMAS))
    for col, (chave, cfg) in zip(cols, bc.PROBLEMAS.items()):
        with col:
            card(f"<b>{tr(cfg['titulo'])}</b><br><span style='font-size:13px;opacity:.85'>{tr(cfg['descricao'])}</span>")
            if st.button("Começar", key=f"bn_prob_{chave}", use_container_width=True):
                st.session_state["bn_problema"] = chave
                ir_para("business_analises")

    if analises:
        section_title("Análises recentes")
        for a in analises[:5]:
            rot = tr(bc.PROBLEMAS.get(a["problema"], {}).get("titulo", ""))
            st.markdown(f"- **{a['titulo']}** · {rot} · {a['criado_por']} · {_dt_curta(a['created_at'])}")

    if pode(m, "equipa"):
        section_title("Convidar a equipa")
        sucesso_box(bc.T("Código de convite da empresa: **{}**. Quem o usar no Perfil entra como membro; podes mudar o papel em «Equipa».",
                         m["codigo_convite"]))


# ══════════════════════════════════════════════════════
# DADOS
# ══════════════════════════════════════════════════════
def render_dados(username: str, name: str):
    m = empresa_activa(username)
    if not m:
        _sem_empresa()
        return
    page_header("Dados da empresa", "Carrega, verifica e guarda os dados com que a equipa vai trabalhar")
    aba_lista, aba_novo = st.tabs([tr("Os nossos dados"), tr("Carregar dados")])

    with aba_novo:
        if not pode(m, "carregar"):
            st.info("O teu papel só permite consultar. Pede ao gestor para te tornar analista se precisares de carregar dados.")
        else:
            _carregar_novo(m, username)

    with aba_lista:
        _lista_dados(m, username)


def _carregar_novo(m: dict, username: str):
    ficheiro = st.file_uploader("Ficheiro CSV ou Excel", type=["csv", "txt", "tsv", "xlsx", "xls"], key="bn_upload")
    st.caption(bc.T("Máximo {} MB. Na primeira linha devem estar os nomes das colunas.", MAX_UPLOAD_MB))
    df, nome_sug = None, ""
    if ficheiro is not None:
        dados = ficheiro.getvalue()
        if len(dados) > MAX_UPLOAD_MB * 1024 * 1024:
            erro_box(bc.T("O ficheiro tem mais de {} MB.", MAX_UPLOAD_MB))
            return
        folha = None
        if ficheiro.name.lower().endswith((".xlsx", ".xls")):
            folhas = bc.folhas_excel(dados)
            if len(folhas) > 1:
                folha = st.selectbox("Folha do Excel", folhas, key="bn_folha")
        try:
            df = bc.ler_tabela(ficheiro.name, dados, folha)
            nome_sug = ficheiro.name.rsplit(".", 1)[0]
        except ValueError as e:
            erro_box(tr(str(e)) if str(e).count(":") == 0 else str(e))
            return
    else:
        st.markdown("**" + tr("Ou experimenta com dados de exemplo") + "**")
        cols = st.columns(len(bc.EXEMPLOS))
        for col, (chave, (titulo, _)) in zip(cols, bc.EXEMPLOS.items()):
            if col.button(titulo, key=f"bn_ex_{chave}", use_container_width=True):
                st.session_state["bn_exemplo"] = chave
        if st.session_state.get("bn_exemplo"):
            ch = st.session_state["bn_exemplo"]
            df, nome_sug = bc.exemplo(ch), tr(bc.EXEMPLOS[ch][0])
    if df is None:
        return

    section_title("Pré-visualização")
    st.dataframe(df.head(50), use_container_width=True, height=260)
    section_title("Diagnóstico automático")
    _mostrar_diagnostico(df)
    with st.expander(tr("Ver as colunas")):
        r = bc.resumo_colunas(df)
        r["Tipo"] = r["Tipo"].map(tr)
        st.dataframe(r, use_container_width=True, hide_index=True)

    section_title("Guardar para a equipa")
    nome = st.text_input("Nome do conjunto de dados", value=nome_sug, key="bn_nome_dados", max_chars=60)
    limpar = st.checkbox("Remover linhas repetidas ao guardar", value=bool(df.duplicated().sum()), key="bn_limpar")
    if st.button("Guardar dados", type="primary", key="bn_guardar"):
        if len(nome.strip()) < 2:
            erro_box("Dá um nome ao conjunto de dados.")
            return
        try:
            _guardar_conjunto(m, username, nome.strip(), df, limpar)
        except (ValueError, db.DBError) as e:
            erro_box(str(e))
            return
        st.session_state.pop("bn_exemplo", None)
        st.session_state["bn_upload_ok"] = True
        st.success(bc.T("«{}» guardado. Já podes escolher o problema em «Análises».", nome.strip()))


def _lista_dados(m: dict, username: str):
    dados = db.bn_datasets(m["org_id"])
    if not dados:
        st.info("Ainda não há dados guardados. Carrega um ficheiro no separador «Carregar dados» ou experimenta com os exemplos.")
        return
    tabela = pd.DataFrame([{"Nome": d["nome"], "Linhas": d["linhas"], "Colunas": d["colunas"], "Carregado por": d["criado_por"],
                            "Data": _dt_curta(d["created_at"])} for d in dados])
    st.dataframe(tabela, use_container_width=True, hide_index=True)
    ids = [d["id"] for d in dados]
    nomes = {d["id"]: d["nome"] for d in dados}
    esc = st.selectbox("Abrir conjunto de dados", ids, format_func=lambda i: nomes[i], key="bn_abrir_dados")
    df = _carregar_tabela(esc)
    if df is None:
        erro_box("Não consegui abrir estes dados.")
        return
    reg = next(d for d in dados if d["id"] == esc)
    if reg.get("reduzido"):
        aviso_box("Este conjunto guardou só uma amostra dos dados originais (eram grandes demais).")
    st.dataframe(df.head(100), use_container_width=True, height=260)
    with st.expander(tr("Diagnóstico"), expanded=False):
        _mostrar_diagnostico(df)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Descarregar CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{reg['nome']}.csv",
                       mime="text/csv", use_container_width=True, key="bn_dl_dados")
    if c2.button("Analisar estes dados", use_container_width=True, key="bn_analisar_dados"):
        st.session_state["bn_dataset_id"] = esc
        ir_para("business_analises")
    pode_apagar = m["papel"] == "gestor" or reg["criado_por"] == username
    if pode_apagar and c3.button("Apagar", use_container_width=True, key="bn_apagar_dados"):
        st.session_state["bn_confirmar_apagar_dados"] = esc
    if st.session_state.get("bn_confirmar_apagar_dados") == esc:
        aviso_box("Apagar estes dados não apaga as análises já feitas com eles. Confirmas?")
        d1, d2 = st.columns(2)
        if d1.button("Sim, apagar", type="primary", key="bn_apagar_sim"):
            db.bn_apagar_dataset(esc)
            _esquecer_cache(esc)
            st.session_state.pop("bn_confirmar_apagar_dados", None)
            st.rerun()
        if d2.button("Cancelar", key="bn_apagar_nao"):
            st.session_state.pop("bn_confirmar_apagar_dados", None)
            st.rerun()


# ══════════════════════════════════════════════════════
# ANÁLISES (escolher o problema e treinar)
# ══════════════════════════════════════════════════════
def render_analises(username: str, name: str):
    m = empresa_activa(username)
    if not m:
        _sem_empresa()
        return
    page_header("Análises", "Escolhe o problema e o DataForge treina o modelo por ti")
    dados = db.bn_datasets(m["org_id"])
    if not dados:
        st.info("Primeiro precisas de dados. Vai a «Dados» e carrega um ficheiro (ou usa um exemplo).")
        if st.button("Ir para Dados", type="primary", key="bn_an_ir_dados"):
            ir_para("business_dados")
        return
    if not pode(m, "treinar"):
        st.info("O teu papel só permite consultar. Vê as análises em «Relatórios» e usa os modelos em «Previsões».")
        return

    ids = [d["id"] for d in dados]
    nomes = {d["id"]: bc.T("{} ({} linhas)", d["nome"], d["linhas"]) for d in dados}
    pre = st.session_state.get("bn_dataset_id")
    idx = ids.index(pre) if pre in ids else 0
    section_title("1. Que dados?")
    did = st.selectbox("Conjunto de dados", ids, index=idx, format_func=lambda i: nomes[i], key="bn_an_dados")
    df = _carregar_tabela(did)
    if df is None:
        erro_box("Não consegui abrir estes dados.")
        return
    nome_dados = next(d["nome"] for d in dados if d["id"] == did)
    _assistente_ia(m, df, did)

    section_title("2. Que problema?")
    chaves = list(bc.PROBLEMAS)
    atual = st.session_state.get("bn_problema")
    prob = st.radio("Problema", chaves, index=chaves.index(atual) if atual in chaves else 0, key="bn_an_problema",
                    format_func=lambda k: tr(bc.PROBLEMAS[k]["titulo"]), horizontal=False)
    cfg = bc.PROBLEMAS[prob]
    tipo = cfg["tipo"]
    st.caption(tr(cfg["descricao"]))

    alvo, excluir, por_data, taxa, k = None, [], False, 0.02, None
    section_title("3. Opções")
    if tipo in bc.TIPOS_SUPERVISIONADOS:
        possiveis = bc.colunas_alvo_possiveis(df, tipo)
        if not possiveis:
            erro_box("Nenhuma coluna serve para este problema. Escolhe outro problema ou outros dados.")
            return
        sug = bc.sugerir_alvo(df, prob)
        alvo = st.selectbox(cfg["alvo"], possiveis, index=possiveis.index(sug) if sug in possiveis else 0, key=f"bn_alvo_{prob}_{did}")
        if sug == alvo:
            st.caption(bc.T("Sugestão automática: a coluna «{}».", alvo))
    outras = [c for c in df.columns if c != alvo]
    ids_auto = [c for c in bc.colunas_id(df) if c != alvo]
    with st.expander(tr("Colunas a ignorar e outras opções")):
        excluir = st.multiselect("Colunas a ignorar (ex.: números de cliente, colunas que já contêm a resposta)", outras,
                                 default=[c for c in ids_auto if c in outras], key=f"bn_excl_{prob}_{did}")
        if tipo in bc.TIPOS_SUPERVISIONADOS and bc.colunas_data(df):
            por_data = st.checkbox("Validar com os dados mais recentes (recomendado para previsões no tempo)", value=tipo == "regressao",
                                   key=f"bn_pordata_{prob}_{did}")
        if tipo == "anomalia":
            taxa = st.slider("Que percentagem das operações esperas que seja invulgar?", 0.5, 10.0, 2.0, 0.5, key=f"bn_taxa_{did}") / 100
        if tipo == "segmentacao":
            auto = st.checkbox("Escolher o número de segmentos automaticamente", value=True, key=f"bn_kauto_{did}")
            k = None if auto else st.slider("Número de segmentos", 2, 8, 4, key=f"bn_k_{did}")
    if tipo == "anomalia":
        rot = [c for c in df.columns if "fraud" in c.lower() and c not in excluir]
        if rot:
            info_box(bc.T("A coluna «{}» parece já dizer quais são fraudes. Para o modelo descobrir sozinho, ignora-a nas opções acima; se quiseres aprender com ela, usa «Clientes que vão sair» com esta coluna.", rot[0]))

    section_title("4. Treinar")
    titulo = st.text_input("Título da análise", value=f"{tr(cfg['titulo'])} - {nome_dados}"[:80], key=f"bn_titulo_{prob}_{did}")
    if st.button("Treinar e analisar", type="primary", key="bn_treinar"):
        with st.spinner(tr("A treinar e a comparar modelos... pode demorar até um minuto.")):
            try:
                if tipo in bc.TIPOS_SUPERVISIONADOS:
                    saida = bc.treinar(df, alvo, tipo, excluir, por_data)
                elif tipo == "anomalia":
                    saida = bc.detectar_anomalias(df, excluir, taxa)
                else:
                    saida = bc.segmentar(df, excluir, k)
            except ValueError as e:
                erro_box(str(e))
                return
            except Exception as e:                       # um erro inesperado nunca deve rebentar o ecrã
                erro_box(bc.T("Não foi possível concluir a análise: {}", str(e)[:200]))
                return
        texto_modelo = bc.modelo_para_texto({"modelo": saida["modelo"]})
        try:
            aid = db.bn_criar_analise(m["org_id"], did, nome_dados, username, prob, titulo.strip() or tr(cfg["titulo"]),
                                      alvo, saida["resultado"], saida["meta"], texto_modelo)
        except db.DBError as e:
            erro_box(str(e))
            return
        st.session_state["bn_ultima"] = {"id": aid, "saida": saida, "problema": prob, "titulo": titulo, "df": _amostra_para_tabela(saida, df)}
        if not texto_modelo:
            aviso_box("O modelo ficou grande demais para ser guardado: os resultados ficam gravados, mas não poderás usá-lo em «Previsões».")

    ultima = st.session_state.get("bn_ultima")
    if ultima and ultima.get("problema") == prob:
        st.divider()
        mostrar_resultado(ultima["saida"]["resultado"], ultima["problema"], tabela=ultima.get("df"), chave="bn_res")
        sucesso_box("Análise guardada em «Relatórios»: lá podes descarregar o PDF para a gestão.")
        c1, c2 = st.columns(2)
        if c1.button("Usar este modelo em Previsões", key="bn_ir_prev", use_container_width=True):
            st.session_state["bn_analise_prev"] = ultima["id"]
            ir_para("business_previsoes")
        if c2.button("Ir para Relatórios", key="bn_ir_rel", use_container_width=True):
            st.session_state["bn_analise_rel"] = ultima["id"]
            ir_para("business_relatorios")


def _amostra_para_tabela(saida: dict, df: pd.DataFrame):
    """Tabela completa de resultados (fraude: linhas suspeitas · segmentos: cliente + segmento) para descarregar já."""
    res = saida["resultado"]
    if res["tipo"] == "anomalia":
        t = saida["tabela"]
        out = df.copy()
        out["pontuacao_anomalia"] = t["pontuacao"].values
        out["suspeito"] = t["suspeito"].map({True: "sim", False: "não"}).values
        return out.sort_values("pontuacao_anomalia", ascending=False).head(5000)
    if res["tipo"] == "segmentacao":
        out = df.copy()
        out["segmento"] = saida["rotulos"]
        return out
    return None


# ══════════════════════════════════════════════════════
# MOSTRAR UM RESULTADO (Análises e Relatórios)
# ══════════════════════════════════════════════════════
def mostrar_resultado(res: dict, problema: str, tabela=None, chave: str = "res"):
    tipo = res["tipo"]
    if tipo in bc.TIPOS_SUPERVISIONADOS:
        _res_supervisionado(res, chave)
    elif tipo == "anomalia":
        _res_anomalia(res, tabela, chave)
    else:
        _res_segmentos(res, tabela, chave)
    _tabela_aviso(res)
    prox = bc.PROBLEMAS.get(problema, {}).get("seguinte", [])
    if prox:
        section_title("Próximos passos")
        for p in prox:
            st.markdown("- " + tr(p))


def _res_supervisionado(res: dict, chave: str):
    import plotly.express as px
    import plotly.graph_objects as go
    rot, cor = bc.qualidade(res)
    st.markdown(f"### {tr('Resultado')} {badge(tr(rot), cor)}", unsafe_allow_html=True)
    st.markdown(bc.manchete(res))
    m = res["metricas"]
    if res["tipo"] == "regressao":
        c1, c2, c3 = st.columns(3)
        c1.metric("Erro médio", bc._num(m["mae"]), help=tr("Quanto a previsão falha, em média, na unidade do valor."))
        c2.metric("Erro sem modelo", bc._num(m["mae_base"]), help=tr("Erro de um palpite pela média: o modelo deve ser bem melhor."))
        c3.metric("Variação explicada", f"{max(m['r2'], 0):.0%}", help=tr("Quanto da variação o modelo explica (1 = perfeito)."))
    else:
        cols = st.columns(4 if "recall" in m else 2)
        cols[0].metric("Taxa de acerto", f"{m['accuracy']:.0%}", help=tr("Percentagem de casos em que o modelo acerta."))
        cols[1].metric("Sem modelo", f"{m['accuracy_base']:.0%}", help=tr("Acerto de quem respondesse sempre o mais comum."))
        if "recall" in m:
            cols[2].metric("Recall", f"{m['recall']:.0%}", help=tr("Dos casos reais, quantos o modelo apanha."))
            cols[3].metric("Precisão", f"{m['precisao']:.0%}", help=tr("Quando o modelo avisa, quantas vezes acerta."))
    st.caption(bc.T("Treinado com {} linhas e verificado em {} linhas que nunca viu · algoritmo escolhido: {} · {} s",
                    res["n_treino"], res["n_teste"], tr(res["melhor"]), res.get("segundos", 0)))
    if res.get("por_data"):
        st.caption(tr("A verificação usou os dados mais recentes, como acontece na vida real."))

    if res.get("importancias"):
        section_title("O que mais pesa na previsão")
        imp = pd.DataFrame(res["importancias"], columns=["Coluna", "Peso"]).iloc[::-1]
        fig = px.bar(imp, x="Peso", y="Coluna", orientation="h")
        fig.update_xaxes(tickformat=".0%", title=tr("Peso na previsão"))
        fig.update_yaxes(title="")
        _fig(fig)
    g = res.get("grafico")
    if g:
        section_title("Previsto contra real")
        fig = go.Figure(go.Scatter(x=g["real"], y=g["previsto"], mode="markers", marker={"opacity": 0.6}, name=tr("Casos")))
        lo, hi = min(g["real"] + g["previsto"]), max(g["real"] + g["previsto"])
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name=tr("Previsão perfeita"), line={"dash": "dash"}))
        fig.update_xaxes(title=tr("Valor real"))
        fig.update_yaxes(title=tr("Valor previsto"))
        _fig(fig)
        st.caption(tr("Quanto mais os pontos estão perto da linha, melhor a previsão."))
    if res.get("matriz"):
        section_title("Onde o modelo acerta e falha")
        mx = res["matriz"]
        fig = px.imshow(mx["valores"], x=mx["classes"], y=mx["classes"], text_auto=True, aspect="auto",
                        labels={"x": tr("Previsto"), "y": tr("Real"), "color": tr("Casos")})
        _fig(fig)
    with st.expander(tr("Algoritmos comparados")):
        t = pd.DataFrame([{"Algoritmo": tr(c["nome"]), "Pontuação na validação": c["pontuacao"]} for c in res["candidatos"]])
        st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption(tr("O DataForge testa 3 algoritmos e fica com o melhor; a nota final é medida em dados que o modelo nunca viu."))


def _res_anomalia(res: dict, tabela, chave: str):
    import plotly.graph_objects as go
    st.markdown(f"### {tr('Resultado')}")
    st.markdown(bc.T("Foram analisadas **{}** operações e **{}** ({}) saem do normal e merecem ser investigadas.",
                     res["n_linhas"], res["n_suspeitos"], f"{res['n_suspeitos'] / max(res['n_linhas'], 1):.1%}"))
    g = res["grafico"]
    fig = go.Figure(go.Histogram(x=g["pontuacoes"], nbinsx=40, name=tr("Operações")))
    fig.add_vline(x=g["limiar"], line_dash="dash", annotation_text=tr("Limite de suspeita"))
    fig.update_xaxes(title=tr("Pontuação de anomalia (maior = mais invulgar)"))
    fig.update_yaxes(title=tr("Operações"))
    _fig(fig)
    section_title("As mais invulgares")
    linhas = [{"Linha": t["linha"], "Pontuação": t["pontuacao"],
               "Porquê": "; ".join(bc.texto_motivo(x) for x in t["motivos"]) or "-"} for t in res["top"][:25]]
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
    st.caption(tr("A «Linha» é o número da linha no ficheiro original (a linha 1 são os nomes das colunas)."))
    if tabela is not None:
        st.download_button("Descarregar lista completa (CSV)", tabela.to_csv(index=False).encode("utf-8-sig"),
                           file_name="anomalias.csv", mime="text/csv", key=f"{chave}_dl")


def _res_segmentos(res: dict, tabela, chave: str):
    import plotly.express as px
    st.markdown(f"### {tr('Resultado')}")
    st.markdown(bc.T("Os clientes foram agrupados em **{}** segmentos.", res["k"]))
    g = res["grafico"]
    dfp = pd.DataFrame({"x": g["x"], "y": g["y"], "Segmento": [str(s) for s in g["segmento"]]})
    fig = px.scatter(dfp, x="x", y="y", color="Segmento", opacity=0.7)
    fig.update_xaxes(title=tr("Eixo 1 (resumo das colunas)"))
    fig.update_yaxes(title=tr("Eixo 2 (resumo das colunas)"))
    _fig(fig)
    section_title("Quem é cada segmento")
    for p in res["perfis"]:
        st.markdown(f"**{bc.nome_segmento(p)}** · {p['tamanho']} ({p['percentagem']:.0%})")
        if p["caracteristicas"]:
            st.caption("; ".join(bc.T("{}: média {} (geral {})", c, v, geral) for c, _, v, geral in p["caracteristicas"]))
    if tabela is not None:
        st.download_button("Descarregar clientes com segmento (CSV)", tabela.to_csv(index=False).encode("utf-8-sig"),
                           file_name="segmentos.csv", mime="text/csv", key=f"{chave}_dl")


# ══════════════════════════════════════════════════════
# PREVISÕES
# ══════════════════════════════════════════════════════
def render_previsoes(username: str, name: str):
    m = empresa_activa(username)
    if not m:
        _sem_empresa()
        return
    page_header("Previsões", "Usa um modelo já treinado em dados novos")
    analises = [a for a in db.bn_analises(m["org_id"]) if a.get("modelo_guardado")]
    if not analises:
        st.info("Ainda não há modelos prontos. Treina um em «Análises» e volta aqui.")
        if pode(m, "treinar") and st.button("Ir para Análises", type="primary", key="bn_pv_ir"):
            ir_para("business_analises")
        return
    ids = [a["id"] for a in analises]
    rotulo = {a["id"]: f"{a['titulo']} · {_dt_curta(a['created_at'])}" for a in analises}
    pre = st.session_state.get("bn_analise_prev")
    esc = st.selectbox("Modelo", ids, index=ids.index(pre) if pre in ids else 0, format_func=lambda i: rotulo[i], key="bn_pv_modelo")
    cache = st.session_state.setdefault("_bn_cache_modelos", {})
    if esc not in cache:
        row = db.bn_get_analise(esc)
        try:
            cache[esc] = (row["meta"], bc.texto_para_modelo(row["modelo"])["modelo"], row)
        except Exception:
            erro_box("Não consegui abrir este modelo. Treina-o de novo em «Análises».")
            return
    meta, modelo, row = cache[esc]
    tipo = meta["tipo"]
    nome_prob = tr(bc.PROBLEMAS.get(row["problema"], {}).get("titulo", ""))
    st.caption(bc.T("{} · dados de origem: {} · colunas usadas: {}", nome_prob, row.get("dataset_nome", ""),
                    ", ".join(meta["colunas_entrada"][:10]) + ("..." if len(meta["colunas_entrada"]) > 10 else "")))

    aba_ficheiro, aba_guardados, aba_caso = st.tabs([tr("Ficheiro novo"), tr("Dados guardados"), tr("Um caso à mão")])
    with aba_ficheiro:
        f = st.file_uploader("Ficheiro CSV ou Excel com os dados novos", type=["csv", "txt", "tsv", "xlsx", "xls"], key=f"bn_pv_up_{esc}")
        if f is not None:
            try:
                novo = bc.ler_tabela(f.name, f.getvalue())
            except ValueError as e:
                erro_box(str(e))
            else:
                _aplicar_e_mostrar(meta, modelo, novo, f.name.rsplit(".", 1)[0], tipo, f"bn_pv_f_{esc}")
    with aba_guardados:
        dados = db.bn_datasets(m["org_id"])
        if not dados:
            st.info("Não há dados guardados.")
        else:
            ids_d = [d["id"] for d in dados]
            nomes_d = {d["id"]: d["nome"] for d in dados}
            did = st.selectbox("Conjunto de dados", ids_d, format_func=lambda i: nomes_d[i], key=f"bn_pv_dados_{esc}")
            if st.button("Aplicar o modelo", key=f"bn_pv_aplicar_{esc}"):
                st.session_state[f"bn_pv_g_{esc}"] = did
            if st.session_state.get(f"bn_pv_g_{esc}") == did:
                novo = _carregar_tabela(did)
                if novo is not None:
                    _aplicar_e_mostrar(meta, modelo, novo, nomes_d[did], tipo, f"bn_pv_g_{esc}_out")
    with aba_caso:
        _caso_a_mao(meta, modelo, esc, tipo)


def _aplicar_e_mostrar(meta, modelo, novo: pd.DataFrame, nome: str, tipo: str, chave: str):
    em_falta = bc.colunas_em_falta(meta, novo)
    if len(em_falta) == len(meta["colunas_entrada"]):
        erro_box(bc.T("Este ficheiro não tem nenhuma das colunas do modelo: {}.", ", ".join(meta["colunas_entrada"][:8])))
        return
    if em_falta:
        aviso_box(bc.T("Faltam estas colunas e vão ficar vazias (a previsão pode piorar): {}.", ", ".join(em_falta)))
    try:
        saida = bc.aplicar_modelo(meta, modelo, novo)
    except Exception as e:
        erro_box(bc.T("Não foi possível aplicar o modelo: {}", str(e)[:200]))
        return
    st.markdown(bc.T("Previsões para **{}** linhas.", len(saida)))
    if tipo == "classificacao" and "risco" in saida.columns:
        cont = saida["risco"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Risco alto", int(cont.get("alto", 0)))
        c2.metric("Risco médio", int(cont.get("médio", 0)))
        c3.metric("Risco baixo", int(cont.get("baixo", 0)))
        ordem = [c for c in saida.columns if c.startswith("probabilidade_")][0]
        saida = saida.sort_values(ordem, ascending=False)
    elif tipo == "regressao":
        st.metric("Soma das previsões", bc._num(saida["previsao"].sum()))
    elif tipo == "anomalia":
        st.metric("Operações suspeitas", int((saida["suspeito"] == "sim").sum()))
        saida = saida.sort_values("pontuacao_anomalia", ascending=False)
    elif tipo == "segmentacao":
        st.dataframe(saida["segmento"].value_counts().sort_index().rename(tr("Clientes")), use_container_width=True)
    st.dataframe(saida.head(500), use_container_width=True, height=320)
    c1, c2 = st.columns(2)
    c1.download_button("Descarregar CSV", saida.to_csv(index=False).encode("utf-8-sig"), file_name=f"previsoes_{nome}.csv",
                       mime="text/csv", use_container_width=True, key=f"{chave}_csv")
    buf = io.BytesIO()
    try:
        saida.to_excel(buf, index=False, sheet_name="Previsoes")
        c2.download_button("Descarregar Excel", buf.getvalue(), file_name=f"previsoes_{nome}.xlsx", use_container_width=True,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{chave}_xlsx")
    except Exception:
        pass


def _caso_a_mao(meta, modelo, esc: str, tipo: str):
    st.caption(tr("Preenche os valores e vê a previsão para esse caso. Útil para testar «e se...?»."))
    valores = {}
    cols = st.columns(2)
    for i, c in enumerate(meta["colunas_entrada"]):
        col = cols[i % 2]
        t = meta["tipos"].get(c)
        ref = meta["referencia"].get(c, {})
        chave = f"bn_caso_{esc}_{c}"
        if t == "numero":
            inteiro = ref.get("inteiro")
            val = float(ref.get("mediana", 0.0))
            valores[c] = col.number_input(c, value=int(val) if inteiro else val, step=1 if inteiro else None, key=chave,
                                          help=bc.T("Valores vistos nos dados: de {} a {}", bc._num(ref.get("min", 0)), bc._num(ref.get("max", 0))))
        elif t == "data":
            try:
                padrao = datetime.date.fromisoformat(ref.get("max") or str(datetime.date.today()))
            except ValueError:
                padrao = datetime.date.today()
            valores[c] = str(col.date_input(c, value=padrao, key=chave))
        else:
            opcoes = ref.get("valores") or [""]
            valores[c] = col.selectbox(c, opcoes, key=chave)
    if st.button("Calcular previsão", type="primary", key=f"bn_caso_btn_{esc}"):
        try:
            r = bc.prever_um_caso(meta, modelo, valores)
        except Exception as e:
            erro_box(bc.T("Não foi possível calcular: {}", str(e)[:200]))
            return
        if tipo == "regressao":
            sucesso_box(bc.T("Previsão: **{}**", bc._num(r["previsao"])))
        elif tipo == "classificacao":
            pcol = [k for k in r if str(k).startswith("probabilidade_")]
            if pcol:
                p = float(r[pcol[0]])
                sucesso_box(bc.T("Probabilidade de «{}»: **{}** · risco **{}**", meta["positiva"], f"{p:.0%}", tr(bc.nivel_risco(p))))
            else:
                sucesso_box(bc.T("Previsão: **{}** (confiança {})", r["previsao"], f"{float(r.get('confianca', 0)):.0%}"))
        elif tipo == "anomalia":
            if r["suspeito"] == "sim":
                aviso_box(bc.T("Este caso é invulgar (pontuação {}): merece ser investigado.", r["pontuacao_anomalia"]))
            else:
                sucesso_box(bc.T("Este caso parece normal (pontuação {}).", r["pontuacao_anomalia"]))
        else:
            sucesso_box(bc.T("Este caso pertence ao segmento **{}**.", int(r["segmento"])))


# ══════════════════════════════════════════════════════
# RELATÓRIOS
# ══════════════════════════════════════════════════════
def render_relatorios(username: str, name: str):
    m = empresa_activa(username)
    if not m:
        _sem_empresa()
        return
    page_header("Relatórios", "Todas as análises da empresa, prontas a partilhar com a gestão")
    analises = db.bn_analises(m["org_id"])
    if not analises:
        st.info("Ainda não há análises. Faz a primeira em «Análises».")
        return
    f1, f2 = st.columns(2)
    probs = ["todos"] + list(bc.PROBLEMAS)
    filtro = f1.selectbox("Problema", probs, format_func=lambda k: tr("Todos") if k == "todos" else tr(bc.PROBLEMAS[k]["titulo"]), key="bn_rel_filtro")
    so_minhas = f2.checkbox("Só as minhas", key="bn_rel_minhas")
    lista = [a for a in analises if (filtro == "todos" or a["problema"] == filtro) and (not so_minhas or a["criado_por"] == username)]
    if not lista:
        st.info("Nenhuma análise com este filtro.")
        return
    st.dataframe(pd.DataFrame([{"Título": a["titulo"], "Problema": tr(bc.PROBLEMAS.get(a["problema"], {}).get("titulo", "")),
                                "Dados": a.get("dataset_nome", ""), "Autor": a["criado_por"], "Data": _dt_curta(a["created_at"])}
                               for a in lista]), use_container_width=True, hide_index=True)
    ids = [a["id"] for a in lista]
    pre = st.session_state.get("bn_analise_rel")
    esc = st.selectbox("Abrir análise", ids, index=ids.index(pre) if pre in ids else 0,
                       format_func=lambda i: next(a["titulo"] for a in lista if a["id"] == i), key="bn_rel_abrir")
    a = db.bn_get_analise(esc)
    if not a:
        return
    mostrar_resultado(a["resumo"], a["problema"], chave=f"bn_rel_{esc}")
    section_title("Comentário para a gestão")
    texto_ia = (a["resumo"] or {}).get("texto_ia")
    if texto_ia:
        st.markdown(texto_ia["texto"])
        st.caption(bc.T("Escrito pela IA ({}) em {}. Confirma antes de decidir.", texto_ia.get("modelo", ""), texto_ia.get("em", "")))
    if not ia.configurada():
        st.caption(tr("A IA está desligada. Quem administra a aplicação pode ligá-la com a chave «ia_api_key» nos segredos."))
    elif pode(m, "treinar"):
        if st.button("Escrever outra vez com IA" if texto_ia else "Escrever comentário com IA", key=f"bn_rel_ia_{esc}"):
            with st.spinner(tr("A escrever o comentário...")):
                try:
                    texto = ia.escrever_comentario(a, m["nome"], m["org_id"], idioma_actual())
                except ia.IAErro as e:
                    erro_box(tr(str(e)))
                else:
                    novo = dict(a["resumo"] or {})
                    novo["texto_ia"] = {"texto": texto, "modelo": ia.modelo(), "em": str(datetime.date.today())}
                    db.bn_actualizar_analise(esc, novo)
                    st.rerun()
        st.caption(bc.T("Pedidos à IA que restam hoje: {} · o comentário usa só os números da análise, nunca as linhas dos dados.",
                        ia.pedidos_restantes(m["org_id"])))

    section_title("Exportar")
    autor = (db.get_user(a["criado_por"]) or {}).get("name") or a["criado_por"]
    try:
        pdf = bc.relatorio_pdf(m["nome"], a, autor)
        st.download_button("Descarregar relatório em PDF", pdf, file_name=f"relatorio_{a['id']}.pdf", mime="application/pdf",
                           type="primary", key=f"bn_rel_pdf_{esc}")
    except Exception as e:
        erro_box(bc.T("Não foi possível gerar o PDF: {}", str(e)[:150]))
    if a.get("modelo_guardado"):
        if st.button("Usar este modelo em Previsões", key=f"bn_rel_prev_{esc}"):
            st.session_state["bn_analise_prev"] = esc
            ir_para("business_previsoes")
    if m["papel"] == "gestor" or a["criado_por"] == username:
        if st.button("Apagar análise", key=f"bn_rel_del_{esc}"):
            st.session_state["bn_rel_confirmar"] = esc
        if st.session_state.get("bn_rel_confirmar") == esc:
            aviso_box("Esta análise e o seu modelo serão apagados. Confirmas?")
            d1, d2 = st.columns(2)
            if d1.button("Sim, apagar", type="primary", key=f"bn_rel_sim_{esc}"):
                db.bn_apagar_analise(esc)
                st.session_state.pop("bn_rel_confirmar", None)
                st.session_state.pop("bn_analise_rel", None)
                st.session_state.get("_bn_cache_modelos", {}).pop(esc, None)
                st.rerun()
            if d2.button("Cancelar", key=f"bn_rel_nao_{esc}"):
                st.session_state.pop("bn_rel_confirmar", None)
                st.rerun()


# ══════════════════════════════════════════════════════
# EQUIPA
# ══════════════════════════════════════════════════════
def render_equipa(username: str, name: str):
    m = empresa_activa(username)
    if not m:
        _sem_empresa()
        return
    org = m["org_id"]
    page_header("Equipa", m["nome"])
    membros = db.list_membros(org_id=org)
    analises = db.bn_analises(org)
    por_autor = {}
    for a in analises:
        por_autor[a["criado_por"]] = por_autor.get(a["criado_por"], 0) + 1
    gestores = [x for x in membros if x["papel"] == "gestor"]

    linhas = []
    for x in sorted(membros, key=lambda r: (r["papel"] != "gestor", r["username"])):
        u = db.get_user(x["username"]) or {}
        linhas.append({"Nome": u.get("name", x["username"]), "Utilizador": x["username"], "Papel": tr(PAPEIS.get(x["papel"], x["papel"])),
                       "Análises": por_autor.get(x["username"], 0), "Desde": _dt_curta(x.get("created_at"))})
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

    section_title("O que cada papel pode fazer")
    st.markdown("- **" + tr("Gestor") + "**: " + tr("tudo, incluindo gerir a equipa e apagar dados e análises.") + "\n"
                "- **" + tr("Analista") + "**: " + tr("carrega dados, treina modelos e gera relatórios.") + "\n"
                "- **" + tr("Membro") + "**: " + tr("consulta análises e relatórios e usa os modelos em Previsões."))

    if pode(m, "equipa"):
        section_title("Convidar pessoas")
        sucesso_box(bc.T("Código de convite: **{}**. Quem o escrever no Perfil («Juntar-me a uma empresa») entra como membro.", m["codigo_convite"]))
        if st.button("Gerar um código novo", key="bn_novo_codigo"):
            from modules.acesso import codigo_novo
            db.update_org(org, {"codigo_convite": codigo_novo()})
            st.session_state.pop("_membros_cache", None)
            st.rerun()
        st.caption(tr("Gerar um código novo deixa o anterior de ser válido. Quem já está na equipa não é afectado."))

        section_title("Gerir a equipa")
        outros = [x["username"] for x in membros if x["username"] != username]
        if not outros:
            st.info("Ainda és a única pessoa da equipa. Partilha o código de convite.")
        else:
            alvo = st.selectbox("Pessoa", outros, key="bn_eq_pessoa",
                                format_func=lambda u: f"{(db.get_user(u) or {}).get('name', u)} ({u})")
            atual = next(x["papel"] for x in membros if x["username"] == alvo)
            novo = st.selectbox("Papel", list(PAPEIS), index=list(PAPEIS).index(atual) if atual in PAPEIS else 2,
                                format_func=lambda p: tr(PAPEIS[p]), key=f"bn_eq_papel_{alvo}")
            c1, c2 = st.columns(2)
            if c1.button("Guardar papel", key="bn_eq_guardar", use_container_width=True) and novo != atual:
                db.add_membro(org, alvo, novo)
                st.session_state.pop("_membros_cache", None)
                st.success(bc.T("{} agora é {}.", alvo, tr(PAPEIS[novo])))
                st.rerun()
            if c2.button("Remover da empresa", key="bn_eq_remover", use_container_width=True):
                st.session_state["bn_eq_confirmar"] = alvo
            if st.session_state.get("bn_eq_confirmar") == alvo:
                aviso_box(bc.T("Remover {} da empresa? Perde o acesso aos dados e às análises.", alvo))
                d1, d2 = st.columns(2)
                if d1.button("Sim, remover", type="primary", key="bn_eq_sim"):
                    db.remove_membro(org, alvo)
                    st.session_state.pop("bn_eq_confirmar", None)
                    st.rerun()
                if d2.button("Cancelar", key="bn_eq_nao"):
                    st.session_state.pop("bn_eq_confirmar", None)
                    st.rerun()
    else:
        if st.button("Sair desta empresa", key="bn_eq_sair"):
            st.session_state["bn_eq_sair_conf"] = True
        if st.session_state.get("bn_eq_sair_conf"):
            aviso_box("Vais perder o acesso aos dados e às análises desta empresa. Confirmas?")
            d1, d2 = st.columns(2)
            if d1.button("Sim, sair", type="primary", key="bn_eq_sair_sim"):
                db.remove_membro(org, username)
                st.session_state.pop("bn_eq_sair_conf", None)
                st.session_state.pop("_membros_cache", None)
                st.session_state.pop("org_activa", None)
                st.rerun()
            if d2.button("Cancelar", key="bn_eq_sair_nao"):
                st.session_state.pop("bn_eq_sair_conf", None)
                st.rerun()
    if len(gestores) == 1 and gestores[0]["username"] == username and len(membros) > 1:
        st.caption(tr("És o único gestor. Promove outra pessoa a gestor antes de deixares a empresa."))


# ══════════════════════════════════════════════════════
# PERFIL (sem turma, ranking nem nada de aluno)
# ══════════════════════════════════════════════════════
def render_perfil(username: str, name: str):
    import bcrypt
    from modules.auth import get_authenticator
    from modules.utils import render_seletor_tema_perfil
    page_header("O Meu Perfil", "A tua conta, a empresa e as preferências")
    registo = db.get_user(username)
    m = empresa_activa(username)
    st.markdown(f"**{tr('Nome')}:** {name}  \n**{tr('Utilizador')}:** {username}  \n"
                f"**{tr('Email')}:** {(registo or {}).get('email') or '—'}")
    if m:
        st.markdown(f"**{tr('Empresa')}:** {m['nome']}  \n**{tr('Papel')}:** {tr(PAPEIS.get(m['papel'], 'Membro'))}")

    st.markdown("### " + tr("Preferências"))
    tam = st.select_slider("Tamanho do texto", options=["Pequeno", "Normal", "Grande", "Muito grande"],
                           value=st.session_state.get("tamanho_texto", "Normal"), key="perfil_tam")
    st.session_state["tamanho_texto"] = tam
    render_seletor_tema_perfil(username)
    if registo is None:
        st.info("As contas demo não podem ser alteradas. Cria uma conta real para gravar a palavra-passe.")
        return

    st.markdown("### " + tr("As minhas empresas"))
    for e in empresas_do(username):
        st.markdown(f"- **{e['nome']}** ({tr(PAPEIS.get(e['papel'], 'Membro'))})")
    cod = st.text_input("Juntar-me a outra empresa (código de convite)", key="bn_perfil_cod").strip().upper()
    if st.button("Juntar-me", key="bn_perfil_juntar") and cod:
        from modules.admin import adicionar_membro_por_codigo
        ok, msg = adicionar_membro_por_codigo(username, cod, "membro")
        if ok:
            st.session_state.pop("_membros_cache", None)
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.markdown("### " + tr("Mudar palavra-passe"))
    with st.form("bn_form_pw"):
        atual_pw = st.text_input("Palavra-passe actual", type="password")
        nova = st.text_input("Nova palavra-passe (mínimo 6)", type="password")
        nova2 = st.text_input("Repete a nova", type="password")
        ok = st.form_submit_button("Mudar palavra-passe")
    if ok:
        if not bcrypt.checkpw(atual_pw.encode(), registo["password_hash"].encode()):
            st.error("A palavra-passe actual está errada.")
        elif len(nova) < 6 or nova != nova2:
            st.error("A nova palavra-passe deve ter 6+ caracteres e ser igual nas duas caixas.")
        else:
            from modules.acesso import hash_pw
            db.update_user(username, {"password_hash": hash_pw(nova)})
            get_authenticator(force=True)
            st.success("Palavra-passe alterada.")
