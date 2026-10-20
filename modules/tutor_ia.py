"""
Tutor de IA para o aluno: «Perguntar à IA sobre isto».

Aparece em três sítios:
- nas lições da Escola de ML (a dúvida sobre o que acabou de ler);
- nos resultados dos mini-testes (porque errei? o que quer dizer a pergunta?);
- depois de cada treino (o que significam estes números? como melhoro?).

O tutor só recebe texto das lições e NÚMEROS dos resultados, nunca dados pessoais.
Só aparece se a IA estiver ligada (chave nos segredos).

Para poupar o limite da chave: nas lições e nos testes as respostas ficam em cache e são partilhadas entre alunos;
depois de um treino as explicações padrão dos números (as que já estão escritas no sistema) aparecem primeiro,
sem IA, e a IA só é chamada quando o aluno pergunta mais.
"""
import numbers
import re

import html
import streamlit as st

from modules import ia
from modules.i18n import idioma_actual, tr

# Chaves da sessão onde cada módulo de treino guarda o último resultado
RESULTADOS_POR_PAGINA = {
    "primeiro_modelo": ["pm_res"],
    "clustering": ["cl_result", "cl_livre_res"],
    "reducao": ["rd_result"],
    "anomalias": ["an_result"],
    "mlp": ["mlp_clf_res", "mlp_reg_res"],
    "visao": ["vis_result"],
    "nlp": ["nlp_bow_res", "nlp_lstm_res"],
    "series": ["st_result", "arima_result"],
    "generativo": ["ae_result", "vae_result", "gan_result"],
    "reforco": ["dqn_result", "ql_result"],
    "comparador": ["cmp_result"],
    "classificacao": [],        # estes dois registam o contexto em explicacoes.py
    "regressao": [],
    "projeto": [],
}

# Que métrica do glossário (explicacoes.AJUDA_METRICAS) aparece num texto de resultado: nome -> padrão
_METRICAS_NO_TEXTO = {
    "Accuracy": r"accuracy|\bacc\b|taxa de acerto",
    "Precision": r"precision|\bprec\b",
    "Recall": r"recall|\brec\b",
    "F1-Score": r"\bf1\b",
    "ROC-AUC": r"roc-auc|\bauc\b",
    "R²": r"r²|\br2\b",
    "RMSE": r"rmse",
    "MAE": r"\bmae\b",
    "MSE": r"\bmse\b",
    "Média CV": r"valida[çc][ãa]o cruzada|\bcv\b|média cv",
    "Desvio Padrão": r"variação ±|desvio",
}
EXPLICA_NUMEROS = "Explica estes números"


def explicacoes_padrao(contexto: str) -> list:
    """Explicações já escritas no sistema para as métricas que aparecem no resultado: [(nome, texto)]. Não usam IA."""
    from modules.explicacoes import AJUDA_METRICAS
    t = (contexto or "").lower()
    return [(nome, AJUDA_METRICAS[nome]) for nome, padrao in _METRICAS_NO_TEXTO.items()
            if nome in AJUDA_METRICAS and re.search(padrao, t)]


SUGESTOES_LICAO = ["Explica de outra forma", "Dá-me um exemplo do dia-a-dia", "Porque é que isto é importante?"]
SUGESTOES_TESTE = ["Porque é que esta é a resposta certa?", "Explica-me o conceito desta pergunta"]
SUGESTOES_TREINO = ["Explica estes números", "O resultado é bom ou mau?", "Como posso melhorar?"]


# ══════════════════════════════════════════════════════
# CONTEXTO: o que a IA pode ver
# ══════════════════════════════════════════════════════
def _escalar(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Number):
        try:
            return f"{float(v):.4g}"
        except (TypeError, ValueError):
            return None
    if isinstance(v, str) and len(v) <= 80:
        return v
    return None


def resumir_resultado(res, limite: int = 1500) -> str:
    """Transforma o resultado de um treino em texto curto: só números e textos curtos (sem tabelas, figuras nem modelos)."""
    if not isinstance(res, dict):
        return ""
    linhas = []
    for k, v in res.items():
        if str(k).startswith("_"):
            continue
        e = _escalar(v)
        if e is not None:
            linhas.append(f"{k}: {e}")
            continue
        if not (isinstance(v, (list, tuple)) or (type(v).__name__ == "ndarray" and getattr(v, "ndim", 0) == 1)):
            continue                                      # tabelas, figuras, modelos... não seguem para a IA
        try:                                              # listas ou arrays pequenos de números
            lista = list(v)
            if 0 < len(lista) <= 12:
                itens = [_escalar(x) for x in lista]
                if all(i is not None for i in itens):
                    linhas.append(f"{k}: [{', '.join(itens)}]")
                    continue
            if len(lista) > 12:
                nums = [float(x) for x in lista if isinstance(x, numbers.Number)]
                if len(nums) == len(lista):
                    linhas.append(f"{k}: {len(nums)} valores, de {min(nums):.4g} a {max(nums):.4g}, média {sum(nums) / len(nums):.4g}")
        except (TypeError, ValueError):
            pass
    return "\n".join(linhas)[:limite]


def registar(titulo: str, texto: str):
    """Um módulo diz ao tutor o que o aluno está a ver (resultado do treino actual)."""
    st.session_state["_ia_ctx"] = {"pagina": st.session_state.get("pagina"), "titulo": titulo, "texto": texto}


def _contexto_da_pagina(pagina: str):
    ctx = st.session_state.get("_ia_ctx")
    if ctx and ctx.get("pagina") == pagina and ctx.get("texto"):
        return ctx["titulo"], ctx["texto"]
    for chave in RESULTADOS_POR_PAGINA.get(pagina, []):
        texto = resumir_resultado(st.session_state.get(chave))
        if texto:
            return f"Resultado do último treino ({chave})", texto
    return None


# ══════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════
def render_tutor(username: str, chave: str, titulo: str, contexto: str, sugestoes: list = None,
                 rotulo: str = "Perguntar à IA sobre isto", nao_revelar: bool = False, aberto: bool = False,
                 usar_cache: bool = False, padrao_primeiro: bool = False, estado: str = "", compacto: bool = False):
    """Caixa de perguntas ao tutor. `chave` tem de ser única por sítio (lição, teste, página).
    `usar_cache`: lições e testes (respostas partilhadas, sem gastar limite). `padrao_primeiro`: resultados de treino,
    mostra primeiro as explicações padrão dos números (sem IA) e tira o botão «Explica estes números»."""
    if not ia.configurada():
        return
    hist_key = f"_tutor_h_{chave}"
    historico = st.session_state.setdefault(hist_key, [])
    with (st.container() if compacto else st.expander(tr(rotulo), expanded=aberto)):
        st.caption(tr("Pergunta o que não percebeste. A IA responde com base no que estás a ver e pode enganar-se: confirma com a lição ou com o professor."))
        if padrao_primeiro:
            padrao = explicacoes_padrao(contexto)
            if padrao:
                st.markdown(f"**{tr('O que significam estes números (sem IA)')}**")
                for nome, texto_padrao in padrao:
                    st.markdown(f"- **{nome}:** {tr(texto_padrao)}")
                st.caption(tr("Ainda tens dúvidas? Pergunta abaixo: a IA só é usada quando perguntas."))
                sugestoes = [x for x in (sugestoes or []) if x != EXPLICA_NUMEROS]
        pergunta = None
        if sugestoes:
            cols = st.columns(len(sugestoes))
            for i, (col, s) in enumerate(zip(cols, sugestoes)):
                if col.button(s, key=f"_tutor_s_{chave}_{i}", use_container_width=True):
                    pergunta = tr(s)
        with st.form(f"_tutor_f_{chave}", clear_on_submit=True):
            texto = st.text_input("A tua dúvida", max_chars=400, key=f"_tutor_t_{chave}",
                                  placeholder=tr("ex.: não percebi o que é overfitting"))
            enviar = st.form_submit_button("Perguntar")
        if enviar and texto.strip():
            pergunta = texto.strip()
        if pergunta:
            with st.spinner(tr("A pensar...")):
                try:
                    resposta = ia.perguntar_tutor(pergunta, titulo, contexto, historico, username, idioma_actual(), nao_revelar,
                                                  usar_cache=usar_cache, estado=estado)
                except ia.IAErro as e:
                    st.error(tr(str(e)))
                else:
                    historico.append((pergunta, resposta))
                    del historico[:-6]
        for p, r in reversed(historico[-3:]):
            st.markdown(f"**{tr('Tu')}:** {p}")
            st.markdown(r)
            st.divider()
        st.caption(tr("Perguntas à IA que restam hoje: {}").format(ia.pedidos_restantes(ia.id_aluno(username), ia.limite_aluno())))


def render_tutor_pagina(username: str, pagina: str, rotulo_pagina: str):
    """Tutor no fim das páginas de treino: usa o último resultado que o aluno gerou. Sem resultado, não aparece."""
    ctx = _contexto_da_pagina(pagina)
    if not ctx:
        return
    titulo, texto = ctx
    render_tutor(username, f"pag_{pagina}", f"{rotulo_pagina}: {titulo}", texto, SUGESTOES_TREINO,
                 rotulo="Perguntar à IA sobre este resultado", padrao_primeiro=True)     # números do aluno: sem cache


# ══════════════════════════════════════════════════════
# PAINEL DA IA NA LATERAL DIREITA
# A página regista o que o aluno está a ver (registar_painel); o painel abre à direita, a página encolhe para lhe dar
# espaço e, ao fechar, tudo volta ao que era. O estado (aberto/fechado) fica na sessão.
# ══════════════════════════════════════════════════════
LARGURA_PAINEL = 400


def registar_painel(chave: str, titulo: str, contexto: str, sugestoes: list = None, usar_cache: bool = False,
                    estado: str = "", padrao_primeiro: bool = False):
    """Uma página diz ao painel da IA o que o aluno está a ver. Chamada em cada execução da página."""
    st.session_state["_ia_pagina"] = dict(chave=chave, titulo=titulo, contexto=contexto, sugestoes=sugestoes,
                                          usar_cache=usar_cache, estado=estado, padrao_primeiro=padrao_primeiro)


def _alternar_painel():
    st.session_state["ia_painel_aberto"] = not st.session_state.get("ia_painel_aberto", False)


def _css_painel(aberto: bool) -> str:
    L = LARGURA_PAINEL
    css = ("<style>"
           ".st-key-ia_fab { position: fixed; right: 18px; bottom: 18px; z-index: 1001; width: auto !important; }"
           ".st-key-ia_fab button { border-radius: 999px !important; box-shadow: 0 6px 20px rgba(0,0,0,.25) !important; }")
    if not aberto:
        return css + "</style>"
    css += ("[data-testid='stMainBlockContainer'], .block-container { padding-right: %dpx !important; max-width: none !important; }"
            # painel: cabeçalho em cima, mensagens ao meio (a rolar), caixa de escrever em baixo
            ".st-key-ia_painel { position: fixed; top: 0; right: 0; width: %dpx; height: 100vh; z-index: 1000; display: flex;"
            " flex-direction: column; gap: 0 !important; padding: 0 !important; overflow: hidden; background: var(--df-bg);"
            " border-left: 1px solid var(--df-border); box-shadow: -6px 0 24px rgba(0,0,0,.12); }"
            ".st-key-ia_head { flex: 0 0 auto; padding: .7rem 1rem; border-bottom: 1px solid var(--df-border); background: var(--df-surface); }"
            ".st-key-ia_head button, .st-key-ia_head button:hover, .st-key-ia_head [data-testid^='stBaseButton'] { background: transparent !important;"
            " border: none !important; box-shadow: none !important; min-height: 32px !important; padding: 0 .5rem !important; transform: none !important; }"
            ".st-key-ia_head button * { color: var(--df-text-sec) !important; font-size: 18px !important; }"
            # os blocos do painel ficam dentro de invólucros do Streamlit: o das mensagens é o que cresce
            ".st-key-ia_painel > div { min-height: 0; }"
            ".st-key-ia_painel > div:has(.st-key-ia_msgs) { flex: 1 1 0 !important; position: relative; overflow: hidden; }"
            ".st-key-ia_painel > div:has(.st-key-ia_head), .st-key-ia_painel > div:has(.st-key-ia_comp) { flex: 0 0 auto !important; }"
            "[data-testid='InputInstructions'] { display: none !important; }"
            # mensagens: column-reverse faz a conversa ficar encostada em baixo e a mais recente sempre visível
            ".st-key-ia_msgs { position: absolute !important; inset: 0; overflow-y: scroll !important; overscroll-behavior: contain; scrollbar-width: thin; display: flex; flex-direction: column-reverse;"
            " gap: .7rem !important; padding: 1rem; background: var(--df-bg); }"
            "[class*='st-key-ia_u_'], [class*='st-key-ia_a_'] { width: fit-content !important; max-width: 88%%; padding: .6rem .9rem;"
            " border-radius: 16px; font-size: 14.5px; line-height: 1.5; gap: .3rem !important; }"
            "[class*='st-key-ia_u_'] { margin-left: auto; background: var(--df-accent); border-bottom-right-radius: 4px; }"
            "[class*='st-key-ia_u_'] * { color: #FFFFFF !important; }"
            "[class*='st-key-ia_a_'] { margin-right: auto; background: var(--df-surface2); border: 1px solid var(--df-border);"
            " border-bottom-left-radius: 4px; }"
            "[class*='st-key-ia_a_'] p, [class*='st-key-ia_a_'] li { margin: 0 0 .4rem; }"
            ".st-key-ia_comp { flex: 0 0 auto; padding: .6rem 1rem .5rem; border-top: 1px solid var(--df-border); background: var(--df-surface); }"
            ".st-key-ia_comp [data-testid='stForm'] { border: none !important; padding: 0 !important; }"
            ".st-key-ia_comp input { border-radius: 999px !important; padding-left: 1rem !important; }"
            ".st-key-ia_comp [data-testid='stFormSubmitButton'] button { border-radius: 50%% !important; min-width: 42px !important;"
            " width: 42px !important; padding: 0 !important; }"
            ".st-key-ia_chips { gap: .35rem !important; margin-bottom: .4rem; }"
            ".st-key-ia_chips .stButton > button, .st-key-ia_chips .stButton > button:hover { min-height: 34px !important; font-size: 13px !important;"
            " font-weight: 500 !important; border-radius: 999px !important; padding: .25rem .9rem !important; background: transparent !important;"
            " border: 1px solid var(--df-accent) !important; box-shadow: none !important; transform: none !important; }"
            ".st-key-ia_chips .stButton > button:not([kind='secondary']) * { color: var(--df-accent) !important; }"
            "@media (max-width: 900px) { [data-testid='stMainBlockContainer'], .block-container { padding-right: 1rem !important; }"
            " .st-key-ia_painel { width: 100vw; } }") % (L + 24, L)
    return css + "</style>"


def _bolha(papel: str, chave: str, texto: str, recipiente=None):
    """Uma mensagem da conversa. papel: 'u' (aluno) ou 'a' (tutor)."""
    alvo = recipiente if recipiente is not None else st
    with alvo.container(key=f"ia_{papel}_{chave}"):
        st.markdown(texto)


def render_painel_ia(username: str, pagina: str, rotulo_pagina: str):
    """Botão flutuante «Tutor IA» e, quando aberto, o painel de conversa à direita. Chamar depois de desenhar a página."""
    if not ia.configurada():
        return
    aberto = bool(st.session_state.get("ia_painel_aberto", False))
    st.markdown(_css_painel(aberto), unsafe_allow_html=True)
    if not aberto:
        with st.container(key="ia_fab"):
            st.button(tr("Tutor IA"), key="ia_abrir", on_click=_alternar_painel, icon=":material/smart_toy:")
        return

    ctx = st.session_state.get("_ia_pagina")
    if not ctx:
        ctx = dict(chave=f"geral_{pagina}", titulo=rotulo_pagina, contexto=f"O aluno está na página: {rotulo_pagina}.",
                   sugestoes=["Explica-me o que posso fazer aqui", "Por onde devo começar?"], usar_cache=False,
                   estado="", padrao_primeiro=False)
    chave = ctx["chave"]
    historico = st.session_state.setdefault(f"_tutor_h_{chave}", [])
    sugestoes = [x for x in (ctx["sugestoes"] or []) if not (ctx["padrao_primeiro"] and x == EXPLICA_NUMEROS)]

    with st.container(key="ia_painel"):
        with st.container(key="ia_head"):
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(
                '<div style="display:flex;align-items:center;gap:10px;">'
                '<div style="width:36px;height:36px;border-radius:50%;background:var(--df-accent);color:#fff;display:flex;'
                'align-items:center;justify-content:center;font-weight:800;font-size:13px;">IA</div>'
                f'<div><div style="font-weight:700;font-size:15px;line-height:1.2;">{tr("Tutor IA")}</div>'
                f'<div style="font-size:12px;color:var(--df-text-sec);line-height:1.3;">{html.escape(tr(ctx["titulo"])[:60])}</div></div></div>',
                unsafe_allow_html=True)
            c2.button("✕", key="ia_fechar_b", on_click=_alternar_painel)
        msgs = st.container(key="ia_msgs")
        comp = st.container(key="ia_comp")

    pergunta = None
    with comp:
        if not historico and sugestoes:
            with st.container(key="ia_chips"):
                for i, sug in enumerate(sugestoes):
                    if st.button(sug, key=f"ia_chip_{chave}_{i}", use_container_width=True):
                        pergunta = tr(sug)
        with st.form(f"ia_form_{chave}", clear_on_submit=True, border=False):
            ca, cb = st.columns([5, 1], vertical_alignment="center")
            texto = ca.text_input("A tua dúvida", max_chars=400, label_visibility="collapsed",
                                  placeholder=tr("Escreve a tua dúvida..."))
            enviar = cb.form_submit_button("Enviar")
        if enviar and texto.strip():
            pergunta = texto.strip()
        st.caption(tr("Perguntas à IA que restam hoje: {}").format(ia.pedidos_restantes(ia.id_aluno(username), ia.limite_aluno())))

    with msgs:                      # a ordem é da mais recente para a mais antiga (column-reverse põe a mais recente em baixo)
        nova = st.empty() if pergunta else None
        if pergunta:
            _bolha("u", "novo", pergunta)
        for i in range(len(historico) - 1, -1, -1):
            p, r = historico[i]
            _bolha("a", str(i), r)
            _bolha("u", str(i), p)
        with st.container(key="ia_a_boasvindas"):
            st.markdown(f"{tr('Olá! Sou o teu tutor de Machine Learning.')} {tr('Estou a ver')}: **{tr(ctx['titulo'])}**. "
                        f"{tr('Pergunta o que não percebeste. A IA pode enganar-se: confirma com a lição ou com o professor.')}")
            if ctx["padrao_primeiro"]:
                padrao = explicacoes_padrao(ctx["contexto"])
                if padrao:
                    st.markdown(f"**{tr('O que significam estes números (sem IA)')}**")
                    for nome, texto_padrao in padrao:
                        st.markdown(f"- **{nome}:** {tr(texto_padrao)}")
        if pergunta:
            _bolha("a", "pensa", "_" + tr("A pensar...") + "_", nova)
            try:
                resposta = ia.perguntar_tutor(pergunta, ctx["titulo"], ctx["contexto"], historico, username, idioma_actual(),
                                              False, usar_cache=ctx["usar_cache"], estado=ctx["estado"])
            except ia.IAErro as e:
                nova.empty()
                _bolha("a", "erro", "Aviso: " + tr(str(e)), nova)
            else:
                historico.append((pergunta, resposta))
                del historico[:-6]
                nova.empty()
                _bolha("a", "novo", resposta, nova)
