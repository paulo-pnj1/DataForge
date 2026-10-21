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
import hashlib
import json
import numbers
import re
import threading
import time

import html
import streamlit as st

from modules import db, ia
from modules.settings import usando_supabase
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
                if col.button(s, key=f"_tutor_s_{chave}_{i}", width="stretch"):
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
            ".st-key-ia_painel { position: fixed; top: 0; right: 0; width: %dpx; height: 100vh; height: 100dvh; z-index: 1000; display: flex;"
            " flex-direction: column; gap: 0 !important; padding: 0 !important; overflow: hidden; background: var(--df-bg);"
            " border-left: 1px solid var(--df-border); box-shadow: -6px 0 24px rgba(0,0,0,.12); }"
            ".st-key-ia_head { flex: 0 0 auto; padding: .7rem 1rem; border-bottom: 1px solid var(--df-border); background: var(--df-surface); }"
            ".st-key-ia_head button, .st-key-ia_head button:hover, .st-key-ia_head [data-testid^='stBaseButton'] { background: transparent !important;"
            " border: none !important; box-shadow: none !important; min-height: 32px !important; padding: 0 .5rem !important; transform: none !important; }"
            ".st-key-ia_head button * { color: var(--df-text-sec) !important; font-size: 18px !important; }"
            # os blocos do painel ficam dentro de invólucros do Streamlit: o das mensagens é o que cresce
            ".st-key-ia_painel > div { min-height: 0; }"
            ".st-key-ia_painel > div:has(.st-key-ia_msgs), .st-key-ia_painel > div:has(.st-key-ia_hist) { flex: 1 1 0 !important; position: relative; overflow: hidden; }"
            ".st-key-ia_painel > div:has(.st-key-ia_head), .st-key-ia_painel > div:has(.st-key-ia_comp) { flex: 0 0 auto !important; }"
            "[data-testid='InputInstructions'] { display: none !important; }"
            # mensagens: coluna normal que rola (o scroll para a mensagem nova é feito por um pequeno script, _scroll_js).
            # flex-shrink:0 nos filhos: sem isto o flexbox ENCOLHE e corta as mensagens em vez de as deixar transbordar.
            ".st-key-ia_msgs, .st-key-ia_hist { position: absolute !important; inset: 0; overflow-y: auto !important; overflow-x: hidden !important;"
            " overscroll-behavior: contain; -webkit-overflow-scrolling: touch; scrollbar-width: auto; scrollbar-color: var(--df-accent) var(--df-surface2);"
            " display: flex !important; flex-direction: column !important; gap: .7rem !important; padding: 1rem; background: var(--df-bg); }"
            ".st-key-ia_msgs > *, .st-key-ia_hist > * { flex: 0 0 auto !important; min-height: min-content !important; }"
            # em ecrãs estreitos o Streamlit empilha as colunas: aqui o cabeçalho e a caixa de escrever ficam sempre numa linha
            ".st-key-ia_painel [data-testid='stHorizontalBlock'] { flex-wrap: nowrap !important; flex-direction: row !important; }"
            ".st-key-ia_js { position: fixed !important; width: 0 !important; height: 0 !important; overflow: hidden !important; }"
            ".st-key-ia_msgs::-webkit-scrollbar, .st-key-ia_hist::-webkit-scrollbar { width: 12px; }"
            ".st-key-ia_msgs::-webkit-scrollbar-track, .st-key-ia_hist::-webkit-scrollbar-track { background: var(--df-surface2); border-left: 1px solid var(--df-border); }"
            ".st-key-ia_msgs::-webkit-scrollbar-thumb, .st-key-ia_hist::-webkit-scrollbar-thumb { background: var(--df-accent); border-radius: 8px; border: 3px solid var(--df-surface2); min-height: 40px; }"
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
            " .st-key-ia_painel { width: 100vw; } }"
            ".st-key-ia_painel [data-testid='stColumn'] { min-width: 0 !important; }") % (L + 24, L)
    return css + "</style>"


def _bolha(papel: str, chave: str, texto: str, recipiente=None):
    """Uma mensagem da conversa. papel: 'u' (aluno) ou 'a' (tutor)."""
    alvo = recipiente if recipiente is not None else st
    with alvo.container(key=f"ia_{papel}_{chave}"):
        st.markdown(texto)


# ── Histórico das conversas: guardado na conta (tabela progress, chave «chat:<utilizador>») ──
_MAX_TROCAS = 16        # trocas guardadas por conversa
_MAX_CONVERSAS = 10     # conversas guardadas por aluno (as mais antigas saem)


def _doc_chat(username: str) -> dict:
    """Conversas do aluno: {'c': {chave: {'titulo', 't', 'm': [[pergunta, resposta], ...]}}}. Lê da base de dados 1 vez por sessão."""
    d = st.session_state.get("_chat_doc")
    if d is None or d.get("_u") != username:
        try:
            bruto = (db.get_progress(f"chat:{username}") or {}).get("c", {})
        except Exception:
            bruto = {}
        d = {"_u": username, "c": bruto if isinstance(bruto, dict) else {}}
        st.session_state["_chat_doc"] = d
    return d


def _guardar_chat(username: str, doc: dict):
    """Grava as conversas sem fazer o aluno esperar (fio à parte quando há Supabase)."""
    c = doc["c"]
    for k in sorted(c, key=lambda k: c[k].get("t", 0))[:max(0, len(c) - _MAX_CONVERSAS)]:
        c.pop(k, None)
    copia = {"c": json.loads(json.dumps(c))}

    def _gravar():
        try:
            db.save_progress(f"chat:{username}", copia)
        except Exception:
            pass
    if usando_supabase():
        threading.Thread(target=_gravar, daemon=True).start()
    else:
        _gravar()


def _restantes(username: str) -> int:
    """Perguntas que restam hoje: guardado na sessão 2 min (em vez de 1 pedido à base de dados por interacção)."""
    c = st.session_state.get("_ia_rest")
    if c and c[0] == username and time.time() - c[1] < 120:
        return c[2]
    n = ia.pedidos_restantes(ia.id_aluno(username), ia.limite_aluno())
    st.session_state["_ia_rest"] = (username, time.time(), n)
    return n


def _gastou_uma(username: str):
    c = st.session_state.get("_ia_rest")
    if c and c[0] == username:
        st.session_state["_ia_rest"] = (username, c[1], max(0, c[2] - 1))


def _alt_historico():
    st.session_state["_ia_ver_hist"] = not st.session_state.get("_ia_ver_hist", False)


def _fechar_painel():
    st.session_state["ia_painel_aberto"] = False
    st.rerun()                      # a página inteira volta a ocupar a largura toda (fora do fragmento)


def _nova_conversa(username: str, skey: str):
    doc = _doc_chat(username)
    ent = doc["c"].pop(skey, None)
    if ent and ent.get("m"):
        doc["c"][f"{skey}#{int(time.time())}"] = ent         # a conversa antiga fica no histórico
        _guardar_chat(username, doc)
    st.session_state["_ia_ver_hist"] = False


def _scroll_js(foco: str):
    """Leva o scroll da conversa: «pergunta» = início da pergunta acabada de fazer (para ler a resposta de cima para baixo);
    «fim» = última mensagem. Corre num iframe invisível com o mesmo domínio; o carimbo de tempo força-o a correr de novo."""
    import streamlit.components.v1 as components
    components.html(
        "<script>(function(){var d=window.parent.document;function go(){var m=d.querySelector('.st-key-ia_msgs');if(!m)return;"
        "var f=%s;if(f){var u=m.querySelectorAll(\"[class*='st-key-ia_u_']\");var l=u[u.length-1];"
        "if(l){m.scrollTop+=l.getBoundingClientRect().top-m.getBoundingClientRect().top-10;return;}}"
        "m.scrollTop=m.scrollHeight;}go();setTimeout(go,80);setTimeout(go,300);})();</script><!--%s-->"
        % ("true" if foco == "pergunta" else "false", time.time()), height=0)


def _data_curta(t) -> str:
    try:
        return time.strftime("%d/%m %H:%M", time.localtime(float(t)))
    except (TypeError, ValueError):
        return ""


@st.fragment
def _painel_ia(username: str, ctx: dict):
    """O painel é um fragmento: perguntar, abrir o histórico ou começar nova conversa só volta a desenhar o painel,
    não a página inteira por trás (antes cada pergunta re-executava toda a página)."""
    chave = ctx["chave"]
    skey = f"{chave}~{hashlib.md5((ctx['contexto'] or '').encode('utf-8')).hexdigest()[:6]}"     # resultado novo = conversa nova
    doc = _doc_chat(username)
    conv = doc["c"]
    historico = (conv.get(skey) or {}).get("m", [])
    ver_hist = bool(st.session_state.get("_ia_ver_hist", False))
    sugestoes = [x for x in (ctx["sugestoes"] or []) if not (ctx["padrao_primeiro"] and x == EXPLICA_NUMEROS)]

    with st.container(key="ia_painel"):
        with st.container(key="ia_head"):
            c1, c2, c3, c4 = st.columns([6, 1, 1, 1], vertical_alignment="center")
            c1.markdown(
                '<div style="display:flex;align-items:center;gap:10px;">'
                '<div style="width:36px;height:36px;border-radius:50%;background:var(--df-accent);color:#fff;display:flex;'
                'align-items:center;justify-content:center;font-weight:800;font-size:13px;flex-shrink:0;">IA</div>'
                f'<div style="min-width:0;"><div style="font-weight:700;font-size:15px;line-height:1.2;">{tr("Tutor IA")}</div>'
                f'<div style="font-size:12px;color:var(--df-text-sec);line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'{html.escape(tr(ctx["titulo"])[:60])}</div></div></div>', unsafe_allow_html=True)
            c2.button(" ", key="ia_hist_b", icon=":material/history:", on_click=_alt_historico,
                      help=tr("Conversas anteriores"))
            c3.button(" ", key="ia_nova_b", icon=":material/add_comment:", on_click=_nova_conversa, args=(username, skey),
                      help=tr("Nova conversa"), disabled=not historico)
            if c4.button("✕", key="ia_fechar_b", help=tr("Fechar")):
                _fechar_painel()
        msgs = st.container(key="ia_hist" if ver_hist else "ia_msgs")
        comp = st.container(key="ia_comp")

    pergunta = None
    with comp:
        if ver_hist:
            st.button(tr("Voltar à conversa"), key="ia_voltar_b", on_click=_alt_historico, width="stretch")
            cap = None
        else:
            chips = st.empty()
            if not historico and sugestoes:
                with chips.container(key="ia_chips"):
                    for i, sug in enumerate(sugestoes):
                        if st.button(sug, key=f"ia_chip_{chave}_{i}", width="stretch"):
                            pergunta = tr(sug)
            with st.form(f"ia_form_{chave}", clear_on_submit=True, border=False):
                ca, cb = st.columns([5, 1], vertical_alignment="center")
                texto = ca.text_input("A tua dúvida", max_chars=400, label_visibility="collapsed",
                                      placeholder=tr("Escreve a tua dúvida..."))
                enviar = cb.form_submit_button("", icon=":material/send:", help=tr("Enviar"))
            if enviar and texto.strip():
                pergunta = texto.strip()
            if pergunta:
                chips.empty()        # as sugestões desaparecem logo que há pergunta
            cap = st.empty()

    if ver_hist:
        with msgs:
            antigas = sorted(((k, v) for k, v in conv.items() if k != skey and v.get("m")),
                             key=lambda kv: kv[1].get("t", 0), reverse=True)
            if not antigas:
                st.markdown(f"_{tr('Ainda não tens conversas anteriores.')}_")
            for n, (k, v) in enumerate(antigas):
                with st.expander(f"{v.get('titulo', '')[:55]} · {_data_curta(v.get('t'))}", expanded=(n == 0)):
                    for j, (p, r) in enumerate(v["m"]):
                        _bolha("u", f"h{n}_{j}", p)
                        _bolha("a", f"h{n}_{j}", r)
        return

    with msgs:
        with st.container(key="ia_a_boasvindas"):
            if historico or pergunta:
                st.markdown(f"{tr('Estou a ver')}: **{tr(ctx['titulo'])}**. "
                            f"{tr('A IA pode enganar-se: confirma com a lição ou com o professor.')}")
            else:
                st.markdown(f"{tr('Olá! Sou o teu tutor de Machine Learning.')} {tr('Estou a ver')}: **{tr(ctx['titulo'])}**. "
                            f"{tr('Pergunta o que não percebeste. A IA pode enganar-se: confirma com a lição ou com o professor.')}")
                if ctx["padrao_primeiro"]:
                    padrao = explicacoes_padrao(ctx["contexto"])
                    if padrao:
                        st.markdown(f"**{tr('O que significam estes números (sem IA)')}**")
                        for nome, texto_padrao in padrao:
                            st.markdown(f"- **{nome}:** {tr(texto_padrao)}")
        for i, (p, r) in enumerate(historico):
            _bolha("u", str(i), p)
            _bolha("a", str(i), r)
        if pergunta:
            _bolha("u", "novo", pergunta)
            nova = st.empty()
            _bolha("a", "pensa", "_" + tr("A pensar...") + "_", nova)
            try:
                resposta = ia.perguntar_tutor(pergunta, ctx["titulo"], ctx["contexto"], historico, username, idioma_actual(),
                                              False, usar_cache=ctx["usar_cache"], estado=ctx["estado"])
            except ia.IAErro as e:
                nova.empty()
                _bolha("a", "erro", "Aviso: " + tr(str(e)), nova)
            else:
                ent = conv.setdefault(skey, {"titulo": ctx["titulo"], "m": []})
                ent["t"] = time.time()
                ent["m"].append([pergunta, resposta])
                del ent["m"][:-_MAX_TROCAS]
                _guardar_chat(username, doc)
                _gastou_uma(username)
                nova.empty()
                _bolha("a", "novo", resposta, nova)
    cap.caption(tr("Perguntas à IA que restam hoje: {}").format(_restantes(username)))
    with st.container(key="ia_js"):
        _scroll_js("pergunta" if pergunta else "fim")


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
    _painel_ia(username, ctx)
