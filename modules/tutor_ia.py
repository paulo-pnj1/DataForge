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
                 usar_cache: bool = False, padrao_primeiro: bool = False):
    """Caixa de perguntas ao tutor. `chave` tem de ser única por sítio (lição, teste, página).
    `usar_cache`: lições e testes (respostas partilhadas, sem gastar limite). `padrao_primeiro`: resultados de treino,
    mostra primeiro as explicações padrão dos números (sem IA) e tira o botão «Explica estes números»."""
    if not ia.configurada():
        return
    hist_key = f"_tutor_h_{chave}"
    historico = st.session_state.setdefault(hist_key, [])
    with st.expander(tr(rotulo), expanded=aberto):
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
                                                  usar_cache=usar_cache)
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
