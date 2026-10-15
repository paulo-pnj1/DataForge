"""
DataForge EDU - Dashboard Principal
"""
import streamlit as st
import datetime
from modules.utils import (
    progresso_bar, section_title, badge,
    get_user_progress, ir_para, PAGINA_PARA_MODULO,
    C_TEXT, C_TEXT_SEC, C_TEXT_MUTE,
    C_SURFACE, C_SURFACE2, C_BORDER, C_ACCENT,
    C_GREEN, C_AMBER, C_ACCENT2, C_TEAL, C_RED, C_BG
)

# ── Phosphor Icons (mesmo sistema do app.py) ──────────
def _ico(name, size=18, color="currentColor"):
    P = {
        "home":    '<path d="M218.83 103.77l-80-75.48-.06-.06a16 16 0 0 0-21.54 0l-.06.06-80 75.48A16 16 0 0 0 32 115.55V208a16 16 0 0 0 16 16H96a16 16 0 0 0 16-16V160h32v48a16 16 0 0 0 16 16h48a16 16 0 0 0 16-16V115.55a16 16 0 0 0-5.17-11.78z"/>',
        "book":    '<path d="M48 80h112v160H48a16 16 0 0 1-16-16V96a16 16 0 0 1 16-16zm112 0h48a16 16 0 0 1 16 16v128a16 16 0 0 1-16 16H160m-112-96h112m-112 40h112" fill="none" stroke-width="16"/>',
        "star":    '<path d="M128 12.5l27.8 55.5 62 8.9-44.9 43.6 10.6 61.6L128 153.5l-55.5 28.6 10.6-61.6L38.2 76.9l62-8.9z" fill="none" stroke-width="16"/>',
        "layers":  '<path d="M16 128l112 64 112-64M16 176l112 64 112-64M128 16l112 64-112 64L16 80z" fill="none" stroke-width="16"/>',
        "award":   '<path d="M128 168a72 72 0 1 0-72-72 72 72 0 0 0 72 72zm0 0v72M96 224h64M80 176l-32 56M176 176l32 56" fill="none" stroke-width="16"/>',
        "activity":'<polyline points="16,128 60,128 92,48 164,208 196,128 240,128" fill="none" stroke-width="16"/>',
        "target":  '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><circle cx="128" cy="128" r="48" fill="none" stroke-width="16"/><circle cx="128" cy="128" r="16"/>',
        "trend":   '<path d="M232 56L144 144l-40-40-80 80M232 56h-56M232 56v56" fill="none" stroke-width="16"/>',
        "grid":    '<rect x="40" y="40" width="64" height="64" rx="8" fill="none" stroke-width="16"/><rect x="152" y="40" width="64" height="64" rx="8" fill="none" stroke-width="16"/><rect x="40" y="152" width="64" height="64" rx="8" fill="none" stroke-width="16"/><rect x="152" y="152" width="64" height="64" rx="8" fill="none" stroke-width="16"/>',
        "shrink":  '<path d="M48 40h168v176H48z" fill="none" stroke-width="16"/><path d="M88 88h80v80H88z" fill="none" stroke-width="16"/>',
        "radar":   '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><circle cx="128" cy="128" r="48" fill="none" stroke-width="16"/><circle cx="128" cy="128" r="12"/><path d="M128 128L80 80" fill="none" stroke-width="16"/>',
        "network": '<circle cx="200" cy="56" r="24" fill="none" stroke-width="16"/><circle cx="56" cy="128" r="24" fill="none" stroke-width="16"/><circle cx="200" cy="200" r="24" fill="none" stroke-width="16"/><path d="M80 128h96M103 103l54-23M103 153l54 23" fill="none" stroke-width="16"/>',
        "cpu":     '<rect x="80" y="80" width="96" height="96" rx="8" fill="none" stroke-width="16"/><path d="M160 80V48m-32 32V48m-32 32V48m0 160v32m32-32v32m32-32v32m48-80h32m-32 32h32m-32-64h32M48 160H16m32-32H16m32-64H16" fill="none" stroke-width="16"/>',
        "compass": '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><path d="M166 90l-38 38-38-38 38-38zm-76 76l38-38 38 38-38 38z"/>',
        "play":    '<path d="M80 56l96 72-96 72z"/>',
        "lock":    '<rect x="40" y="120" width="176" height="112" rx="16" fill="none" stroke-width="16"/><path d="M88 120V88a40 40 0 0 1 80 0v32" fill="none" stroke-width="16"/>',
        "check":   '<path d="M40 128l64 64L224 56" fill="none" stroke-width="20" stroke-linecap="round"/>',
        "info":    '<circle cx="128" cy="128" r="96" fill="none" stroke-width="16"/><path d="M120 120h16v72m-16 0h32" fill="none" stroke-width="16"/>',
        "eye":     '<path d="M16 128s40-80 112-80 112 80 112 80-40 80-112 80S16 128 16 128z" fill="none" stroke-width="16"/><circle cx="128" cy="128" r="32" fill="none" stroke-width="16"/>',
        "book2":   '<path d="M32 56a16 16 0 0 1 16-16h160a16 16 0 0 1 16 16v160a16 16 0 0 1-16 16H48a16 16 0 0 1-16-16zm96 0v176M80 96h32M80 136h32" fill="none" stroke-width="16"/>',
        "shuffle": '<path d="M32 72h48a80 80 0 0 1 80 80m0-112h64l-32-32m32 32-32 32M32 184h48a80 80 0 0 0 80-80m0 112h64l-32-32m32 32-32 32" fill="none" stroke-width="16"/>',
        "chart":   '<path d="M32 200h192M72 200v-80m48 80V72m48 128v-48m48 48V40" fill="none" stroke-width="16"/>',
    }
    path = P.get(name, P["chart"])
    sw = 'stroke-width="16"' if 'stroke-width' not in path else ''
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 256 256" fill="{color}" stroke="{color}" '
        f'stroke-linecap="round" stroke-linejoin="round" {sw} '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;">'
        f'{path}</svg>'
    )


PERCURSO = [
    {"id":"supervisionado_classificacao","titulo":"Classificação","categoria":"Supervisionado",
     "cor":C_ACCENT,"nivel":"Iniciante","algoritmos":10,"icon":"target",
     "descricao":"KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM, AdaBoost, Naive Bayes, Logistic Regression",
     "prerequisitos":[]},
    {"id":"supervisionado_regressao","titulo":"Regressão","categoria":"Supervisionado",
     "cor":C_GREEN,"nivel":"Iniciante","algoritmos":8,"icon":"trend",
     "descricao":"Linear, Ridge, Lasso, ElasticNet, SVR, KNN Regressor, RF Regressor, GB Regressor",
     "prerequisitos":["supervisionado_classificacao"]},
    {"id":"nao_supervisionado_clustering","titulo":"Clustering","categoria":"Não Supervisionado",
     "cor":C_TEAL,"nivel":"Intermédio","algoritmos":7,"icon":"grid",
     "descricao":"KMeans, DBSCAN, HDBSCAN, Agglomerative, Birch, Mean Shift, GMM",
     "prerequisitos":["supervisionado_classificacao"]},
    {"id":"nao_supervisionado_reducao","titulo":"Redução de Dimensionalidade","categoria":"Não Supervisionado",
     "cor":C_ACCENT2,"nivel":"Intermédio","algoritmos":6,"icon":"shrink",
     "descricao":"PCA, Kernel PCA, t-SNE, UMAP, LDA, ICA",
     "prerequisitos":["supervisionado_classificacao"]},
    {"id":"nao_supervisionado_anomalias","titulo":"Detecção de Anomalias","categoria":"Não Supervisionado",
     "cor":C_AMBER,"nivel":"Intermédio","algoritmos":5,"icon":"radar",
     "descricao":"Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, Autoencoder",
     "prerequisitos":["nao_supervisionado_clustering"]},
    {"id":"redes_neurais_mlp","titulo":"Redes Neurais (MLP)","categoria":"Redes Neurais",
     "cor":C_ACCENT,"nivel":"Avançado","algoritmos":4,"icon":"network",
     "descricao":"MLP Classifier, MLP Regressor, Rede Personalizada PyTorch, Dropout + BatchNorm",
     "prerequisitos":["supervisionado_regressao"]},
    {"id":"deep_learning_visao","titulo":"Visão Computacional","categoria":"Deep Learning",
     "cor":C_ACCENT2,"nivel":"Avançado","algoritmos":6,"icon":"eye",
     "descricao":"CNN personalizada, ResNet, VGG16, MobileNetV2, EfficientNet, U-Net",
     "prerequisitos":["redes_neurais_mlp"]},
    {"id":"deep_learning_nlp","titulo":"Processamento de Linguagem","categoria":"Deep Learning",
     "cor":C_GREEN,"nivel":"Avançado","algoritmos":7,"icon":"book2",
     "descricao":"BoW, TF-IDF, Word2Vec, LSTM, GRU, Transformer, BERT",
     "prerequisitos":["redes_neurais_mlp"]},
    {"id":"deep_learning_series","titulo":"Séries Temporais","categoria":"Deep Learning",
     "cor":C_TEAL,"nivel":"Avançado","algoritmos":5,"icon":"activity",
     "descricao":"LSTM, GRU, TCN, Prophet, ARIMA",
     "prerequisitos":["redes_neurais_mlp"]},
    {"id":"deep_learning_generativo","titulo":"Modelos Generativos","categoria":"Deep Learning",
     "cor":C_AMBER,"nivel":"Expert","algoritmos":3,"icon":"shuffle",
     "descricao":"Autoencoder Variacional (VAE), GAN, DCGAN",
     "prerequisitos":["deep_learning_visao"]},
    {"id":"reforco","titulo":"Aprendizagem por Reforço","categoria":"Reforço",
     "cor":C_RED,"nivel":"Expert","algoritmos":5,"icon":"compass",
     "descricao":"Q-Learning, DQN, Policy Gradient, Actor-Critic, CartPole",
     "prerequisitos":["redes_neurais_mlp"]},
]

NIVEL_CORES = {
    "Iniciante":  C_GREEN,
    "Intermédio": C_AMBER,
    "Avançado":   C_ACCENT,
    "Expert":     C_RED,
}

CAT_ICONS = {
    "Supervisionado":     "target",
    "Não Supervisionado": "layers",
    "Redes Neurais":      "network",
    "Deep Learning":      "cpu",
    "Reforço":            "compass",
}



MODULO_PARA_PAGINA = {v: k for k, v in PAGINA_PARA_MODULO.items()}


def _total_licoes() -> int:
    from modules.escola import LICOES
    return len(LICOES)


def _proximo_passo(progress: dict, completados: list) -> dict:
    """Decide a melhor sugestão para o utilizador, em linguagem simples."""
    licoes = progress.get("licoes", [])
    fez_primeiro = "primeiro_modelo" in progress.get("modulos", {})

    if not licoes and not fez_primeiro:
        return {"titulo": "Bem-vindo! Começa aqui",
                "texto": "Nunca fizeste Machine Learning? Sem stress: a Escola de ML explica tudo do zero em lições de 5 minutos, "
                         "com experiências que podes mexer. Preferes ver já um modelo a funcionar? Há um botão para isso.",
                "botao": "Começar a Escola de ML", "pagina": "escola", "abrir_licao": "l1",
                "botao2": "Treinar um modelo já", "pagina2": "primeiro_modelo"}
    if not fez_primeiro:
        return {"titulo": "Próximo passo: treina o teu primeiro modelo",
                "texto": "Já viste a teoria. Agora experimenta na prática: escolhes um problema, carregas num botão e o computador aprende.",
                "botao": "Treinar o meu primeiro modelo", "pagina": "primeiro_modelo",
                "botao2": "Continuar a Escola de ML", "pagina2": "escola"}
    if len(licoes) < _total_licoes():
        return {"titulo": f"Continua a Escola de ML ({len(licoes)}/{_total_licoes()} lições)",
                "texto": "Cada lição leva uns 5 minutos e tem uma experiência interactiva. Há mais conceitos para dominares.",
                "botao": "Continuar a aprender", "pagina": "escola",
                "botao2": "Que algoritmo devo usar?", "pagina2": "assistente"}
    for m in PERCURSO:
        if m["id"] not in completados and m["id"] in MODULO_PARA_PAGINA:
            return {"titulo": f"Próximo módulo: {m['titulo']}",
                    "texto": m["descricao"],
                    "botao": f"Abrir {m['titulo']}", "pagina": MODULO_PARA_PAGINA[m["id"]],
                    "botao2": "Dicionário de ML", "pagina2": "dicionario"}
    return {"titulo": "Completaste o percurso!",
            "texto": "Podes comparar algoritmos, rever lições ou treinar com os teus próprios dados (CSV).",
            "botao": "Comparar modelos", "pagina": "comparador",
            "botao2": "Rever a Escola de ML", "pagina2": "escola"}


def _cartao_proximo_passo(passo: dict):
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(63,185,80,.10),rgba(88,166,255,.10));'
        f'border:1px solid {C_GREEN};border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:1rem;">'
        f'<div style="font-size:11px;font-weight:800;color:{C_GREEN};text-transform:uppercase;letter-spacing:.08em;">O teu próximo passo</div>'
        f'<div style="font-size:22px;font-weight:800;color:{C_TEXT};margin:4px 0 6px;">{passo["titulo"]}</div>'
        f'<div style="font-size:15px;color:{C_TEXT_SEC};line-height:1.65;">{passo["texto"]}</div></div>',
        unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.3, 1.3, 2])
    with c1:
        if st.button(passo["botao"], type="primary", key="dash_passo1", width="stretch"):
            if passo.get("abrir_licao"):
                st.session_state.escola_aberta = passo["abrir_licao"]
            ir_para(passo["pagina"])
    with c2:
        if st.button(passo["botao2"], key="dash_passo2", width="stretch"):
            ir_para(passo["pagina2"])


def render_dashboard(name: str, username: str):
    progress    = get_user_progress(username)
    completados = progress.get("completados", [])
    pontos      = progress.get("pontos", 0)
    total       = len(PERCURSO)
    feitos      = len(completados)
    pct         = feitos / total if total > 0 else 0
    nivel_actual = (
        "Expert"    if feitos >= 9 else
        "Avançado"  if feitos >= 5 else
        "Intermédio" if feitos >= 2 else
        "Iniciante"
    )

    # ── PAGE HEADER ──────────────────────────────────
    st.markdown(f"""<div style="display:flex;align-items:center;gap:14px;margin-bottom:1.8rem; padding-bottom:1.2rem;border-bottom:1px solid {C_BORDER};"><div style="width:48px;height:48px;background:linear-gradient(135deg,{C_ACCENT},{C_ACCENT2}); border-radius:14px;display:flex;align-items:center;justify-content:center;flex-shrink:0; box-shadow:0 4px 16px rgba(88,166,255,.25);">{_ico("home", 24, "#0D1117")}</div><div><div style="font-size:22px;font-weight:800;color:{C_TEXT};line-height:1.2;letter-spacing:-.3px;">Dashboard</div><div style="font-size:14px;color:{C_TEXT_SEC};margin-top:3px;">Bem-vindo de volta, {name.split()[0]}. Continua o teu percurso.</div></div></div>""", unsafe_allow_html=True)

    # ── PRÓXIMO PASSO ────────────────────────────────
    _cartao_proximo_passo(_proximo_passo(progress, completados))

    # ── KPIs ─────────────────────────────────────────
    kpis = [
        {"icon":"book",    "label":"Módulos",    "valor":f"{feitos}/{total}",  "cor":C_ACCENT},
        {"icon":"star",    "label":"Pontuação",  "valor":f"{pontos} pts",      "cor":C_AMBER},
        {"icon":"layers",  "label":"Algoritmos", "valor":"50+",                "cor":C_TEAL},
        {"icon":"award",   "label":"Nível",      "valor":nivel_actual,         "cor":C_GREEN},
    ]
    cols = st.columns(4)
    for col, k in zip(cols, kpis):
        with col:
            st.markdown(f"""<div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:12px; padding:1rem 1.1rem;display:flex;align-items:center;gap:12px;"><div style="width:40px;height:40px;background:var(--df-overlay,rgba(255,255,255,.04));border-radius:10px; border:1px solid {C_BORDER};display:flex;align-items:center;justify-content:center;flex-shrink:0;">{_ico(k["icon"], 20, k["cor"])}</div><div><div style="font-size:20px;font-weight:800;color:{C_TEXT};line-height:1.1;">{k["valor"]}</div><div style="font-size:11px;font-weight:600;color:{C_TEXT_MUTE};text-transform:uppercase; letter-spacing:.06em;margin-top:2px;">{k["label"]}</div></div></div>""", unsafe_allow_html=True)

    # ── PROGRESSO ────────────────────────────────────
    st.markdown("<div style='margin:1.4rem 0 .2rem;'>", unsafe_allow_html=True)
    progresso_bar(pct, "Progresso geral do percurso")
    progresso_bar(len(progress.get("licoes", [])) / _total_licoes(), "Escola de ML (lições)")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── PERCURSO ─────────────────────────────────────
    section_title("PERCURSO DE APRENDIZAGEM")

    cats = {}
    for m in PERCURSO:
        cats.setdefault(m["categoria"], []).append(m)

    for cat, modulos in cats.items():
        cat_ico = CAT_ICONS.get(cat, "layers")
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px; font-size:11px;font-weight:700;color:{C_TEXT_SEC};text-transform:uppercase; letter-spacing:.08em;margin:1.6rem 0 .8rem;padding-bottom:8px; border-bottom:1px solid {C_BORDER};">{_ico(cat_ico, 14, C_TEXT_SEC)}&nbsp;{cat}</div>""", unsafe_allow_html=True)

        cols = st.columns(min(len(modulos), 3))
        for i, m in enumerate(modulos):
            with cols[i % 3]:
                _card_modulo(m, completados)

    # ── ACTIVIDADE RECENTE ────────────────────────────
    section_title("ACTIVIDADE RECENTE")
    historico = progress.get("historico_pontos", [])
    if historico:
        for h in reversed(historico[-5:]):
            data   = h.get("data", "")[:10]
            motivo = h.get("motivo", "-")
            pts    = h.get("pontos", 0)
            st.markdown(f"""<div style="display:flex;align-items:center;gap:12px; padding:.65rem 0;border-bottom:1px solid {C_BORDER};"><div style="width:32px;height:32px;background:rgba(88,166,255,.1); border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{_ico("star", 15, C_AMBER)}</div><div style="flex:1;"><div style="font-size:13px;color:{C_TEXT};">{motivo}</div><div style="font-size:11px;color:{C_TEXT_MUTE};margin-top:2px;">{data}</div></div><div style="font-size:13px;font-weight:700;color:{C_ACCENT};">+{pts} pts</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style="text-align:center;padding:2rem;background:{C_SURFACE}; border:1px solid {C_BORDER};border-radius:12px;"><div style="margin-bottom:12px;">{_ico("activity", 32, C_TEXT_MUTE)}</div><div style="font-size:14px;color:{C_TEXT_MUTE};">Ainda sem actividade. Começa por Classificação!</div></div>""", unsafe_allow_html=True)

    # ── DICA DO DIA ───────────────────────────────────
    dicas = [
        "Normaliza sempre as features antes de usar KNN ou SVM.",
        "Random Forest raramente precisa de tuning intensivo - excelente ponto de partida.",
        "Overfitting? Adiciona regularização (Ridge/Lasso) ou reduz a complexidade do modelo.",
        "Usa curvas de aprendizagem para diagnosticar overfitting vs underfitting.",
        "Para dados desbalanceados, usa F1-score em vez de accuracy.",
        "t-SNE é óptimo para visualizar - nunca uses as suas coordenadas como features.",
        "Gradient Boosting com learning_rate baixo e mais árvores tende a generalizar melhor.",
        "Em séries temporais, nunca mistures dados de treino e teste por ordem cronológica.",
    ]
    dica = dicas[datetime.date.today().toordinal() % len(dicas)]
    st.markdown(f"""<div style="background:rgba(88,166,255,.07);border:1px solid rgba(88,166,255,.2); border-left:3px solid {C_ACCENT};border-radius:10px;padding:1rem 1.2rem; margin-top:1.4rem;display:flex;gap:12px;align-items:flex-start;"><div style="flex-shrink:0;margin-top:1px;">{_ico("info", 16, C_ACCENT)}</div><div><div style="font-size:11px;font-weight:700;color:{C_ACCENT};text-transform:uppercase; letter-spacing:.06em;margin-bottom:5px;">Dica do Dia</div><div style="font-size:14px;color:{C_TEXT_SEC};line-height:1.65;">{dica}</div></div></div>""", unsafe_allow_html=True)


def _card_modulo(modulo: dict, completados: list):
    mid        = modulo["id"]
    feito      = mid in completados
    cor        = modulo["cor"]
    nivel_cor  = NIVEL_CORES.get(modulo["nivel"], C_ACCENT)
    prereqs_ok = all(p in completados for p in modulo["prerequisitos"])
    bloqueado  = not prereqs_ok and bool(modulo["prerequisitos"])
    opac       = "0.78" if bloqueado else "1"
    nomes_pre  = [m["titulo"] for m in PERCURSO if m["id"] in modulo["prerequisitos"] and m["id"] not in completados]

    if feito:
        status_bg  = "rgba(63,185,80,.15)"
        status_ico = _ico("check", 14, C_GREEN)
        bl         = f"3px solid {cor}"
    elif bloqueado:
        status_bg  = "var(--df-overlay,rgba(255,255,255,.04))"
        status_ico = _ico("lock", 14, C_TEXT_MUTE)
        bl         = f"1px solid {C_BORDER}"
    else:
        status_bg  = "rgba(88,166,255,.12)"
        status_ico = _ico("play", 14, C_ACCENT)
        bl         = f"1px solid {C_BORDER}"

    mod_ico = _ico(modulo.get("icon", "chart"), 18, cor)

    bloq_html = ""
    if bloqueado:
        bloq_html = (
            f'<div style="display:flex;align-items:center;gap:5px;font-size:11px;'
            f'color:{C_TEXT_MUTE};margin-top:.6rem;">'
            f'{_ico("lock", 11, C_TEXT_MUTE)}'
            f'&nbsp;Recomendado fazer antes: {", ".join(nomes_pre)}</div>'
        )

    st.markdown(f"""<div style="background:{C_SURFACE};border-left:{bl}; border-top:1px solid {C_BORDER};border-right:1px solid {C_BORDER}; border-bottom:1px solid {C_BORDER};border-radius:12px; padding:1rem 1.1rem;margin-bottom:.8rem;opacity:{opac};"><div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.8rem;"><div style="width:36px;height:36px;background:var(--df-overlay,rgba(255,255,255,.04)); border-radius:9px;display:flex;align-items:center;justify-content:center;">{mod_ico}</div><div style="width:26px;height:26px;background:{status_bg}; border-radius:7px;display:flex;align-items:center;justify-content:center;">{status_ico}</div></div><div style="font-size:14px;font-weight:700;color:{C_TEXT};margin-bottom:3px;">{modulo["titulo"]}</div><div style="font-size:11px;color:{C_TEXT_MUTE};margin-bottom:.7rem; overflow:hidden;display:-webkit-box;-webkit-line-clamp:2; -webkit-box-orient:vertical;line-height:1.6;">{modulo["descricao"]}</div><div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center;"><span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:8px; color:{nivel_cor};background:color-mix(in srgb, {nivel_cor} 10%, transparent);border:1px solid color-mix(in srgb, {nivel_cor} 20%, transparent);">{modulo["nivel"]}</span><span style="font-size:10px;padding:2px 8px;border-radius:8px; color:{C_TEXT_MUTE};background:var(--df-overlay,rgba(255,255,255,.04));">{modulo["algoritmos"]} algoritmos</span></div>{bloq_html}</div>""", unsafe_allow_html=True)

    pagina = MODULO_PARA_PAGINA.get(mid)
    if pagina and st.button("Abrir" if not feito else "Praticar de novo", key=f"abrir_{mid}", width="stretch"):
        ir_para(pagina)
