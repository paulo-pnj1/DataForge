"""
DataForge EDU - Escola de ML
Curso para quem começa do zero: 11 lições curtas, cada uma com
analogia do dia-a-dia, demonstração interactiva e mini-teste.
"""
import random
import zlib

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from modules.utils import (
    page_header, progresso_bar, get_licoes_concluidas, concluir_licao, ir_para,
    C_BG, C_SURFACE, C_SURFACE2, C_BORDER, C_ACCENT, C_GREEN, C_AMBER, C_RED,
    C_TEAL, C_TEXT, C_TEXT_SEC, C_TEXT_MUTE,
)

COR_A = C_ACCENT   # azul
COR_B = C_AMBER    # âmbar


# ══════════════════════════════════════════════════════
# COMPONENTES VISUAIS
# ══════════════════════════════════════════════════════
def _caixa(titulo: str, corpo: str, cor: str = C_ACCENT):
    st.markdown(
        f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-left:4px solid {cor};'
        f'border-radius:10px;padding:1rem 1.2rem;margin:.9rem 0;">'
        f'<div style="font-size:11px;font-weight:800;color:{cor};text-transform:uppercase;'
        f'letter-spacing:.07em;margin-bottom:6px;">{titulo}</div>'
        f'<div style="font-size:15px;color:{C_TEXT};line-height:1.75;">{corpo}</div></div>',
        unsafe_allow_html=True,
    )


def _secao(numero: int, titulo: str):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:1.8rem 0 .6rem;">'
        f'<div style="width:26px;height:26px;border-radius:50%;background:{C_ACCENT};color:#0D1117;'
        f'font-weight:800;font-size:13px;display:flex;align-items:center;justify-content:center;">{numero}</div>'
        f'<div style="font-size:18px;font-weight:800;color:{C_TEXT};">{titulo}</div></div>',
        unsafe_allow_html=True,
    )


def _fig(w=7.2, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    for s in ax.spines.values():
        s.set_color(C_BORDER)
    ax.tick_params(colors=C_TEXT_SEC)
    ax.xaxis.label.set_color(C_TEXT_SEC)
    ax.yaxis.label.set_color(C_TEXT_SEC)
    ax.title.set_color(C_TEXT)
    ax.grid(alpha=.15, color=C_TEXT_SEC)
    return fig, ax


def _mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


def _legenda(ax, **kw):
    leg = ax.legend(facecolor=C_SURFACE2, edgecolor=C_BORDER, labelcolor=C_TEXT, **kw)
    return leg


# ══════════════════════════════════════════════════════
# LIÇÃO 1 - O que é Machine Learning?
# ══════════════════════════════════════════════════════
def _dados_frutas():
    rng = np.random.default_rng(5)
    n = 40
    lim = np.column_stack([rng.normal(5.4, .8, n), rng.normal(3.5, 1.5, n)])
    lar = np.column_stack([rng.normal(7.4, .8, n), rng.normal(6.5, 1.5, n)])
    X = np.vstack([lim, lar])
    y = np.array([0] * n + [1] * n)
    return X, y


def _licao_1():
    _caixa("Analogia",
           "Uma zungueira experiente sabe quanto peixe deve levar para vender, sem ninguém lhe dar uma fórmula. "
           "Aprendeu com <b>muitos dias de experiência</b>: dias de chuva, fins de mês, dias de festa. "
           "O Machine Learning faz o mesmo: o computador <b>aprende com exemplos</b> em vez de receber regras escritas.",
           C_TEAL)

    _secao(1, "Duas formas de resolver um problema")
    c1, c2 = st.columns(2)
    with c1:
        _caixa("Programação clássica",
               "Tu escreves as <b>regras</b>.<br>Dados + Regras → <b>Resposta</b><br><br>"
               "<i>Ex.: \"Se doçura &gt; 5, é laranja.\"</i>", C_TEXT_SEC)
    with c2:
        _caixa("Machine Learning",
               "O computador descobre as <b>regras</b>.<br>Dados + Respostas → <b>Regras</b><br><br>"
               "<i>Ex.: vê 80 frutas já identificadas e percebe sozinho o que distingue uma laranja de um limão.</i>", C_ACCENT)

    _secao(2, "Experimenta: tu contra o computador")
    st.markdown("Cada ponto é uma fruta. Azul = **limão**, laranja = **laranja**. "
                "Escreve uma regra e vê quantas frutas ela acerta. Depois deixa o computador tentar.")
    X, y = _dados_frutas()
    c1, c2 = st.columns(2)
    with c1:
        a = st.slider("Regra: é laranja se a doçura for maior que...", 0.0, 10.0, 5.0, 0.1, key="l1_a")
    with c2:
        b = st.slider("...e o tamanho (cm) for maior que...", 3.0, 10.0, 3.0, 0.1, key="l1_b")

    pred_tu = ((X[:, 1] > a) & (X[:, 0] > b)).astype(int)
    acc_tu = float((pred_tu == y).mean())

    fig, ax = _fig()
    ax.scatter(*X[y == 0].T, c=COR_A, s=42, label="Limão", edgecolor="white", linewidth=.4)
    ax.scatter(*X[y == 1].T, c=COR_B, s=42, label="Laranja", edgecolor="white", linewidth=.4)
    ax.axhline(a, color=C_GREEN, ls="--", lw=1.5)
    ax.axvline(b, color=C_GREEN, ls="--", lw=1.5)
    ax.set_xlabel("Tamanho (cm)")
    ax.set_ylabel("Doçura (0 a 10)")
    ax.set_title("A tua regra (linhas verdes)")
    _legenda(ax, loc="upper left")
    _mostrar(fig)

    if st.button("Deixar o computador aprender com os exemplos", key="l1_btn", type="primary"):
        st.session_state["l1_pc"] = True
    if st.session_state.get("l1_pc"):
        modelo = LogisticRegression().fit(X, y)
        acc_pc = float(modelo.score(X, y))
        gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
        zz = modelo.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        fig, ax = _fig()
        ax.contourf(gx, gy, zz, alpha=.18, colors=[COR_A, COR_B], levels=[-.5, .5, 1.5])
        ax.scatter(*X[y == 0].T, c=COR_A, s=42, edgecolor="white", linewidth=.4)
        ax.scatter(*X[y == 1].T, c=COR_B, s=42, edgecolor="white", linewidth=.4)
        ax.set_xlabel("Tamanho (cm)")
        ax.set_ylabel("Doçura (0 a 10)")
        ax.set_title("A regra que o computador descobriu sozinho")
        _mostrar(fig)
        m1, m2 = st.columns(2)
        m1.metric("A tua regra acertou", f"{acc_tu:.0%}")
        m2.metric("O computador acertou", f"{acc_pc:.0%}")
        st.info("Ninguém disse ao computador o que é uma laranja. Ele **olhou para os exemplos** e encontrou a fronteira. "
                "Isto é Machine Learning.")
    else:
        st.metric("A tua regra acertou", f"{acc_tu:.0%}")


# ══════════════════════════════════════════════════════
# LIÇÃO 2 - Os dados
# ══════════════════════════════════════════════════════
def _licao_2():
    import pandas as pd
    _caixa("Analogia",
           "Pensa num caderno de uma loja: cada <b>linha</b> é uma venda e cada <b>coluna</b> é um detalhe "
           "(produto, preço, hora). Os modelos de ML lêem exactamente este tipo de tabela.", C_TEAL)

    df = pd.DataFrame({
        "Bairro": ["Talatona", "Maianga", "Viana", "Cazenga", "Kilamba", "Benfica", "Ingombota", "Camama"],
        "Quartos": [4, 3, 2, 2, 3, 3, 4, 2],
        "Área (m²)": [180, 110, 70, 65, 95, 120, 150, 75],
        "Preço (milhões Kz)": [95, 60, 22, 18, 40, 48, 85, 24],
    })
    _secao(1, "Uma tabela de casas (dados fictícios)")
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("""
- Cada **linha** é um **exemplo** (uma casa).
- Cada **coluna** é uma **característica** - em ML chama-se *feature*.
- A coluna que queremos prever chama-se **alvo** (*target*).
""")

    _secao(2, "Experimenta")
    alvo = st.radio("Queremos prever o preço das casas. Qual é o ALVO?", list(df.columns),
                    index=None, key="l2_alvo", horizontal=True)
    if alvo:
        if alvo == "Preço (milhões Kz)":
            st.success("Certo! O alvo é aquilo que queremos descobrir: o preço.")
        else:
            st.error("Quase! O alvo é a coluna que queremos **prever**. Aqui queremos prever o preço.")

    feats = st.multiselect("Que colunas podem AJUDAR a prever o preço? (as features)", list(df.columns), key="l2_feat")
    if feats:
        if "Preço (milhões Kz)" in feats:
            st.warning("Cuidado: o preço é a **resposta**. Se o puseres como feature, o modelo faz batota "
                       "(isto chama-se *fuga de informação*) e depois não funciona em casas novas.")
        elif feats:
            st.success(f"Boa escolha: {', '.join(feats)}. Quanto mais informação útil, melhor o modelo aprende.")

    _caixa("Regra de ouro",
           "<b>Lixo entra, lixo sai.</b> Um modelo só é tão bom quanto os dados com que aprende. "
           "Dados errados, em falta ou pouco representativos dão previsões más.", C_AMBER)


# ══════════════════════════════════════════════════════
# LIÇÃO 3 - Três maneiras de aprender
# ══════════════════════════════════════════════════════
def _licao_3():
    c1, c2, c3 = st.columns(3)
    with c1:
        _caixa("Supervisionada",
               "Aprendes <b>com professor</b>: tens os exemplos e a resposta certa.<br><br>"
               "<i>Como estudar com exercícios que têm gabarito.</i>", C_ACCENT)
    with c2:
        _caixa("Não supervisionada",
               "Aprendes <b>sem professor</b>: só tens os dados e procuras padrões.<br><br>"
               "<i>Como arrumar roupa sem ninguém dizer em que montes.</i>", C_TEAL)
    with c3:
        _caixa("Por reforço",
               "Aprendes <b>por tentativa e erro</b>, com prémios e castigos.<br><br>"
               "<i>Como treinar um cão com petiscos.</i>", C_AMBER)

    _secao(1, "Jogo rápido: que tipo de aprendizagem é?")
    TIPOS = ["Supervisionada", "Não supervisionada", "Por reforço"]
    cenarios = [
        ("Prever se um cliente vai pagar o crédito, usando milhares de clientes antigos cujo resultado já se conhece.", 0),
        ("Separar os clientes de uma loja em grupos parecidos, sem ninguém ter dito quais são os grupos.", 1),
        ("Um robô aprende a sair de um labirinto: ganha pontos quando se aproxima da saída.", 2),
        ("Prever o preço de uma casa a partir da área e do bairro.", 0),
        ("Descobrir transacções bancárias estranhas sem ter exemplos de fraude.", 1),
    ]
    escolhas = []
    for i, (txt, _) in enumerate(cenarios):
        escolhas.append(st.selectbox(f"{i + 1}. {txt}", ["Escolhe..."] + TIPOS, key=f"l3_c{i}"))
    if st.button("Verificar o jogo", key="l3_btn"):
        certas = 0
        for i, (txt, certa) in enumerate(cenarios):
            if escolhas[i] == TIPOS[certa]:
                certas += 1
                st.success(f"{i + 1}. Certo: {TIPOS[certa]}.")
            elif escolhas[i] == "Escolhe...":
                st.warning(f"{i + 1}. Ainda sem resposta.")
            else:
                st.error(f"{i + 1}. Não é bem isso. A resposta é: {TIPOS[certa]}.")
        st.metric("Acertaste", f"{certas} de {len(cenarios)}")

    _caixa("Pista para decidir",
           "Pergunta a ti próprio: <b>tenho a resposta certa nos meus dados?</b> "
           "Se sim → supervisionada. Se não e quero achar grupos/padrões → não supervisionada. "
           "Se há um agente a agir e a receber prémios → reforço.", C_GREEN)


# ══════════════════════════════════════════════════════
# LIÇÃO 4 - Regressão
# ══════════════════════════════════════════════════════
def _dados_casas():
    rng = np.random.default_rng(3)
    area = rng.uniform(40, 200, 35)
    preco = 0.45 * area - 5 + rng.normal(0, 7, 35)
    return area, preco


def _licao_4():
    _caixa("Analogia",
           "Quando um taxista diz \"para o Kilamba são uns 4.500 Kz\", está a fazer uma <b>previsão de um número</b> "
           "com base na distância. Prever números (preços, temperaturas, vendas) chama-se <b>regressão</b>.", C_TEAL)

    area, preco = _dados_casas()
    _secao(1, "Desenha a linha que melhor passa pelos pontos")
    st.markdown("Cada ponto é uma casa (dados fictícios). A linha é a tua **previsão**: dada a área, que preço esperas? "
                "Mexe nos controlos para a ajustar.")
    c1, c2 = st.columns(2)
    with c1:
        incl = st.slider("Inclinação (quanto sobe o preço por cada m²)", 0.0, 1.0, 0.20, 0.01, key="l4_m")
    with c2:
        base = st.slider("Ponto de partida (preço base)", -40.0, 40.0, 0.0, 0.5, key="l4_b")
    mostrar = st.checkbox("Mostrar a melhor linha encontrada pelo computador", key="l4_best")

    pred = incl * area + base
    mae_tu = float(np.mean(np.abs(preco - pred)))

    fig, ax = _fig()
    ax.scatter(area, preco, c=COR_A, s=40, edgecolor="white", linewidth=.4, label="Casas reais")
    xs = np.linspace(40, 200, 50)
    ax.plot(xs, incl * xs + base, color=C_GREEN, lw=2.4, label="A tua linha")
    for xi, yi, pi in zip(area, preco, pred):
        ax.plot([xi, xi], [yi, pi], color=C_RED, alpha=.35, lw=1)
    if mostrar:
        m, b = np.polyfit(area, preco, 1)
        ax.plot(xs, m * xs + b, color=COR_B, lw=2.4, ls="--", label="Linha do computador")
    ax.set_xlabel("Área (m²)")
    ax.set_ylabel("Preço (milhões Kz)")
    ax.set_title("As riscas vermelhas são os erros")
    _legenda(ax, loc="upper left")
    _mostrar(fig)

    c1, c2 = st.columns(2)
    c1.metric("Erro médio da tua linha", f"{mae_tu:.1f} milhões Kz",
              help="Média das riscas vermelhas. Quanto menor, melhor a linha.")
    if mostrar:
        m, b = np.polyfit(area, preco, 1)
        mae_pc = float(np.mean(np.abs(preco - (m * area + b))))
        c2.metric("Erro médio do computador", f"{mae_pc:.1f} milhões Kz")
        st.info("O computador procura a linha que **torna as riscas vermelhas o mais pequenas possível**. "
                "Isto é aprender: ajustar até o erro ser mínimo.")
        a_nova = st.number_input("Quanto custa uma casa nova de quantos m²?", 40, 250, 120, key="l4_nova")
        st.success(f"Previsão do modelo para {a_nova} m²: **{m * a_nova + b:.0f} milhões de Kz**")


# ══════════════════════════════════════════════════════
# LIÇÃO 5 - Classificação (KNN)
# ══════════════════════════════════════════════════════
def _licao_5():
    _caixa("Analogia",
           "Chegas a um bairro novo e queres saber se o restaurante é bom. Perguntas aos <b>5 vizinhos mais próximos</b>: "
           "se 4 dizem que é bom, tu acreditas. É exactamente assim que funciona o algoritmo <b>KNN</b>. "
           "Prever uma <b>categoria</b> (sim/não, spam/normal, espécie A/B/C) chama-se <b>classificação</b>.", C_TEAL)

    rng = np.random.default_rng(7)
    A = rng.normal([3.0, 3.5], 1.0, (25, 2))
    B = rng.normal([6.5, 7.0], 1.2, (25, 2))
    X = np.vstack([A, B])
    y = np.array([0] * 25 + [1] * 25)
    nomes = ["Não compra", "Compra"]

    _secao(1, "Um cliente novo entra na banca. Vai comprar?")
    st.markdown("Cada ponto é um cliente antigo. Coloca o **cliente novo** (estrela) e escolhe quantos vizinhos vão votar.")
    c1, c2, c3 = st.columns(3)
    with c1:
        px = st.slider("Tempo na banca (min)", 0.0, 10.0, 5.0, 0.1, key="l5_x")
    with c2:
        py = st.slider("Interesse (0 a 10)", 0.0, 10.0, 5.0, 0.1, key="l5_y")
    with c3:
        k = st.slider("K = número de vizinhos", 1, 15, 5, key="l5_k")

    dist = np.sqrt((X[:, 0] - px) ** 2 + (X[:, 1] - py) ** 2)
    ordem = np.argsort(dist)[:k]
    votos = np.bincount(y[ordem], minlength=2)
    if votos[0] == votos[1]:
        decisao = int(y[ordem[0]])  # desempate: o vizinho mais próximo
    else:
        decisao = int(np.argmax(votos))

    fig, ax = _fig()
    ax.scatter(*X[y == 0].T, c=COR_A, s=42, edgecolor="white", linewidth=.4, label=nomes[0])
    ax.scatter(*X[y == 1].T, c=COR_B, s=42, edgecolor="white", linewidth=.4, label=nomes[1])
    for i in ordem:
        ax.plot([px, X[i, 0]], [py, X[i, 1]], color=C_GREEN, lw=1.4, alpha=.8)
    ax.scatter(X[ordem, 0], X[ordem, 1], s=170, facecolors="none", edgecolors=C_GREEN, linewidths=2)
    ax.scatter([px], [py], marker="*", s=420, c=COR_B if decisao else COR_A, edgecolor="white", linewidth=1.4, zorder=5)
    ax.set_xlabel("Tempo na banca (min)")
    ax.set_ylabel("Interesse (0 a 10)")
    ax.set_title(f"Os {k} vizinhos mais próximos estão rodeados a verde")
    _legenda(ax, loc="lower right")
    _mostrar(fig)

    st.markdown(f"**Votação:** {votos[1]} vizinhos dizem *Compra* e {votos[0]} dizem *Não compra*.")
    if decisao:
        st.success("Decisão do modelo: **este cliente vai comprar.**")
    else:
        st.error("Decisão do modelo: **este cliente provavelmente não vai comprar.**")
    if votos[0] == votos[1]:
        st.caption("Houve empate, por isso ganhou o vizinho mais próximo.")

    _caixa("Repara",
           "Com <b>K=1</b> o modelo olha só para um vizinho e fica nervoso: um ponto estranho muda tudo. "
           "Com <b>K muito grande</b> ele deixa de ver os detalhes. Escolher K é encontrar o equilíbrio. "
           "Experimenta mover o cliente para a fronteira entre os dois grupos e vai mudando o K.", C_AMBER)


# ══════════════════════════════════════════════════════
# LIÇÃO 6 - Treino e teste
# ══════════════════════════════════════════════════════
def _licao_6():
    _caixa("Analogia",
           "Um professor dá-te 100 exercícios para estudar. Se na prova aparecerem <b>exactamente os mesmos</b>, "
           "qualquer um tira 20. Para saber se aprendeste mesmo, a prova tem de ter <b>perguntas novas</b>. "
           "Em ML separamos os dados em <b>treino</b> (para estudar) e <b>teste</b> (a prova).", C_TEAL)

    X, y = make_classification(n_samples=300, n_features=6, n_informative=3, n_redundant=0,
                               flip_y=0.12, class_sep=1.0, random_state=3)
    _secao(1, "Estuda e depois faz a prova")
    c1, c2 = st.columns(2)
    with c1:
        pct = st.slider("Percentagem dos dados guardada para a prova (teste)", 10, 50, 30, 5, key="l6_pct")
    with c2:
        tipo = st.radio("Que tipo de aluno é o modelo?",
                        ["Simples (aprende o essencial)", "Decorador (decora tudo, até os erros)"], key="l6_tipo")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=pct / 100, random_state=1)
    clf = DecisionTreeClassifier(max_depth=3 if tipo.startswith("Simples") else None, random_state=0).fit(X_tr, y_tr)
    a_tr, a_te = clf.score(X_tr, y_tr), clf.score(X_te, y_te)

    fig, ax = _fig(6, 3.4)
    barras = ax.bar(["Exercícios de estudo\n(treino)", "Prova com perguntas novas\n(teste)"], [a_tr, a_te],
                    color=[COR_A, COR_B], width=.5)
    for b, v in zip(barras, [a_tr, a_te]):
        ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.0%}", ha="center", color=C_TEXT, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Percentagem de acertos")
    _mostrar(fig)

    st.markdown(f"Treino: **{len(X_tr)} casos**. Teste: **{len(X_te)} casos** que o modelo nunca viu.")
    if a_tr - a_te > 0.12:
        st.error(f"Repara na diferença: **{a_tr:.0%}** no estudo e só **{a_te:.0%}** na prova. "
                 "O modelo **decorou** os exemplos em vez de perceber a matéria. Isto chama-se *overfitting*.")
    else:
        st.success(f"Notas parecidas ({a_tr:.0%} e {a_te:.0%}). O modelo percebeu a matéria e deve funcionar bem com casos novos.")

    _caixa("Nunca faças isto",
           "Avaliar o modelo com os mesmos dados que usou para aprender. É como o aluno ver a prova antes de estudar: "
           "a nota fica bonita mas não diz nada sobre o mundo real.", C_RED)


# ══════════════════════════════════════════════════════
# LIÇÃO 7 - Overfitting / Underfitting
# ══════════════════════════════════════════════════════
def _dados_curva():
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 1, 40))
    y = np.sin(2 * np.pi * x) + rng.normal(0, .3, 40)
    idx = rng.permutation(40)
    tr, te = np.sort(idx[:15]), np.sort(idx[15:])
    return x[tr], y[tr], x[te], y[te]


def _ajusta_polinomio(x, y, grau):
    return np.polynomial.Polynomial.fit(x, y, grau)


def _licao_7():
    _caixa("Analogia",
           "Um alfaiate que faz o fato <b>tão justo</b> ao teu corpo de hoje que já não serve se engordares 1 kg "
           "(overfitting). Outro faz um saco enorme que serve a toda a gente mas não assenta em ninguém (underfitting). "
           "O fato bom assenta bem e <b>aguenta pequenas mudanças</b>.", C_TEAL)

    xtr, ytr, xte, yte = _dados_curva()
    _secao(1, "Quão complicado deve ser o modelo?")
    st.markdown("A curva cinzenta é a verdade escondida. Os pontos azuis são o que o modelo vê para **aprender**; "
                "os pontos laranja são a **prova** (ele não os vê). Aumenta a complexidade e observa.")
    grau = st.slider("Complexidade do modelo (grau do polinómio)", 1, 14, 1, key="l7_grau")

    p = _ajusta_polinomio(xtr, ytr, grau)
    e_tr = float(np.mean((p(xtr) - ytr) ** 2))
    e_te = float(np.mean((p(xte) - yte) ** 2))
    xs = np.linspace(0, 1, 300)

    fig, ax = _fig()
    ax.plot(xs, np.sin(2 * np.pi * xs), color=C_TEXT_MUTE, lw=1.6, ls=":", label="Verdade escondida")
    ax.scatter(xtr, ytr, c=COR_A, s=46, edgecolor="white", linewidth=.4, label="Treino (estudo)", zorder=4)
    ax.scatter(xte, yte, c=COR_B, s=34, alpha=.8, label="Teste (prova)", zorder=3)
    ax.plot(xs, np.clip(p(xs), -2.2, 2.2), color=C_GREEN, lw=2.4, label=f"Modelo (grau {grau})")
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _legenda(ax, loc="upper right", fontsize=8)
    _mostrar(fig)

    c1, c2 = st.columns(2)
    c1.metric("Erro no treino", f"{e_tr:.2f}", help="Quanto o modelo falha nos exemplos que estudou.")
    c2.metric("Erro no teste", f"{e_te:.2f}", help="Quanto falha nos casos novos. É este que importa.")

    if grau <= 2:
        _caixa("Diagnóstico: underfitting (modelo demasiado simples)",
               "O modelo nem sequer acompanha a forma dos dados: erra no treino <b>e</b> no teste. "
               "Falta complexidade.", C_AMBER)
    elif e_te > 3 * e_tr + 0.05 and grau >= 6:
        _caixa("Diagnóstico: overfitting (decorou)",
               "O erro no treino é pequeno mas o erro no teste disparou. A curva persegue o <b>ruído</b> dos "
               "pontos de estudo e falha em dados novos.", C_RED)
    else:
        _caixa("Diagnóstico: bom equilíbrio",
               "A curva acompanha a forma geral e erra pouco tanto no treino como no teste. É aqui que queres estar.", C_GREEN)


# ══════════════════════════════════════════════════════
# LIÇÃO 8 - Métricas
# ══════════════════════════════════════════════════════
def _licao_8():
    _caixa("Analogia",
           "Um teste rápido de malária é usado em 1000 pessoas, das quais 100 estão doentes. "
           "Um teste preguiçoso que diz sempre \"saudável\" <b>acerta em 90% dos casos</b>... e não encontra nenhum doente! "
           "Por isso precisamos de mais do que uma nota.", C_TEAL)

    rng = np.random.default_rng(21)
    doentes = rng.normal(.68, .17, 100)
    saudaveis = rng.normal(.35, .17, 900)
    scores = np.concatenate([doentes, saudaveis])
    real = np.array([1] * 100 + [0] * 900)

    _secao(1, "O teu teste: quão desconfiado queres que seja?")
    st.markdown("O modelo dá a cada pessoa uma pontuação de 0 a 1. Se passar do **limiar**, diz \"doente\". "
                "Dados fictícios.")
    c1, c2 = st.columns([3, 2])
    with c1:
        lim = st.slider("Limiar: a partir de que pontuação damos como doente?", 0.10, 0.90, 0.50, 0.01, key="l8_lim")
    with c2:
        preguicoso = st.checkbox("Usar o teste preguiçoso (diz sempre 'saudável')", key="l8_lazy")

    pred = np.zeros_like(real) if preguicoso else (scores > lim).astype(int)
    vp = int(((pred == 1) & (real == 1)).sum())
    fp = int(((pred == 1) & (real == 0)).sum())
    fn = int(((pred == 0) & (real == 1)).sum())
    vn = int(((pred == 0) & (real == 0)).sum())

    c1, c2 = st.columns(2)
    c1.metric("Doentes encontrados", f"{vp}", help="Verdadeiros positivos: estavam doentes e o teste detectou.")
    c2.metric("Falsos alarmes", f"{fp}", help="Falsos positivos: estavam saudáveis mas o teste disse doente.")
    c3, c4 = st.columns(2)
    c3.metric("Doentes que escaparam", f"{fn}", help="Falsos negativos: estavam doentes mas o teste disse saudável. O erro mais perigoso aqui.")
    c4.metric("Saudáveis correctos", f"{vn}", help="Verdadeiros negativos.")

    acc = (vp + vn) / 1000
    prec = vp / (vp + fp) if (vp + fp) else 0.0
    rec = vp / (vp + fn) if (vp + fn) else 0.0

    _secao(2, "O que cada nota quer dizer")
    st.markdown(f"""
| Nota | Valor | Em linguagem simples |
|---|---|---|
| **Accuracy** | {acc:.0%} | De todas as 1000 pessoas, o teste acertou em {acc:.0%}. |
| **Precision** | {prec:.0%} | Quando o teste diz "doente", acerta {prec:.0%} das vezes. |
| **Recall** | {rec:.0%} | Dos 100 doentes verdadeiros, o teste apanhou {rec:.0%}. |
""")
    if preguicoso:
        st.error("Repara: **accuracy 90%** parece óptima, mas o **recall é 0%**. O teste não ajuda ninguém. "
                 "Por isso a accuracy sozinha engana, sobretudo quando uma classe é rara.")
    elif rec < .6:
        st.warning("Muitos doentes a escapar. Baixa o limiar para o teste ficar mais desconfiado.")
    elif prec < .6:
        st.warning("Muitos falsos alarmes. Sobe o limiar para o teste ser mais exigente.")
    else:
        st.success("Bom equilíbrio entre apanhar doentes e não dar falsos alarmes.")

    _caixa("Qual escolher?",
           "Se <b>falhar um caso é grave</b> (doença, fraude) → vigia o <b>Recall</b>. "
           "Se <b>um falso alarme é caro</b> (bloquear um cliente honesto) → vigia a <b>Precision</b>. "
           "O <b>F1</b> junta as duas.", C_AMBER)


# ══════════════════════════════════════════════════════
# LIÇÃO 9 - Clustering
# ══════════════════════════════════════════════════════
def _licao_9():
    _caixa("Analogia",
           "Num mercado grande ninguém te diz quais são as zonas, mas ao andar vês que há a zona do peixe, "
           "a da fruta, a da roupa. Descobriste <b>grupos</b> sozinho. O <b>K-Means</b> faz isto com dados: "
           "tu dizes quantos grupos (K) queres e ele encontra os centros.", C_TEAL)

    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.1, random_state=8)
    _secao(1, "Quantos grupos existem nestes clientes?")
    k = st.slider("K = número de grupos pedidos", 1, 8, 2, key="l9_k")
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    cores = [C_ACCENT, C_AMBER, C_GREEN, C_RED, C_TEAL, "#BC8CFF", "#F778BA", "#8B949E"]

    fig, ax = _fig()
    for g in range(k):
        pts = X[km.labels_ == g]
        ax.scatter(*pts.T, c=cores[g], s=36, edgecolor="white", linewidth=.3, alpha=.9)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=240, c="white",
               edgecolor="black", linewidth=1.4, zorder=5, label="Centro do grupo")
    ax.set_xlabel("Gasto mensal")
    ax.set_ylabel("Visitas por mês")
    ax.set_title(f"Clientes separados em {k} grupo(s)")
    _legenda(ax, loc="upper left")
    _mostrar(fig)

    if k >= 2:
        s = silhouette_score(X, km.labels_)
        qual = "muito bem separados" if s > .6 else "razoavelmente separados" if s > .4 else "mal separados"
        st.metric("Qualidade da separação (silhouette)", f"{s:.2f}",
                  help="Vai de -1 a 1. Perto de 1 = grupos bem definidos e distintos.")
        st.markdown(f"Com {k} grupos, os clientes ficam **{qual}**.")
    else:
        st.info("Com 1 grupo não há separação nenhuma. Aumenta o K.")

    _caixa("Dica",
           "Não existe um K \"certo\" garantido. Experimenta vários e vê onde a qualidade é melhor. "
           "Aqui os dados foram feitos com <b>4 grupos reais</b>: vê o que acontece com K=4.", C_AMBER)


# ══════════════════════════════════════════════════════
# LIÇÃO 10 - Neurónio
# ══════════════════════════════════════════════════════
def _licao_10():
    _caixa("Analogia",
           "Decides se vais ao Mussulo no sábado. Pesas: <b>o tempo está bom?</b> <b>os amigos vão?</b> "
           "<b>tens dinheiro para o transporte?</b> Cada factor tem uma <b>importância</b> diferente para ti. "
           "Um neurónio artificial faz exactamente esta conta.", C_TEAL)

    _secao(1, "Constrói um neurónio de decisão")
    st.markdown("**1. As situações deste sábado** (0 = nada, 10 = perfeito)")
    c1, c2, c3 = st.columns(3)
    x = [c1.slider("Tempo bom", 0, 10, 8, key="l10_x1"),
         c2.slider("Amigos vão", 0, 10, 6, key="l10_x2"),
         c3.slider("Dinheiro para o transporte", 0, 10, 3, key="l10_x3")]
    st.markdown("**2. A importância que dás a cada factor** (os *pesos*; negativo = atrapalha)")
    c1, c2, c3 = st.columns(3)
    w = [c1.slider("Peso do tempo", -1.0, 1.0, 0.6, 0.1, key="l10_w1"),
         c2.slider("Peso dos amigos", -1.0, 1.0, 0.8, 0.1, key="l10_w2"),
         c3.slider("Peso do dinheiro", -1.0, 1.0, 0.4, 0.1, key="l10_w3")]
    b = st.slider("3. A tua preguiça (quanto maior, mais difícil é convencer-te)", 0.0, 15.0, 8.0, 0.5, key="l10_b")

    contrib = [wi * xi for wi, xi in zip(w, x)]
    z = sum(contrib) - b
    prob = 1 / (1 + np.exp(-z))

    fig, ax = _fig(7, 3.2)
    nomes = ["Tempo", "Amigos", "Dinheiro", "Preguiça"]
    vals = contrib + [-b]
    ax.barh(nomes, vals, color=[C_GREEN if v >= 0 else C_RED for v in vals])
    ax.axvline(0, color=C_TEXT_SEC, lw=1)
    ax.set_xlabel("Contributo para a decisão (peso × situação)")
    ax.invert_yaxis()
    _mostrar(fig)

    st.markdown(f"Soma total: **{z:+.1f}**. A rede transforma a soma numa probabilidade: **{prob:.0%}** de ir.")
    if prob >= .5:
        st.success("Decisão do neurónio: **vou ao Mussulo!**")
    else:
        st.error("Decisão do neurónio: **fico em casa.**")

    _secao(2, "Como se chega a uma rede neuronal")
    st.markdown("""
- Uma **rede neuronal** é muitos neurónios ligados em camadas: a saída de uns é a entrada de outros.
- O segredo está nos **pesos**. Ninguém os escolhe à mão: a rede começa com pesos ao acaso, vê onde erra e **ajusta os pesos pouco a pouco** até errar menos.
- É a mesma ideia da lição da linha recta, só que com milhões de ajustes. No módulo **MLP / Redes Neurais** podes ver isto a acontecer.
""")


# ══════════════════════════════════════════════════════
# LIÇÃO 11 - A receita completa
# ══════════════════════════════════════════════════════
def _licao_11():
    _caixa("A receita do funge",
           "Todo o projecto de ML segue uma receita. Se perceberes estes 7 passos, percebes qualquer projecto.", C_TEAL)
    passos = [
        ("Perceber o problema", "Que pergunta queres responder? Prever um número, uma categoria, achar grupos?"),
        ("Juntar os dados", "Uma tabela com exemplos suficientes e representativos."),
        ("Limpar os dados", "Tratar valores em falta, erros e colunas inúteis."),
        ("Separar treino e teste", "Guardar uma parte para a \"prova\"."),
        ("Treinar o modelo", "Deixar o algoritmo aprender com os dados de treino."),
        ("Avaliar com o teste", "Ver notas honestas: accuracy, recall, erro médio..."),
        ("Melhorar e usar", "Ajustar, comparar algoritmos e finalmente fazer previsões reais."),
    ]
    for i, (t, d) in enumerate(passos, 1):
        st.markdown(
            f'<div style="display:flex;gap:12px;align-items:flex-start;background:{C_SURFACE};border:1px solid {C_BORDER};'
            f'border-radius:10px;padding:.7rem 1rem;margin-bottom:.5rem;">'
            f'<div style="min-width:28px;height:28px;border-radius:50%;background:{C_ACCENT};color:#0D1117;font-weight:800;'
            f'display:flex;align-items:center;justify-content:center;">{i}</div>'
            f'<div><div style="font-weight:700;color:{C_TEXT};">{t}</div>'
            f'<div style="font-size:14px;color:{C_TEXT_SEC};">{d}</div></div></div>',
            unsafe_allow_html=True,
        )
    _caixa("Parabéns",
           "Já sabes o que a maioria das pessoas nunca aprende. Agora vai <b>praticar</b>: a melhor forma de aprender ML é "
           "treinar o teu primeiro modelo a sério. É só um clique.", C_GREEN)
    c1, c2, c3 = st.columns(3)
    if c1.button("Treinar o meu primeiro modelo", type="primary", key="l11_p1", width="stretch"):
        ir_para("primeiro_modelo")
    if c2.button("Qual algoritmo devo usar?", key="l11_p2", width="stretch"):
        ir_para("assistente")
    if c3.button("Explorar Classificação", key="l11_p3", width="stretch"):
        ir_para("classificacao")


# ══════════════════════════════════════════════════════
# CATÁLOGO DE LIÇÕES
# ══════════════════════════════════════════════════════
LICOES = [
    {"id": "l1", "titulo": "O que é Machine Learning?", "tempo": "5 min", "render": _licao_1,
     "ideia": "É ensinar o computador a aprender com exemplos, em vez de lhe dar regras escritas.",
     "resumo": ["Na programação clássica escreves as regras; no ML o computador descobre-as a partir de exemplos.",
                "Quanto melhores e mais variados os exemplos, melhor o modelo aprende."],
     "pratica": ("Experimentar na plataforma", "primeiro_modelo"),
     "quiz": [("Qual é a grande diferença entre a programação clássica e o Machine Learning?",
               "Na clássica escrevemos as regras; no ML o computador descobre as regras a partir de exemplos",
               ["O ML não precisa de dados", "O ML só funciona em telemóveis"]),
              ("Para ensinar um modelo a distinguir laranjas de limões, o que é mais importante?",
               "Muitos exemplos já identificados correctamente",
               ["Um computador muito caro", "Uma regra escrita à mão"])]},
    {"id": "l2", "titulo": "Os dados: linhas, colunas e alvo", "tempo": "4 min", "render": _licao_2,
     "ideia": "Os modelos aprendem com tabelas: cada linha é um exemplo e cada coluna é uma característica.",
     "resumo": ["Linha = um exemplo. Coluna = uma característica (feature).",
                "O alvo (target) é a coluna que queremos prever. Nunca a uses como feature.",
                "Dados maus dão modelos maus."],
     "pratica": None,
     "quiz": [("Numa tabela de dados, o que representa cada LINHA?",
               "Um exemplo (um caso, uma casa, um cliente)",
               ["Uma característica", "A resposta certa"]),
              ("Queremos prever o preço de casas. Qual é o alvo?",
               "A coluna Preço",
               ["A coluna Área", "O nome do bairro"])]},
    {"id": "l3", "titulo": "Três maneiras de aprender", "tempo": "4 min", "render": _licao_3,
     "ideia": "Há aprendizagem com professor, sem professor e por tentativa e erro.",
     "resumo": ["Supervisionada: tens exemplos com a resposta certa.",
                "Não supervisionada: só tens dados e procuras padrões ou grupos.",
                "Por reforço: um agente aprende com prémios e castigos."],
     "pratica": ("Ver os módulos", "dashboard"),
     "quiz": [("Tens milhares de clientes e queres separá-los em grupos parecidos, sem etiquetas. Que aprendizagem é esta?",
               "Não supervisionada", ["Supervisionada", "Por reforço"]),
              ("Um robô aprende a jogar ganhando pontos por boas jogadas. Isso é:",
               "Aprendizagem por reforço", ["Aprendizagem supervisionada", "Aprendizagem não supervisionada"])]},
    {"id": "l4", "titulo": "Prever números (Regressão)", "tempo": "6 min", "render": _licao_4,
     "ideia": "Regressão é prever um valor numérico, como um preço, uma temperatura ou umas vendas.",
     "resumo": ["O modelo traça uma linha (ou curva) que passa o mais perto possível dos pontos.",
                "O erro é a distância entre a linha e cada ponto. Aprender é reduzir esse erro."],
     "pratica": ("Praticar Regressão", "regressao"),
     "quiz": [("Qual destes problemas é de REGRESSÃO?",
               "Prever o preço de uma casa em Kwanzas",
               ["Dizer se um email é spam ou não", "Separar fotos de gatos e cães"]),
              ("Quando o erro médio da linha diminui, o modelo está...",
               "A ficar melhor", ["A ficar pior", "Na mesma"])]},
    {"id": "l5", "titulo": "Prever categorias (Classificação)", "tempo": "6 min", "render": _licao_5,
     "ideia": "Classificação é escolher uma categoria. O KNN decide pela votação dos vizinhos mais próximos.",
     "resumo": ["Classificar = responder \"de que grupo é este caso?\".",
                "No KNN, os K vizinhos mais próximos votam e a maioria ganha.",
                "K pequeno é nervoso; K grande é desatento."],
     "pratica": ("Praticar Classificação", "classificacao"),
     "quiz": [("No KNN com K=5, como é decidida a classe de um ponto novo?",
               "Pela votação dos 5 vizinhos mais próximos", ["Por sorteio", "Pela cor do ponto"]),
              ("Com K=1, o modelo fica...",
               "Muito sensível a pontos isolados (pode decorar)", ["Sempre perfeito", "Impossível de usar"])]},
    {"id": "l6", "titulo": "Treino e teste: a prova com perguntas novas", "tempo": "5 min", "render": _licao_6,
     "ideia": "Para saber se o modelo aprendeu mesmo, testa-o com casos que nunca viu.",
     "resumo": ["Treino = exercícios de estudo. Teste = prova com perguntas novas.",
                "Nunca avalies o modelo com os mesmos dados do treino."],
     "pratica": None,
     "quiz": [("Porque guardamos parte dos dados só para teste?",
               "Para ver como o modelo se sai com casos que nunca viu, como numa prova",
               ["Para poupar memória", "Porque os dados de treino estão errados"]),
              ("Um modelo tem 100% no treino e 70% no teste. O que isso sugere?",
               "Que decorou o treino (overfitting)", ["Que é perfeito", "Que os dados de teste estão errados"])]},
    {"id": "l7", "titulo": "Overfitting e underfitting", "tempo": "6 min", "render": _licao_7,
     "ideia": "Um modelo pode ser simples demais, complicado demais ou estar no ponto certo.",
     "resumo": ["Underfitting: simples demais, erra em tudo.",
                "Overfitting: decorou o treino, falha no teste.",
                "O objectivo é o equilíbrio: erro baixo no treino E no teste."],
     "pratica": None,
     "quiz": [("Um modelo demasiado simples, que erra no treino e no teste, sofre de:",
               "Underfitting", ["Overfitting", "Nada, está óptimo"]),
              ("Um modelo que acerta tudo no treino mas falha muito no teste sofre de:",
               "Overfitting", ["Underfitting", "Falta de memória"])]},
    {"id": "l8", "titulo": "Como saber se o modelo é bom", "tempo": "7 min", "render": _licao_8,
     "ideia": "Uma só nota engana. Accuracy, precision e recall contam histórias diferentes.",
     "resumo": ["Accuracy: percentagem total de acertos (engana quando uma classe é rara).",
                "Precision: quando diz \"sim\", quantas vezes tem razão.",
                "Recall: dos casos reais, quantos encontrou."],
     "pratica": ("Ver métricas num modelo real", "primeiro_modelo"),
     "quiz": [("Um teste diz sempre \"saudável\". Em 1000 pessoas, 100 estão doentes. A accuracy é 90%. É um bom teste?",
               "Não: não encontra nenhum doente (recall 0%)", ["Sim, 90% é óptimo", "Sim, porque acertou mais do que falhou"]),
              ("Quando falhar um caso é muito grave (ex.: doença), que nota devemos vigiar?",
               "O Recall", ["A cor do gráfico", "O número de colunas"])]},
    {"id": "l9", "titulo": "Agrupar sem professor (Clustering)", "tempo": "5 min", "render": _licao_9,
     "ideia": "O clustering encontra grupos nos dados sem ninguém dizer quais são.",
     "resumo": ["K-Means: tu escolhes K (nº de grupos) e ele encontra os centros.",
                "Não precisa de respostas certas, por isso é não supervisionado."],
     "pratica": ("Praticar Clustering", "clustering"),
     "quiz": [("No K-Means, o que significa K?",
               "O número de grupos que pedimos", ["O número de colunas", "A velocidade do computador"]),
              ("O K-Means precisa de etiquetas (respostas certas)?",
               "Não: descobre os grupos sozinho", ["Sim: de todas", "Sim: de metade"])]},
    {"id": "l10", "titulo": "O neurónio: o tijolo das redes neuronais", "tempo": "6 min", "render": _licao_10,
     "ideia": "Um neurónio soma factores multiplicados por pesos e decide. Uma rede junta muitos neurónios.",
     "resumo": ["Peso = a importância de cada entrada.",
                "A rede aprende ajustando os pesos automaticamente até errar menos."],
     "pratica": ("Ver uma rede a aprender", "mlp"),
     "quiz": [("O que faz um \"peso\" num neurónio?",
               "Diz quanta importância tem cada entrada", ["Guarda os dados", "Desliga o neurónio"]),
              ("Como aprende uma rede neuronal?",
               "Ajustando os pesos automaticamente para errar menos", ["Alguém escreve os pesos à mão", "Não aprende, só repete"])]},
    {"id": "l11", "titulo": "A receita completa e o teu próximo passo", "tempo": "3 min", "render": _licao_11,
     "ideia": "Todo o projecto de ML segue a mesma receita de 7 passos.",
     "resumo": ["Problema → Dados → Limpeza → Treino/Teste → Treinar → Avaliar → Melhorar e usar."],
     "pratica": None,
     "quiz": []},
]
LICAO_POR_ID = {l["id"]: l for l in LICOES}


# ══════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════
def _opcoes_embaralhadas(chave: str, certa: str, erradas: list) -> list:
    """Mistura as opções de forma estável (a certa nem sempre é a primeira)."""
    opcoes = [certa] + list(erradas)
    random.Random(zlib.crc32(chave.encode())).shuffle(opcoes)
    return opcoes


def _render_quiz(username: str, licao: dict):
    quiz = licao["quiz"]
    if not quiz:
        if st.button("Concluir a Escola de ML", type="primary", key=f"fim_{licao['id']}"):
            if concluir_licao(username, licao["id"], licao["titulo"], 10):
                st.balloons()
            st.rerun()
        return

    st.markdown("### Testa o que aprendeste")
    respostas = []
    for i, (pergunta, certa, erradas) in enumerate(quiz):
        chave = f"quiz_{licao['id']}_{i}"
        opcoes = _opcoes_embaralhadas(chave, certa, erradas)
        respostas.append(st.radio(f"{i + 1}. {pergunta}", opcoes, index=None, key=chave))

    if st.button("Verificar respostas", key=f"ver_{licao['id']}", type="primary"):
        st.session_state[f"verificado_{licao['id']}"] = True

    if st.session_state.get(f"verificado_{licao['id']}"):
        certas = 0
        for i, (pergunta, certa, _) in enumerate(quiz):
            if respostas[i] == certa:
                certas += 1
                st.success(f"Pergunta {i + 1}: certo!")
            elif respostas[i] is None:
                st.warning(f"Pergunta {i + 1}: ainda sem resposta.")
            else:
                st.error(f"Pergunta {i + 1}: não é bem isso. Resposta certa: **{certa}**.")
        if certas == len(quiz):
            if concluir_licao(username, licao["id"], licao["titulo"], 8):
                st.balloons()
                st.success("Lição concluída! +8 pontos.")
            else:
                st.success("Lição já concluída anteriormente.")


# ══════════════════════════════════════════════════════
# PÁGINAS
# ══════════════════════════════════════════════════════
def _proxima_nao_feita(concluidas: list):
    for l in LICOES:
        if l["id"] not in concluidas:
            return l["id"]
    return None


def _render_trilha(username: str):
    concluidas = get_licoes_concluidas(username)
    feitas = len([l for l in LICOES if l["id"] in concluidas])
    proxima = _proxima_nao_feita(concluidas)

    page_header("Escola de ML", "Aprende Machine Learning do zero, sem precisar de matemática avançada")
    progresso_bar(feitas / len(LICOES), f"{feitas} de {len(LICOES)} lições concluídas")

    if proxima:
        pl = LICAO_POR_ID[proxima]
        _caixa("A tua próxima lição" if feitas else "Começa por aqui",
               f"<b>{pl['titulo']}</b> - {pl['tempo']}<br><span style='color:{C_TEXT_SEC};'>{pl['ideia']}</span>", C_GREEN)
        if st.button("Continuar a aprender", type="primary", key="esc_continuar"):
            st.session_state.escola_aberta = proxima
            st.rerun()
    else:
        _caixa("Terminaste a Escola de ML!", "Agora pratica nos módulos da plataforma.", C_GREEN)

    st.markdown("### Todas as lições")
    cols = st.columns(2)
    for i, l in enumerate(LICOES):
        feita = l["id"] in concluidas
        cor = C_GREEN if feita else (C_ACCENT if l["id"] == proxima else C_BORDER)
        estado = "Concluída" if feita else ("Próxima" if l["id"] == proxima else l["tempo"])
        with cols[i % 2]:
            st.markdown(
                f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-left:4px solid {cor};'
                f'border-radius:10px;padding:.8rem 1rem;margin-top:.6rem;">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;font-weight:700;color:{C_TEXT_MUTE};">'
                f'<span>LIÇÃO {i + 1}</span><span style="color:{cor if cor != C_BORDER else C_TEXT_MUTE};">{estado}</span></div>'
                f'<div style="font-size:15px;font-weight:700;color:{C_TEXT};margin:3px 0;">{l["titulo"]}</div>'
                f'<div style="font-size:12.5px;color:{C_TEXT_SEC};line-height:1.5;">{l["ideia"]}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Rever" if feita else "Abrir", key=f"esc_abrir_{l['id']}", width="stretch"):
                st.session_state.escola_aberta = l["id"]
                st.rerun()


def _render_licao(username: str, lid: str):
    licao = LICAO_POR_ID[lid]
    idx = LICOES.index(licao)
    concluidas = get_licoes_concluidas(username)

    topo1, topo2 = st.columns([1, 3])
    with topo1:
        if st.button("← Todas as lições", key="esc_voltar"):
            st.session_state.escola_aberta = None
            st.rerun()
    with topo2:
        progresso_bar((idx + 1) / len(LICOES), f"Lição {idx + 1} de {len(LICOES)}")

    estado = "  (concluída)" if lid in concluidas else ""
    page_header(f"{idx + 1}. {licao['titulo']}{estado}", f"Tempo estimado: {licao['tempo']}")
    _caixa("A ideia numa frase", licao["ideia"], C_ACCENT)

    licao["render"]()

    st.markdown("---")
    st.markdown("### Em resumo")
    for ponto in licao["resumo"]:
        st.markdown(f"- {ponto}")

    _render_quiz(username, licao)

    if licao["pratica"]:
        rotulo, destino = licao["pratica"]
        st.markdown("---")
        if st.button(f"{rotulo} →", key=f"pratica_{lid}"):
            ir_para(destino)

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if idx > 0 and st.button("← Anterior", key="esc_ant", width="stretch"):
            st.session_state.escola_aberta = LICOES[idx - 1]["id"]
            st.rerun()
    with c3:
        if idx < len(LICOES) - 1 and st.button("Seguinte →", key="esc_seg", type="primary", width="stretch"):
            st.session_state.escola_aberta = LICOES[idx + 1]["id"]
            st.rerun()


def render_escola(username: str):
    aberta = st.session_state.get("escola_aberta")
    if aberta in LICAO_POR_ID:
        _render_licao(username, aberta)
    else:
        _render_trilha(username)
