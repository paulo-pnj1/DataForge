"""
DataForge EDU - O Teu Primeiro Modelo
Treina um modelo em 1 clique, sem configurar nada, e explica o resultado
em linguagem simples. Ideal para quem nunca treinou um modelo.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from modules.explicacoes import (
    em_cada_10, explicar_classificacao, explicar_regressao, frase_acerto, nivel_qualidade, nivel_r2,
)
from modules.utils import (
    page_header, section_title, carregar_dataset_embutido, ir_para,
    add_pontos, save_historico_modelo, save_user_progress, get_user_progress,
    C_SURFACE, C_BORDER, C_ACCENT, C_GREEN, C_AMBER, C_RED, C_TEAL,
    C_TEXT, C_TEXT_SEC,
)

# ══════════════════════════════════════════════════════
# PROBLEMAS DISPONÍVEIS (linguagem do dia-a-dia)
# ══════════════════════════════════════════════════════
PROBLEMAS = {
    "Prever o preço de uma casa em Luanda": {
        "dataset": "Casas em Luanda (fictício) - Preços", "tipo": "reg", "alvo": "preco_milhoes_kz",
        "pergunta": "Quanto custa esta casa?", "unidade": "milhões de Kz",
        "texto": "400 casas fictícias com bairro, quartos, área, distância ao centro, gerador e quintal. "
                 "O modelo aprende a estimar o preço. Funciona sem internet.",
    },
    "Prever se um aluno passa de ano": {
        "dataset": "Alunos em Angola (fictício) - Aprovação", "tipo": "clf", "alvo": "passou_de_ano",
        "pergunta": "Este aluno passa de ano?",
        "texto": "500 alunos fictícios com horas de estudo, faltas, nota anterior, se trabalha e se tem internet em casa. Funciona sem internet.",
    },
    "Prever as vendas de uma banca de rua": {
        "dataset": "Banca de rua (fictício) - Vendas", "tipo": "reg", "alvo": "vendas_kz",
        "pergunta": "Quanto vai vender a banca hoje?", "unidade": "Kz",
        "texto": "365 dias fictícios com dia da semana, chuva, fim de mês, temperatura e preço. Funciona sem internet.",
    },
    "Descobrir se um tumor é benigno ou maligno": {
        "dataset": "Cancro - Diagnóstico", "tipo": "clf", "alvo": "target",
        "pergunta": "Este tumor é benigno ou maligno?",
        "texto": "569 casos reais com 30 medidas de células. Um exemplo clássico de como o ML ajuda no diagnóstico.",
    },
    "Reconhecer a espécie de uma flor (Iris)": {
        "dataset": "Iris - Classificação de flores", "tipo": "clf", "alvo": "target",
        "pergunta": "Qual é a espécie desta flor?",
        "texto": "150 flores de 3 espécies, medidas em pétalas e sépalas. O exemplo mais famoso do Machine Learning.",
    },
    "Adivinhar quem sobreviveu ao Titanic": {
        "dataset": "Titanic - Sobrevivência", "tipo": "clf", "alvo": "survived",
        "pergunta": "Este passageiro sobreviveu?", "rotulos": {0: "Não sobreviveu", 1: "Sobreviveu"},
        "texto": "714 passageiros com classe, sexo, idade e bilhete. Precisa de internet na primeira utilização.",
    },
    "Classificar o tipo de um vinho": {
        "dataset": "Vinho - Qualidade", "tipo": "clf", "alvo": "target",
        "pergunta": "De que tipo é este vinho?", "rotulos": {0: "Tipo 1", 1: "Tipo 2", 2: "Tipo 3"},
        "texto": "178 vinhos com 13 propriedades químicas e 3 tipos.",
    },
    "Estimar a progressão da diabetes": {
        "dataset": "Diabetes - Progressão", "tipo": "reg", "alvo": "target",
        "pergunta": "Qual será a progressão da doença?", "unidade": "pontos de progressão",
        "texto": "442 pacientes com 10 medidas clínicas. O modelo estima a progressão ao fim de um ano.",
    },
}


def _fig_barras(nomes, valores, titulo):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    ax.barh(nomes[::-1], valores[::-1], color=C_ACCENT)
    ax.set_title(titulo, color=C_TEXT, fontsize=11)
    ax.tick_params(colors=C_TEXT_SEC, labelsize=9)
    for s in ax.spines.values():
        s.set_color(C_BORDER)
    ax.grid(alpha=.15, axis="x")
    fig.tight_layout()
    return fig


def _preparar(df: pd.DataFrame, alvo: str, rotulos=None):
    y = df[alvo]
    if rotulos:
        y = y.map(rotulos)
    X = pd.get_dummies(df.drop(columns=[alvo]), drop_first=False).astype(float)
    return X, y


# ══════════════════════════════════════════════════════
# TREINO
# ══════════════════════════════════════════════════════
def _treinar(nome: str, username: str) -> dict:
    cfg = PROBLEMAS[nome]
    df, _, _ = carregar_dataset_embutido(cfg["dataset"])
    if df is None:
        raise RuntimeError("Não foi possível carregar os dados.")
    df = df.dropna()
    X, y = _preparar(df, cfg["alvo"], cfg.get("rotulos"))
    clf = cfg["tipo"] == "clf"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y if clf else None)

    if clf:
        modelo = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)
        base = DummyClassifier(strategy="most_frequent")
    else:
        modelo = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=1)
        base = DummyRegressor(strategy="mean")
    modelo.fit(X_tr, y_tr)
    base.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    y_base = base.predict(X_te)

    res = {
        "nome": nome, "cfg": cfg, "modelo": modelo, "X_te": X_te, "y_te": y_te, "y_pred": y_pred,
        "n_treino": len(X_tr), "n_teste": len(X_te), "colunas": list(X.columns),
        "importancias": modelo.feature_importances_,
    }

    if clf:
        media = "weighted"
        res.update({
            "acc": accuracy_score(y_te, y_pred),
            "prec": precision_score(y_te, y_pred, average=media, zero_division=0),
            "rec": recall_score(y_te, y_pred, average=media, zero_division=0),
            "f1": f1_score(y_te, y_pred, average=media, zero_division=0),
            "acc_base": accuracy_score(y_te, y_base),
            "cv_scores": cross_val_score(modelo, X, y, cv=5, scoring="accuracy"),
            "classes": list(modelo.classes_), "auc": None,
        })
        try:
            proba = modelo.predict_proba(X_te)
            if len(modelo.classes_) == 2:
                res["auc"] = roc_auc_score(y_te, proba[:, 1])
            else:
                res["auc"] = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
        except Exception:
            pass
        principal = res["acc"]
        historico = {"tipo": "classificacao", "algoritmo": "Random Forest (Primeiro Modelo)",
                     "accuracy": round(res["acc"], 4), "f1": round(res["f1"], 4)}
    else:
        mse = mean_squared_error(y_te, y_pred)
        res.update({
            "r2": r2_score(y_te, y_pred), "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_te, y_pred),
            "mae_base": mean_absolute_error(y_te, y_base),
            "cv_scores": cross_val_score(modelo, X, y, cv=5, scoring="r2"),
        })
        principal = res["r2"]
        historico = {"tipo": "regressao", "algoritmo": "Random Forest (Primeiro Modelo)",
                     "r2": round(res["r2"], 4), "rmse": round(res["rmse"], 4)}

    # Guardar progresso (os pontos só contam da 1ª vez)
    primeira_vez = "primeiro_modelo" not in get_user_progress(username).get("modulos", {})
    save_historico_modelo(username, historico)
    if primeira_vez:
        add_pontos(username, 10, f"Primeiro modelo treinado: {nome}")
    save_user_progress(username, "primeiro_modelo", {"ultimo": nome, "resultado": round(float(principal), 4)})
    res["primeira_vez"] = primeira_vez
    return res


# ══════════════════════════════════════════════════════
# APRESENTAÇÃO
# ══════════════════════════════════════════════════════
def _cartao_destaque(titulo: str, grande: str, detalhe: str, cor: str):
    st.markdown(
        f'<div style="background:{C_SURFACE};border:2px solid {cor};border-radius:16px;padding:1.4rem 1.6rem;margin:.8rem 0;">'
        f'<div style="font-size:11px;font-weight:800;color:{cor};text-transform:uppercase;letter-spacing:.08em;">{titulo}</div>'
        f'<div style="font-size:26px;font-weight:800;color:{C_TEXT};margin:6px 0;line-height:1.3;">{grande}</div>'
        f'<div style="font-size:15px;color:{C_TEXT_SEC};line-height:1.65;">{detalhe}</div></div>',
        unsafe_allow_html=True,
    )


def _mostrar_resultado(r: dict):
    cfg = r["cfg"]
    clf = cfg["tipo"] == "clf"

    if r.get("primeira_vez"):
        st.success("Parabéns, treinaste o teu primeiro modelo! +10 pontos.")

    st.caption(f"O modelo estudou {r['n_treino']} casos e foi testado em {r['n_teste']} casos que nunca tinha visto.")

    if clf:
        rotulo, cor = nivel_qualidade(r["acc"])
        cor_hex = {"green": C_GREEN, "amber": C_AMBER, "red": C_RED}[cor]
        ganho = r["acc"] - r["acc_base"]
        _cartao_destaque(
            f"Resultado: {rotulo}",
            f"Acertou {em_cada_10(r['acc'])} casos novos ({r['acc']:.0%})",
            f"Um palpite sem inteligência (responder sempre a classe mais comum) acertaria <b>{r['acc_base']:.0%}</b>. "
            + (f"O teu modelo é <b>{ganho * 100:.0f} pontos percentuais melhor</b>." if ganho > .005
               else "Neste caso o modelo não bate o palpite simples: ou os dados dizem pouco, ou a classe mais comum domina."),
            cor_hex)
    else:
        rotulo, cor = nivel_r2(r["r2"])
        cor_hex = {"green": C_GREEN, "amber": C_AMBER, "red": C_RED}[cor]
        reducao = 1 - r["mae"] / r["mae_base"] if r["mae_base"] else 0
        _cartao_destaque(
            f"Resultado: {rotulo}",
            f"Erra em média {r['mae']:,.1f} {cfg['unidade']}",
            f"Se previssemos sempre a média, o erro médio seria <b>{r['mae_base']:,.1f}</b>. "
            f"O teu modelo reduziu o erro em <b>{reducao:.0%}</b> e explica <b>{max(r['r2'], 0):.0%}</b> do que faz o valor variar.",
            cor_hex)

    # Importância das colunas
    section_title("O QUE O MODELO ACHOU MAIS IMPORTANTE")
    ordem = np.argsort(r["importancias"])[::-1][:6]
    nomes = [r["colunas"][i] for i in ordem]
    vals = [float(r["importancias"][i]) for i in ordem]
    st.pyplot(_fig_barras(nomes, vals, "As colunas que mais pesaram nas decisões"))
    plt.close("all")
    st.markdown(f"A coluna que mais pesou foi **{nomes[0]}** ({vals[0]:.0%} da influência). "
                "Isto ajuda a perceber *porquê* o modelo decide como decide.")

    # Testa um caso
    section_title("TESTA TU: UM CASO REAL")
    st.markdown(f"*{cfg['pergunta']}* Sorteia um caso do conjunto de teste e vê se o modelo acerta.")
    if st.button("Sortear um caso", key="pm_sortear", type="primary"):
        st.session_state["pm_caso"] = int(np.random.randint(0, r["n_teste"]))
    i = st.session_state.get("pm_caso")
    if i is not None and i < r["n_teste"]:
        linha = r["X_te"].iloc[[i]]
        real, prev = r["y_te"].iloc[i], r["y_pred"][i]
        with st.expander("Ver os dados deste caso", expanded=False):
            st.dataframe(linha.T.rename(columns={linha.index[0]: "valor"}), width="stretch")
        if clf:
            proba = r["modelo"].predict_proba(linha)[0]
            confianca = float(np.max(proba))
            c1, c2 = st.columns(2)
            c1.metric("O modelo previu", str(prev), f"{confianca:.0%} de confiança", delta_color="off")
            c2.metric("A resposta certa era", str(real))
            if prev == real:
                st.success("Acertou!")
            else:
                st.error("Errou neste caso. Acontece: nenhum modelo é perfeito. Olha para a confiança: quando é baixa, o modelo já 'desconfiava'.")
        else:
            erro = abs(prev - real)
            c1, c2, c3 = st.columns(3)
            c1.metric("O modelo previu", f"{prev:,.1f}")
            c2.metric("O valor real era", f"{real:,.1f}")
            c3.metric("Diferença", f"{erro:,.1f}")
            if erro <= r["mae"]:
                st.success(f"Boa previsão: o erro ({erro:,.1f}) ficou abaixo do erro médio do modelo ({r['mae']:,.1f}).")
            else:
                st.warning(f"Aqui errou mais do que o habitual (o erro médio é {r['mae']:,.1f}).")

    # Explicação
    section_title("PERCEBER OS NÚMEROS")
    if clf:
        explicar_classificacao(r)
    else:
        explicar_regressao(r["y_te"], r["y_pred"], r["r2"], r["rmse"], r["mae"], r["cv_scores"], cfg["pergunta"].rstrip("?").lower())

    # Próximos passos
    section_title("E AGORA?")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Aprender o que isto significa", key="pm_escola", width="stretch"):
            st.session_state.escola_aberta = "l8"
            ir_para("escola")
    with c2:
        destino = "classificacao" if clf else "regressao"
        if st.button("Experimentar outros algoritmos", key="pm_mais", width="stretch"):
            ir_para(destino)
    with c3:
        if st.button("Comparar algoritmos", key="pm_comp", width="stretch"):
            ir_para("comparador")


def render_primeiro_modelo(username: str):
    page_header("O Teu Primeiro Modelo", "Treina um modelo de Machine Learning em 1 clique, sem configurar nada")

    st.markdown(
        f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-left:4px solid {C_TEAL};border-radius:10px;'
        f'padding:1rem 1.2rem;margin-bottom:1rem;font-size:15px;color:{C_TEXT};line-height:1.7;">'
        '<b>Como funciona:</b> 1) escolhes um problema, 2) carregas no botão, 3) o computador estuda uma parte dos dados '
        'e faz uma <i>prova</i> com a outra parte. No fim explico-te o resultado em palavras simples.</div>',
        unsafe_allow_html=True)

    st.markdown("#### 1. Escolhe um problema")
    nome = st.radio("Problema", list(PROBLEMAS.keys()), key="pm_problema", label_visibility="collapsed")
    st.caption(PROBLEMAS[nome]["texto"])

    st.markdown("#### 2. Treina")
    if st.button("Treinar o modelo agora", type="primary", key="pm_treinar"):
        with st.spinner("O computador está a estudar..."):
            try:
                st.session_state["pm_res"] = _treinar(nome, username)
                st.session_state.pop("pm_caso", None)
            except Exception as e:
                st.session_state.pop("pm_res", None)
                st.error("Não consegui treinar este problema. Se for o Titanic, precisas de internet na primeira vez; "
                         f"tenta outro problema. (Detalhe: {e})")

    res = st.session_state.get("pm_res")
    if res and res["nome"] == nome:
        st.markdown("#### 3. Resultado")
        _mostrar_resultado(res)
    elif res:
        st.info("Mudaste de problema. Carrega em **Treinar o modelo agora** para ver o novo resultado.")
