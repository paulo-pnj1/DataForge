"""
DataForge EDU - Projecto Final guiado
Um projecto completo em 5 passos: dados -> colunas -> comparar -> avaliar -> relatório.
"""
import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from modules.diagnostico import mostrar_diagnostico
from modules.settings import PONTOS_PROJETO
from modules.utils import (
    page_header, carregar_dataset_embutido, add_pontos, marcar_marco, get_user_progress, _gravar_progress,
    progresso_bar, ir_para,
)

PROJETOS = {
    "Prever se um aluno passa de ano (classificação)": ("Alunos em Angola (fictício) - Aprovação", "clf",
        "Uma escola quer identificar cedo os alunos em risco para os apoiar."),
    "Prever o preço de uma casa em Luanda (regressão)": ("Casas em Luanda (fictício) - Preços", "reg",
        "Uma imobiliária quer estimar preços justos para os seus anúncios."),
    "Prever as vendas de uma banca de rua (regressão)": ("Banca de rua (fictício) - Vendas", "reg",
        "Uma vendedora quer saber quanto leva para vender e quanto stock preparar."),
}
PASSOS = ["Problema e dados", "Escolher colunas", "Comparar algoritmos", "Avaliar e concluir", "Relatório"]


def _modelos(tipo):
    if tipo == "clf":
        return {"Logistic Regression": lambda: LogisticRegression(max_iter=1000),
                "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
                "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42)}
    return {"Linear Regression": lambda: LinearRegression(),
            "Decision Tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=42),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42)}


def _prep(df, feats, alvo, tipo):
    X = pd.get_dummies(df[feats], drop_first=False).astype(float)
    y = df[alvo]
    if tipo == "clf":
        y = LabelEncoder().fit_transform(y.astype(str))
    return X, np.asarray(y)


def _l1(t):
    from modules.i18n import tr
    return str(tr(str(t))).replace("**", "").encode("latin-1", "replace").decode("latin-1")


def _pdf(P, nome_aluno):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()

    def tit(t, tam=13):
        pdf.set_font("Helvetica", "B", tam)
        pdf.set_text_color(20, 60, 120)
        pdf.multi_cell(0, 8, _l1(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    def txt(t):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _l1(t), new_x="LMARGIN", new_y="NEXT")

    tit("Projecto Final de Machine Learning", 17)
    txt(f"Autor: {nome_aluno}  |  Data: {datetime.date.today():%d/%m/%Y}  |  DataForge EDU")
    pdf.ln(2)
    tit("1. Problema")
    txt(P["contexto"] + f" Dados: {P['dataset']} ({P['n']} exemplos). Alvo: {P['alvo']}.")
    pdf.ln(1)
    tit("2. Colunas usadas")
    txt(", ".join(P["feats"]))
    pdf.ln(1)
    tit("3. Comparação de algoritmos (média de várias provas)")
    for n, v in P["cv"].items():
        txt(f"- {n}: {v:.3f}")
    pdf.ln(1)
    tit("4. Modelo escolhido e avaliação")
    txt(f"Modelo: {P['escolhido']}.  " + "  ".join(f"{k}: {v:.3f}" for k, v in P["metricas"].items()))
    pdf.ln(1)
    tit("5. A minha conclusão")
    txt(P["conclusao"])
    pdf.ln(1)
    tit("6. O que cada linha dos dados representa")
    txt(P["linha"] or "-")
    return bytes(pdf.output())


def render_projeto(username: str, nome: str):
    page_header("Projecto Final", "Um projecto completo de Machine Learning, passo a passo")
    P = st.session_state.setdefault("proj", {"passo": 0})
    passo = P["passo"]
    progresso_bar(passo / (len(PASSOS) - 1), f"Passo {passo + 1} de {len(PASSOS)}: {PASSOS[passo]}")

    # ── 1 ──
    if passo == 0:
        esc = st.radio("Escolhe o teu projecto", list(PROJETOS.keys()), key="proj_esc")
        ds, tipo, ctx = PROJETOS[esc]
        st.info(ctx)
        df, _, _ = carregar_dataset_embutido(ds)
        st.dataframe(df.head(8), width="stretch", hide_index=True)
        mostrar_diagnostico(df, expandido=False)
        P["linha"] = st.text_area("Com as tuas palavras: o que representa cada linha desta tabela?", value=P.get("linha", ""), key="proj_linha")
        if st.button("Começar este projecto", type="primary", key="proj_p0") and len(P["linha"].strip()) >= 5:
            P.update({"dataset": ds, "tipo": tipo, "contexto": ctx, "alvo": df.columns[-1], "n": len(df), "passo": 1})
            st.rerun()
        st.caption("Escreve a descrição (mínimo 5 caracteres) para avançar.")

    # ── 2 ──
    elif passo == 1:
        df, _, _ = carregar_dataset_embutido(P["dataset"])
        cols = [c for c in df.columns if c != P["alvo"]]
        st.markdown(f"O alvo é **{P['alvo']}**. Que colunas queres usar para o prever?")
        st.caption("Dica: lembra-te da lição sobre fuga de informação. Usa só colunas que terias disponíveis na hora de prever.")
        P["feats"] = st.multiselect("Colunas", cols, default=P.get("feats", cols), key="proj_feats")
        c1, c2 = st.columns(2)
        if c1.button("Voltar", key="proj_b1"):
            P["passo"] = 0
            st.rerun()
        if c2.button("Seguinte", type="primary", key="proj_p1"):
            if P["feats"]:
                P["passo"] = 2
                st.rerun()
            st.warning("Escolhe pelo menos uma coluna.")

    # ── 3 ──
    elif passo == 2:
        df, _, _ = carregar_dataset_embutido(P["dataset"])
        X, y = _prep(df, P["feats"], P["alvo"], P["tipo"])
        cv = (StratifiedKFold(5, shuffle=True, random_state=42) if P["tipo"] == "clf" else KFold(5, shuffle=True, random_state=42))
        sc = "accuracy" if P["tipo"] == "clf" else "r2"
        st.markdown("Vamos pôr 3 algoritmos a fazer a mesma prova (5 provas repetidas).")
        if st.button("Comparar agora", type="primary", key="proj_cmp"):
            with st.spinner("A comparar..."):
                P["cv"] = {n: float(cross_val_score(make_pipeline(StandardScaler(), f()), X, y, cv=cv, scoring=sc).mean())
                           for n, f in _modelos(P["tipo"]).items()}
        if P.get("cv"):
            st.bar_chart(pd.Series(P["cv"], name="Nota (CV)"))
            melhor = max(P["cv"], key=P["cv"].get)
            st.success(f"Melhor nas provas: **{melhor}** ({P['cv'][melhor]:.3f}).")
            P["escolhido"] = st.selectbox("Que modelo escolhes?", list(P["cv"].keys()),
                                          index=list(P["cv"].keys()).index(P.get("escolhido", melhor)), key="proj_esc_mod")
        c1, c2 = st.columns(2)
        if c1.button("Voltar", key="proj_b2"):
            P["passo"] = 1
            st.rerun()
        if P.get("cv") and c2.button("Seguinte", type="primary", key="proj_p2"):
            P["passo"] = 3
            st.rerun()

    # ── 4 ──
    elif passo == 3:
        df, _, _ = carregar_dataset_embutido(P["dataset"])
        X, y = _prep(df, P["feats"], P["alvo"], P["tipo"])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=42, stratify=y if P["tipo"] == "clf" else None)
        m = make_pipeline(StandardScaler(), _modelos(P["tipo"])[P["escolhido"]]()).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        if P["tipo"] == "clf":
            P["metricas"] = {"Accuracy": accuracy_score(y_te, pred), "F1": f1_score(y_te, pred, average="weighted")}
        else:
            P["metricas"] = {"R2": r2_score(y_te, pred), "Erro médio (MAE)": mean_absolute_error(y_te, pred)}
        st.markdown(f"**{P['escolhido']}** avaliado em casos que nunca viu:")
        cols = st.columns(len(P["metricas"]))
        for c, (k, v) in zip(cols, P["metricas"].items()):
            c.metric(k, f"{v:.3f}")
        P["conclusao"] = st.text_area("Conclusão (mínimo 40 caracteres): que modelo escolheste e porquê? Que cuidado terias antes de o usar na vida real?",
                                      value=P.get("conclusao", ""), key="proj_concl", height=140)
        c1, c2 = st.columns(2)
        if c1.button("Voltar", key="proj_b3"):
            P["passo"] = 2
            st.rerun()
        if c2.button("Concluir o projecto", type="primary", key="proj_p3"):
            if len(P["conclusao"].strip()) < 40:
                st.warning("Escreve uma conclusão com pelo menos 40 caracteres.")
            else:
                if marcar_marco(username, "projeto"):
                    add_pontos(username, PONTOS_PROJETO, "Projecto final concluído")
                    st.balloons()
                prog = get_user_progress(username)
                prog["projeto"] = {"dataset": P["dataset"], "modelo": P["escolhido"], "conclusao": P["conclusao"],
                                   "data": datetime.date.today().isoformat()}
                _gravar_progress(username, prog)
                P["passo"] = 4
                st.rerun()

    # ── 5 ──
    else:
        st.success("Projecto concluído! Parabéns.")
        st.download_button("Descarregar o relatório do projecto (PDF)", _pdf(P, nome), file_name="projecto_final_ml.pdf",
                           mime="application/pdf", key="proj_pdf")
        c1, c2 = st.columns(2)
        if c1.button("Ver os meus certificados", type="primary", key="proj_cert"):
            ir_para("certificados")
        if c2.button("Fazer outro projecto", key="proj_novo"):
            st.session_state["proj"] = {"passo": 0}
            st.rerun()
