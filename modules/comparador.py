"""
DataForge EDU - Comparador de Modelos
Treina até 3 algoritmos lado a lado e compara métricas
"""

import io
import pickle
import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER, C_SURFACE2,
    PALETTE, carregar_dataset_embutido, add_pontos, TEORIA, marcar_marco
)
from modules.settings import PONTOS_COMPARACAO, PONTOS_PALPITE
from modules.diagnostico import mostrar_diagnostico

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

CLF_MODELS = {
    "KNN":                  lambda: KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":        lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":        lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression":  lambda: LogisticRegression(max_iter=500, random_state=42),
    "SVM (RBF)":            lambda: SVC(probability=True, random_state=42),
    "Naive Bayes":          lambda: GaussianNB(),
}
if HAS_XGB:
    CLF_MODELS["XGBoost"] = lambda: XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss")
if HAS_LGB:
    CLF_MODELS["LightGBM"] = lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

REG_MODELS = {
    "Linear Regression":    lambda: LinearRegression(),
    "Ridge":                lambda: Ridge(alpha=1.0),
    "KNN Regressor":        lambda: KNeighborsRegressor(n_neighbors=5),
    "Decision Tree":        lambda: DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":        lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR":                  lambda: SVR(),
}
if HAS_XGB:
    REG_MODELS["XGBoost"] = lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
if HAS_LGB:
    REG_MODELS["LightGBM"] = lambda: LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)


def _bar_comparison(nomes, valores, titulo, metrica):
    cores = [PALETTE[i % len(PALETTE)] for i in range(len(nomes))]
    fig = go.Figure(go.Bar(
        x=nomes, y=valores, marker_color=cores,
        text=[f"{v:.4f}" for v in valores],
        textposition="auto",
        textfont=dict(color="#FFFFFF", size=13, family="Atkinson Hyperlegible")
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(title=metrica, color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _radar_chart(nomes, metricas_dict):
    """Radar chart para comparação multidimensional"""
    cats = list(metricas_dict.keys())
    fig = go.Figure()
    for i, nome in enumerate(nomes):
        vals = [metricas_dict[c][i] for c in cats]
        vals.append(vals[0])  # fechar o polígono
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=nome,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fillcolor=f"rgba({','.join(str(int(PALETTE[i%len(PALETTE)].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.15)"
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color="#7A8BA8", gridcolor=C_BORDER),
            angularaxis=dict(color="#FFFFFF")
        ),
        paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
        legend=dict(font=dict(color="#FFFFFF")),
        title=dict(text="Comparação Multidimensional", font=dict(color="#FFFFFF", size=14))
    )
    return fig


from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════
# PERSONALIDADE DE CADA ALGORITMO
# explicabilidade: 3 = fácil de explicar, 1 = caixa negra
# ══════════════════════════════════════════════════════
PERSONALIDADE = {
    "KNN": dict(analogia="Pergunta aos vizinhos mais próximos e segue a maioria.", forte="Simples; bom com poucos dados e fronteiras irregulares.",
                fraco="Guarda todos os dados, por isso fica lento e pesado em tabelas grandes.", expl=3, forma="Manchas à volta dos pontos conhecidos."),
    "Decision Tree": dict(analogia="Jogo das 20 perguntas: cada pergunta divide os casos em dois grupos.", forte="Muito fácil de explicar a qualquer pessoa.",
                fraco="Tende a decorar os dados de estudo (overfitting).", expl=3, forma="Retângulos, porque corta sempre em linha recta."),
    "Random Forest": dict(analogia="100 especialistas, cada um viu dados diferentes, e votam.", forte="Robusto e quase não precisa de ajustes.",
                fraco="Mais pesado e mais difícil de explicar do que uma só árvore.", expl=2, forma="Retângulos suavizados pela votação."),
    "Gradient Boosting": dict(analogia="Aprende com os erros: cada novo modelo corrige o anterior.", forte="Costuma ser dos mais precisos em tabelas.",
                fraco="Mais lento a treinar e sensível a ruído.", expl=1, forma="Retângulos finos e muito ajustados."),
    "Logistic Regression": dict(analogia="Pesa cada coluna, soma tudo e transforma numa probabilidade.", forte="Rápida, leve e explicável.",
                fraco="Só separa classes com linhas rectas.", expl=3, forma="Uma única linha recta a separar."),
    "SVM (RBF)": dict(analogia="Procura a fronteira com a maior margem possível entre as classes.", forte="Ótimo com poucos dados e fronteiras curvas.",
                fraco="Lento em tabelas grandes e difícil de explicar.", expl=1, forma="Curvas suaves."),
    "Naive Bayes": dict(analogia="Cada coluna dá o seu voto, como se não se conhecessem umas às outras.", forte="Muito rápido e funciona com poucos dados.",
                fraco="Assume que as colunas são independentes, o que raramente é verdade.", expl=3, forma="Elipses suaves à volta de cada classe."),
    "XGBoost": dict(analogia="Boosting turbinado: muitas árvores pequenas, cada uma a corrigir as anteriores.", forte="Muito preciso em tabelas.",
                fraco="Tem muitos botões para ajustar e é difícil de explicar.", expl=1, forma="Retângulos finos e muito ajustados."),
    "LightGBM": dict(analogia="Boosting rápido: árvores pequenas a corrigir os erros das anteriores.", forte="Muito rápido em tabelas grandes.",
                fraco="Pode decorar com poucos dados.", expl=1, forma="Retângulos finos e muito ajustados."),
    "Linear Regression": dict(analogia="Traça a melhor linha recta pelos pontos.", forte="Simples, rápida e muito fácil de explicar.",
                fraco="Só apanha relações em linha recta.", expl=3, forma="Uma recta."),
    "Ridge": dict(analogia="Linha recta com um travão: não deixa nenhum peso ficar exagerado.", forte="Estável quando as colunas se parecem entre si.",
                fraco="Continua a ser uma recta.", expl=3, forma="Uma recta, ligeiramente mais calma."),
    "KNN Regressor": dict(analogia="Faz a média dos vizinhos mais parecidos.", forte="Simples e flexível.",
                fraco="Pesado e lento com muitos dados.", expl=3, forma="Uma curva em degraus que segue os vizinhos."),
    "SVR": dict(analogia="Procura uma curva que passa 'dentro de um tubo' à volta dos pontos.", forte="Bom com poucos dados.",
                fraco="Lento e difícil de explicar.", expl=1, forma="Curva suave."),
}
PERSONALIDADE_BASE = dict(analogia="Algoritmo de aprendizagem.", forte="-", fraco="-", expl=2, forma="-")
ENSEMBLES = {"Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"}
LINEARES = {"Logistic Regression", "Linear Regression", "Ridge"}
EXPL_TXT = {1: "Baixa", 2: "Média", 3: "Alta"}


def _pers(nome):
    return PERSONALIDADE.get(nome, PERSONALIDADE_BASE)


# ══════════════════════════════════════════════════════
# HIPERPARÂMETRO PRINCIPAL (para "Mexer e ver")
# (parâmetro, rótulo, valores, fábrica, complexidade_cresce, dica)
# ══════════════════════════════════════════════════════
HYPER_CLF = {
    "KNN": ("K", "K (nº de vizinhos)", list(range(1, 31)), lambda v: KNeighborsClassifier(n_neighbors=v), False,
            "K pequeno olha para poucos vizinhos e decora; K grande olha para tantos que perde os detalhes."),
    "Decision Tree": ("depth", "Profundidade máxima da árvore", list(range(1, 16)), lambda v: DecisionTreeClassifier(max_depth=v, random_state=42), True,
                      "Poucas perguntas = simples demais; muitas perguntas = decora os exemplos."),
    "Random Forest": ("depth", "Profundidade de cada árvore", list(range(1, 16)), lambda v: RandomForestClassifier(n_estimators=60, max_depth=v, random_state=42), True,
                      "Árvores curtas aprendem pouco; árvores fundas aprendem muito, mas a votação ajuda a controlar o excesso."),
    "Gradient Boosting": ("n", "Número de rondas de correção", [5, 10, 20, 40, 60, 80, 120, 200], lambda v: GradientBoostingClassifier(n_estimators=v, random_state=42), True,
                          "Cada ronda corrige os erros da anterior. Demasiadas rondas podem começar a decorar."),
    "Logistic Regression": ("C", "C (liberdade do modelo)", [0.001, 0.01, 0.1, 1, 10, 100, 1000], lambda v: LogisticRegression(C=v, max_iter=1000), True,
                            "C pequeno trava o modelo; C grande dá-lhe liberdade total."),
    "SVM (RBF)": ("C", "C (liberdade do modelo)", [0.001, 0.01, 0.1, 1, 10, 100, 1000], lambda v: SVC(C=v), True,
                  "C pequeno faz uma fronteira simples; C grande tenta acertar todos os pontos de estudo."),
}
HYPER_REG = {
    "KNN Regressor": ("K", "K (nº de vizinhos)", list(range(1, 31)), lambda v: KNeighborsRegressor(n_neighbors=v), False,
                      "K pequeno segue cada ponto; K grande faz médias demasiado largas."),
    "Decision Tree": ("depth", "Profundidade máxima da árvore", list(range(1, 16)), lambda v: DecisionTreeRegressor(max_depth=v, random_state=42), True,
                      "Poucas perguntas = simples demais; muitas = decora os exemplos."),
    "Random Forest": ("depth", "Profundidade de cada árvore", list(range(1, 16)), lambda v: RandomForestRegressor(n_estimators=60, max_depth=v, random_state=42), True,
                      "Árvores curtas aprendem pouco; árvores fundas aprendem muito."),
    "Gradient Boosting": ("n", "Número de rondas de correção", [5, 10, 20, 40, 60, 80, 120, 200], lambda v: GradientBoostingRegressor(n_estimators=v, random_state=42), True,
                          "Cada ronda corrige os erros da anterior."),
    "Ridge": ("alpha", "Alpha (força do travão)", [0.001, 0.01, 0.1, 1, 10, 100, 1000], lambda v: Ridge(alpha=v), False,
              "Alpha grande trava tanto o modelo que ele fica simples demais."),
    "SVR": ("C", "C (liberdade do modelo)", [0.1, 1, 10, 100, 1000], lambda v: SVR(C=v), True,
            "C pequeno dá uma curva muito rígida; C grande segue os pontos de estudo."),
}


def _pipe(factory):
    return make_pipeline(StandardScaler(), factory())


def _fmt(tipo, v):
    return f"{v:.1%}" if tipo == "Classificação" else f"{v:.2f}"


# ══════════════════════════════════════════════════════
# TREINO E ANÁLISE
# ══════════════════════════════════════════════════════
def _preparar_dados(df, features, target, tipo):
    X = df[features].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    y = df[target].copy()
    classes = None
    if tipo == "Classificação":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        classes = [str(c) for c in le.classes_]
    else:
        y = pd.to_numeric(y, errors="coerce")
        y = y.fillna(y.median()).to_numpy(dtype=float)
    return X, np.asarray(y), classes


def _run_comparison(df, features, target, algos, models_dict, tipo, test_size, cv_folds, username, dataset, palpite):
    try:
        X, y, classes = _preparar_dados(df, features, target, tipo)
        clf = tipo == "Classificação"
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42,
                                                      stratify=y if clf else None)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
        cv = (StratifiedKFold(cv_folds, shuffle=True, random_state=42) if clf
              else KFold(cv_folds, shuffle=True, random_state=42))

        resultados, preds, modelos = [], {}, {}
        barra = st.progress(0)
        for i, nome in enumerate(algos):
            with st.spinner(f"A treinar {nome}..."):
                m = _pipe(models_dict[nome])
                t0 = time.perf_counter()
                m.fit(X_tr, y_tr)
                tempo = time.perf_counter() - t0
                y_pred = m.predict(X_te)
                treino = float(m.score(X_tr, y_tr))
                kb = len(pickle.dumps(m)) / 1024
                cv_val = float(cross_val_score(_pipe(models_dict[nome]), X, y, cv=cv,
                                               scoring="accuracy" if clf else "r2").mean())
                if clf:
                    linha = {"Algoritmo": nome, "Accuracy": accuracy_score(y_te, y_pred),
                             "F1 Score": f1_score(y_te, y_pred, average="weighted", zero_division=0),
                             "CV Accuracy": cv_val, "Treino": treino}
                else:
                    linha = {"Algoritmo": nome, "R²": r2_score(y_te, y_pred),
                             "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
                             "CV R²": cv_val, "Treino": treino}
                linha["Tempo (s)"] = tempo
                linha["Tamanho (KB)"] = kb
                resultados.append(linha)
                preds[nome] = y_pred
                modelos[nome] = m
            barra.progress((i + 1) / len(algos))

        corr = X.corr().abs().to_numpy()
        corr_media = float(np.nanmean(corr[np.triu_indices_from(corr, k=1)])) if X.shape[1] > 1 else 0.0

        res = {
            "id": time.time(), "tipo": tipo, "dataset": dataset, "alvo": target, "resultados": resultados, "algos": algos,
            "palpite": palpite, "features": list(X.columns), "classes": classes,
            "X": X, "y": y, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
            "preds": preds, "modelos": modelos, "cv": cv,
            "n_linhas": len(X), "n_feat": X.shape[1], "corr_media": corr_media,
        }
        add_pontos(username, PONTOS_COMPARACAO, f"Comparação de {len(algos)} modelos")
        marcar_marco(username, "comparador")
        an = _analise(res)
        if palpite and palpite != "(sem palpite)" and palpite == an["campeao"]:
            add_pontos(username, PONTOS_PALPITE, "Acertou no palpite do comparador")
            res["ganhou_palpite"] = True
        st.session_state.cmp_result = res
        st.session_state.pop("cmp_sweep", None)
        st.rerun()
    except Exception as e:
        erro_box(f"Erro: {e}")


def _analise(res) -> dict:
    """Toda a interpretação em texto simples (usada no ecrã e no relatório PDF)."""
    tipo = res["tipo"]
    clf = tipo == "Classificação"
    df = pd.DataFrame(res["resultados"]).set_index("Algoritmo")
    c_teste, c_cv = ("Accuracy", "CV Accuracy") if clf else ("R²", "CV R²")
    ordem = df.sort_values(c_cv, ascending=False)
    campeao = ordem.index[0]
    segundo = ordem.index[1] if len(ordem) > 1 else None
    rapido = df["Tempo (s)"].idxmin()
    dif = float(ordem[c_cv].iloc[0] - ordem[c_cv].iloc[1]) if segundo else 1.0

    # veredicto
    if segundo is None:
        veredicto = f"**{campeao}** teve {_fmt(tipo, ordem[c_cv].iloc[0])} nas provas. Escolhe mais algoritmos para comparar."
    elif dif < 0.01:
        veredicto = (f"**Empate técnico.** {campeao} e {segundo} ficaram praticamente iguais ({_fmt(tipo, ordem[c_cv].iloc[0])}). "
                     f"Quando é assim, escolhe o mais simples ou o mais rápido (**{rapido}**).")
    else:
        veredicto = (f"**{campeao}** ganhou com {_fmt(tipo, ordem[c_cv].iloc[0])} nas provas, à frente de {segundo} "
                     f"({_fmt(tipo, ordem[c_cv].iloc[1])}). O mais rápido a treinar foi **{rapido}**.")

    avisos = []
    if clf and (df[c_teste] >= 0.999).all():
        avisos.append("Todos tiveram 100% na prova: os dados são fáceis demais para distinguir os algoritmos (acontece muito com o Iris). Experimenta o Vinho ou o Titanic.")
    if (df[c_teste] - df[c_cv]).max() > 0.08:
        avisos.append("Um modelo tem nota bem mais alta na prova do que nas provas repetidas. Pode ter sido sorte: confia mais na coluna CV.")

    # porquê
    porque, limite = [], (0.10 if clf else 0.15)
    for nome, r in df.iterrows():
        if r["Treino"] - r[c_teste] > limite:
            porque.append(f"**{nome}** tirou {_fmt(tipo, r['Treino'])} nos exemplos de estudo mas só {_fmt(tipo, r[c_teste])} na prova: **decorou** os dados (overfitting).")
    if campeao in ENSEMBLES:
        simples = [n for n in df.index if n not in ENSEMBLES and df.loc[n, c_cv] >= df.loc[campeao, c_cv] - 0.01]
        if simples:
            porque.append(f"{campeao} ganhou, mas **{simples[0]}** ficou a menos de 1% com um modelo muito mais simples. Aqui a complexidade extra compensa pouco.")
        else:
            porque.append(f"{campeao} ganhou porque combina muitos modelos pequenos, o que apanha padrões que um modelo simples não vê.")
    elif campeao in LINEARES:
        porque.append(f"{campeao} ganhou: a relação entre as colunas e o resultado é **quase em linha recta**, por isso um modelo simples chega.")
    elif campeao.startswith("KNN"):
        porque.append(f"O KNN ganhou: casos parecidos têm resultados parecidos nestes dados, e com {res['n_linhas']} exemplos a proximidade funciona bem.")
    elif campeao == "Decision Tree":
        porque.append("A árvore ganhou: bastam poucas perguntas simples para separar os casos, e isso é fácil de explicar.")
    elif campeao == "Naive Bayes":
        porque.append("O Naive Bayes ganhou: é rápido e bom com poucos dados, mesmo ignorando as relações entre colunas.")
    elif campeao == "SVM (RBF)" or campeao == "SVR":
        porque.append(f"{campeao} ganhou: a fronteira ideal é curva e suave, e este modelo lida bem com isso.")
    if "Naive Bayes" in df.index and campeao != "Naive Bayes" and res["corr_media"] > 0.4:
        porque.append(f"O Naive Bayes ficou atrás porque assume colunas independentes, mas aqui as colunas estão correlacionadas (correlação média {res['corr_media']:.2f}).")
    if "Decision Tree" in df.index and "Random Forest" in df.index and df.loc["Random Forest", c_cv] > df.loc["Decision Tree", c_cv] + 0.01:
        porque.append("A floresta bateu a árvore sozinha: uma árvore decora facilmente, mas muitas árvores a votar compensam os erros umas das outras.")
    if res["n_linhas"] < 300:
        porque.append(f"Com só {res['n_linhas']} exemplos, os resultados mexem facilmente com outra divisão dos dados. Não tires conclusões definitivas.")
    if not porque:
        porque.append("Os modelos ficaram muito próximos. Experimenta outros algoritmos ou mais colunas para ver diferenças.")

    # recomendações (custo real)
    melhor = df[c_cv].max()
    perto = df[df[c_cv] >= melhor - 0.03].copy()
    perto["expl"] = [_pers(n)["expl"] for n in perto.index]
    banco = perto.sort_values(["expl", c_cv], ascending=[False, False]).index[0]
    leve = perto.sort_values("Tamanho (KB)").index[0]
    veloz = perto.sort_values("Tempo (s)").index[0]
    recomend = [
        ("Banco, hospital ou tribunal (é preciso explicar a decisão)", banco),
        ("App no telemóvel (modelo leve e pequeno)", leve),
        ("Resultados rápidos / pouco tempo de treino", veloz),
        ("Melhor nota possível, custe o que custar", campeao),
    ]

    # palpite
    p = res.get("palpite")
    palpite_txt = None
    if p and p != "(sem palpite)":
        if p == campeao:
            palpite_txt = (True, f"Acertaste! Apostaste em **{p}** e ganhou. {_pers(p)['forte']}")
        else:
            pos = list(ordem.index).index(p) + 1 if p in ordem.index else None
            palpite_txt = (False, f"Surpresa! Apostaste em **{p}** (ficou em {pos}º lugar), mas ganhou **{campeao}**. "
                                  f"{_pers(campeao)['forte']} Isto acontece muito: o melhor algoritmo depende dos dados, não da fama.")

    return {"df": df, "c_teste": c_teste, "c_cv": c_cv, "campeao": campeao, "veredicto": veredicto,
            "avisos": avisos, "porque": porque, "recomend": recomend, "palpite": palpite_txt}


# ══════════════════════════════════════════════════════
# RELATÓRIO PDF
# ══════════════════════════════════════════════════════
def _l1(t):
    return str(t).replace("**", "").encode("latin-1", "replace").decode("latin-1")


def _gerar_pdf(res, an):
    from fpdf import FPDF
    tipo = res["tipo"]
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    def titulo(t, tam=13):
        pdf.set_font("Helvetica", "B", tam)
        pdf.set_text_color(20, 60, 120)
        pdf.multi_cell(0, 8, _l1(t), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    def texto(t, tam=10):
        pdf.set_font("Helvetica", "", tam)
        pdf.multi_cell(0, 5.5, _l1(t), new_x="LMARGIN", new_y="NEXT")

    def tabela(cab, linhas, larguras):
        pdf.set_font("Helvetica", "B", 9)
        for c, w in zip(cab, larguras):
            pdf.cell(w, 7, _l1(c), border=1, fill=False)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 9)
        for l in linhas:
            for c, w in zip(l, larguras):
                pdf.cell(w, 6.5, _l1(c), border=1)
            pdf.ln(6.5)

    titulo("Relatorio de Comparacao de Modelos", 17)
    texto(f"DataForge EDU - {datetime.date.today():%d/%m/%Y}")
    texto(f"Problema: {tipo}. Dados: {res['dataset']} ({res['n_linhas']} exemplos, {res['n_feat']} colunas). Alvo: {res['alvo']}.")
    pdf.ln(2)
    titulo("1. Veredicto")
    texto(an["veredicto"])
    for a in an["avisos"]:
        texto("Nota: " + a)
    if an["palpite"]:
        texto("Palpite: " + an["palpite"][1])
    pdf.ln(2)

    titulo("2. Resultados")
    df = an["df"]
    c_t, c_c = an["c_teste"], an["c_cv"]
    cab = ["Algoritmo", "Estudo", "Prova", "Provas (CV)", "Tempo (s)", "Tam. (KB)"]
    linhas = [[n, _fmt(tipo, r["Treino"]), _fmt(tipo, r[c_t]), _fmt(tipo, r[c_c]), f"{r['Tempo (s)']:.3f}", f"{r['Tamanho (KB)']:.0f}"]
              for n, r in df.iterrows()]
    tabela(cab, linhas, [48, 24, 24, 30, 28, 28])
    pdf.ln(3)

    titulo("3. Porque ganhou quem ganhou")
    for p in an["porque"]:
        texto("- " + p)
    pdf.ln(2)

    titulo("4. Qual escolher conforme o uso")
    for uso, alg in an["recomend"]:
        texto(f"- {uso}: {alg}")
    pdf.ln(2)

    titulo("5. Os algoritmos em linguagem simples")
    for n in df.index:
        p = _pers(n)
        texto(f"{n}: {p['analogia']} Forte: {p['forte']} Fraco: {p['fraco']}")
    pdf.ln(2)
    texto("Como ler: 'Estudo' e a nota nos exemplos que o modelo viu; 'Prova' e a nota em casos novos; 'Provas (CV)' e a media de varias provas diferentes e e a mais fiavel.", 9)
    return bytes(pdf.output())


# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
def _tab_resultado(res, an):
    tipo = res["tipo"]
    df = an["df"]
    nomes = list(df.index)

    st.success(an["veredicto"])
    for a in an["avisos"]:
        st.info(a)
    if an["palpite"]:
        (st.balloons() if an["palpite"][0] and res.get("ganhou_palpite") and not res.get("_festejado") else None)
        res["_festejado"] = True
        (st.success if an["palpite"][0] else st.warning)(an["palpite"][1] + (" +5 pontos!" if an["palpite"][0] else ""))

    cfg = {
        "Treino": st.column_config.NumberColumn("Nota no estudo", help="Nota nos exemplos que o modelo já viu. Se for muito maior do que na prova, decorou."),
        "Accuracy": st.column_config.NumberColumn("Accuracy (prova)", help="Percentagem de acertos em casos novos."),
        "F1 Score": st.column_config.NumberColumn("F1 Score", help="Nota que junta Precision e Recall."),
        "CV Accuracy": st.column_config.NumberColumn("CV Accuracy", help="Média de várias provas diferentes. Mais fiável."),
        "R²": st.column_config.NumberColumn("R² (prova)", help="Quanto da variação dos dados o modelo explica (1 = perfeito)."),
        "RMSE": st.column_config.NumberColumn("RMSE", help="Erro típico, na unidade do alvo. Menor é melhor."),
        "CV R²": st.column_config.NumberColumn("CV R²", help="R² médio em várias provas. Mais fiável."),
        "Tempo (s)": st.column_config.NumberColumn("Tempo (s)", help="Tempo que demorou a treinar."),
        "Tamanho (KB)": st.column_config.NumberColumn("Tamanho (KB)", help="Espaço que o modelo ocupa guardado."),
    }
    st.dataframe(df.style.format("{:.4f}"), width="stretch", column_config=cfg)

    if tipo == "Classificação":
        st.plotly_chart(_bar_comparison(nomes, df["Accuracy"].tolist(), "Accuracy - prova (casos novos)", "Accuracy"), width="stretch")
        st.plotly_chart(_bar_comparison(nomes, df["CV Accuracy"].tolist(), "Accuracy - média de várias provas (CV)", "CV Accuracy"), width="stretch")
        if len(nomes) >= 2:
            st.plotly_chart(_radar_chart(nomes, {"Accuracy": df["Accuracy"].tolist(), "F1 Score": df["F1 Score"].tolist(),
                                                 "CV Accuracy": df["CV Accuracy"].tolist()}), width="stretch")
    else:
        st.plotly_chart(_bar_comparison(nomes, df["R²"].tolist(), "R² - prova", "R²"), width="stretch")
        st.plotly_chart(_bar_comparison(nomes, df["RMSE"].tolist(), "RMSE - quanto menor, melhor", "RMSE"), width="stretch")
    teoria_box("Como decidir?",
               "Olha primeiro para a coluna <b>CV</b>: é a média de várias provas, por isso é mais honesta do que uma só nota. "
               "Se dois modelos estiverem quase empatados, escolhe o <b>mais simples e mais rápido</b>.")


def _tab_porque(res, an):
    st.markdown("### Porque é que os resultados ficaram assim?")
    st.caption("O sistema compara as notas com as características dos teus dados para te dar a causa, não só o número.")
    for p in an["porque"]:
        st.markdown(f"- {p}")
    st.markdown("### Estudo vs prova")
    df = an["df"]
    fig = go.Figure()
    fig.add_bar(x=list(df.index), y=df["Treino"], name="Nota no estudo", marker_color=PALETTE[0])
    fig.add_bar(x=list(df.index), y=df[an["c_teste"]], name="Nota na prova", marker_color=PALETTE[2])
    fig.update_layout(barmode="group", plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
                      yaxis=dict(gridcolor=C_BORDER), legend=dict(font=dict(color="#FFFFFF")))
    st.plotly_chart(fig, width="stretch")
    st.caption("Quando a barra do estudo é muito mais alta do que a da prova, o modelo decorou em vez de aprender.")


def _tab_erros(res, an):
    tipo, preds, nomes = res["tipo"], res["preds"], res["algos"]
    y_te, X_raw = res["y_te"], res["X_te"].reset_index(drop=True)
    mostrar = X_raw.columns[:3]
    if tipo == "Classificação":
        cl = res["classes"]
        errou = np.array([preds[n] != y_te for n in nomes])
        n_err = errou.sum(axis=0)
        st.markdown(f"De **{len(y_te)}** casos de prova: **{int((n_err == 0).sum())}** foram acertados por todos, "
                    f"**{int(((n_err > 0) & (n_err < len(nomes))).sum())}** dividiram os modelos e **{int((n_err == len(nomes)).sum())}** enganaram todos.")
        idx = [i for i in np.argsort(-n_err) if n_err[i] > 0][:10]
        if not idx:
            st.success("Nenhum modelo errou nenhum caso. Os dados são fáceis demais para esta comparação.")
            return
        linhas = []
        for i in idx:
            l = {"Resposta certa": cl[int(y_te[i])]}
            for n in nomes:
                p = int(preds[n][i])
                l[n] = "certo" if p == int(y_te[i]) else f"errou (previu {cl[p]})"
            for c in mostrar:
                l[c] = X_raw.loc[i, c]
            linhas.append(l)
        st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)
        if len(nomes) > 1 and (((n_err > 0) & (n_err < len(nomes))).sum() > 0):
            st.info("Repara: modelos diferentes erram em **casos diferentes**. É por isso que se juntam vários modelos a votar (como na Random Forest): "
                    "os erros de uns são corrigidos pelos acertos de outros.")
    else:
        erros = np.array([np.abs(preds[n] - y_te) for n in nomes])
        media = erros.mean(axis=0)
        idx = np.argsort(-media)[:8]
        linhas = []
        for i in idx:
            l = {"Valor real": round(float(y_te[i]), 2)}
            for n in nomes:
                l[f"{n} (erro)"] = round(float(np.abs(preds[n][i] - y_te[i])), 2)
            for c in mostrar:
                l[c] = X_raw.loc[i, c]
            linhas.append(l)
        st.markdown("Os **8 casos mais difíceis** da prova (onde, em média, os modelos mais falharam):")
        st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)
        st.info("Os casos mais difíceis costumam ser valores **extremos** ou raros, com poucos exemplos parecidos nos dados de estudo.")


def _tab_visual(res, an):
    tipo, nomes, feats = res["tipo"], res["algos"], res["features"]
    models_dict = CLF_MODELS if tipo == "Classificação" else REG_MODELS
    if tipo == "Classificação":
        st.markdown("### O que cada algoritmo 'vê': as fronteiras de decisão")
        st.caption("Cada cor é uma classe prevista. Os pontos são casos de prova. Cada algoritmo desenha fronteiras com um estilo próprio.")
        if len(feats) < 2:
            st.info("Precisas de pelo menos 2 colunas.")
            return
        c1, c2 = st.columns(2)
        f1 = c1.selectbox("Coluna do eixo X", feats, index=0, key="cmp_f1")
        f2 = c2.selectbox("Coluna do eixo Y", [f for f in feats if f != f1], index=0, key="cmp_f2")
        cl, n = res["classes"], len(res["classes"])
        Xtr, Xte = res["X_tr"][[f1, f2]], res["X_te"][[f1, f2]]
        xs = np.linspace(res["X"][f1].min(), res["X"][f1].max(), 90)
        ys = np.linspace(res["X"][f2].min(), res["X"][f2].max(), 90)
        gx, gy = np.meshgrid(xs, ys)
        grid = pd.DataFrame({f1: gx.ravel(), f2: gy.ravel()})
        cores = [PALETTE[i % len(PALETTE)] for i in range(n)]
        escala = []
        for k, c in enumerate(cores):
            escala += [[k / n, c], [(k + 1) / n, c]]
        ajust = []
        for nome in nomes:
            m = _pipe(models_dict[nome]).fit(Xtr, res["y_tr"])
            ajust.append((nome, m.predict(grid).reshape(gx.shape), accuracy_score(res["y_te"], m.predict(Xte))))
        titulos = [f"{nome} ({acc:.0%}): {_pers(nome)['forma']}" for nome, _, acc in ajust]
        fig = make_subplots(rows=1, cols=len(nomes), subplot_titles=titulos)
        for j, (nome, z, acc) in enumerate(ajust, 1):
            fig.add_trace(go.Heatmap(x=xs, y=ys, z=z, zmin=-0.5, zmax=n - 0.5, colorscale=escala, opacity=0.4, showscale=False,
                                     hoverinfo="skip"), row=1, col=j)
            for k in range(n):
                msk = res["y_te"] == k
                fig.add_trace(go.Scatter(x=Xte[f1][msk], y=Xte[f2][msk], mode="markers", name=cl[k], showlegend=(j == 1),
                                         marker=dict(color=cores[k], size=7, line=dict(color="white", width=1))), row=1, col=j)
            fig.update_xaxes(title_text=f1, row=1, col=j)
        fig.update_annotations(font=dict(color="#FFFFFF", size=11))
        fig.update_layout(height=420, plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
                          legend=dict(font=dict(color="#FFFFFF")))
        st.plotly_chart(fig, width="stretch")
        st.caption("A nota entre parênteses usa só estas 2 colunas. Repara na forma: a árvore faz retângulos, o KNN faz manchas, o SVM faz curvas, a regressão logística faz uma recta.")
    else:
        st.markdown("### Como cada algoritmo desenha a previsão")
        st.caption("Escolhe uma coluna: cada linha mostra o que o modelo prevê ao variar só essa coluna.")
        f = st.selectbox("Coluna", feats, key="cmp_fr")
        xs = np.linspace(res["X"][f].min(), res["X"][f].max(), 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["X_te"][f], y=res["y_te"], mode="markers", name="Casos de prova",
                                 marker=dict(color="#8B949E", size=6)))
        for i, nome in enumerate(nomes):
            m = _pipe(models_dict[nome]).fit(res["X_tr"][[f]], res["y_tr"])
            fig.add_trace(go.Scatter(x=xs, y=m.predict(pd.DataFrame({f: xs})), mode="lines", name=f"{nome}: {_pers(nome)['forma']}",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=3)))
        fig.update_layout(xaxis_title=f, yaxis_title=res["alvo"], height=430, plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                          font=dict(color="#FFFFFF"), xaxis=dict(gridcolor=C_BORDER), yaxis=dict(gridcolor=C_BORDER),
                          legend=dict(font=dict(color="#FFFFFF")))
        st.plotly_chart(fig, width="stretch")
        st.caption("A recta é rígida; a árvore faz degraus; o KNN segue os vizinhos. Quanto mais flexível a linha, mais perto dos pontos, mas maior o risco de decorar.")


def _sweep(res, algo, hyper):
    cache = st.session_state.setdefault("cmp_sweep", {})
    k = (res["id"], algo)
    if k not in cache:
        _, _, valores, fab, _, _ = hyper[algo]
        tr, te = [], []
        for v in valores:
            m = make_pipeline(StandardScaler(), fab(v)).fit(res["X_tr"], res["y_tr"])
            tr.append(float(m.score(res["X_tr"], res["y_tr"])))
            te.append(float(m.score(res["X_te"], res["y_te"])))
        cache[k] = (tr, te)
    return cache[k]


def _tab_mexer(res, an):
    tipo = res["tipo"]
    hyper = HYPER_CLF if tipo == "Classificação" else HYPER_REG
    disp = [n for n in res["algos"] if n in hyper]
    st.markdown("### Mexe no botão e vê o modelo mudar")
    if not disp:
        st.info("Nenhum dos algoritmos escolhidos tem um botão principal para mexer. Experimenta KNN, Árvore, Random Forest ou Ridge.")
        return
    algo = st.selectbox("Algoritmo", disp, key="cmp_mx_algo")
    _, rotulo, valores, _, cresce, dica = hyper[algo]
    st.caption(dica)
    with st.spinner("A calcular todas as opções..."):
        tr, te = _sweep(res, algo, hyper)
    v = st.select_slider(rotulo, options=valores, value=valores[len(valores) // 2], key=f"cmp_mx_{algo}")
    i = valores.index(v)
    melhor_i = int(np.argmax(te))

    fig = go.Figure()
    xs = [str(x) for x in valores]
    fig.add_trace(go.Scatter(x=xs, y=tr, mode="lines+markers", name="Nota no estudo", line=dict(color=PALETTE[0], width=3)))
    fig.add_trace(go.Scatter(x=xs, y=te, mode="lines+markers", name="Nota na prova", line=dict(color=PALETTE[2], width=3)))
    fig.add_trace(go.Scatter(x=[xs[melhor_i]], y=[te[melhor_i]], mode="markers", name="Melhor ponto",
                             marker=dict(symbol="star", size=18, color=C_GREEN)))
    fig.add_vline(x=xs[i], line_dash="dash", line_color="#FFFFFF")
    fig.update_layout(xaxis_title=rotulo, yaxis_title="Accuracy" if tipo == "Classificação" else "R²", height=380,
                      plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"),
                      xaxis=dict(gridcolor=C_BORDER), yaxis=dict(gridcolor=C_BORDER), legend=dict(font=dict(color="#FFFFFF")))
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Nota no estudo", _fmt(tipo, tr[i]))
    c2.metric("Nota na prova", _fmt(tipo, te[i]))
    c3.metric("Melhor valor encontrado", str(valores[melhor_i]))

    gap = tr[i] - te[i]
    lado_complexo = (i > melhor_i) if cresce else (i < melhor_i)
    lado_simples = (i < melhor_i) if cresce else (i > melhor_i)
    if te[i] >= te[melhor_i] - 0.01:
        st.success("Bom equilíbrio: estás no ponto (ou muito perto) em que o modelo funciona melhor com casos novos.")
    elif lado_complexo and gap > 0.05:
        st.error("**Overfitting:** o modelo está complexo demais. Vai muito bem no estudo mas desce na prova, porque decorou.")
    elif lado_simples:
        st.warning("**Underfitting:** o modelo está simples demais. Falha tanto no estudo como na prova, porque não aprendeu o suficiente.")
    else:
        st.info("Está perto do ponto bom, mas ainda dá para melhorar. Procura a estrela verde.")


def _tab_dados(res, an):
    tipo, nomes = res["tipo"], res["algos"]
    models_dict = CLF_MODELS if tipo == "Classificação" else REG_MODELS
    st.markdown("### Mais dados ajudavam?")
    st.caption("O modelo é treinado com 20%, 40%... 100% dos dados. Se a nota na prova continua a subir, juntar mais dados compensa.")
    if st.button("Calcular a curva de aprendizagem", key="cmp_lc_btn"):
        out = {}
        with st.spinner("A treinar várias vezes..."):
            for n in nomes:
                cv = (StratifiedKFold(3, shuffle=True, random_state=42) if tipo == "Classificação" else KFold(3, shuffle=True, random_state=42))
                sz, tr, va = learning_curve(_pipe(models_dict[n]), res["X"], res["y"], train_sizes=[.2, .4, .6, .8, 1.0], cv=cv, shuffle=True, random_state=42,
                                            scoring="accuracy" if tipo == "Classificação" else "r2")
                out[n] = (sz, tr.mean(axis=1), va.mean(axis=1))
        res["lc"] = out
    lc = res.get("lc")
    if not lc:
        return
    fig = go.Figure()
    for i, (n, (sz, tr, va)) in enumerate(lc.items()):
        cor = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=sz, y=va, mode="lines+markers", name=f"{n} (prova)", line=dict(color=cor, width=3)))
        fig.add_trace(go.Scatter(x=sz, y=tr, mode="lines", name=f"{n} (estudo)", line=dict(color=cor, width=1.5, dash="dash")))
    fig.update_layout(xaxis_title="Número de exemplos usados no estudo", yaxis_title="Nota", height=420, plot_bgcolor=C_SURFACE,
                      paper_bgcolor=C_SURFACE, font=dict(color="#FFFFFF"), xaxis=dict(gridcolor=C_BORDER),
                      yaxis=dict(gridcolor=C_BORDER), legend=dict(font=dict(color="#FFFFFF")))
    st.plotly_chart(fig, width="stretch")
    for n, (sz, tr, va) in lc.items():
        ganho = va[-1] - va[-2]
        if ganho > 0.01:
            st.markdown(f"- **{n}**: a nota ainda sobe no fim (+{ganho:.2f}). **Mais dados ajudariam.**")
        else:
            st.markdown(f"- **{n}**: a nota já estabilizou. Mais dados do mesmo tipo ajudariam pouco; vale mais melhorar as colunas ou mudar de modelo.")


def _tab_custo(res, an):
    df = an["df"]
    st.markdown("### Nota não é tudo: o custo real de cada modelo")
    linhas = pd.DataFrame({
        "Nota (CV)": [_fmt(res["tipo"], v) for v in df[an["c_cv"]]],
        "Tempo de treino (s)": [f"{v:.3f}" for v in df["Tempo (s)"]],
        "Tamanho guardado (KB)": [f"{v:.0f}" for v in df["Tamanho (KB)"]],
        "Facilidade de explicar": [EXPL_TXT[_pers(n)["expl"]] for n in df.index],
    }, index=df.index)
    st.dataframe(linhas, width="stretch")
    st.caption("O KNN é pesado porque guarda todos os exemplos; a regressão linear é minúscula porque guarda só alguns números.")
    st.markdown("### O que escolher conforme o uso?")
    st.caption("Entre os modelos com nota a menos de 3 pontos do melhor.")
    for uso, alg in an["recomend"]:
        st.markdown(f"- **{uso}:** {alg}")


def _tab_conhecer(res, an):
    st.markdown("### Os teus algoritmos em linguagem simples")
    for nome in res["algos"]:
        p = _pers(nome)
        st.markdown(
            f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};border-left:4px solid {C_ACCENT};border-radius:10px;'
            f'padding:.9rem 1.1rem;margin-bottom:.7rem;"><div style="font-size:17px;font-weight:800;color:#FFFFFF;">{nome}</div>'
            f'<div style="font-size:14px;color:#D0D8F0;margin-top:4px;"><b>Analogia:</b> {p["analogia"]}</div>'
            f'<div style="font-size:14px;color:#D0D8F0;"><b>Forte:</b> {p["forte"]}</div>'
            f'<div style="font-size:14px;color:#D0D8F0;"><b>Fraco:</b> {p["fraco"]}</div>'
            f'<div style="font-size:14px;color:#D0D8F0;"><b>Desenha fronteiras assim:</b> {p["forma"]}</div></div>',
            unsafe_allow_html=True)


def _show_comparison_result(tipo):
    res = st.session_state.get("cmp_result")
    if not res or res.get("tipo") != tipo:
        st.markdown(f"""<div style="text-align:center;padding:3.5rem;color:#7A8BA8;border:2px dashed #3A4560;border-radius:16px;margin-top:1rem;">
        Configura a comparação acima e carrega em <b>Comparar Modelos</b>.</div>""", unsafe_allow_html=True)
        return
    an = _analise(res)

    c1, c2 = st.columns([3, 1])
    c1.caption(f"Dados: {res['dataset']} · {res['n_linhas']} exemplos · {res['n_feat']} colunas · alvo: {res['alvo']}")
    with c2:
        try:
            st.download_button("Descarregar relatório (PDF)", data=_gerar_pdf(res, an),
                               file_name="comparacao_de_modelos.pdf", mime="application/pdf", key="cmp_pdf", width="stretch")
        except Exception:
            st.caption("Instala fpdf2 para gerar o relatório em PDF.")

    visual = "Fronteiras de decisão" if tipo == "Classificação" else "Curvas de previsão"
    abas = st.tabs(["Resultado", "Porquê?", "Onde erram", visual, "Mexer e ver", "Mais dados?", "Custo real", "Conhecer os algoritmos"])
    for aba, fn in zip(abas, [_tab_resultado, _tab_porque, _tab_erros, _tab_visual, _tab_mexer, _tab_dados, _tab_custo, _tab_conhecer]):
        with aba:
            fn(res, an)


# ══════════════════════════════════════════════════════
# PÁGINA
# ══════════════════════════════════════════════════════
def render_comparador(username: str):
    inject_css()
    page_header("Comparador de Modelos", "Põe até 3 algoritmos a fazer a mesma prova e descobre porque um ganha", "")

    teoria_box("Porquê comparar?",
               "Não existe um algoritmo que seja sempre o melhor. É como escolher um carro: depende da estrada. "
               "Aqui pões até 3 algoritmos a fazer <b>a mesma prova</b>, vês quem ganha e, mais importante, <b>porquê</b>.")

    tipo = st.radio("Tipo de problema", ["Classificação", "Regressão"], horizontal=True, key="cmp_tipo")
    atual = st.session_state.get("cmp_result")
    tem = bool(atual and atual.get("tipo") == tipo)

    with st.expander("1. Configura a comparação", expanded=not tem):
        col_a, col_b = st.columns(2)
        df, nm = None, "Upload"
        with col_a:
            section_title("Dados")
            fonte = st.radio("Fonte", ["Embutido", "Upload"], horizontal=True, key="cmp_fonte")
            if fonte == "Embutido":
                if tipo == "Classificação":
                    opcoes = ["Alunos em Angola (fictício) - Aprovação", "Iris - Classificação de flores", "Vinho - Qualidade", "Cancro - Diagnóstico", "Titanic - Sobrevivência"]
                else:
                    opcoes = ["Casas em Luanda (fictício) - Preços", "Banca de rua (fictício) - Vendas", "Diabetes - Progressão", "Boston Housing - Preços"]
                nm = st.selectbox("Dataset", opcoes, key="cmp_ds")
                try:
                    df, desc, _ = carregar_dataset_embutido(nm)
                except Exception as e:
                    df, desc = None, ""
                    erro_box(f"Não foi possível carregar este dataset (precisa de internet?): {e}")
                if df is not None and desc:
                    st.caption(desc)
            else:
                f = st.file_uploader("CSV", type=["csv"], key="cmp_up")
                df = pd.read_csv(f) if f else None
                nm = f.name if f else "Upload"
                if df is not None:
                    mostrar_diagnostico(df, expandido=False)

            if df is not None:
                cols = df.columns.tolist()
                target = st.selectbox("Alvo (o que queres prever)", cols, index=len(cols) - 1, key=f"cmp_tgt_{nm}")
                features = st.multiselect("Colunas a usar", [c for c in cols if c != target],
                                          default=[c for c in cols if c != target][:8], key=f"cmp_feats_{nm}_{target}")
        with col_b:
            if df is not None:
                section_title("Prova")
                test_size = st.slider("% dos dados guardados para a prova", 10, 40, 20, key="cmp_ts") / 100
                cv_folds = st.slider("Número de provas repetidas (CV)", 3, 10, 5, key="cmp_cv")
                section_title("Algoritmos (escolhe até 3)")
                models_dict = CLF_MODELS if tipo == "Classificação" else REG_MODELS
                algos = st.multiselect("Algoritmos", list(models_dict.keys()), default=list(models_dict.keys())[:3],
                                       max_selections=3, key="cmp_algos")
                st.caption("Dica: escolhe famílias diferentes (ex.: KNN, Árvore e Random Forest).")
                palpite = st.selectbox("O teu palpite: qual vai ganhar?", ["(sem palpite)"] + algos, key="cmp_palpite")
                if st.button("Comparar Modelos", width="stretch", key="cmp_run", type="primary"):
                    if not features or not algos:
                        aviso_box("Escolhe colunas e pelo menos 1 algoritmo.")
                    else:
                        _run_comparison(df, features, target, algos, models_dict, tipo, test_size, cv_folds, username, nm, palpite)

    _show_comparison_result(tipo)
