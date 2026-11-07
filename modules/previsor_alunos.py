"""
Previsor de Alunos (KNN) - página do DataForge EDU.

Carrega o modelo treinado (artifacts/knn_model.pkl) e prevê "sim" ou "não" a partir de cinco características
do aluno. A lógica (carregar, validar, prever) não depende do Streamlit, por isso é testável;
o ecrã está em render_previsor_alunos().

Segurança: um .pkl executa código ao ser aberto. Antes de o abrir, comparamos o SHA-256 do ficheiro com o
valor guardado em knn_model.pkl.sha256. Se o ficheiro for trocado ou alterado, recusa-se a abrir.
"""
import hashlib
import os

import numpy as np

# ── O que significa a resposta do modelo (muda aqui se for outra coisa, ex.: "Aprovado" / "Em risco") ──
TITULO_RESULTADO = "Resposta do modelo"
TEXTO_SIM = "SIM"
TEXTO_NAO = "NÃO"

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(_RAIZ, "artifacts", "knn_model.pkl")
CAMINHO_HASH = CAMINHO_MODELO + ".sha256"

# Nomes legíveis e intervalos (os intervalos vêm dos dados de treino do modelo)
CAMPOS = {
    "horas_estudo_semana": "Horas de estudo por semana",
    "faltas": "Faltas",
    "nota_anterior": "Nota anterior",
    "trabalha": "Trabalha?",
    "tem_internet": "Tem internet em casa?",
}


class ModeloInvalido(Exception):
    """O ficheiro do modelo falta, foi alterado ou não tem o formato esperado."""


def _sha256(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def verificar_integridade(caminho: str = CAMINHO_MODELO, caminho_hash: str = CAMINHO_HASH) -> None:
    """Recusa abrir o ficheiro se o SHA-256 não bater com o registado."""
    if not os.path.exists(caminho):
        raise ModeloInvalido(f"Modelo não encontrado: {caminho}")
    if not os.path.exists(caminho_hash):
        raise ModeloInvalido("Falta o ficheiro .sha256 do modelo: por segurança, não é aberto.")
    esperado = open(caminho_hash, encoding="utf8").read().strip().lower()
    if _sha256(caminho) != esperado:
        raise ModeloInvalido("O ficheiro do modelo foi alterado (SHA-256 não confere): não é aberto.")


def carregar_modelo(caminho: str = CAMINHO_MODELO, caminho_hash: str = CAMINHO_HASH) -> dict:
    """Devolve o dicionário guardado: model, scaler, encoders, le_target, features."""
    verificar_integridade(caminho, caminho_hash)
    import joblib
    pacote = joblib.load(caminho)
    if not isinstance(pacote, dict) or "model" not in pacote or "features" not in pacote:
        raise ModeloInvalido("Formato inesperado: esperava um dicionário com 'model' e 'features'.")
    return pacote


def intervalos_treino(pacote: dict) -> dict:
    """Mínimo e máximo de cada variável nos dados de treino (para avisar quando se sai do que o modelo viu)."""
    X = np.asarray(pacote["model"]._fit_X)
    return {f: (float(X[:, i].min()), float(X[:, i].max())) for i, f in enumerate(pacote["features"])}


def montar_linha(pacote: dict, valores: dict) -> np.ndarray:
    """Ordena os valores pela lista de variáveis do modelo e aplica o scaler, se existir."""
    em_falta = [f for f in pacote["features"] if f not in valores]
    if em_falta:
        raise ValueError(f"Faltam variáveis: {em_falta}")
    x = np.array([[float(valores[f]) for f in pacote["features"]]])
    if pacote.get("scaler") is not None:
        x = pacote["scaler"].transform(x)
    return x


def prever(pacote: dict, valores: dict) -> dict:
    """Previsão para um aluno: classe, probabilidade de cada classe e os vizinhos que decidiram."""
    modelo = pacote["model"]
    x = montar_linha(pacote, valores)
    classe = str(modelo.predict(x)[0])
    probas = {str(c): float(p) for c, p in zip(modelo.classes_, modelo.predict_proba(x)[0])}
    dist, idx = modelo.kneighbors(x)
    y = np.asarray(modelo._y)
    vizinhos = []
    for d, i in zip(dist[0], idx[0]):
        linha = {f: float(modelo._fit_X[i][j]) for j, f in enumerate(pacote["features"])}
        linha["distância"] = float(d)
        linha["resposta"] = str(modelo.classes_[y[i]]) if y.ndim == 1 else str(y[i])
        vizinhos.append(linha)
    return {"classe": classe, "probabilidades": probas, "vizinhos": vizinhos}


def fora_do_intervalo(pacote: dict, valores: dict) -> list:
    """Variáveis cujo valor está fora do que o modelo viu no treino."""
    avisos = []
    for f, (lo, hi) in intervalos_treino(pacote).items():
        v = float(valores[f])
        if v < lo or v > hi:
            avisos.append((f, v, lo, hi))
    return avisos


# ── Interface ─────────────────────────────────────────────────────────────────────────────────────
def render_previsor_alunos(username: str = ""):
    import pandas as pd
    import streamlit as st

    try:
        from modules.utils import page_header
        page_header("Previsor de Alunos", "Usa um modelo KNN já treinado para fazer previsões")
    except Exception:
        st.title("Previsor de Alunos")
        st.caption("Usa um modelo KNN já treinado para fazer previsões.")

    @st.cache_resource(show_spinner=False)
    def _modelo():
        return carregar_modelo()

    try:
        pacote = _modelo()
    except ModeloInvalido as e:
        st.error(str(e))
        return

    lim = intervalos_treino(pacote)
    st.markdown("Preenche os dados de um aluno e carrega em **Prever**. "
                "O modelo compara-o com os 400 alunos com que foi treinado e vai à procura dos mais parecidos.")

    c1, c2 = st.columns(2)
    with c1:
        horas = st.slider(CAMPOS["horas_estudo_semana"], 0.0, 30.0, 10.0, 0.5, key="pa_horas")
        faltas = st.slider(CAMPOS["faltas"], 0, 25, 3, 1, key="pa_faltas")
        nota = st.slider(CAMPOS["nota_anterior"], 0.0, 20.0, 12.0, 0.1, key="pa_nota")
    with c2:
        trabalha = st.radio(CAMPOS["trabalha"], ["Não", "Sim"], horizontal=True, key="pa_trabalha")
        internet = st.radio(CAMPOS["tem_internet"], ["Não", "Sim"], horizontal=True, key="pa_internet")

    valores = {
        "horas_estudo_semana": horas, "faltas": faltas, "nota_anterior": nota,
        "trabalha": 1 if trabalha == "Sim" else 0, "tem_internet": 1 if internet == "Sim" else 0,
    }

    if st.button("Prever", type="primary", key="pa_prever"):
        st.session_state["pa_resultado"] = (dict(valores), prever(pacote, valores))

    guardado = st.session_state.get("pa_resultado")
    if not guardado:
        return
    val_usados, res = guardado

    for f, v, lo, hi in fora_do_intervalo(pacote, val_usados):
        st.warning(f"**{CAMPOS.get(f, f)}** = {v:g} está fora do que o modelo viu no treino ({lo:g} a {hi:g}). "
                   "A previsão pode ser pouco fiável.")

    p_sim = res["probabilidades"].get("sim", 0.0)
    st.subheader(TITULO_RESULTADO)
    st.metric("Previsão", TEXTO_SIM if res["classe"] == "sim" else TEXTO_NAO)
    st.progress(p_sim, text=f"Dos 5 vizinhos mais parecidos, {round(p_sim * 5)} responderam «sim» ({p_sim:.0%})")

    with st.expander("Como o modelo decidiu (os 5 alunos mais parecidos)", expanded=True):
        st.caption("O KNN não tem fórmulas: olha para os alunos mais próximos nos dados de treino e vota. "
                   "Distância menor = aluno mais parecido.")
        tabela = pd.DataFrame(res["vizinhos"]).rename(columns=CAMPOS)
        st.dataframe(tabela, hide_index=True, width="stretch")

    with st.expander("Antes de confiar neste modelo"):
        st.markdown(
            "- **Não mede o desempenho verdadeiro.** O ficheiro só traz os dados de treino, e um aluno é sempre vizinho "
            "de si próprio. Para saber quanto acerta a sério, falta um conjunto de teste separado.\n"
            "- **Sem normalização.** O KNN mede distâncias; faltas (0 a 20) e horas (0 a 25) pesam muito mais do que "
            "«trabalha» e «tem internet» (0 ou 1), que quase não influenciam a decisão.\n"
            "- **É uma ajuda, não uma sentença.** Nunca decidas sozinho sobre um aluno com base nisto."
        )
