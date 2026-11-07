"""Embrulho para guardar QUALQUER modelo tabular treinado em qualquer página do DataForge.

As páginas de Classificação, Regressão, MLP, Clustering, etc. treinam com matrizes já codificadas e normalizadas
(LabelEncoder, StandardScaler...). Para o modelo poder ser guardado e testado mais tarde com dados NOVOS, é preciso
guardar também esses passos. `PacoteTabular` junta tudo num só objecto que recebe uma tabela com as colunas
originais e faz sozinho: codificar categorias, preencher em falta, normalizar e prever.

Sem Streamlit: testável. A assinatura e o isolamento por utilizador continuam a ser de `model_persistence`.
"""
import numpy as np
import pandas as pd

TAREFAS_SUPERVISIONADAS = ("classificacao", "regressao")
TAREFAS = TAREFAS_SUPERVISIONADAS + ("agrupamento", "anomalias")
NOMES_TAREFA = {"classificacao": "Classificação", "regressao": "Regressão", "agrupamento": "Agrupamento (clustering)",
                "anomalias": "Detecção de anomalias"}


class ErroEmpacotar(Exception):
    """O modelo não se pode guardar (mensagem pronta a mostrar)."""


class PacoteTabular:
    """Estimador + passos de preparação. `predict` recebe um DataFrame com as colunas originais."""

    def __init__(self, estimador, features, tarefa, scaler=None, encoders=None, le_target=None,
                 colunas_dummy=None, cat_cols=None, preenchimento=None):
        self.estimador = estimador
        self.features = list(features)
        self.tarefa = tarefa
        self.scaler = scaler
        self.encoders = dict(encoders or {})
        self.le_target = le_target
        self.colunas_dummy = list(colunas_dummy) if colunas_dummy is not None else None
        self.cat_cols = list(cat_cols if cat_cols is not None else self.encoders)
        self.preenchimento = {str(k): float(v) for k, v in (preenchimento or {}).items()}

    # ── preparação ────────────────────────────────────────────────────────────
    def _matriz(self, df: pd.DataFrame):
        X = df[self.features].copy()
        if self.colunas_dummy is not None:                       # modo get_dummies (O Meu Primeiro Modelo)
            for c in self.features:
                if c not in self.cat_cols:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
            d = pd.get_dummies(X, columns=self.cat_cols).astype(float)
            d = d.reindex(columns=self.colunas_dummy, fill_value=0.0)
            return d.fillna(pd.Series(self.preenchimento)).fillna(0.0)
        cols = {}
        for c in self.features:
            if c in self.encoders:
                conhecidas = {str(k): i for i, k in enumerate(self.encoders[c].classes_)}
                cols[c] = X[c].map(lambda v, m=conhecidas: m.get(str(v), np.nan) if pd.notna(v) else np.nan)
            else:
                cols[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        M = pd.DataFrame(cols, index=X.index).astype(float)
        M = M.fillna(pd.Series(self.preenchimento)).fillna(0.0)
        A = M.to_numpy(dtype=float)
        if self.scaler is not None:
            nomes = getattr(self.scaler, "feature_names_in_", None)      # scaler treinado com DataFrame: usa os mesmos nomes
            if nomes is not None and len(nomes) == A.shape[1]:
                A = self.scaler.transform(pd.DataFrame(A, columns=list(nomes)))
            else:
                A = self.scaler.transform(A)
        nomes_est = getattr(self.estimador, "feature_names_in_", None)    # modelo treinado com DataFrame (ex.: Pipeline)
        if nomes_est is not None and len(nomes_est) == A.shape[1]:
            return pd.DataFrame(A, columns=list(nomes_est))
        return A

    # ── previsão ──────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame):
        """Classificação: posição da classe (0, 1, 2...) na lista `classes` dos metadados. Os outros: valor bruto."""
        X = self._matriz(df)
        p = self.estimador.predict(X)
        if self.tarefa == "classificacao":
            pos = {c: i for i, c in enumerate(self.estimador.classes_)}
            return np.array([pos[v] for v in p], dtype=int)
        return np.asarray(p)

    def predict_proba(self, df: pd.DataFrame):
        return self.estimador.predict_proba(self._matriz(df))

    def pontuacao(self, df: pd.DataFrame):
        """Anomalias: quanto menor, mais estranho (decision_function ou score_samples)."""
        X = self._matriz(df)
        for nome in ("decision_function", "score_samples"):
            if hasattr(self.estimador, nome):
                return np.asarray(getattr(self.estimador, nome)(X), dtype=float)
        return None

    def rotulos_classes(self) -> list:
        """Nome de cada classe, pela mesma ordem de `predict_proba`."""
        cls = list(self.estimador.classes_)
        if self.le_target is not None:
            alvo = list(self.le_target.classes_)
            return [str(alvo[int(c)]) for c in cls]
        return [str(c) for c in cls]


# ── Verificações e metadados ──────────────────────────────────────────────────
def pode_guardar(estimador, tarefa: str) -> str:
    """'' se o modelo se pode guardar para prever dados novos; senão o motivo."""
    if tarefa not in TAREFAS:
        return "Este tipo de modelo ainda não se pode guardar."
    if tarefa == "agrupamento" and not hasattr(estimador, "predict"):
        return ("Este algoritmo de clustering não consegue colocar pontos novos num grupo (só agrupa os dados com que foi treinado). "
                "Experimenta K-Means ou Mistura Gaussiana, que sabem fazê-lo.")
    if tarefa == "anomalias" and not hasattr(estimador, "predict"):
        return ("Este detector (Local Outlier Factor, tal como está configurado) só avalia os dados do treino, "
                "por isso não pode ser usado em dados novos. Experimenta Isolation Forest, One-Class SVM ou Elliptic Envelope.")
    if tarefa in TAREFAS_SUPERVISIONADAS and not hasattr(estimador, "predict"):
        return "Este modelo não tem previsão (predict)."
    return ""


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def construir(estimador, *, tarefa: str, features, alvo: str, algoritmo: str, nome: str, origem: str,
              X_ref: pd.DataFrame, scaler=None, encoders=None, le_target=None, colunas_dummy=None,
              metricas=None, parametros=None, n_treino=None, n_teste=None) -> tuple:
    """Devolve (pacote, meta) prontos para `model_persistence.guardar_modelo`.

    `X_ref`: as variáveis de treino SEM normalizar, com as colunas originais (categorias já codificadas se
    `encoders` existir; texto original se `colunas_dummy` existir). Serve para os intervalos e valores de substituição.
    """
    motivo = pode_guardar(estimador, tarefa)
    if motivo:
        raise ErroEmpacotar(motivo)
    features = list(features)
    encoders = dict(encoders or {})
    modo_dummy = colunas_dummy is not None
    X_ref = pd.DataFrame(X_ref)[features]
    cat = (list(X_ref.select_dtypes(include=["object", "string", "category"]).columns) if modo_dummy else list(encoders))
    num = [c for c in features if c not in cat]
    pre = {}
    for c in num:
        s = pd.to_numeric(X_ref[c], errors="coerce")
        pre[c] = float(s.median()) if s.notna().any() else 0.0
    for c in cat:
        if not modo_dummy and c in X_ref:                      # código mais frequente da categoria
            m = pd.Series(X_ref[c]).mode()
            pre[c] = float(m.iloc[0]) if len(m) else 0.0
    intervalos = {}
    for c in num:
        s = pd.to_numeric(X_ref[c], errors="coerce").dropna()
        if len(s):
            intervalos[c] = [float(s.min()), float(s.max())]
    if modo_dummy:
        conhecidas = {c: sorted(map(str, X_ref[c].dropna().unique())) for c in cat}
    else:
        conhecidas = {c: [str(k) for k in encoders[c].classes_] for c in cat}

    pacote = PacoteTabular(estimador, features, tarefa, scaler=scaler, encoders=encoders, le_target=le_target,
                           colunas_dummy=colunas_dummy, cat_cols=cat, preenchimento=pre)
    classes = pacote.rotulos_classes() if tarefa == "classificacao" else None
    meta = {
        "tipo": tarefa, "nome": nome, "algoritmo_id": str(algoritmo), "algoritmo": str(algoritmo),
        "alvo": alvo, "features": features, "colunas_data": [], "colunas_modelo": list(colunas_dummy or features),
        "num_cols": num, "cat_cols": cat, "classes": classes, "categorias_conhecidas": conhecidas,
        "intervalos_numericos": intervalos, "parametros": parametros or {}, "metricas_teste": metricas or {}, "cv": {},
        "n_treino": n_treino, "n_teste": n_teste, "random_state": 42, "origem": origem, "avisos": [],
    }
    if tarefa == "agrupamento":
        labels = getattr(estimador, "labels_", None)
        meta["n_grupos"] = int(len(set(labels.tolist())) - (1 if labels is not None and -1 in set(labels.tolist()) else 0)) \
            if labels is not None else None
    return pacote, meta


def metricas_simples(d: dict) -> dict:
    """Só números simples (para os metadados em JSON)."""
    out = {}
    for k, v in (d or {}).items():
        n = _num(v)
        if n is not None:
            out[str(k)] = n
    return out
