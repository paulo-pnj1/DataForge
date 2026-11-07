"""Invólucros de modelos treinados nas várias páginas do DataForge, SEM depender do resto da plataforma.

Só usa numpy, pandas e scikit-learn. Por isso este ficheiro vai dentro do ZIP de cada modelo descarregado: com ele,
`carregar("modelo.joblib")` devolve o modelo pronto a usar em qualquer computador, e `.aplicar(tabela)` dá os resultados.

Classes:  ModeloLegado (classificação/regressão com codificadores + scaler), ModeloDummies (O Meu Primeiro Modelo),
          ModeloGrupos (clustering), ModeloAnomalias, ModeloReducao, PoliticaQ (Q-Learning), PoliticaDQN (rede do DQN).
"""
import numpy as np
import pandas as pd


# ── utilitários ────────────────────────────────────────────────────────────────
def _numeros(df: pd.DataFrame, features, preenchimento=None) -> pd.DataFrame:
    """Colunas numéricas, valores ilegíveis ou infinitos viram falta, e a falta é preenchida com a mediana do treino."""
    faltam = [c for c in features if c not in df.columns]
    if faltam:
        raise ValueError("Faltam colunas: " + ", ".join(map(str, faltam)))
    out = pd.DataFrame(index=df.index)
    for c in features:
        s = pd.to_numeric(df[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        out[c] = s.fillna((preenchimento or {}).get(c, 0.0))
    return out


def _com_probabilidades(modelo, M, valores_alvo, nomes):
    """Probabilidades alinhadas com as classes do modelo (uma classe que faltou no treino tem probabilidade 0)."""
    bruto = modelo.predict_proba(M)
    classes = getattr(modelo, "classes_", None)
    if classes is not None and len(classes) != len(valores_alvo):
        completo = np.zeros((bruto.shape[0], len(valores_alvo)))
        completo[:, np.searchsorted(valores_alvo, classes)] = bruto
        bruto = completo
    return bruto


class _Base:
    def para_pacote(self) -> dict:
        """Só objectos simples (scikit-learn, numpy, dicionários): abre-se em qualquer computador com este ficheiro."""
        return {"classe": type(self).__name__, "estado": dict(self.__dict__)}


# ══════════════════════════════════════════════════════════════════════════════
# Classificação e regressão
# ══════════════════════════════════════════════════════════════════════════════
class ModeloLegado(_Base):
    """Pré-processamento (codificadores + scaler) + algoritmo já treinado, num só objecto."""

    def __init__(self, features, cat_cols, encoders, scaler, modelo, tipo, valores_alvo=None, preenchimento=None, modas=None,
                 com_nomes=False, nomes_classes=None):
        self.features = list(features)
        self.cat_cols = list(cat_cols)
        self.encoders = dict(encoders or {})
        self.scaler = scaler
        self.modelo = modelo
        self.tipo = tipo
        self.valores_alvo = None if valores_alvo is None else np.asarray(valores_alvo)
        self.preenchimento = dict(preenchimento or {})
        self.modas = dict(modas or {})
        self.com_nomes = com_nomes          # o Comparador ajusta os modelos com DataFrames (com nomes de colunas)
        self.nomes_classes = None if nomes_classes is None else [str(c) for c in nomes_classes]

    def _matriz(self, X):
        X = pd.DataFrame(X)
        colunas = []
        for c in self.features:
            if c not in X.columns:
                raise ValueError(f"Falta a coluna «{c}».")
            s = X[c]
            if c in self.encoders:
                le = self.encoders[c]
                v = s.astype(str).str.strip()
                v = v.where(v.isin(set(map(str, le.classes_))), self.modas.get(c, str(le.classes_[0])))
                colunas.append(le.transform(v).astype(float))
            else:
                n = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
                colunas.append(n.fillna(self.preenchimento.get(c, 0.0)).to_numpy())
        M = np.column_stack(colunas)
        M = self.scaler.transform(M) if self.scaler is not None else M
        return pd.DataFrame(M, columns=self.features) if getattr(self, "com_nomes", False) else M

    def predict(self, X):
        p = np.asarray(self.modelo.predict(self._matriz(X)))
        if self.tipo == "classificacao":
            return np.searchsorted(self.valores_alvo, p).astype(int)      # índice da classe, como nos Pipelines do Laboratório
        return p.astype(float)

    def predict_proba(self, X):
        if self.tipo != "classificacao" or not hasattr(self.modelo, "predict_proba"):
            raise AttributeError("Este modelo não calcula probabilidades.")
        return _com_probabilidades(self.modelo, self._matriz(X), self.valores_alvo, self.nomes_classes)

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tabela com a previsão (e a probabilidade de cada classe, se o algoritmo a calcula)."""
        out = df.copy()
        pred = self.predict(df)
        if self.tipo == "classificacao":
            nomes = self.nomes_classes or [str(v) for v in self.valores_alvo]
            out["previsao"] = [nomes[i] for i in pred]
            try:
                p = self.predict_proba(df)
                for j, n in enumerate(nomes[: p.shape[1]]):
                    out[f"prob_{n}"] = np.round(p[:, j], 4)
            except AttributeError:
                pass
        else:
            out["previsao"] = pred
        return out


class ModeloDummies(_Base):
    """«O Meu Primeiro Modelo»: colunas de texto viram colunas 0/1 (get_dummies) e o resto segue como número."""

    def __init__(self, features, cat_cols, colunas_modelo, modelo, tipo, valores_alvo=None, preenchimento=None, nomes_classes=None):
        self.features = list(features)
        self.cat_cols = list(cat_cols)
        self.colunas_modelo = list(colunas_modelo)
        self.modelo = modelo
        self.tipo = tipo
        self.valores_alvo = None if valores_alvo is None else np.asarray(valores_alvo)
        self.preenchimento = dict(preenchimento or {})
        self.nomes_classes = None if nomes_classes is None else [str(c) for c in nomes_classes]

    def _matriz(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        faltam = [c for c in self.features if c not in X.columns]
        if faltam:
            raise ValueError("Faltam colunas: " + ", ".join(map(str, faltam)))
        d = pd.DataFrame(index=X.index)
        for c in self.features:
            if c in self.cat_cols:
                d[c] = X[c].astype(object).map(lambda v: "nan" if pd.isna(v) else str(v).strip())
            else:
                n = pd.to_numeric(X[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
                d[c] = n.fillna(self.preenchimento.get(c, 0.0))
        d = pd.get_dummies(d, drop_first=False).astype(float)
        return d.reindex(columns=self.colunas_modelo, fill_value=0.0)

    def predict(self, X):
        p = np.asarray(self.modelo.predict(self._matriz(X)))
        if self.tipo == "classificacao":
            return np.searchsorted(self.valores_alvo, p).astype(int)
        return p.astype(float)

    def predict_proba(self, X):
        if self.tipo != "classificacao" or not hasattr(self.modelo, "predict_proba"):
            raise AttributeError("Este modelo não calcula probabilidades.")
        return _com_probabilidades(self.modelo, self._matriz(X), self.valores_alvo, self.nomes_classes)

    aplicar = ModeloLegado.aplicar


# ══════════════════════════════════════════════════════════════════════════════
# Não supervisionado
# ══════════════════════════════════════════════════════════════════════════════
class _NaoSup(_Base):
    def __init__(self, features, scaler, modelo, preenchimento=None):
        self.features = list(features)
        self.scaler = scaler
        self.modelo = modelo
        self.preenchimento = dict(preenchimento or {})

    def _matriz(self, df):
        M = _numeros(pd.DataFrame(df), self.features, self.preenchimento).to_numpy()
        return self.scaler.transform(M) if self.scaler is not None else M


class ModeloGrupos(_NaoSup):
    """Clustering: diz a que grupo pertence cada caso novo. Algoritmos sem `predict` (DBSCAN, Agglomerative, Spectral...)
    usam o vizinho mais próximo entre os casos do treino, que levam o grupo que o algoritmo lhes deu."""

    def __init__(self, features, scaler, modelo, preenchimento=None, vizinho=None, algoritmo=""):
        super().__init__(features, scaler, modelo, preenchimento)
        self.vizinho = vizinho
        self.algoritmo = algoritmo

    def _grupos(self, M):
        if self.vizinho is None and hasattr(self.modelo, "predict"):
            return np.asarray(self.modelo.predict(M)).astype(int)
        return np.asarray(self.vizinho.predict(M)).astype(int)

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        M = self._matriz(df)
        g = self._grupos(M)
        out = pd.DataFrame(df).copy()
        out["grupo"] = ["ruído" if v == -1 else int(v) for v in g]
        centros = getattr(self.modelo, "cluster_centers_", None)
        if centros is not None and self.vizinho is None:
            out["distancia_ao_centro"] = np.round(np.linalg.norm(M - np.asarray(centros)[g], axis=1), 4)
        if hasattr(self.modelo, "predict_proba") and self.vizinho is None:
            out["probabilidade"] = np.round(self.modelo.predict_proba(M).max(axis=1), 4)
        return out


class ModeloAnomalias(_NaoSup):
    """Deteção de anomalias: «Anomalia» ou «Normal» para cada caso novo, com a pontuação do algoritmo (mais baixa = mais estranho)."""

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        M = self._matriz(df)
        out = pd.DataFrame(df).copy()
        out["resultado"] = np.where(np.asarray(self.modelo.predict(M)) == -1, "Anomalia", "Normal")
        pont = self.modelo.decision_function(M) if hasattr(self.modelo, "decision_function") else self.modelo.score_samples(M)
        out["pontuacao"] = np.round(np.asarray(pont, dtype=float), 4)
        return out


class ModeloReducao(_NaoSup):
    """Redução de dimensionalidade: projecta casos novos nas componentes aprendidas no treino."""

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        Z = np.asarray(self.modelo.transform(self._matriz(df)))
        out = pd.DataFrame(df).copy()
        for j in range(Z.shape[1]):
            out[f"componente_{j + 1}"] = np.round(Z[:, j], 5)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Reforço
# ══════════════════════════════════════════════════════════════════════════════
ACCOES_GRELHA = ["cima", "baixo", "esquerda", "direita"]
_PASSOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class PoliticaQ(_Base):
    """Q-Learning no mundo em grelha: guarda a tabela Q. Para cada casa diz a melhor acção e simula o caminho até ao objectivo."""

    def __init__(self, Q, n, obstaculos=(), armadilhas=(), inicio=(0, 0), objectivo=None):
        self.Q = np.asarray(Q, dtype=float)
        self.n = int(n)
        self.obstaculos = [tuple(o) for o in obstaculos]
        self.armadilhas = [tuple(a) for a in armadilhas]
        self.inicio = tuple(inicio)
        self.objectivo = tuple(objectivo) if objectivo else (self.n - 1, self.n - 1)

    def melhor_accao(self, linha: int, coluna: int) -> dict:
        q = self.Q[linha * self.n + coluna]
        return {"accao": ACCOES_GRELHA[int(np.argmax(q))], "valores_q": {a: float(v) for a, v in zip(ACCOES_GRELHA, q)}}

    def caminho(self, inicio=None, max_passos: int = 100) -> list:
        pos = tuple(inicio or self.inicio)
        trilho = [pos]
        for _ in range(max_passos):
            if pos == self.objectivo or pos in self.armadilhas:
                break
            a = int(np.argmax(self.Q[pos[0] * self.n + pos[1]]))
            nr, nc = pos[0] + _PASSOS[a][0], pos[1] + _PASSOS[a][1]
            if not (0 <= nr < self.n and 0 <= nc < self.n) or (nr, nc) in self.obstaculos:
                nr, nc = pos
            if (nr, nc) == pos:               # preso: a política manda bater na parede
                break
            pos = (nr, nc)
            trilho.append(pos)
        return trilho

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe colunas «linha» e «coluna» (0 a n-1) e devolve a melhor acção e se o caminho chega ao objectivo."""
        for c in ("linha", "coluna"):
            if c not in df.columns:
                raise ValueError("Faltam colunas: linha, coluna")
        out = pd.DataFrame(df).copy()
        accoes, chega = [], []
        for l, c in zip(out["linha"], out["coluna"]):
            l, c = int(l), int(c)
            if not (0 <= l < self.n and 0 <= c < self.n):
                raise ValueError(f"A grelha tem {self.n}×{self.n}: linha e coluna têm de estar entre 0 e {self.n - 1}.")
            accoes.append(self.melhor_accao(l, c)["accao"])
            chega.append(self.caminho((l, c))[-1] == self.objectivo)
        out["melhor_accao"] = accoes
        out["chega_ao_objectivo"] = chega
        return out


class PoliticaDQN(_Base):
    """Rede treinada do DQN (CartPole). Só precisa de numpy: as camadas são Linear + ReLU."""

    def __init__(self, camadas, accoes=("esquerda", "direita"), nomes_estado=("posicao", "velocidade", "angulo", "velocidade_angular")):
        self.camadas = [(np.asarray(W, dtype=float), np.asarray(b, dtype=float)) for W, b in camadas]
        self.accoes = list(accoes)
        self.nomes_estado = list(nomes_estado)

    def valores_q(self, estado) -> np.ndarray:
        x = np.asarray(estado, dtype=float).reshape(-1, len(self.nomes_estado))
        for i, (W, b) in enumerate(self.camadas):
            x = x @ W.T + b
            if i < len(self.camadas) - 1:
                x = np.maximum(x, 0.0)
        return x

    def aplicar(self, df: pd.DataFrame) -> pd.DataFrame:
        faltam = [c for c in self.nomes_estado if c not in df.columns]
        if faltam:
            raise ValueError("Faltam colunas: " + ", ".join(faltam))
        q = self.valores_q(df[self.nomes_estado].to_numpy(dtype=float))
        out = pd.DataFrame(df).copy()
        out["melhor_accao"] = [self.accoes[i] for i in q.argmax(axis=1)]
        for j, a in enumerate(self.accoes):
            out[f"q_{a}"] = np.round(q[:, j], 4)
        return out


_CLASSES = {c.__name__: c for c in (ModeloLegado, ModeloDummies, ModeloGrupos, ModeloAnomalias, ModeloReducao, PoliticaQ, PoliticaDQN)}


def de_pacote(pacote: dict):
    obj = _CLASSES[pacote["classe"]].__new__(_CLASSES[pacote["classe"]])
    obj.__dict__.update(pacote["estado"])
    return obj


def carregar(caminho: str = "modelo.joblib"):
    """Abre um modelo exportado do DataForge (ficheiro .joblib do ZIP). Só abras ficheiros em que confias."""
    import joblib
    return de_pacote(joblib.load(caminho))
