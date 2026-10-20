"""Séries temporais: preparação, divisão cronológica, modelos de previsão e comparação.

Regras que este módulo garante:
- Os dados são SEMPRE ordenados por data e nunca baralhados.
- Divisão cronológica: treino < validação < teste. O teste são os últimos `horizonte` pontos.
- A selecção usa validação progressiva (janela que cresce, «rolling origin»): cada bloco só treina com o passado.
- Nenhum atributo usa valores futuros (lags só olham para trás; previsões multi-passo são recursivas).
- Intervalos de previsão só existem nos modelos que os calculam (ARIMA, SARIMAX); nos outros ficam em branco.
- Os valores em falta preenchem-se com o último valor conhecido (ffill): interpolar usaria o futuro.
"""
import logging
import re
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modules.ml_engine import model_evaluation as ev
from modules.ml_engine import model_registry as reg
from modules.ml_engine.data_loader import ErroDados
from modules.ml_engine.model_comparison import ROTULO_ESTADO, Recomendacao
from modules.ml_engine.model_training import (FALHOU, IGNORADO, INCOMPATIVEL, INDISPONIVEL, OK, REFERENCIA, _curto, _vaga)

log = logging.getLogger("dataforge.ml")

MIN_OBSERVACOES = 20
PERIODO_POR_FREQUENCIA = {"D": 7, "B": 5, "W": 52, "M": 12, "MS": 12, "ME": 12, "SM": 24, "Q": 4, "QS": 4, "QE": 4, "H": 24,
                          "h": 24, "T": 60, "min": 60, "Y": 1, "YS": 1, "YE": 1, "A": 1, "AS": 1, "AE": 1}


# ══════════════════════════════════════════════════════════════════════════════
# Preparação da série
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SerieTemporal:
    datas: pd.DatetimeIndex
    valores: np.ndarray
    freq: str
    periodo: int
    col_data: str
    col_valor: str
    avisos: list = field(default_factory=list)
    notas: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.valores)

    @property
    def passo_mediano(self) -> pd.Timedelta:
        return pd.Series(self.datas).diff().median() if self.n > 1 else pd.Timedelta(days=1)


def detectar_coluna_data(df: pd.DataFrame):
    """Primeira coluna que já é data (ou texto que se lê como data em ≥ 90% dos valores)."""
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            amostra = s.dropna().astype(str).head(200)
            if len(amostra) >= 5:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        if pd.to_datetime(amostra, errors="coerce").notna().mean() >= 0.9:
                            return c
                    except Exception:
                        pass
    return None


def _periodo_da_frequencia(freq) -> int:
    if not freq:
        return 1
    base = re.split(r"[-]", str(freq))[0]
    base = re.sub(r"^\d+", "", base)
    return PERIODO_POR_FREQUENCIA.get(base, PERIODO_POR_FREQUENCIA.get(base.upper(), 1))


def _inferir_frequencia(idx: pd.DatetimeIndex):
    if len(idx) < 3:
        return None
    try:
        f = pd.infer_freq(idx)
        if f:
            return f
    except (ValueError, TypeError):
        pass
    deltas = pd.Series(idx).diff().dropna()
    # calendário com buracos (infer_freq falha): testa frequências de calendário e aceita a que explica ≥ 95% das datas
    mediana = deltas.median()
    dias = mediana / pd.Timedelta(days=1)
    candidatas = []
    if 0.99 <= dias <= 1.01:
        candidatas = ["D", "B"]
    elif 6.5 <= dias <= 7.5:
        candidatas = ["W-" + idx[0].strftime("%a").upper()]
    elif 27 <= dias <= 32:
        candidatas = ["MS", "ME"]
    elif 85 <= dias <= 95:
        candidatas = ["QS", "QE"]
    elif 360 <= dias <= 370:
        candidatas = ["YS", "YE"]
    for c in candidatas:
        try:
            ref = pd.date_range(idx[0], idx[-1], freq=c)
        except Exception:
            continue
        if len(ref) and idx.isin(ref).mean() >= 0.95:
            return c
    moda = deltas.mode()
    if len(moda) and (deltas == moda.iloc[0]).mean() >= 0.8 and moda.iloc[0] > pd.Timedelta(0):
        try:
            return pd.tseries.frequencies.to_offset(moda.iloc[0]).freqstr
        except Exception:
            return None
    return None


def preparar_serie(df: pd.DataFrame, col_data: str, col_valor: str, duplicados: str = "media", nulos: str = "ffill",
                   freq: str = None, periodo: int = None) -> SerieTemporal:
    """Ordena, trata datas repetidas e em falta, regulariza a frequência e devolve a série pronta a modelar.

    `duplicados`: 'media' | 'soma' | 'ultimo' (o que fazer com várias linhas na mesma data).
    `nulos`: 'ffill' (último valor conhecido, sem usar o futuro) | 'interpolar' (usa valores vizinhos, incluindo futuros).
    """
    for c in (col_data, col_valor):
        if c not in df.columns:
            raise ErroDados(f"A coluna «{c}» não existe na tabela.")
    if col_data == col_valor:
        raise ErroDados("A coluna de datas e a coluna a prever têm de ser diferentes.")
    avisos, notas = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pd.to_datetime(df[col_data], errors="coerce")
    y = pd.to_numeric(df[col_valor], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    invalidas = int(t.isna().sum())
    if invalidas / max(len(t), 1) > 0.2:
        raise ErroDados(f"A coluna «{col_data}» tem {invalidas} de {len(t)} valores que não são datas válidas: verifica o formato.")
    if invalidas:
        notas.append(f"{invalidas} linhas com data inválida foram retiradas.")
    d = pd.DataFrame({"t": t, "y": y})[t.notna()]
    if d.empty:
        raise ErroDados("Não há datas válidas na coluna escolhida.")
    if not d["t"].is_monotonic_increasing:
        notas.append("Os dados não estavam por ordem cronológica: foram ordenados por data (nunca se baralham).")
    d = d.sort_values("t", kind="stable")
    dup = int(d["t"].duplicated().sum())
    if dup:
        agg = {"media": "mean", "soma": "sum", "ultimo": "last"}.get(duplicados, "mean")
        d = d.groupby("t", sort=True)["y"].agg(agg).reset_index()
        avisos.append(f"Havia {dup} datas repetidas; os valores foram combinados ({'média' if agg == 'mean' else 'soma' if agg == 'sum' else 'último'}).")
    serie = pd.Series(d["y"].to_numpy(), index=pd.DatetimeIndex(d["t"]))

    f = freq or _inferir_frequencia(serie.index)
    if f:
        regular = serie.asfreq(f)
        if freq and serie.index.isin(regular.index).mean() < 0.9:
            raise ErroDados(f"A frequência «{freq}» não corresponde às datas dos dados.")
        novos = len(regular) - len(serie)
        if novos > 0:
            notas.append(f"Faltavam {novos} datas na sequência «{f}»; foram acrescentadas como valores em falta.")
        serie = regular
    else:
        avisos.append("Não foi possível identificar uma frequência regular (diária, mensal, ...). As datas futuras são estimadas pelo intervalo mediano.")

    n_nulos = int(serie.isna().sum())
    if n_nulos:
        serie = serie.interpolate(limit_direction="forward") if nulos == "interpolar" else serie.ffill()
        notas.append(f"{n_nulos} valores em falta foram preenchidos com {'interpolação' if nulos == 'interpolar' else 'o último valor conhecido'}.")
        if nulos == "interpolar":
            avisos.append("A interpolação usa valores vizinhos, incluindo futuros; prefere «último valor conhecido» para avaliar previsões.")
    inicio_vazio = int(serie.isna().cumprod().sum())
    if inicio_vazio:
        serie = serie.iloc[inicio_vazio:]
        notas.append(f"{inicio_vazio} valores iniciais sem dados foram retirados.")
    serie = serie.dropna()
    if len(serie) < MIN_OBSERVACOES:
        raise ErroDados(f"A série tem só {len(serie)} observações; são precisas pelo menos {MIN_OBSERVACOES}.")
    if serie.nunique() < 2:
        avisos.append("A série tem sempre o mesmo valor: a previsão é trivial.")

    per = int(periodo) if periodo else _periodo_da_frequencia(f)
    if per > len(serie) // 2:
        avisos.append(f"A sazonalidade esperada ({per}) é demasiado longa para esta série; foi ignorada.")
        per = 1
    if per > 1 and len(serie) < 3 * per:
        avisos.append(f"Menos de 3 ciclos sazonais completos ({len(serie)} observações para período {per}): a sazonalidade será pouco fiável.")
    return SerieTemporal(serie.index, serie.to_numpy(dtype=float), f, per, col_data, col_valor, avisos, notas)


def datas_futuras_de(ultima_data, freq, passo: pd.Timedelta, h: int) -> pd.DatetimeIndex:
    if freq:
        try:
            return pd.date_range(start=ultima_data, periods=h + 1, freq=freq)[1:]
        except Exception:
            pass
    return pd.DatetimeIndex([pd.Timestamp(ultima_data) + passo * (k + 1) for k in range(h)])


def datas_futuras(serie: SerieTemporal, h: int) -> pd.DatetimeIndex:
    return datas_futuras_de(serie.datas[-1], serie.freq, serie.passo_mediano, h)


# ══════════════════════════════════════════════════════════════════════════════
# Divisão cronológica
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DivisaoTemporal:
    n: int
    horizonte: int
    fim_treino: int                  # treino = [0, fim_treino)
    fim_validacao: int               # validação = [fim_treino, fim_validacao); teste = [fim_validacao, n)
    origens_cv: list                 # origens da validação progressiva (treino = [0, o); bloco = [o, o + h))
    avisos: list = field(default_factory=list)

    @property
    def tem_validacao(self) -> bool:
        return self.fim_validacao > self.fim_treino


def dividir_temporal(n: int, horizonte: int, periodo: int = 1, n_splits: int = 3) -> DivisaoTemporal:
    """Teste = últimos h pontos; validação = h pontos antes; o resto é treino. Validação progressiva sobre treino+validação."""
    if horizonte < 1:
        raise ErroDados("O horizonte de previsão tem de ser pelo menos 1.")
    minimo = max(10, 2 * periodo)
    b = n - horizonte
    if b < minimo:
        raise ErroDados(f"A série é curta demais para prever {horizonte} passos: precisa de pelo menos {minimo + horizonte} observações (tem {n}).")
    avisos = []
    a = b - horizonte
    if a < minimo:
        a = b
        avisos.append("Série curta: não há um bloco de validação separado; a selecção usa só a validação progressiva.")
    k = min(n_splits, (b - minimo) // horizonte)
    origens = [b - j * horizonte for j in range(k, 0, -1)] if k >= 1 else []
    if not origens:
        avisos.append("Série curta: sem validação progressiva; a escolha do modelo usa só o teste e é pouco fiável.")
    return DivisaoTemporal(n, horizonte, a, b, origens, avisos)


# ══════════════════════════════════════════════════════════════════════════════
# Modelos
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PrevisaoSerie:
    media: np.ndarray
    inferior: np.ndarray = None
    superior: np.ndarray = None


class ModeloSerie:
    """Interface comum: `ajustar(y, periodo)` com o passado e `prever(h)` com os h passos seguintes."""
    suporta_intervalos = False

    def __init__(self, params=None, random_state: int = 42):
        self.params = params or {}
        self.random_state = random_state

    def ajustar(self, y: np.ndarray, periodo: int = 1):
        raise NotImplementedError

    def prever(self, h: int) -> PrevisaoSerie:
        raise NotImplementedError


class NaiveSerie(ModeloSerie):
    def ajustar(self, y, periodo=1):
        self.ultimo = float(np.asarray(y)[-1])
        return self

    def prever(self, h):
        return PrevisaoSerie(np.full(h, self.ultimo))


class NaiveSazonalSerie(ModeloSerie):
    def ajustar(self, y, periodo=1):
        y = np.asarray(y, dtype=float)
        self.m = max(int(periodo), 1)
        if len(y) < self.m:
            raise ValueError("A série é mais curta do que um ciclo sazonal.")
        self.ciclo = y[-self.m:].copy()
        return self

    def prever(self, h):
        return PrevisaoSerie(np.array([self.ciclo[k % self.m] for k in range(h)]))


def _fourier(pos, periodo: int, k_max: int) -> list:
    """Termos seno/co-seno que dependem só da posição (nunca de valores futuros)."""
    if periodo <= 1:
        return []
    out = []
    for k in range(1, k_max + 1):
        if 2 * k > periodo:
            break
        out += [np.sin(2 * np.pi * k * pos / periodo), np.cos(2 * np.pi * k * pos / periodo)]
    return out


class RegressaoTemporalSerie(ModeloSerie):
    """Regressão linear sobre a posição no tempo (tendência) e termos sazonais."""

    def _x(self, pos):
        return np.array([[p] + _fourier(p, self.m, self.K) for p in pos], dtype=float)

    def ajustar(self, y, periodo=1):
        from sklearn.linear_model import LinearRegression
        y = np.asarray(y, dtype=float)
        self.m, self.K, self.n = max(int(periodo), 1), int(self.params.get("harmonicos", 2)), len(y)
        self.modelo = LinearRegression().fit(self._x(np.arange(self.n)), y)
        return self

    def prever(self, h):
        return PrevisaoSerie(self.modelo.predict(self._x(np.arange(self.n, self.n + h))))


def _forca_tendencia(y: np.ndarray) -> float:
    t = np.arange(len(y), dtype=float)
    if np.std(y) < 1e-12:
        return 0.0
    r = np.corrcoef(t, y)[0, 1]
    return float(r * r) if np.isfinite(r) else 0.0


class _ModeloLags(ModeloSerie):
    """Prevê o próximo valor a partir dos L valores anteriores (e do ciclo sazonal), passo a passo.

    Com tendência forte o alvo é a variação (y_t − y_{t−1}), porque as árvores não extrapolam tendências.
    """

    def _estimador(self):
        raise NotImplementedError

    def _linha(self, z_janela, pos):
        return list(z_janela) + _fourier(pos, self.m, 2)

    def ajustar(self, y, periodo=1):
        y = np.asarray(y, dtype=float)
        self.m = max(int(periodo), 1)
        L = int(self.params.get("lags", 0)) or min(max(self.m, 5), 24)
        L = max(1, min(L, len(y) // 3))
        self.L = L
        self.dif = _forca_tendencia(y) > 0.3
        z = np.diff(y) if self.dif else y.copy()
        desl = 1 if self.dif else 0                          # z[i] corresponde à posição i + desl em y
        if len(z) <= L + 3:
            raise ValueError("Série curta demais para este modelo.")
        X = np.array([self._linha(z[i - L:i], i + desl) for i in range(L, len(z))], dtype=float)
        self.est = self._estimador().fit(X, z[L:])
        self.z_cauda = list(z[-L:])
        self.ultimo = float(y[-1])
        self.n = len(y)
        return self

    def prever(self, h):
        z, nivel, saidas = list(self.z_cauda), self.ultimo, []
        for k in range(h):
            pos = self.n + k
            zhat = float(self.est.predict(np.array([self._linha(z[-self.L:], pos)], dtype=float))[0])
            z.append(zhat)
            nivel = nivel + zhat if self.dif else zhat
            saidas.append(nivel)
        return PrevisaoSerie(np.array(saidas))


class RandomForestSerie(_ModeloLags):
    def _estimador(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=int(self.params.get("n_estimators", 200)), min_samples_leaf=2,
                                     random_state=self.random_state, n_jobs=1)


class XGBoostSerie(_ModeloLags):
    def _estimador(self):
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=int(self.params.get("n_estimators", 300)), learning_rate=0.05, max_depth=4,
                            subsample=0.9, random_state=self.random_state, n_jobs=1, verbosity=0, tree_method="hist")


class _Estatistico(ModeloSerie):
    def _silencioso(self):
        w = warnings.catch_warnings()
        w.__enter__()
        warnings.simplefilter("ignore")
        return w


class ArimaSerie(_Estatistico):
    suporta_intervalos = True

    def ajustar(self, y, periodo=1):
        from statsmodels.tsa.arima.model import ARIMA
        p = self.params
        w = self._silencioso()
        try:
            self.res = ARIMA(np.asarray(y, dtype=float), order=(int(p.get("p", 1)), int(p.get("d", 1)), int(p.get("q", 1)))).fit()
        finally:
            w.__exit__(None, None, None)
        return self

    def prever(self, h):
        fc = self.res.get_forecast(h)
        ci = np.asarray(fc.conf_int(alpha=0.05))
        return PrevisaoSerie(np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1])


class SarimaxSerie(_Estatistico):
    suporta_intervalos = True

    def ajustar(self, y, periodo=1):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        p, m = self.params, max(int(periodo), 1)
        sazonal = (int(p.get("P", 1)), int(p.get("D", 0)), int(p.get("Q", 1)), m) if m > 1 else (0, 0, 0, 0)
        w = self._silencioso()
        try:
            self.res = SARIMAX(np.asarray(y, dtype=float), order=(int(p.get("p", 1)), int(p.get("d", 1)), int(p.get("q", 1))),
                               seasonal_order=sazonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
        finally:
            w.__exit__(None, None, None)
        return self

    def prever(self, h):
        fc = self.res.get_forecast(h)
        ci = np.asarray(fc.conf_int(alpha=0.05))
        return PrevisaoSerie(np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1])


class SuavizacaoExponencialSerie(_Estatistico):
    def ajustar(self, y, periodo=1):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        tendencia = "add" if self.params.get("tendencia", "nenhuma") == "add" else None
        w = self._silencioso()
        try:
            self.res = ExponentialSmoothing(np.asarray(y, dtype=float), trend=tendencia, seasonal=None,
                                            initialization_method="estimated").fit()
        finally:
            w.__exit__(None, None, None)
        return self

    def prever(self, h):
        return PrevisaoSerie(np.asarray(self.res.forecast(h)))


class HoltWintersSerie(_Estatistico):
    def ajustar(self, y, periodo=1):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = np.asarray(y, dtype=float)
        saz = self.params.get("sazonalidade", "add")
        if saz == "mul" and y.min() <= 0:
            raise ValueError("A sazonalidade multiplicativa exige valores sempre positivos.")
        tendencia = "add" if self.params.get("tendencia", "add") == "add" else None
        w = self._silencioso()
        try:
            self.res = ExponentialSmoothing(y, trend=tendencia, seasonal=saz, seasonal_periods=max(int(periodo), 2),
                                            initialization_method="estimated").fit()
        finally:
            w.__exit__(None, None, None)
        return self

    def prever(self, h):
        return PrevisaoSerie(np.asarray(self.res.forecast(h)))


class _RedeSerie(ModeloSerie):
    """LSTM/GRU em PyTorch (CPU). O estado guarda-se como arrays NumPy, por isso o modelo pode ir para disco com joblib."""
    celula = "lstm"

    def _criar_rede(self, unidades: int):
        import torch.nn as nn
        celula = self.celula

        class Rede(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = (nn.LSTM if celula == "lstm" else nn.GRU)(1, unidades, batch_first=True)
                self.saida = nn.Linear(unidades, 1)

            def forward(self, x):
                o, _ = self.rnn(x)
                return self.saida(o[:, -1, :])
        return Rede()

    def ajustar(self, y, periodo=1):
        import torch
        torch.manual_seed(self.random_state)
        torch.set_num_threads(2)                         # evita competir com o resto do servidor
        y = np.asarray(y, dtype=float)
        self.mu, self.sigma = float(y.mean()), float(y.std()) or 1.0
        z = (y - self.mu) / self.sigma
        W = int(self.params.get("janela", 0)) or min(max(int(periodo), 8), 30)
        W = max(2, min(W, len(z) // 3))
        self.W, self.unidades = W, int(self.params.get("unidades", 32))
        X = np.array([z[i - W:i] for i in range(W, len(z))], dtype=np.float32)[:, :, None]
        t = z[W:].astype(np.float32)[:, None]
        rede = self._criar_rede(self.unidades)
        otim = torch.optim.Adam(rede.parameters(), lr=0.01)
        perda = torch.nn.MSELoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(t)
        rng = np.random.default_rng(self.random_state)
        for _ in range(int(self.params.get("epocas", 60))):
            ordem = rng.permutation(len(Xt))
            for i in range(0, len(ordem), 32):
                b = torch.from_numpy(ordem[i:i + 32])
                otim.zero_grad()
                perda(rede(Xt[b]), yt[b]).backward()
                otim.step()
        self._rede = rede
        self.pesos = {k: v.detach().cpu().numpy().copy() for k, v in rede.state_dict().items()}
        self.cauda = list(z[-W:])
        return self

    def _garantir_rede(self):
        import torch
        if getattr(self, "_rede", None) is None:
            self._rede = self._criar_rede(self.unidades)
            self._rede.load_state_dict({k: torch.from_numpy(v) for k, v in self.pesos.items()})
        self._rede.eval()

    def prever(self, h):
        import torch
        self._garantir_rede()
        janela, saidas = list(self.cauda), []
        with torch.no_grad():
            for _ in range(h):
                x = torch.tensor(np.array(janela[-self.W:], dtype=np.float32)[None, :, None])
                v = float(self._rede(x).item())
                janela.append(v)
                saidas.append(v * self.sigma + self.mu)
        return PrevisaoSerie(np.array(saidas))

    def __getstate__(self):
        estado = dict(self.__dict__)
        estado["_rede"] = None
        return estado


class LstmSerie(_RedeSerie):
    celula = "lstm"


class GruSerie(_RedeSerie):
    celula = "gru"


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação e comparação
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ResultadoSerie:
    id: str
    nome: str
    estado: str = OK
    motivo: str = ""
    params: dict = field(default_factory=dict)
    metricas_validacao: dict = field(default_factory=dict)
    cv: dict = field(default_factory=dict)               # {metrica: (media, desvio)} nos blocos da validação progressiva
    metricas_teste: dict = field(default_factory=dict)
    pred_teste: np.ndarray = None
    previsao_futura: PrevisaoSerie = None
    modelo_final: object = None                          # ajustado a TODA a série (para guardar e voltar a prever)
    tempo_treino: float = 0.0
    avisos: list = field(default_factory=list)
    complexidade: int = 1
    referencia: bool = False

    @property
    def ok(self) -> bool:
        return self.estado == OK


def _resumo(valores: list) -> dict:
    out = {}
    for k in ("mae", "rmse", "mape", "mase"):
        v = [x[k] for x in valores if x.get(k) is not None]
        out[k] = (float(np.mean(v)), float(np.std(v))) if len(v) == len(valores) and v else None
    return {k: v for k, v in out.items() if v is not None}


def avaliar_modelo_serie(modelo_id: str, serie: SerieTemporal, div: DivisaoTemporal, params: dict = None,
                         random_state: int = 42) -> ResultadoSerie:
    """Validação progressiva, avaliação final no teste e ajuste final à série toda. Nunca levanta erro do algoritmo."""
    try:
        info = reg.obter(modelo_id)
    except KeyError as e:
        return ResultadoSerie(modelo_id, modelo_id, FALHOU, str(e).strip("'\""))
    r = ResultadoSerie(info.id, info.nome, complexidade=info.complexidade, referencia=info.categoria == "Referência")
    if info.tipo_modelo != "serie":
        r.estado, r.motivo = INCOMPATIVEL, "Este algoritmo não é para séries temporais."
        return r
    if not info.disponivel():
        r.estado, r.motivo = INDISPONIVEL, info.motivo_indisponivel()
        return r
    try:
        ok, motivo = info.verifica(serie) if info.verifica else (True, "")
    except Exception as e:
        ok, motivo = False, _curto(e)
    if not ok:
        r.estado, r.motivo = INCOMPATIVEL, motivo
        return r
    r.params = reg.completar_parametros(info, params)
    y, h, m = serie.valores, div.horizonte, serie.periodo
    a, b = div.fim_treino, div.fim_validacao
    try:
        with _vaga(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            folds = []
            for o in div.origens_cv:                     # cada bloco só treina com o passado
                mod = info.construir(r.params, random_state).ajustar(y[:o], m)
                folds.append(ev.metricas_serie(y[o:o + h], mod.prever(h).media, y[:o], m))
            r.cv = _resumo(folds)
            if folds and div.tem_validacao:
                r.metricas_validacao = folds[-1]         # o último bloco é exactamente a validação [fim_treino, fim_validacao)
            mod = info.construir(r.params, random_state).ajustar(y[:b], m)
            r.pred_teste = mod.prever(h).media
            r.metricas_teste = ev.metricas_serie(y[b:], r.pred_teste, y[:b], m)
            final = info.construir(r.params, random_state).ajustar(y, m)
            r.previsao_futura = final.prever(h)
            r.modelo_final = final
            r.tempo_treino = time.perf_counter() - t0
        if not info.construir(r.params, random_state).suporta_intervalos:
            r.previsao_futura.inferior = r.previsao_futura.superior = None     # nunca inventar intervalos
    except TimeoutError as e:
        r.estado, r.motivo = IGNORADO, str(e)
    except Exception as e:
        log.exception("Falha na série com %s", info.id)
        r.estado, r.motivo, r.modelo_final = FALHOU, _curto(e), None
    return r


@dataclass
class ComparacaoSerie:
    serie: SerieTemporal
    divisao: DivisaoTemporal
    resultados: list
    tempo_total: float = 0.0
    avisos: list = field(default_factory=list)
    interrompido: bool = False

    @property
    def horizonte(self) -> int:
        return self.divisao.horizonte

    def validos(self) -> list:
        return [r for r in self.resultados if r.ok]

    def por_id(self, modelo_id: str) -> ResultadoSerie:
        for r in self.resultados:
            if r.id == modelo_id:
                return r
        raise KeyError(modelo_id)

    def _melhor_referencia(self):
        refs = [r for r in self.validos() if r.referencia and r.cv.get("mae")]
        return min(refs, key=lambda r: r.cv["mae"][0]) if refs else None

    def tabela(self, ordenar_por: str = "mae") -> pd.DataFrame:
        """Métricas no TESTE (últimos h pontos), mais o MAE médio da validação progressiva e o ganho face à melhor referência."""
        ref = self._melhor_referencia()
        linhas = []
        for r in self.resultados:
            ok = r.ok
            mt = r.metricas_teste if ok else {}
            ganho = None
            if ok and ref and r.cv.get("mae") and ref.cv["mae"][0] > 0:
                ganho = 1 - r.cv["mae"][0] / ref.cv["mae"][0]
            linhas.append({
                "Modelo": r.nome + (" (referência)" if r.referencia else ""),
                "MAE": mt.get("mae"), "RMSE": mt.get("rmse"), "MAPE (%)": mt.get("mape"), "MASE": mt.get("mase"),
                "MAE (validação progressiva)": r.cv["mae"][0] if ok and r.cv.get("mae") else None,
                "Ganho vs. referência": ganho,
                "Tempo de treino (s)": r.tempo_treino if ok else None,
                "Estado": ROTULO_ESTADO[REFERENCIA if (ok and r.referencia) else r.estado] + (f": {r.motivo}" if r.motivo and not ok else ""),
                "_id": r.id, "_ok": ok})
        df = pd.DataFrame(linhas)
        if len(df) and ordenar_por:
            col = {"mae": "MAE", "rmse": "RMSE", "mase": "MASE", "mape": "MAPE (%)", "tempo": "Tempo de treino (s)",
                   "cv": "MAE (validação progressiva)"}.get(ordenar_por, "MAE")
            df["_o"] = ~df["_ok"].astype(bool)
            df = df.sort_values(["_o", col], na_position="last").drop(columns="_o")
        return df.reset_index(drop=True)

    def tabela_visivel(self, ordenar_por: str = "mae") -> pd.DataFrame:
        return self.tabela(ordenar_por).drop(columns=["_id", "_ok"])

    def previsoes_dataframe(self, modelo_id: str) -> pd.DataFrame:
        """Histórico + previsão do teste + futuro, com intervalos quando o modelo os suporta (para exportar)."""
        r = self.por_id(modelo_id)
        if not r.ok:
            raise ValueError("Este modelo não foi treinado com sucesso.")
        s, b = self.serie, self.divisao.fim_validacao
        hist = pd.DataFrame({"data": s.datas, "valor_real": s.valores, "previsao": np.nan, "inferior": np.nan, "superior": np.nan,
                             "fase": "historico"})
        hist.loc[hist.index[b:], "previsao"] = r.pred_teste
        hist.loc[hist.index[b:], "fase"] = "teste"
        fut = r.previsao_futura
        futuro = pd.DataFrame({"data": datas_futuras(s, len(fut.media)), "valor_real": np.nan, "previsao": fut.media,
                               "inferior": fut.inferior if fut.inferior is not None else np.nan,
                               "superior": fut.superior if fut.superior is not None else np.nan, "fase": "futuro"})
        return pd.concat([hist, futuro], ignore_index=True)


def comparar_series(serie: SerieTemporal, ids: list, horizonte: int, n_splits: int = 3, params_por_modelo: dict = None,
                    random_state: int = 42, max_modelos: int = None, orcamento_s: float = None, progresso=None,
                    cancelar=None, incluir_referencias: bool = True) -> ComparacaoSerie:
    """Compara modelos de séries com a mesma divisão cronológica. Inclui sempre as referências ingénuas para comparar."""
    params_por_modelo = params_por_modelo or {}
    div = dividir_temporal(serie.n, horizonte, serie.periodo, n_splits)
    ids = [i for k, i in enumerate(ids) if i not in ids[:k]]
    if incluir_referencias:
        refs = ["serie_naive"] + (["serie_snaive"] if serie.periodo > 1 else [])
        ids = refs + [i for i in ids if i not in refs]
    comp = ComparacaoSerie(serie, div, [], avisos=list(serie.avisos) + list(div.avisos))
    t0, n = time.perf_counter(), len(ids)
    for k, mid in enumerate(ids):
        try:
            nome = reg.obter(mid).nome
        except KeyError:
            nome = mid
        eh_ref = mid in ("serie_naive", "serie_snaive")
        if not eh_ref and max_modelos is not None and sum(1 for r in comp.resultados if not r.referencia) >= max_modelos:
            comp.resultados.append(ResultadoSerie(mid, nome, IGNORADO, f"Limite de {max_modelos} modelos atingido."))
            continue
        if cancelar is not None and cancelar():
            comp.interrompido = True
            comp.resultados.append(ResultadoSerie(mid, nome, IGNORADO, "Cancelado pelo utilizador."))
            continue
        if orcamento_s is not None and time.perf_counter() - t0 > orcamento_s and not eh_ref:
            comp.interrompido = True
            comp.resultados.append(ResultadoSerie(mid, nome, IGNORADO, f"Limite de tempo ({orcamento_s:.0f} s) atingido."))
            continue
        if progresso:
            progresso(k, n, nome)
        comp.resultados.append(avaliar_modelo_serie(mid, serie, div, params_por_modelo.get(mid), random_state))
    comp.tempo_total = time.perf_counter() - t0
    if comp.interrompido:
        comp.avisos.append("A comparação foi interrompida antes de avaliar todos os modelos.")
    if progresso:
        progresso(n, n, "")
    return comp


def recomendar_serie(comp: ComparacaoSerie, criterio: str = "mae") -> Recomendacao:
    """Recomenda pelo erro médio da validação progressiva (só passado), com a regra do um desvio-padrão e comparando com a ingénua."""
    rec = Recomendacao(criterio=ev.rotulo("serie", criterio))
    cands = [r for r in comp.validos() if not r.referencia]
    refs = [r for r in comp.validos() if r.referencia]
    if not cands and not refs:
        rec.justificacao = "Nenhum modelo foi treinado com sucesso."
        return rec

    def cv_de(r):
        v = r.cv.get(criterio) or r.cv.get("mae")
        return v

    usar_cv = bool(cands) and all(cv_de(r) for r in cands)
    base = "validação progressiva (cada bloco só treina com o passado)" if usar_cv else "conjunto de teste"
    rec.base = base

    def valor(r):
        if usar_cv:
            return cv_de(r)[0]
        return r.metricas_teste.get(criterio) if r.metricas_teste.get(criterio) is not None else r.metricas_teste["mae"]

    pool = cands or refs
    melhor = min(pool, key=valor)
    vm = valor(melhor)
    if usar_cv and cands:
        tol = cv_de(melhor)[1]
        proximos = [r for r in cands if valor(r) - vm <= tol]
        escolhido = min(proximos, key=lambda r: (r.complexidade, r.tempo_treino))
    else:
        proximos, escolhido = [melhor], melhor
        if cands:
            rec.avisos.append("Sem validação progressiva (série curta), a escolha usa o teste e é pouco fiável.")
    ve = valor(escolhido)
    melhor_ref = min(refs, key=lambda r: cv_de(r)[0] if (usar_cv and cv_de(r)) else valor(r)) if refs else None
    vr = None
    if melhor_ref:
        vr = cv_de(melhor_ref)[0] if (usar_cv and cv_de(melhor_ref)) else melhor_ref.metricas_teste.get(criterio, melhor_ref.metricas_teste["mae"])
    if cands and melhor_ref and vr is not None and ve >= vr:
        rec.modelo_id, rec.nome = melhor_ref.id, melhor_ref.nome
        rec.justificacao = (f"Nenhum modelo superou a referência «{melhor_ref.nome}» ({rec.criterio} = {vr:.4g} na {base}). "
                            f"O melhor foi {escolhido.nome} com {ve:.4g}. Esta série pode ser pouco previsível, ou os dados são insuficientes. "
                            "Recomenda-se usar a referência simples, ou juntar mais dados.")
        rec.avisos.append("Nenhum modelo supera a previsão ingénua.")
        return rec
    rec.modelo_id, rec.nome = escolhido.id, escolhido.nome
    partes = [f"**{escolhido.nome}** é o recomendado segundo o critério **{rec.criterio}** (menor é melhor), calculado pela {base}: {ve:.4g}"
              + (f" ± {cv_de(escolhido)[1]:.2g}" if usar_cv else "") + "."]
    if usar_cv and len(proximos) > 1:
        partes.append(f"{len(proximos)} modelos ficaram dentro de um desvio-padrão do melhor ({melhor.nome}: {vm:.4g}); escolheu-se o mais simples e rápido.")
    if vr is not None:
        partes.append(f"A referência «{melhor_ref.nome}» teve {vr:.4g}, ou seja o modelo reduz o erro em {1 - ve / vr:.0%}." if vr > 0 else "")
    mt = escolhido.metricas_teste
    partes.append(f"Nos últimos {comp.horizonte} pontos (teste final) obteve MAE {mt['mae']:.4g} e RMSE {mt['rmse']:.4g}"
                  + (f", MASE {mt['mase']:.3g}" if mt.get("mase") is not None else "") + ".")
    partes.append(f"É o melhor entre os {len(cands)} modelos avaliados, não necessariamente o melhor possível.")
    rec.justificacao = " ".join(p for p in partes if p)
    rec.alternativas = [(r.nome, valor(r)) for r in sorted(cands, key=valor) if r.id != escolhido.id][:5]
    if escolhido.previsao_futura is not None and escolhido.previsao_futura.inferior is None:
        rec.avisos.append("Este modelo dá só previsões pontuais (sem intervalo de confiança).")
    if mt.get("mase") is not None and mt["mase"] > 1:
        rec.avisos.append("MASE acima de 1 no teste: neste período o modelo errou mais do que a previsão ingénua.")
    return rec
