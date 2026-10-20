"""Catálogo de algoritmos.

Cada algoritmo é um `ModeloInfo`: nome, categoria, descrição, tipos de problema, parâmetros principais (com limites,
para a interface construir os controlos), se precisa de escala, de que biblioteca depende e como se constrói.
Os algoritmos cuja biblioteca opcional não existe aparecem no catálogo como «indisponível» com o motivo; nunca
partem a aplicação.
"""
import functools
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

REGRESSAO, CLASSIFICACAO, SERIE = "regressao", "classificacao", "serie"
NOMES_PROBLEMA = {REGRESSAO: "Regressão", CLASSIFICACAO: "Classificação", SERIE: "Série temporal"}


def P(tipo, padrao, minimo=None, maximo=None, opcoes=None, ajuda=""):
    """Especificação de um parâmetro: tipo 'int' | 'float' | 'choice'."""
    return dict(tipo=tipo, padrao=padrao, min=minimo, max=maximo, opcoes=opcoes, ajuda=ajuda)


@dataclass(frozen=True)
class ModeloInfo:
    id: str
    nome: str
    categoria: str
    descricao: str
    problemas: tuple
    construir: Callable                       # (params, random_state) -> estimador (ou ModeloSerie)
    parametros: dict = field(default_factory=dict)
    escalar: bool = False                     # precisa de números na mesma escala
    nao_negativo: bool = False                # precisa de entradas >= 0 (Naive Bayes multinomial)
    dependencia: Optional[str] = None         # biblioteca opcional
    complexidade: int = 1                     # 1 (simples) a 5 (muito complexo): desempata recomendações
    verifica: Optional[Callable] = None       # (dados) -> (ok, motivo)
    tipo_modelo: str = "tabular"              # 'tabular' | 'serie'

    def disponivel(self) -> bool:
        return self.dependencia is None or _dependencia_ok(self.dependencia)

    def motivo_indisponivel(self) -> str:
        return "" if self.disponivel() else f"Precisa da biblioteca «{self.dependencia}», que não está instalada neste ambiente."


@functools.lru_cache(maxsize=None)
def _dependencia_ok(nome: str) -> bool:
    """True se a biblioteca existe e importa. O torch só é procurado (importá-lo é pesado)."""
    try:
        if importlib.util.find_spec(nome) is None:
            return False
        if nome == "torch":
            return True
        importlib.import_module(nome)
        return True
    except Exception:
        return False


_REGISTO: dict = {}


def registar(info: ModeloInfo) -> ModeloInfo:
    _REGISTO[info.id] = info
    return info


def remover(modelo_id: str):
    _REGISTO.pop(modelo_id, None)


def obter(modelo_id: str) -> ModeloInfo:
    if modelo_id not in _REGISTO:
        raise KeyError(f"Algoritmo desconhecido: {modelo_id}")
    return _REGISTO[modelo_id]


def listar(problema: str = None, apenas_disponiveis: bool = False, categoria: str = None) -> list:
    out = []
    for m in _REGISTO.values():
        if problema and problema not in m.problemas:
            continue
        if categoria and m.categoria != categoria:
            continue
        if apenas_disponiveis and not m.disponivel():
            continue
        out.append(m)
    return out


def categorias(problema: str = None) -> list:
    vistas = []
    for m in listar(problema):
        if m.categoria not in vistas:
            vistas.append(m.categoria)
    return vistas


def completar_parametros(info: ModeloInfo, params: dict = None) -> dict:
    """Junta os valores dados aos padrões, convertendo tipos e respeitando limites. Ignora chaves desconhecidas."""
    params = params or {}
    out = {}
    for nome, spec in info.parametros.items():
        v = params.get(nome, spec["padrao"])
        try:
            if spec["tipo"] == "int":
                v = int(v)
            elif spec["tipo"] == "float":
                v = float(v)
            elif spec["tipo"] == "choice":
                if v not in spec["opcoes"]:
                    v = spec["padrao"]
        except (TypeError, ValueError):
            v = spec["padrao"]
        if spec["tipo"] in ("int", "float"):
            if spec["min"] is not None:
                v = max(spec["min"], v)
            if spec["max"] is not None:
                v = min(spec["max"], v)
        out[nome] = v
    return out


def catalogo(problema: str = None) -> pd.DataFrame:
    """Tabela para a biblioteca de algoritmos."""
    linhas = []
    for m in listar(problema):
        linhas.append({
            "Algoritmo": m.nome,
            "Categoria": m.categoria,
            "Descrição": m.descricao,
            "Problema": " / ".join(NOMES_PROBLEMA[p] for p in m.problemas),
            "Parâmetros principais": ", ".join(f"{k} = {v['padrao']}" for k, v in m.parametros.items()) or "—",
            "Disponível": "Sim" if m.disponivel() else "Não: " + m.motivo_indisponivel(),
            "id": m.id,
        })
    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# Construtores (imports dentro das funções: só se paga o que se usa)
# ══════════════════════════════════════════════════════════════════════════════
def _prof(v):
    return None if int(v) <= 0 else int(v)


def _ttr(est):
    """Regressores sensíveis à escala do alvo (SVR, MLP): o alvo é normalizado só durante o treino."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def _camadas(texto):
    return tuple(int(x) for x in str(texto).split(",") if x.strip())


# ── verificações de compatibilidade ──
def _verifica_polinomial(dados):
    n = _n_colunas(dados)
    if n > 40:
        return False, f"Regressão polinomial com {n} variáveis geraria centenas de colunas. Escolhe menos variáveis (máx. 40)."
    return True, ""


def _verifica_lento(limite):
    def f(dados):
        if dados.n_treino > limite:
            return False, f"Com {dados.n_treino} exemplos de treino este algoritmo seria lento demais (limite {limite})."
        return True, ""
    return f


def _verifica_nao_negativo(dados):
    if dados.num_cols:
        minimo = dados.X_train[dados.num_cols].min().min()
        if pd.notna(minimo) and minimo < 0:
            return False, "O Naive Bayes multinomial só aceita valores ≥ 0 (contagens ou frequências); há valores negativos."
    return True, ""


def _n_colunas(dados):
    from modules.ml_engine.preprocessing import estimar_n_colunas
    return estimar_n_colunas(dados)


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSÃO
# ══════════════════════════════════════════════════════════════════════════════
def _reg_linear(p, rs):
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _reg_poly(p, rs):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    return make_pipeline(PolynomialFeatures(degree=p["grau"], include_bias=False), LinearRegression())


def _reg_ridge(p, rs):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=p["alpha"])


def _reg_lasso(p, rs):
    from sklearn.linear_model import Lasso
    return Lasso(alpha=p["alpha"], max_iter=20000, random_state=rs)


def _reg_enet(p, rs):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=20000, random_state=rs)


def _reg_bayes(p, rs):
    from sklearn.linear_model import BayesianRidge
    return BayesianRidge()


def _reg_tree(p, rs):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=_prof(p["max_depth"]), min_samples_leaf=p["min_samples_leaf"], random_state=rs)


def _reg_rf(p, rs):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=_prof(p["max_depth"]), random_state=rs, n_jobs=1)


def _reg_et(p, rs):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=p["n_estimators"], max_depth=_prof(p["max_depth"]), random_state=rs, n_jobs=1)


def _reg_gb(p, rs):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                     max_depth=p["max_depth"], random_state=rs)


def _reg_hgb(p, rs):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                         max_depth=_prof(p["max_depth"]), random_state=rs)


def _reg_svr(p, rs):
    from sklearn.svm import SVR
    return _ttr(SVR(C=p["C"], kernel=p["kernel"], epsilon=p["epsilon"]))


def _reg_knn(p, rs):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=p["n_neighbors"], weights=p["weights"])


def _reg_mlp(p, rs):
    from sklearn.neural_network import MLPRegressor
    return _ttr(MLPRegressor(hidden_layer_sizes=_camadas(p["camadas"]), max_iter=p["max_iter"], random_state=rs,
                             early_stopping=False, n_iter_no_change=20))


def _reg_xgb(p, rs):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                        subsample=p["subsample"], random_state=rs, n_jobs=1, verbosity=0, tree_method="hist")


def _reg_lgbm(p, rs):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                         random_state=rs, n_jobs=1, verbose=-1)


_NEST = lambda d=200: P("int", d, 10, 1000, ajuda="Número de árvores.")
_PROF0 = lambda d=0: P("int", d, 0, 50, ajuda="Profundidade máxima (0 = sem limite).")
_LR = lambda d=0.1: P("float", d, 0.001, 1.0, ajuda="Passo de aprendizagem (menor = mais lento e mais cuidadoso).")

for _m in [
    ModeloInfo("linear_reg", "Linear Regression", "Modelos lineares", "Ajusta uma recta (ou plano) aos dados. Simples e fácil de explicar.",
               (REGRESSAO,), _reg_linear, {}, escalar=True, complexidade=1),
    ModeloInfo("poly_reg", "Polynomial Regression", "Modelos lineares", "Regressão linear sobre potências das variáveis: apanha curvas simples.",
               (REGRESSAO,), _reg_poly, {"grau": P("int", 2, 2, 4, ajuda="Grau do polinómio.")}, escalar=True, complexidade=2,
               verifica=_verifica_polinomial),
    ModeloInfo("ridge", "Ridge Regression", "Modelos lineares", "Regressão linear com penalização que encolhe os coeficientes (menos sobreajuste).",
               (REGRESSAO,), _reg_ridge, {"alpha": P("float", 1.0, 0.0001, 1000.0, ajuda="Força da penalização.")}, escalar=True, complexidade=1),
    ModeloInfo("lasso", "Lasso Regression", "Modelos lineares", "Como a Ridge, mas pode anular variáveis inúteis (selecção automática).",
               (REGRESSAO,), _reg_lasso, {"alpha": P("float", 0.1, 0.0001, 100.0, ajuda="Força da penalização.")}, escalar=True, complexidade=1),
    ModeloInfo("elasticnet", "Elastic Net", "Modelos lineares", "Mistura de Ridge e Lasso.",
               (REGRESSAO,), _reg_enet, {"alpha": P("float", 0.1, 0.0001, 100.0, ajuda="Força da penalização."),
                                         "l1_ratio": P("float", 0.5, 0.0, 1.0, ajuda="0 = Ridge, 1 = Lasso.")}, escalar=True, complexidade=2),
    ModeloInfo("bayesian_ridge", "Bayesian Ridge", "Modelos lineares", "Regressão linear bayesiana: ajusta sozinha a penalização.",
               (REGRESSAO,), _reg_bayes, {}, escalar=True, complexidade=2),
    ModeloInfo("tree_reg", "Decision Tree Regressor", "Árvores e florestas", "Faz perguntas sucessivas sobre as variáveis. Fácil de visualizar; tende a decorar.",
               (REGRESSAO,), _reg_tree, {"max_depth": P("int", 8, 0, 50, ajuda="Profundidade máxima (0 = sem limite)."),
                                         "min_samples_leaf": P("int", 1, 1, 100, ajuda="Mínimo de exemplos por folha.")}, complexidade=2),
    ModeloInfo("rf_reg", "Random Forest Regressor", "Árvores e florestas", "Muitas árvores a votar. Robusta e boa em quase tudo.",
               (REGRESSAO,), _reg_rf, {"n_estimators": _NEST(), "max_depth": _PROF0()}, complexidade=3),
    ModeloInfo("extra_trees_reg", "Extra Trees Regressor", "Árvores e florestas", "Como a Random Forest, com cortes mais aleatórios (mais rápida).",
               (REGRESSAO,), _reg_et, {"n_estimators": _NEST(), "max_depth": _PROF0()}, complexidade=3),
    ModeloInfo("gb_reg", "Gradient Boosting Regressor", "Boosting", "Árvores pequenas em sequência, cada uma corrige o erro da anterior.",
               (REGRESSAO,), _reg_gb, {"n_estimators": _NEST(), "learning_rate": _LR(), "max_depth": P("int", 3, 1, 10)}, complexidade=4),
    ModeloInfo("hist_gb_reg", "HistGradientBoosting Regressor", "Boosting", "Boosting rápido do scikit-learn; aceita valores em falta e muitos dados.",
               (REGRESSAO,), _reg_hgb, {"max_iter": P("int", 200, 10, 1000), "learning_rate": _LR(), "max_depth": _PROF0()}, complexidade=4),
    ModeloInfo("svr", "Support Vector Regression", "Máquinas de vectores", "Procura uma «faixa» que contenha os dados; boa com poucos exemplos.",
               (REGRESSAO,), _reg_svr, {"C": P("float", 1.0, 0.01, 1000.0, ajuda="Rigidez (maior = ajusta mais)."),
                                        "kernel": P("choice", "rbf", opcoes=["rbf", "linear", "poly"]),
                                        "epsilon": P("float", 0.1, 0.0, 5.0, ajuda="Largura da faixa sem penalização.")},
               escalar=True, complexidade=3, verifica=_verifica_lento(20000)),
    ModeloInfo("knn_reg", "K-Nearest Neighbors Regressor", "Vizinhos", "Prevê a média dos K casos mais parecidos.",
               (REGRESSAO,), _reg_knn, {"n_neighbors": P("int", 5, 1, 50, ajuda="Número de vizinhos (K)."),
                                        "weights": P("choice", "uniform", opcoes=["uniform", "distance"])}, escalar=True, complexidade=2),
    ModeloInfo("mlp_reg", "MLP Regressor", "Redes neuronais", "Rede neuronal com camadas escondidas. Flexível; precisa de mais dados.",
               (REGRESSAO,), _reg_mlp, {"camadas": P("choice", "64,32", opcoes=["32", "64,32", "128,64", "100"], ajuda="Neurónios por camada."),
                                        "max_iter": P("int", 500, 50, 2000, ajuda="Épocas máximas.")}, escalar=True, complexidade=4),
    ModeloInfo("xgb_reg", "XGBoost Regressor", "Boosting", "Boosting optimizado, muito usado em competições.",
               (REGRESSAO,), _reg_xgb, {"n_estimators": P("int", 300, 10, 2000), "learning_rate": _LR(0.05),
                                        "max_depth": P("int", 6, 1, 15), "subsample": P("float", 0.9, 0.3, 1.0)},
               dependencia="xgboost", complexidade=5),
    ModeloInfo("lgbm_reg", "LightGBM Regressor", "Boosting", "Boosting muito rápido em tabelas grandes.",
               (REGRESSAO,), _reg_lgbm, {"n_estimators": P("int", 300, 10, 2000), "learning_rate": _LR(0.05),
                                         "num_leaves": P("int", 31, 4, 256)}, dependencia="lightgbm", complexidade=5),
]:
    registar(_m)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
_CW = lambda: P("choice", "nenhum", opcoes=["nenhum", "balanced"], ajuda="«balanced» dá mais peso às classes raras.")


def _cw(p):
    return None if p.get("class_weight", "nenhum") == "nenhum" else "balanced"


def _clf_logistic(p, rs):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=p["C"], class_weight=_cw(p), max_iter=3000, random_state=rs)


def _clf_tree(p, rs):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=_prof(p["max_depth"]), min_samples_leaf=p["min_samples_leaf"],
                                  class_weight=_cw(p), random_state=rs)


def _clf_rf(p, rs):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=_prof(p["max_depth"]), class_weight=_cw(p),
                                  random_state=rs, n_jobs=1)


def _clf_et(p, rs):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=p["n_estimators"], max_depth=_prof(p["max_depth"]), class_weight=_cw(p),
                                random_state=rs, n_jobs=1)


def _clf_gb(p, rs):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                      max_depth=p["max_depth"], random_state=rs)


def _clf_hgb(p, rs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                          max_depth=_prof(p["max_depth"]), class_weight=_cw(p), random_state=rs)


def _clf_svc(p, rs):
    from sklearn.svm import SVC
    return SVC(C=p["C"], kernel=p["kernel"], class_weight=_cw(p), random_state=rs)


def _clf_knn(p, rs):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"])


def _clf_gnb(p, rs):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(var_smoothing=p["var_smoothing"])


def _clf_mnb(p, rs):
    from sklearn.naive_bayes import MultinomialNB
    return MultinomialNB(alpha=p["alpha"])


def _clf_mlp(p, rs):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=_camadas(p["camadas"]), max_iter=p["max_iter"], random_state=rs,
                         early_stopping=False, n_iter_no_change=20)


def _clf_xgb(p, rs):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                         subsample=p["subsample"], random_state=rs, n_jobs=1, verbosity=0, tree_method="hist")


def _clf_lgbm(p, rs):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                          class_weight=_cw(p), random_state=rs, n_jobs=1, verbose=-1)


for _m in [
    ModeloInfo("logistic", "Logistic Regression", "Modelos lineares", "Calcula a probabilidade de cada classe com uma fórmula linear. Base de comparação clássica.",
               (CLASSIFICACAO,), _clf_logistic, {"C": P("float", 1.0, 0.001, 1000.0, ajuda="Inverso da penalização."), "class_weight": _CW()},
               escalar=True, complexidade=1),
    ModeloInfo("tree_clf", "Decision Tree Classifier", "Árvores e florestas", "Perguntas sim/não sucessivas até chegar a uma classe.",
               (CLASSIFICACAO,), _clf_tree, {"max_depth": P("int", 8, 0, 50, ajuda="Profundidade máxima (0 = sem limite)."),
                                             "min_samples_leaf": P("int", 1, 1, 100), "class_weight": _CW()}, complexidade=2),
    ModeloInfo("rf_clf", "Random Forest Classifier", "Árvores e florestas", "Muitas árvores a votar. Robusta e boa em quase tudo.",
               (CLASSIFICACAO,), _clf_rf, {"n_estimators": _NEST(), "max_depth": _PROF0(), "class_weight": _CW()}, complexidade=3),
    ModeloInfo("extra_trees_clf", "Extra Trees Classifier", "Árvores e florestas", "Como a Random Forest, com cortes mais aleatórios.",
               (CLASSIFICACAO,), _clf_et, {"n_estimators": _NEST(), "max_depth": _PROF0(), "class_weight": _CW()}, complexidade=3),
    ModeloInfo("gb_clf", "Gradient Boosting Classifier", "Boosting", "Árvores pequenas em sequência, cada uma corrige a anterior.",
               (CLASSIFICACAO,), _clf_gb, {"n_estimators": _NEST(), "learning_rate": _LR(), "max_depth": P("int", 3, 1, 10)}, complexidade=4),
    ModeloInfo("hist_gb_clf", "HistGradientBoosting Classifier", "Boosting", "Boosting rápido do scikit-learn; aceita valores em falta.",
               (CLASSIFICACAO,), _clf_hgb, {"max_iter": P("int", 200, 10, 1000), "learning_rate": _LR(), "max_depth": _PROF0(),
                                            "class_weight": _CW()}, complexidade=4),
    ModeloInfo("svc", "Support Vector Classifier", "Máquinas de vectores", "Procura a fronteira com a maior margem entre as classes.",
               (CLASSIFICACAO,), _clf_svc, {"C": P("float", 1.0, 0.01, 1000.0), "kernel": P("choice", "rbf", opcoes=["rbf", "linear", "poly"]),
                                            "class_weight": _CW()}, escalar=True, complexidade=3, verifica=_verifica_lento(20000)),
    ModeloInfo("knn_clf", "K-Nearest Neighbors Classifier", "Vizinhos", "Os K casos mais parecidos votam.",
               (CLASSIFICACAO,), _clf_knn, {"n_neighbors": P("int", 5, 1, 50, ajuda="Número de vizinhos (K)."),
                                            "weights": P("choice", "uniform", opcoes=["uniform", "distance"])}, escalar=True, complexidade=2),
    ModeloInfo("gaussian_nb", "Gaussian Naive Bayes", "Naive Bayes", "Probabilidades assumindo variáveis independentes e em forma de sino. Muito rápido.",
               (CLASSIFICACAO,), _clf_gnb, {"var_smoothing": P("float", 1e-9, 1e-12, 1e-3)}, complexidade=1),
    ModeloInfo("multinomial_nb", "Multinomial Naive Bayes", "Naive Bayes", "Para contagens e frequências (≥ 0), como palavras num texto.",
               (CLASSIFICACAO,), _clf_mnb, {"alpha": P("float", 1.0, 0.0001, 10.0, ajuda="Suavização.")},
               nao_negativo=True, complexidade=1, verifica=_verifica_nao_negativo),
    ModeloInfo("mlp_clf", "MLP Classifier", "Redes neuronais", "Rede neuronal com camadas escondidas.",
               (CLASSIFICACAO,), _clf_mlp, {"camadas": P("choice", "64,32", opcoes=["32", "64,32", "128,64", "100"]),
                                            "max_iter": P("int", 500, 50, 2000)}, escalar=True, complexidade=4),
    ModeloInfo("xgb_clf", "XGBoost Classifier", "Boosting", "Boosting optimizado, muito usado em competições.",
               (CLASSIFICACAO,), _clf_xgb, {"n_estimators": P("int", 300, 10, 2000), "learning_rate": _LR(0.05),
                                            "max_depth": P("int", 6, 1, 15), "subsample": P("float", 0.9, 0.3, 1.0)},
               dependencia="xgboost", complexidade=5),
    ModeloInfo("lgbm_clf", "LightGBM Classifier", "Boosting", "Boosting muito rápido em tabelas grandes.",
               (CLASSIFICACAO,), _clf_lgbm, {"n_estimators": P("int", 300, 10, 2000), "learning_rate": _LR(0.05),
                                             "num_leaves": P("int", 31, 4, 256), "class_weight": _CW()},
               dependencia="lightgbm", complexidade=5),
]:
    registar(_m)


# ══════════════════════════════════════════════════════════════════════════════
# SÉRIES TEMPORAIS (as classes estão em forecasting.py; importadas só quando se constrói)
# ══════════════════════════════════════════════════════════════════════════════
def _serie(nome_classe):
    def construir(params, rs):
        from modules.ml_engine import forecasting
        return getattr(forecasting, nome_classe)(params, rs)
    return construir


def _verifica_serie(minimo_pontos=0, periodo_minimo=0, dobro_periodo=False):
    def f(serie):
        n = len(serie.valores)
        if n < minimo_pontos:
            return False, f"Precisa de pelo menos {minimo_pontos} observações (a série tem {n})."
        if periodo_minimo and serie.periodo < periodo_minimo:
            return False, "A série não tem uma sazonalidade identificada (período < 2)."
        if dobro_periodo and n < 2 * serie.periodo + 2:
            return False, f"Precisa de pelo menos 2 ciclos sazonais completos ({2 * serie.periodo + 2} observações)."
        return True, ""
    return f


_SH = lambda: P("choice", "add", opcoes=["add", "mul"], ajuda="Sazonalidade aditiva ou multiplicativa (mul exige valores > 0).")

for _m in [
    ModeloInfo("serie_naive", "Previsão ingénua (último valor)", "Referência", "Repete o último valor observado. Serve de comparação: qualquer modelo tem de a superar.",
               (SERIE,), _serie("NaiveSerie"), {}, tipo_modelo="serie", complexidade=1),
    ModeloInfo("serie_snaive", "Previsão ingénua sazonal", "Referência", "Repete o valor do mesmo ponto do ciclo anterior (ex.: a mesma semana).",
               (SERIE,), _serie("NaiveSazonalSerie"), {}, tipo_modelo="serie", complexidade=1, verifica=_verifica_serie(periodo_minimo=2)),
    ModeloInfo("serie_regressao", "Regressão com variáveis temporais", "Regressão temporal", "Regressão linear sobre a posição no tempo (tendência) e ciclos sazonais.",
               (SERIE,), _serie("RegressaoTemporalSerie"), {"harmonicos": P("int", 2, 1, 6, ajuda="Termos sazonais (seno/co-seno).")},
               tipo_modelo="serie", complexidade=2, verifica=_verifica_serie(minimo_pontos=12)),
    ModeloInfo("serie_rf", "Random Forest para séries", "Aprendizagem automática", "Prevê o próximo valor a partir dos anteriores (lags), passo a passo.",
               (SERIE,), _serie("RandomForestSerie"), {"lags": P("int", 0, 0, 48, ajuda="Valores anteriores usados (0 = automático)."),
                                                      "n_estimators": P("int", 200, 10, 1000)}, tipo_modelo="serie", complexidade=3,
               verifica=_verifica_serie(minimo_pontos=30)),
    ModeloInfo("serie_xgb", "XGBoost para séries", "Aprendizagem automática", "Boosting sobre valores anteriores (lags), passo a passo.",
               (SERIE,), _serie("XGBoostSerie"), {"lags": P("int", 0, 0, 48, ajuda="Valores anteriores usados (0 = automático)."),
                                                 "n_estimators": P("int", 300, 10, 2000)}, dependencia="xgboost", tipo_modelo="serie",
               complexidade=4, verifica=_verifica_serie(minimo_pontos=30)),
    ModeloInfo("serie_arima", "ARIMA", "Estatísticos", "Modelo clássico: passado + erros passados, após diferenciar. Dá intervalos de previsão.",
               (SERIE,), _serie("ArimaSerie"), {"p": P("int", 1, 0, 8), "d": P("int", 1, 0, 2), "q": P("int", 1, 0, 8)},
               dependencia="statsmodels", tipo_modelo="serie", complexidade=3, verifica=_verifica_serie(minimo_pontos=20)),
    ModeloInfo("serie_sarimax", "SARIMAX", "Estatísticos", "ARIMA com sazonalidade. Dá intervalos de previsão.",
               (SERIE,), _serie("SarimaxSerie"), {"p": P("int", 1, 0, 5), "d": P("int", 1, 0, 2), "q": P("int", 1, 0, 5),
                                                 "P": P("int", 1, 0, 2), "D": P("int", 0, 0, 1), "Q": P("int", 1, 0, 2)},
               dependencia="statsmodels", tipo_modelo="serie", complexidade=4, verifica=_verifica_serie(minimo_pontos=30, periodo_minimo=2, dobro_periodo=True)),
    ModeloInfo("serie_ets", "Exponential Smoothing", "Estatísticos", "Média ponderada que dá mais peso ao recente (com tendência opcional).",
               (SERIE,), _serie("SuavizacaoExponencialSerie"), {"tendencia": P("choice", "nenhuma", opcoes=["nenhuma", "add"], ajuda="Tendência aditiva?")},
               dependencia="statsmodels", tipo_modelo="serie", complexidade=2, verifica=_verifica_serie(minimo_pontos=12)),
    ModeloInfo("serie_holt_winters", "Holt-Winters", "Estatísticos", "Suavização com tendência e sazonalidade.",
               (SERIE,), _serie("HoltWintersSerie"), {"sazonalidade": _SH(), "tendencia": P("choice", "add", opcoes=["nenhuma", "add"])},
               dependencia="statsmodels", tipo_modelo="serie", complexidade=3, verifica=_verifica_serie(minimo_pontos=24, periodo_minimo=2, dobro_periodo=True)),
    ModeloInfo("serie_lstm", "LSTM (PyTorch)", "Aprendizagem profunda", "Rede recorrente com memória. Só com séries longas; funciona em CPU.",
               (SERIE,), _serie("LstmSerie"), {"janela": P("int", 0, 0, 120, ajuda="Passos olhados para trás (0 = automático)."),
                                              "epocas": P("int", 60, 5, 300), "unidades": P("int", 32, 4, 128)},
               dependencia="torch", tipo_modelo="serie", complexidade=5, verifica=_verifica_serie(minimo_pontos=100)),
    ModeloInfo("serie_gru", "GRU (PyTorch)", "Aprendizagem profunda", "Rede recorrente mais leve que a LSTM.",
               (SERIE,), _serie("GruSerie"), {"janela": P("int", 0, 0, 120, ajuda="Passos olhados para trás (0 = automático)."),
                                             "epocas": P("int", 60, 5, 300), "unidades": P("int", 32, 4, 128)},
               dependencia="torch", tipo_modelo="serie", complexidade=5, verifica=_verifica_serie(minimo_pontos=100)),
]:
    registar(_m)
