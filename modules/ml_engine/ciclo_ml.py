"""Ciclo completo de ML depois do treino: avaliar, diagnosticar, optimizar, exportar, prever, implementar e monitorizar.

Tudo aqui funciona sem Streamlit (testável) e REUTILIZA o motor existente: `preparar_dados` (divisão e pré-processamento),
`treinar_modelo` (pipelines e métricas), `comparar_modelos`/`recomendar`, `model_persistence` e `prediction_service`.

Regras que este módulo garante:
- O conjunto de teste NUNCA é usado para escolher hiperparâmetros: a procura (GridSearchCV / RandomizedSearchCV) só vê
  `X_train`, com validação cruzada, e o pré-processamento está dentro do Pipeline (ajustado só com o treino de cada bloco).
- O modelo optimizado é treinado e avaliado pelo mesmo caminho dos outros (`treinar_modelo`), por isso é comparável.
- Os diagnósticos são indícios, nunca sentenças: com poucos dados ou sem referência dizem «inconclusivo».
- Não se inventa nenhum número: tudo vem das métricas reais calculadas.
"""
import io
import json
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

from modules.ml_engine import model_evaluation as ev
from modules.ml_engine import model_persistence as mp
from modules.ml_engine import model_registry as reg
from modules.ml_engine import prediction_service as ps
from modules.ml_engine.model_training import OK, ResultadoModelo, _vaga, treinar_modelo, treinar_referencia
from modules.ml_engine.preprocessing import DadosPreparados, construir_preprocessador

PASTA_CODIGO = Path(__file__).resolve().parent / "modelos_codigo"
LIMITE_COMBINACOES = 60          # combinações testadas, no máximo
LIMITE_AJUSTES = 400             # combinações × blocos de validação cruzada
ORCAMENTO_PADRAO_S = 240.0
MAX_LINHAS_PREVISAO = 50_000
LIMITE_SOBREAJUSTE = 0.15        # o mesmo limite que `model_comparison.recomendar` usa


class ErroCiclo(Exception):
    """Erro com mensagem pronta a mostrar ao estudante."""


def principal(tipo: str) -> str:
    return "r2" if tipo == "regressao" else "f1"


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Avaliação
# ══════════════════════════════════════════════════════════════════════════════
def metricas_cartoes(tipo: str) -> list:
    """(chave, rótulo) das métricas mostradas em cartões."""
    if tipo == "regressao":
        return [("mae", "MAE"), ("mse", "MSE"), ("rmse", "RMSE"), ("r2", "R²")]
    return [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]


def interpretar_metrica(tipo: str, chave: str, valor, dados: DadosPreparados) -> str:
    """Frase em português simples que diz o que ESTE valor significa para ESTES dados."""
    if valor is None:
        return "Não foi possível calcular esta métrica."
    if tipo == "regressao":
        y = np.asarray(dados.y_test, dtype=float)
        sd = float(np.std(y))
        media = float(np.mean(np.abs(y)))
        if chave == "mae":
            extra = f" Isso é {valor / media:.0%} do valor médio de «{dados.alvo}»." if media > 0 else ""
            return f"Em média, a previsão falha por {valor:.4g} (na unidade de «{dados.alvo}»).{extra}"
        if chave == "mse":
            return (f"Média dos erros ao quadrado ({valor:.4g}). Os erros grandes pesam muito. "
                    "Como está ao quadrado, para comparar com a unidade do alvo usa o RMSE.")
        if chave == "rmse":
            extra = ""
            if sd > 0:
                extra = (f" O desvio-padrão do alvo no teste é {sd:.4g}: um RMSE abaixo disso quer dizer que o modelo "
                         "faz melhor do que prever sempre a média.") if valor < sd else (
                    f" O desvio-padrão do alvo no teste é {sd:.4g}: este RMSE não é melhor do que prever sempre a média.")
            return f"O erro típico é {valor:.4g}, e os erros grandes pesam mais do que no MAE.{extra}"
        if chave == "r2":
            if valor < 0:
                return "R² negativo: neste teste o modelo é pior do que prever sempre a média."
            return f"O modelo explica {valor:.0%} da variação de «{dados.alvo}» nos dados de teste (1 = perfeito, 0 = como a média)."
        return ""
    base = float(np.bincount(np.asarray(dados.y_test, dtype=int)).max() / max(len(dados.y_test), 1))
    if chave == "accuracy":
        txt = f"De cada 100 casos de teste, o modelo acertou em {valor * 100:.0f}."
        if base > 0.6:
            txt += (f" Cuidado: a classe mais comum tem {base:.0%} dos casos; responder sempre essa já acertaria {base:.0%}, "
                    "por isso olha também para F1 e recall.")
        return txt
    if chave == "precision":
        return (f"Quando o modelo prevê uma classe, acerta em média {valor:.0%} das vezes "
                "(média das classes: todas pesam o mesmo).")
    if chave == "recall":
        return f"Dos casos reais de cada classe, o modelo encontra em média {valor:.0%} (média das classes)."
    if chave == "f1":
        return f"Equilíbrio entre precision e recall ({valor:.0%}). Só é alto se os dois forem altos."
    return ""


def relatorio_classificacao(dados: DadosPreparados, r: ResultadoModelo) -> pd.DataFrame:
    """Classification report (por classe + médias) como tabela. Só classificação."""
    if dados.tipo != "classificacao":
        raise ErroCiclo("O relatório de classificação só existe para problemas de classificação.")
    y_pred = r.predicoes.get("y_pred")
    if y_pred is None:
        raise ErroCiclo("Este modelo não guardou as previsões do teste.")
    rep = classification_report(dados.y_test, y_pred, labels=list(range(len(dados.classes))),
                                target_names=[str(c) for c in dados.classes], output_dict=True, zero_division=0)
    linhas = []
    for nome, v in rep.items():
        if nome == "accuracy":
            linhas.append({"Classe": "accuracy", "Precision": None, "Recall": None, "F1-score": float(v), "Exemplos": int(len(dados.y_test))})
        else:
            linhas.append({"Classe": nome, "Precision": float(v["precision"]), "Recall": float(v["recall"]),
                           "F1-score": float(v["f1-score"]), "Exemplos": int(v["support"])})
    return pd.DataFrame(linhas)


def avaliar_treino(r: ResultadoModelo, dados: DadosPreparados) -> dict:
    """As mesmas métricas do teste, calculadas nos dados de treino (para comparar treino × teste)."""
    if r.pipeline is None:
        raise ErroCiclo("O modelo não tem pipeline treinado.")
    pred = np.asarray(r.pipeline.predict(dados.X_train)).ravel()
    if dados.tipo == "regressao":
        m = ev.metricas_regressao(dados.y_train, pred)
        return {k: m[k] for k in ("r2", "mae", "mse", "rmse")}
    m = ev.metricas_classificacao(dados.y_train, pred, None, len(dados.classes))
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1")}


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Generalização: sobreajuste, subajuste, bom ajustamento
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Diagnostico:
    estado: str                       # 'sobreajuste' | 'subajuste' | 'bom' | 'intermedio' | 'inconclusivo'
    titulo: str
    explicacao: str
    sugestoes: list = field(default_factory=list)
    numeros: dict = field(default_factory=dict)
    avisos: list = field(default_factory=list)
    base_cv: bool = False             # a validação cruzada confirma a leitura


def obter_referencia(dados: DadosPreparados) -> ResultadoModelo:
    """Modelo sem inteligência (média / classe mais frequente), para saber o que é «melhor do que nada»."""
    return treinar_referencia(dados, dados.random_state)


def diagnosticar_generalizacao(r: ResultadoModelo, dados: DadosPreparados, referencia: ResultadoModelo = None) -> Diagnostico:
    """Compara treino, teste (e validação cruzada, se existir). Devolve indícios, nunca um veredicto definitivo."""
    p = principal(dados.tipo)
    nome_p = ev.rotulo(dados.tipo, p)
    tr, te = r.metricas_treino.get(p), r.metricas.get(p)
    cv = r.cv.get(p) if r.cv else None
    ref = referencia.metricas.get(p) if referencia is not None and referencia.ok else None
    avisos, nums = [], {"metrica": nome_p, "treino": tr, "teste": te, "referencia": ref,
                        "cv_media": cv[0] if cv else None, "cv_desvio": cv[1] if cv else None}
    if tr is None or te is None:
        return Diagnostico("inconclusivo", "Sem dados suficientes para diagnosticar",
                           "Faltam as métricas de treino ou de teste, por isso não consigo comparar.", numeros=nums)
    gap = float(tr - te)
    nums["diferenca"] = gap
    pequeno = dados.n_teste < 30 or dados.n_treino < 30
    if dados.n_teste < 30:
        avisos.append(f"O teste tem só {dados.n_teste} exemplos: pequenas diferenças podem ser sorte.")
    if dados.n_treino < 30:
        avisos.append(f"O treino tem só {dados.n_treino} exemplos: é difícil tirar conclusões.")
    base_cv = bool(cv) and (abs(cv[0] - te) <= max(2 * cv[1], 0.03))
    if cv and not base_cv:
        avisos.append(f"A validação cruzada ({nome_p} = {cv[0]:.3f}) e o teste ({te:.3f}) discordam: o resultado do teste pode ter sido "
                      "influenciado pela divisão dos dados.")
    if pequeno:
        return Diagnostico("inconclusivo", "Diagnóstico inconclusivo: poucos dados",
                           f"No treino o {nome_p} é {tr:.3f} e no teste {te:.3f}, mas com tão poucos exemplos não dá para dizer se há "
                           "sobreajuste ou subajuste. Com mais dados (ou validação cruzada) a leitura fica mais fiável.",
                           ["Juntar mais exemplos.", "Usar validação cruzada para ter várias notas em vez de uma."], nums, avisos, base_cv)
    if gap > LIMITE_SOBREAJUSTE:
        sug = ["Juntar mais exemplos de treino, se possível.",
               "Usar um modelo mais simples ou limitar a sua complexidade (por exemplo, menor profundidade das árvores, menos camadas).",
               "Aumentar a regularização (Ridge, Lasso, parâmetro C) ou retirar variáveis pouco úteis.",
               "Experimentar a optimização de hiperparâmetros (etapa 4) com validação cruzada."]
        txt = (f"O {nome_p} no treino é {tr:.3f} mas no teste é só {te:.3f} (diferença de {gap:.3f}). O modelo parece decorar os exemplos "
               "de treino em vez de aprender a regra geral: é o que se chama sobreajuste (overfitting).")
        if base_cv:
            txt += " A validação cruzada, feita só com o treino, aponta no mesmo sentido, o que reforça este indício."
        return Diagnostico("sobreajuste", "Indício de sobreajuste (overfitting)", txt, sug, nums, avisos, base_cv)
    if ref is None:
        return Diagnostico("inconclusivo", "Sem referência para comparar",
                           f"Treino {tr:.3f} e teste {te:.3f} estão próximos, mas sem o modelo de referência não consigo dizer se isto é bom "
                           "ou se o modelo aprendeu pouco.", ["Calcula a referência (prever sempre a média ou a classe mais comum)."],
                           nums, avisos, base_cv)
    acima_teste = float(te - ref)
    acima_treino = float(tr - ref)
    nums["acima_da_referencia_teste"] = acima_teste
    if acima_teste <= 0.02 and acima_treino < 0.10:
        sug = ["Experimentar um algoritmo mais flexível (por exemplo, uma floresta aleatória ou boosting).",
               "Acrescentar variáveis que expliquem melhor o que se quer prever, ou criar novas a partir das existentes.",
               "Reduzir a regularização, se estiveres a usar Ridge, Lasso ou um C pequeno.",
               "Confirmar que os dados têm sinal: talvez as variáveis não expliquem o alvo."]
        txt = (f"Nem no treino ({nome_p} = {tr:.3f}) o modelo se afasta muito da referência simples ({ref:.3f}). "
               "O modelo parece não conseguir aprender o padrão: é o que se chama subajuste (underfitting).")
        return Diagnostico("subajuste", "Indício de subajuste (underfitting)", txt, sug, nums, avisos, base_cv)
    if gap <= 0.05 and acima_teste > 0.05:
        txt = (f"Treino {tr:.3f} e teste {te:.3f} estão próximos (diferença {gap:.3f}) e o teste supera a referência simples ({ref:.3f}). "
               "É um sinal de bom ajustamento: o modelo aprendeu algo que também funciona em dados novos.")
        sug = ["Confirma com validação cruzada para ter mais confiança.",
               "Se queres melhorar, experimenta outros algoritmos e optimiza os hiperparâmetros."]
        return Diagnostico("bom", "Sinal de bom ajustamento", txt, sug, nums, avisos, base_cv)
    txt = (f"Treino {tr:.3f} e teste {te:.3f} (diferença {gap:.3f}); a referência simples dá {ref:.3f}. A diferença entre treino e teste é pequena "
           "mas não tão pequena que se possa dizer que o ajustamento é bom sem mais evidência.")
    return Diagnostico("intermedio", "Ajustamento razoável, sem conclusão firme", txt,
                       ["Usar validação cruzada para confirmar.", "Comparar com outros algoritmos (etapa 3)."], nums, avisos, base_cv)


def curva_aprendizagem(r: ResultadoModelo, dados: DadosPreparados, folds: int = 3, pontos: int = 5,
                       orcamento_s: float = 60.0) -> pd.DataFrame:
    """Nota de treino e de validação com cada vez mais exemplos, SÓ com os dados de treino (o teste não entra)."""
    if r.pipeline is None:
        raise ErroCiclo("O modelo não tem pipeline treinado.")
    cv, folds = _objecto_cv(dados, folds)
    if r.tempo_treino * folds * pontos * 0.7 > orcamento_s:
        raise ErroCiclo("Este modelo demora demasiado para desenhar a curva de aprendizagem. Experimenta um modelo mais rápido.")
    scoring = "r2" if dados.tipo == "regressao" else "f1_macro"
    with _vaga(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tam, tr, va = learning_curve(clone(r.pipeline), dados.X_train, dados.y_train, cv=cv, train_sizes=np.linspace(0.25, 1.0, pontos),
                                     scoring=scoring, n_jobs=1, shuffle=True, random_state=dados.random_state, error_score=np.nan)
    df = pd.DataFrame({"exemplos": tam.astype(int), "treino": np.nanmean(tr, axis=1), "validacao": np.nanmean(va, axis=1),
                       "validacao_desvio": np.nanstd(va, axis=1)})
    df = df.dropna(subset=["treino", "validacao"])
    if df.empty:
        raise ErroCiclo("Não foi possível calcular a curva de aprendizagem com estes dados.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — Comparação (usa model_comparison; aqui só o que falta)
# ══════════════════════════════════════════════════════════════════════════════
def melhor_por_metrica(comp) -> pd.DataFrame:
    """Qual é o modelo recomendado para CADA critério. Critérios diferentes podem favorecer modelos diferentes."""
    from modules.ml_engine.model_comparison import recomendar
    linhas = []
    for crit in ev.CRITERIOS[comp.tipo]:
        rec = recomendar(comp, crit)
        valor = None
        if rec.modelo_id:
            valor = comp.por_id(rec.modelo_id).metricas.get(crit)
        linhas.append({"Critério": ev.rotulo(comp.tipo, crit), "Melhor modelo": rec.nome or "—",
                       "Valor no teste": valor, "Escolhido por": rec.base or "—"})
    return pd.DataFrame(linhas)


def divergem(tabela: pd.DataFrame) -> bool:
    nomes = {n for n in tabela["Melhor modelo"] if n != "—"}
    return len(nomes) > 1


def dados_grafico_comparacao(comp, metrica: str) -> pd.DataFrame:
    """Teste, validação cruzada (média ± desvio) e tempo de cada modelo que treinou com sucesso."""
    linhas = []
    for r in comp.validos():
        cv = r.cv.get(metrica) if r.cv else None
        linhas.append({"modelo": r.nome, "id": r.id, "teste": r.metricas.get(metrica), "cv": cv[0] if cv else None,
                       "cv_desvio": cv[1] if cv else None, "tempo": r.tempo_treino})
    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — Optimização de hiperparâmetros
# ══════════════════════════════════════════════════════════════════════════════
SCORING = {
    "regressao": {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
    "classificacao": {"f1": "f1_macro", "accuracy": "accuracy", "precision": "precision_macro", "recall": "recall_macro"},
}


class _AlgoritmoBase(BaseEstimator):
    """Estimador do scikit-learn que constrói o algoritmo pelo CATÁLOGO do DataForge.

    Expõe os parâmetros do catálogo (ex.: `max_depth`, `camadas`) como parâmetros normais, para o GridSearchCV poder
    variá-los com a mesma tradução que `treinar_modelo` usa (0 = sem limite, camadas em texto, alvo normalizado, ...).
    """

    def __init__(self, modelo_id="", random_state=42, **params):
        self.modelo_id = modelo_id
        self.random_state = random_state
        self.params = dict(params)

    def get_params(self, deep=True):
        return {"modelo_id": self.modelo_id, "random_state": self.random_state, **self.params}

    def set_params(self, **kw):
        for k, v in kw.items():
            if k == "modelo_id":
                self.modelo_id = v
            elif k == "random_state":
                self.random_state = v
            else:
                self.params[k] = v
        return self

    def fit(self, X, y):
        info = reg.obter(self.modelo_id)
        self.est_ = info.construir(reg.completar_parametros(info, self.params), self.random_state)
        self.est_.fit(X, y)
        if hasattr(self.est_, "classes_"):
            self.classes_ = self.est_.classes_
        return self

    def predict(self, X):
        return self.est_.predict(X)

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def decision_function(self, X):
        return self.est_.decision_function(X)


class _AlgoritmoClassificador(ClassifierMixin, _AlgoritmoBase):
    pass


class _AlgoritmoRegressor(RegressorMixin, _AlgoritmoBase):
    pass


def espaco_padrao(info) -> dict:
    """Valores sugeridos para cada parâmetro (o estudante pode mudar). {parâmetro: [valores]}"""
    out = {}
    for nome, s in info.parametros.items():
        if s["tipo"] == "choice":
            out[nome] = list(s["opcoes"])
            continue
        p, lo, hi = s["padrao"], s["min"], s["max"]
        if s["tipo"] == "int":
            cand = [0, 5, 10] if p == 0 else [max(1, p // 2), p, p * 2]
        elif lo and lo > 0 and p > 0 and hi and hi / lo >= 100:
            cand = [p / 10, p, p * 10]
        else:
            cand = [p * 0.5, p, p * 1.5]
        vals = []
        for v in cand:
            v = int(v) if s["tipo"] == "int" else float(v)
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            if v not in vals:
                vals.append(v)
        out[nome] = sorted(vals)
    return out


def validar_espaco(info, texto_por_parametro: dict) -> tuple:
    """Converte o que o estudante escreveu («1, 5, 10») em listas de valores válidos.

    Devolve (espaco, erros). Parâmetros deixados vazios não entram na procura (ficam com o valor actual).
    """
    espaco, erros = {}, []
    for nome, texto in texto_por_parametro.items():
        if nome not in info.parametros:
            erros.append(f"«{nome}» não é um parâmetro de {info.nome}.")
            continue
        s = info.parametros[nome]
        if isinstance(texto, (list, tuple)):
            partes = [str(x).strip() for x in texto]
        else:
            partes = [x.strip() for x in str(texto or "").replace(";", ",").split(",")]
        partes = [x for x in partes if x]
        if not partes:
            continue
        vals = []
        for x in partes:
            try:
                if s["tipo"] == "choice":
                    v = next((o for o in s["opcoes"] if str(o) == x), None)
                    if v is None:
                        raise ValueError
                elif s["tipo"] == "int":
                    f = float(x.replace(",", "."))
                    if f != int(f):
                        raise ValueError
                    v = int(f)
                else:
                    v = float(x)
                    if not np.isfinite(v):
                        raise ValueError
            except ValueError:
                erros.append(f"«{nome}»: «{x}» não é um valor válido." + (f" Opções: {', '.join(map(str, s['opcoes']))}." if s["tipo"] == "choice" else ""))
                continue
            if s["tipo"] in ("int", "float") and ((s["min"] is not None and v < s["min"]) or (s["max"] is not None and v > s["max"])):
                erros.append(f"«{nome}»: {v} está fora do intervalo permitido ({s['min']} a {s['max']}).")
                continue
            if v not in vals:
                vals.append(v)
        if vals:
            espaco[nome] = vals
    return espaco, erros


def contar_combinacoes(espaco: dict) -> int:
    n = 1
    for v in espaco.values():
        n *= max(len(v), 1)
    return n if espaco else 0


def _objecto_cv(dados: DadosPreparados, folds: int):
    """Mesmos blocos que `model_training._validacao_cruzada` usa: assim as notas são comparáveis."""
    if dados.tipo == "classificacao":
        folds = min(folds, int(np.bincount(dados.y_train).min()))
        if folds < 2:
            raise ErroCiclo("Há uma classe com menos de 2 exemplos de treino: não dá para fazer validação cruzada.")
        return StratifiedKFold(folds, shuffle=True, random_state=dados.random_state), folds
    folds = min(folds, dados.n_treino // 2)
    if folds < 2:
        raise ErroCiclo("Há poucos exemplos de treino para validação cruzada.")
    return KFold(folds, shuffle=True, random_state=dados.random_state), folds


@dataclass
class ResultadoOtimizacao:
    metodo: str
    modelo_id: str
    nome: str
    criterio: str
    n_combinacoes: int
    n_testadas: int
    cv_folds: int
    params_originais: dict
    melhores_params: dict
    cv_original: float = None
    cv_otimizado: float = None
    tabela: pd.DataFrame = None
    original: ResultadoModelo = None
    otimizado: ResultadoModelo = None
    tempo: float = 0.0
    avisos: list = field(default_factory=list)
    conclusao: list = field(default_factory=list)


def otimizar(modelo_id: str, dados: DadosPreparados, espaco: dict, metodo: str = "grid", criterio: str = None, n_iter: int = 10,
             cv_folds: int = 3, params_base: dict = None, orcamento_s: float = ORCAMENTO_PADRAO_S) -> ResultadoOtimizacao:
    """GridSearchCV ou RandomizedSearchCV sobre o Pipeline completo, SÓ com os dados de treino.

    Devolve o modelo original e o optimizado, ambos avaliados no teste UMA vez (no fim) e na validação cruzada do treino.
    """
    if metodo not in ("grid", "random"):
        raise ErroCiclo("O método tem de ser «grid» ou «random».")
    try:
        info = reg.obter(modelo_id)
    except KeyError:
        raise ErroCiclo("Algoritmo desconhecido.")
    if not espaco:
        raise ErroCiclo("Escolhe pelo menos um parâmetro com valores para testar.")
    if not info.parametros:
        raise ErroCiclo(f"{info.nome} não tem parâmetros para optimizar.")
    desconhecidos = [k for k in espaco if k not in info.parametros]
    if desconhecidos:
        raise ErroCiclo("Parâmetros que não existem neste algoritmo: " + ", ".join(desconhecidos) + ".")
    criterio = criterio or ("rmse" if dados.tipo == "regressao" else "f1")
    if criterio not in SCORING[dados.tipo]:
        raise ErroCiclo("Critério não suportado para este tipo de problema.")
    cv, folds = _objecto_cv(dados, cv_folds)

    original = treinar_modelo(modelo_id, dados, params_base, dados.random_state, folds)
    if original.estado != OK:
        raise ErroCiclo(original.motivo or "Este algoritmo não pôde ser treinado com estes dados.")

    combos = contar_combinacoes(espaco)
    n_testar = combos if metodo == "grid" else max(1, min(int(n_iter), combos))
    if n_testar > LIMITE_COMBINACOES:
        raise ErroCiclo(f"{n_testar} combinações é demais (máximo {LIMITE_COMBINACOES}). "
                        + ("Reduz os valores ou usa a procura aleatória." if metodo == "grid" else "Reduz o número de combinações."))
    if n_testar * folds > LIMITE_AJUSTES:
        raise ErroCiclo("Demasiados treinos (combinações × blocos). Reduz um deles.")
    previsto = original.tempo_treino * (n_testar * folds + 1) * 1.3
    if previsto > orcamento_s:
        raise ErroCiclo(f"Com este modelo a procura demoraria cerca de {previsto:.0f} s (limite {orcamento_s:.0f} s). "
                        "Testa menos combinações ou menos valores.")

    info_pre = reg.obter(modelo_id)
    pre = construir_preprocessador(dados.num_cols, dados.cat_cols, escalar=info_pre.escalar, nao_negativo=info_pre.nao_negativo)
    classe = _AlgoritmoClassificador if dados.tipo == "classificacao" else _AlgoritmoRegressor
    base = reg.completar_parametros(info, params_base)
    pipe = Pipeline([("prep", pre), ("modelo", classe(modelo_id, dados.random_state, **base))])
    grelha = {f"modelo__{k}": list(v) for k, v in espaco.items()}
    scoring = SCORING[dados.tipo][criterio]
    comum = dict(scoring=scoring, cv=cv, n_jobs=1, refit=False, error_score=np.nan)
    busca = (GridSearchCV(pipe, grelha, **comum) if metodo == "grid"
             else RandomizedSearchCV(pipe, grelha, n_iter=n_testar, random_state=dados.random_state, **comum))
    t0 = time.perf_counter()
    with _vaga(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        busca.fit(dados.X_train, dados.y_train)               # só o treino: o teste nunca entra aqui
    tempo = time.perf_counter() - t0
    res = busca.cv_results_
    sinal = -1.0 if scoring.startswith("neg_") else 1.0
    medias = sinal * np.asarray(res["mean_test_score"], dtype=float)
    if not np.isfinite(medias).any():
        raise ErroCiclo("Nenhuma combinação conseguiu treinar. Confirma os valores escolhidos.")
    melhores = {k[len("modelo__"):]: v for k, v in busca.best_params_.items()}
    melhores = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in melhores.items()}
    params_finais = {**base, **melhores}
    otimizado = treinar_modelo(modelo_id, dados, params_finais, dados.random_state, folds)
    if otimizado.estado != OK:
        raise ErroCiclo(otimizado.motivo or "Não foi possível treinar o modelo com os melhores parâmetros.")

    linhas = []
    for i in np.argsort(np.asarray(res["rank_test_score"])):
        linha = {k[len("modelo__"):]: res[f"param_{k}"][i] for k in grelha}
        linha["Nota média (CV)"] = float(medias[i])
        linha["Desvio (CV)"] = float(res["std_test_score"][i])
        linha["Posição"] = int(res["rank_test_score"][i])
        linhas.append(linha)
    tabela = pd.DataFrame(linhas).head(15)

    out = ResultadoOtimizacao(
        metodo=metodo, modelo_id=modelo_id, nome=info.nome, criterio=criterio, n_combinacoes=combos, n_testadas=n_testar, cv_folds=folds,
        params_originais=original.params, melhores_params=otimizado.params, cv_original=(original.cv or {}).get(criterio, (None,))[0],
        cv_otimizado=(otimizado.cv or {}).get(criterio, (None,))[0], tabela=tabela, original=original, otimizado=otimizado, tempo=tempo)
    if not np.isfinite(medias).all():
        out.avisos.append("Algumas combinações falharam e foram ignoradas.")
    out.conclusao = _conclusao_otimizacao(out, dados)
    return out


def _conclusao_otimizacao(o: ResultadoOtimizacao, dados: DadosPreparados) -> list:
    """Frases que comparam antes e depois com os números reais (o teste é a nota honesta)."""
    c = o.criterio
    maior = ev.direccao(dados.tipo, c)
    rot = ev.rotulo(dados.tipo, c)
    te0, te1 = o.original.metricas.get(c), o.otimizado.metricas.get(c)
    linhas = []
    if o.cv_original is not None and o.cv_otimizado is not None:
        melhor_cv = (o.cv_otimizado > o.cv_original) if maior else (o.cv_otimizado < o.cv_original)
        linhas.append(f"Na validação cruzada (só com o treino) o {rot} passou de {o.cv_original:.4g} para {o.cv_otimizado:.4g}"
                      + (" (melhorou)." if melhor_cv else " (não melhorou)."))
    if te0 is None or te1 is None:
        return linhas
    melhorou = (te1 > te0) if maior else (te1 < te0)
    dif = abs(te1 - te0)
    rel = dif / (abs(te0) if te0 else 1.0)
    if rel < 0.01:
        linhas.append(f"No teste o {rot} ficou praticamente igual ({te0:.4g} → {te1:.4g}). Os valores por omissão já eram bons para estes dados.")
    elif melhorou:
        linhas.append(f"No teste o {rot} melhorou de {te0:.4g} para {te1:.4g}.")
    else:
        linhas.append(f"No teste o {rot} piorou de {te0:.4g} para {te1:.4g}. Isto acontece: a procura escolhe o melhor na validação "
                      "cruzada e esse ganho nem sempre se repete em dados novos. O teste é a nota honesta.")
    cv = o.original.cv.get(c) if o.original.cv else None
    if cv and melhorou and dif < cv[1]:
        linhas.append(f"A melhoria ({dif:.3g}) é menor do que o desvio da validação cruzada ({cv[1]:.3g}): pode ser só ruído.")
    return linhas


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — Guardar e exportar (Pipeline completo + metadados)
# ══════════════════════════════════════════════════════════════════════════════
def _limpar_metricas(m: dict) -> dict:
    out = {}
    for k, v in (m or {}).items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def _arestas(valores: pd.Series, n: int = 10):
    v = pd.to_numeric(valores, errors="coerce").dropna()
    if v.nunique() < 2:
        return None
    e = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)))
    return e if len(e) >= 3 else None


def _proporcoes(valores, arestas) -> list:
    v = np.asarray(pd.to_numeric(pd.Series(valores), errors="coerce").dropna(), dtype=float)
    if not len(v):
        return [0.0] * (len(arestas) - 1)
    interior = np.asarray(arestas[1:-1])
    idx = np.searchsorted(interior, v, side="right")
    cont = np.bincount(idx, minlength=len(arestas) - 1).astype(float)
    return (cont / cont.sum()).tolist()


def perfil_referencia(r: ResultadoModelo, dados: DadosPreparados) -> dict:
    """Retrato dos dados de treino e das previsões no teste: serve de «antes» na monitorização (deriva)."""
    perfil = {"n_treino": int(dados.n_treino), "numericas": {}, "categoricas": {}, "previsoes": None}
    for c in dados.num_cols:
        s = dados.X_train[c]
        e = _arestas(s)
        item = {"em_falta": float(s.isna().mean()), "media": float(s.mean()) if s.notna().any() else None,
                "desvio": float(s.std()) if s.notna().sum() > 1 else None}
        if e is not None:
            item.update(arestas=[float(x) for x in e], proporcoes=_proporcoes(s, e))
        perfil["numericas"][c] = item
    for c in dados.cat_cols:
        vc = dados.X_train[c].value_counts(normalize=True, dropna=True)
        perfil["categoricas"][c] = {"em_falta": float(dados.X_train[c].isna().mean()),
                                    "proporcoes": {str(k): float(v) for k, v in vc.head(20).items()}}
    try:
        pred = np.asarray(r.pipeline.predict(dados.X_test)).ravel()
        if dados.tipo == "classificacao":
            vc = pd.Series([dados.classes[int(i)] for i in pred]).value_counts(normalize=True)
            perfil["previsoes"] = {"tipo": "classes", "proporcoes": {str(k): float(v) for k, v in vc.items()}}
        else:
            e = _arestas(pd.Series(pred))
            if e is not None:
                perfil["previsoes"] = {"tipo": "numero", "arestas": [float(x) for x in e], "proporcoes": _proporcoes(pred, e)}
    except Exception:
        perfil["previsoes"] = None
    return perfil


def meta_ciclo(dados: DadosPreparados, r: ResultadoModelo, nome: str, origem: str = "laboratório de avaliação",
               otimizacao: ResultadoOtimizacao = None) -> dict:
    """Metadados completos para guardar: os de `meta_tabular` + treino, hiperparâmetros, perfil de referência."""
    meta = mp.meta_tabular(dados, r, nome, origem)
    meta["metricas_teste"] = mp._simples(_limpar_metricas(r.metricas))
    meta["metricas_treino"] = mp._simples(r.metricas_treino)
    meta["data_treino"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["pipeline_completo"] = True
    meta["referencia_monitorizacao"] = mp._simples(perfil_referencia(r, dados))
    if otimizacao is not None:
        meta["otimizacao"] = mp._simples({"metodo": otimizacao.metodo, "criterio": otimizacao.criterio, "n_testadas": otimizacao.n_testadas,
                                          "cv_folds": otimizacao.cv_folds, "parametros_originais": otimizacao.params_originais})
    return meta


def metadados_legiveis(meta: dict) -> list:
    """Os 8 itens pedidos na etapa de exportação, como (rótulo, valor)."""
    return [("Nome do modelo", meta.get("nome")), ("Algoritmo", meta.get("algoritmo")),
            ("Data do treino", meta.get("data_treino") or meta.get("criado_em")),
            ("Métricas de avaliação (teste)", {k: v for k, v in (meta.get("metricas_teste") or {}).items() if isinstance(v, (int, float))}),
            ("Hiperparâmetros", meta.get("parametros")), ("Características utilizadas", meta.get("features")),
            ("Tipo de problema", meta.get("tipo")), ("Versão do modelo", meta.get("versao", "ainda não guardado"))]


def exportar_joblib(pipeline) -> bytes:
    import joblib
    buf = io.BytesIO()
    joblib.dump(pipeline, buf, compress=3)
    return buf.getvalue()


def _exemplo_linha(meta: dict) -> dict:
    ex = {}
    iv = meta.get("intervalos_numericos") or {}
    cats = meta.get("categorias_conhecidas") or {}
    datas = set(meta.get("colunas_data") or [])
    for f in meta.get("features") or []:
        if f in datas:
            ex[f] = "2024-01-15"
        elif f in cats and cats[f]:
            ex[f] = cats[f][0]
        elif f in iv and iv[f] and iv[f][2] is not None:
            ex[f] = round(float(iv[f][2]), 4)
        else:
            ex[f] = 0
    return ex


def _preencher(modelo: str, meta: dict, extra: dict = None) -> str:
    texto = (PASTA_CODIGO / modelo).read_text(encoding="utf-8")
    valores = {"@@PREPARAR@@": (PASTA_CODIGO / "_preparar.py.txt").read_text(encoding="utf-8"),
               "@@NOME@@": str(meta.get("nome") or "modelo").replace('"', "'"), "@@EXEMPLO@@": repr(_exemplo_linha(meta))}
    valores.update(extra or {})
    for k, v in valores.items():
        texto = texto.replace(k, v)
    base = {"@@TIPO@@": meta.get("tipo"), "@@ALVO@@": meta.get("alvo"), "@@FEATURES@@": meta.get("features") or [],
            "@@NUM_COLS@@": meta.get("num_cols") or [], "@@CAT_COLS@@": meta.get("cat_cols") or [],
            "@@COLUNAS_DATA@@": meta.get("colunas_data") or [], "@@CLASSES@@": meta.get("classes") or None}
    for k, v in base.items():
        texto = texto.replace(k, repr(v))
    return texto


def gerar_codigo(meta: dict, tipo: str) -> str:
    """Código Python pronto a usar: 'uso', 'fastapi', 'streamlit' ou 'cliente'."""
    if meta.get("tipo") not in ("regressao", "classificacao"):
        raise ErroCiclo("Só se gera código para modelos de classificação e regressão.")
    ficheiros = {"uso": "uso.py.txt", "fastapi": "fastapi.py.txt", "streamlit": "streamlit.py.txt", "cliente": "cliente.py.txt"}
    if tipo not in ficheiros:
        raise ErroCiclo("Tipo de exemplo desconhecido.")
    return _preencher(ficheiros[tipo], meta)


def exportar_pacote(pipeline, meta: dict) -> bytes:
    """ZIP com modelo.joblib, metadados.json, usar_modelo.py e LEEME.txt."""
    leeme = ("Modelo exportado do DataForge EDU.\n\n"
             "modelo.joblib    Pipeline completo (imputação, escala, one-hot e algoritmo), pela ordem das características.\n"
             "metadados.json   Algoritmo, hiperparâmetros, métricas, características, classes e versão.\n"
             "usar_modelo.py   Exemplo de previsão com o modelo.\n\n"
             "Usa as mesmas versões das bibliotecas (ver metadados.json). Só abras ficheiros joblib em que confias: abrir um executa código.\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("modelo.joblib", exportar_joblib(pipeline))
        z.writestr("metadados.json", json.dumps({**meta, "versoes_bibliotecas": meta.get("versoes_bibliotecas") or mp._versoes()},
                                                ensure_ascii=False, indent=1, default=str))
        z.writestr("usar_modelo.py", gerar_codigo(meta, "uso"))
        z.writestr("LEEME.txt", leeme)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 6 — Prediction Playground
# ══════════════════════════════════════════════════════════════════════════════
def csv_modelo(meta: dict) -> bytes:
    """CSV de exemplo com as colunas obrigatórias (uma linha de exemplo) para o estudante preencher."""
    return pd.DataFrame([_exemplo_linha(meta)]).to_csv(index=False).encode("utf-8")


def prever_ficheiro(modelo, meta: dict, df: pd.DataFrame) -> tuple:
    """Valida e prevê. Devolve (tabela com previsões, relatório {erros, avisos}). Levanta ErroCiclo se não der."""
    if len(df) > MAX_LINHAS_PREVISAO:
        raise ErroCiclo(f"O ficheiro tem {len(df)} linhas; o máximo é {MAX_LINHAS_PREVISAO}.")
    if df.columns.duplicated().any():
        raise ErroCiclo("O ficheiro tem colunas com o mesmo nome. Cada coluna deve aparecer só uma vez.")
    try:
        return ps.prever_tabela(modelo, meta, df)
    except ps.ErroPrevisao as e:
        raise ErroCiclo(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 8 — Monitorização: deriva dos dados, das previsões e das métricas
# ══════════════════════════════════════════════════════════════════════════════
def psi(esperado, observado, eps: float = 1e-4) -> float:
    """Population Stability Index entre duas distribuições (proporções por classe/intervalo)."""
    e = np.clip(np.asarray(esperado, dtype=float), eps, None)
    o = np.clip(np.asarray(observado, dtype=float), eps, None)
    e, o = e / e.sum(), o / o.sum()
    return float(np.sum((o - e) * np.log(o / e)))


def nivel_deriva(valor: float) -> str:
    if valor < 0.10:
        return "estável"
    return "moderada" if valor < 0.25 else "forte"


def analisar_deriva(meta: dict, df_novo: pd.DataFrame, previsoes=None) -> dict:
    """Compara os dados novos com o perfil guardado no treino. Devolve {variaveis: DataFrame, previsoes: dict|None, avisos: [...]}.

    PSI < 0,10: estável; 0,10 a 0,25: deriva moderada; > 0,25: deriva forte. São regras práticas, não testes estatísticos.
    """
    ref = meta.get("referencia_monitorizacao")
    if not ref:
        raise ErroCiclo("Este modelo foi guardado sem o perfil dos dados de treino, por isso não dá para medir a deriva. "
                        "Volta a guardá-lo no laboratório.")
    avisos = []
    if len(df_novo) < 30:
        avisos.append(f"Só há {len(df_novo)} linhas novas: o PSI varia muito com tão poucos dados.")
    linhas = []
    for c in meta.get("features") or []:
        if c not in df_novo.columns:
            avisos.append(f"A coluna «{c}» não existe nos dados novos.")
            continue
        if c in ref.get("numericas", {}):
            p = ref["numericas"][c]
            s = pd.to_numeric(df_novo[c], errors="coerce")
            falta = float(s.isna().mean())
            if p.get("arestas"):
                valor = psi(p["proporcoes"], _proporcoes(s, p["arestas"]))
                linhas.append({"Variável": c, "Tipo": "número", "PSI": valor, "Nível": nivel_deriva(valor),
                               "Média no treino": p.get("media"), "Média nova": float(s.mean()) if s.notna().any() else None,
                               "Em falta (treino → novo)": f"{p['em_falta']:.0%} → {falta:.0%}"})
            else:
                linhas.append({"Variável": c, "Tipo": "número", "PSI": None, "Nível": "sem referência", "Média no treino": p.get("media"),
                               "Média nova": float(s.mean()) if s.notna().any() else None,
                               "Em falta (treino → novo)": f"{p['em_falta']:.0%} → {falta:.0%}"})
        elif c in ref.get("categoricas", {}):
            p = ref["categoricas"][c]
            novos = df_novo[c].dropna().astype(str).str.strip()
            cats = list(p["proporcoes"]) + ["__novas__"]
            vc = novos.value_counts(normalize=True) if len(novos) else pd.Series(dtype=float)
            obs = [float(vc.get(k, 0.0)) for k in p["proporcoes"]]
            obs.append(max(0.0, 1.0 - sum(obs)))
            valor = psi([p["proporcoes"].get(k, 0.0) for k in cats[:-1]] + [0.0], obs)
            linhas.append({"Variável": c, "Tipo": "categoria", "PSI": valor, "Nível": nivel_deriva(valor), "Média no treino": None, "Média nova": None,
                           "Em falta (treino → novo)": f"{p['em_falta']:.0%} → {float(df_novo[c].isna().mean()):.0%}"})
    tabela = pd.DataFrame(linhas)
    out = {"variaveis": tabela, "previsoes": None, "avisos": avisos}
    rp = ref.get("previsoes")
    if previsoes is not None and rp:
        pr = pd.Series(list(previsoes))
        if rp["tipo"] == "classes":
            vc = pr.astype(str).value_counts(normalize=True)
            cats = list(rp["proporcoes"])
            obs = [float(vc.get(k, 0.0)) for k in cats]
            obs.append(max(0.0, 1.0 - sum(obs)))
            valor = psi([rp["proporcoes"][k] for k in cats] + [0.0], obs)
            out["previsoes"] = {"psi": valor, "nivel": nivel_deriva(valor), "antes": rp["proporcoes"],
                                "agora": {k: float(vc.get(k, 0.0)) for k in cats}}
        else:
            valor = psi(rp["proporcoes"], _proporcoes(pr, rp["arestas"]))
            out["previsoes"] = {"psi": valor, "nivel": nivel_deriva(valor)}
    return out


def metricas_com_valores_reais(meta: dict, previsoes, reais) -> dict:
    """Métricas do modelo em dados novos quando já se conhece a resposta real. Devolve {metricas, n, avisos}."""
    avisos = []
    p = pd.Series(list(previsoes)).reset_index(drop=True)
    y = pd.Series(list(reais)).reset_index(drop=True)
    ok = y.notna() & p.notna()
    if ok.sum() < len(y):
        avisos.append(f"{int((~ok).sum())} linhas sem valor real foram ignoradas.")
    p, y = p[ok], y[ok]
    if len(y) < 2:
        raise ErroCiclo("São precisos pelo menos 2 casos com o valor real.")
    if len(y) < 30:
        avisos.append(f"Só há {len(y)} casos com valor real: as métricas são pouco fiáveis.")
    if meta["tipo"] == "regressao":
        yy = pd.to_numeric(y, errors="coerce")
        bom = yy.notna()
        if bom.sum() < len(yy):
            avisos.append("Valores reais que não são números foram ignorados.")
        if bom.sum() < 2:
            raise ErroCiclo("Os valores reais têm de ser números.")
        m = ev.metricas_regressao(yy[bom].to_numpy(dtype=float), p[bom].to_numpy(dtype=float))
        return {"metricas": {k: m[k] for k in ("r2", "mae", "rmse")}, "n": int(bom.sum()), "avisos": avisos}
    classes = [str(c) for c in meta.get("classes") or []]
    ys, ps_ = y.astype(str), p.astype(str)
    desconhecidas = ~ys.isin(classes)
    if desconhecidas.any():
        avisos.append(f"{int(desconhecidas.sum())} valores reais têm classes que o modelo nunca viu e foram ignorados.")
    ys, ps_ = ys[~desconhecidas], ps_[~desconhecidas]
    if len(ys) < 2:
        raise ErroCiclo("Nenhum valor real corresponde às classes do modelo.")
    idx = {c: i for i, c in enumerate(classes)}
    m = ev.metricas_classificacao(ys.map(idx).to_numpy(), ps_.map(idx).to_numpy(), None, len(classes))
    return {"metricas": {k: m[k] for k in ("accuracy", "precision", "recall", "f1")}, "n": int(len(ys)), "avisos": avisos}


def contexto_tutor(meta_ou_dados: dict) -> str:
    """Texto para o Tutor IA com SÓ os números reais do ecrã (nenhum valor inventado)."""
    linhas = []
    for k, v in meta_ou_dados.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            linhas.append(f"{k}: {v:.4g}")
        elif isinstance(v, str) and len(v) < 200:
            linhas.append(f"{k}: {v}")
    return "\n".join(linhas)[:3000]
