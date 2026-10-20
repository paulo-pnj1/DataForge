"""AutoML: selecção automática do melhor modelo ENTRE os avaliados, em duas fases e sem tocar no teste.

1. Triagem: validação cruzada rápida (numa amostra do treino) de todos os algoritmos compatíveis.
2. Avaliação completa dos melhores da triagem (treino completo, validação cruzada e teste final) e recomendação.
Não faz ajuste automático de hiperparâmetros (por isso não há risco de «viciar» o conjunto de teste).
"""
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from modules.ml_engine import model_registry as reg
from modules.ml_engine.data_loader import ErroDados
from modules.ml_engine.data_validation import RelatorioValidacao, sugerir_tipo_problema, validar_dados
from modules.ml_engine.model_comparison import ComparacaoResultado, Recomendacao, comparar_modelos, recomendar
from modules.ml_engine.model_training import _curto, _vaga
from modules.ml_engine.preprocessing import DadosPreparados, construir_preprocessador, preparar_dados, sugerir_features

# ordem de preferência quando o tempo é curto: primeiro o que é rápido e robusto
PRIORIDADE = {
    "regressao": ["linear_reg", "ridge", "tree_reg", "rf_reg", "hist_gb_reg", "extra_trees_reg", "gb_reg", "knn_reg", "lasso",
                  "elasticnet", "bayesian_ridge", "xgb_reg", "lgbm_reg", "poly_reg", "svr", "mlp_reg"],
    "classificacao": ["logistic", "tree_clf", "rf_clf", "hist_gb_clf", "extra_trees_clf", "gb_clf", "knn_clf", "gaussian_nb",
                      "multinomial_nb", "xgb_clf", "lgbm_clf", "svc", "mlp_clf"],
}
MAX_LINHAS_TRIAGEM = 3000


@dataclass
class ResultadoAutoML:
    tipo: str
    dados: DadosPreparados
    validacao: RelatorioValidacao
    triagem: pd.DataFrame
    comparacao: ComparacaoResultado
    recomendacao: Recomendacao
    passos: list = field(default_factory=list)
    avisos: list = field(default_factory=list)
    puladas: list = field(default_factory=list)       # [(nome, motivo)]
    tempo_total: float = 0.0


def candidatos(dados: DadosPreparados) -> tuple:
    """(ids compatíveis e disponíveis, [(nome, motivo) dos que ficaram de fora])."""
    ok, fora = [], []
    ordem = {m: i for i, m in enumerate(PRIORIDADE[dados.tipo])}
    for info in sorted(reg.listar(dados.tipo), key=lambda m: ordem.get(m.id, 99)):
        if info.tipo_modelo != "tabular":
            continue
        if not info.disponivel():
            fora.append((info.nome, info.motivo_indisponivel()))
            continue
        if info.verifica:
            try:
                bom, motivo = info.verifica(dados)
            except Exception as e:
                bom, motivo = False, _curto(e)
            if not bom:
                fora.append((info.nome, motivo))
                continue
        ok.append(info.id)
    return ok, fora


def _triagem(ids: list, dados: DadosPreparados, folds: int, random_state: int, orcamento_s: float, cancelar, progresso):
    """Validação cruzada rápida de cada candidato numa amostra do treino. Devolve DataFrame ordenado."""
    X, y = dados.X_train, dados.y_train
    if len(X) > MAX_LINHAS_TRIAGEM:
        idx = np.random.default_rng(random_state).choice(len(X), MAX_LINHAS_TRIAGEM, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    if dados.tipo == "classificacao":
        folds = max(2, min(folds, int(np.bincount(y).min()))) if np.bincount(y).min() >= 2 else 0
        cv = StratifiedKFold(folds, shuffle=True, random_state=random_state) if folds else None
        scoring = "f1_macro"
    else:
        folds = max(2, min(folds, len(X) // 2))
        cv = KFold(folds, shuffle=True, random_state=random_state)
        scoring = "r2"
    linhas, t0 = [], time.perf_counter()
    for i, mid in enumerate(ids):
        info = reg.obter(mid)
        linha = {"Modelo": info.nome, "id": mid, "Pontuação (CV)": None, "Desvio": None, "Tempo (s)": None, "Estado": "OK"}
        if cancelar and cancelar():
            linha["Estado"] = "Ignorado: cancelado"
        elif time.perf_counter() - t0 > orcamento_s:
            linha["Estado"] = "Ignorado: limite de tempo da triagem"
        elif cv is None:
            linha["Estado"] = "Ignorado: classes com poucos exemplos"
        else:
            if progresso:
                progresso(i, len(ids), info.nome)
            try:
                pipe = Pipeline([("prep", construir_preprocessador(dados.num_cols, dados.cat_cols, info.escalar, info.nao_negativo)),
                                 ("modelo", info.construir(reg.completar_parametros(info), random_state))])
                t1 = time.perf_counter()
                with _vaga(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = cross_val_score(clone(pipe), X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
                linha.update({"Pontuação (CV)": float(np.mean(s)), "Desvio": float(np.std(s)), "Tempo (s)": time.perf_counter() - t1})
            except Exception as e:
                linha["Estado"] = "Falhou: " + _curto(e)
        linha["_compl"] = info.complexidade
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    if len(df):
        df["_ok"] = df["Pontuação (CV)"].notna()
        df = df.sort_values(["_ok", "Pontuação (CV)"], ascending=[False, False]).drop(columns="_ok").reset_index(drop=True)
    return df


def executar_automl(df: pd.DataFrame, alvo: str, tipo: str = None, features: list = None, max_modelos: int = 6,
                    orcamento_s: float = 120.0, cv_folds: int = 3, test_size: float = 0.2, random_state: int = 42,
                    criterio: str = None, progresso=None, cancelar=None) -> ResultadoAutoML:
    """Fluxo completo do AutoML. Levanta `ErroDados` só se os dados impedirem qualquer treino."""
    t0 = time.perf_counter()
    passos = []
    if alvo not in df.columns:
        raise ErroDados(f"A variável-alvo «{alvo}» não existe.")
    sugestao = sugerir_tipo_problema(df, alvo)
    tipo_final = tipo or sugestao[0]
    passos.append(f"Tipo de problema: {'regressão' if tipo_final == 'regressao' else 'classificação'}"
                  + (f" (sugerido: {sugestao[1]})" if not tipo else " (confirmado por ti)"))
    val = validar_dados(df, alvo, tipo_final)
    if not val.ok:
        raise ErroDados(" ".join(val.erros))
    feats = features or sugerir_features(df, alvo)
    passos.append(f"Variáveis preditoras: {len(feats)}. Validação: {len(val.avisos)} avisos.")
    dados = preparar_dados(df, alvo, tipo_final, feats, test_size, random_state)
    passos.append(f"Dados preparados: {dados.n_treino} exemplos de treino e {dados.n_teste} de teste (o teste só é usado no fim).")
    ids, puladas = candidatos(dados)
    if not ids:
        raise ErroDados("Nenhum algoritmo compatível está disponível para estes dados.")
    passos.append(f"Algoritmos compatíveis e disponíveis: {len(ids)}" + (f" ({len(puladas)} de fora)." if puladas else "."))

    orc_triagem = max(5.0, orcamento_s * 0.4)
    tri = _triagem(ids, dados, cv_folds, random_state, orc_triagem, cancelar, progresso)
    passos.append(f"Avaliação inicial: validação cruzada rápida de {len(ids)} algoritmos"
                  + (f" numa amostra de {MAX_LINHAS_TRIAGEM} exemplos." if dados.n_treino > MAX_LINHAS_TRIAGEM else "."))
    validos = tri[tri["Pontuação (CV)"].notna()]
    if validos.empty:
        raise ErroDados("Nenhum algoritmo conseguiu ser avaliado na triagem. Verifica os dados.")
    finalistas = validos["id"].head(max(1, max_modelos)).tolist()
    passos.append(f"Os {len(finalistas)} melhores da triagem seguem para o treino completo.")

    restante = max(5.0, orcamento_s - (time.perf_counter() - t0))
    comp = comparar_modelos(finalistas, dados, None, cv_folds, random_state, orcamento_s=restante, progresso=progresso, cancelar=cancelar)
    passos.append("Treino completo, validação cruzada e avaliação final no teste.")
    rec = recomendar(comp, criterio)
    passos.append(f"Recomendação: {rec.nome or 'nenhuma'}.")
    res = ResultadoAutoML(tipo_final, dados, val, tri.drop(columns=["_compl"], errors="ignore"), comp, rec, passos,
                          avisos=list(dict.fromkeys(dados.avisos + val.avisos)), puladas=puladas)
    res.tempo_total = time.perf_counter() - t0
    return res
