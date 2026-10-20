"""Comparação de vários modelos com os mesmos dados e as mesmas regras, e recomendação justificada.

Como se recomenda (e porquê):
- A recomendação usa a validação cruzada feita SÓ no conjunto de treino. O conjunto de teste fica reservado
  para a avaliação final e nunca decide qual é o modelo.
- Entre os modelos que ficam dentro de um desvio-padrão do melhor (regra do «um desvio-padrão»), escolhe-se o mais
  simples e, se empatarem, o mais rápido. Não se prefere um modelo só por ser mais complexo.
- Compara-se sempre com uma referência (média / classe mais frequente). Se nenhum a superar, diz-se.
- É o melhor entre os modelos avaliados, não o melhor algoritmo possível.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modules.ml_engine import model_evaluation as ev
from modules.ml_engine import model_registry as reg
from modules.ml_engine.model_training import (FALHOU, IGNORADO, INCOMPATIVEL, INDISPONIVEL, OK, REFERENCIA, ResultadoModelo,
                                              treinar_modelo, treinar_referencia)
from modules.ml_engine.preprocessing import DadosPreparados

ROTULO_ESTADO = {OK: "OK", FALHOU: "Falhou", INDISPONIVEL: "Indisponível", INCOMPATIVEL: "Incompatível",
                 IGNORADO: "Ignorado", REFERENCIA: "Referência"}


@dataclass
class Recomendacao:
    modelo_id: str = None
    nome: str = ""
    criterio: str = ""
    justificacao: str = ""
    avisos: list = field(default_factory=list)
    alternativas: list = field(default_factory=list)      # [(nome, valor)]
    base: str = ""                                        # 'validação cruzada' | 'teste'


@dataclass
class ComparacaoResultado:
    tipo: str
    dados: DadosPreparados
    resultados: list
    referencia: ResultadoModelo = None
    cv_folds: int = 0
    tempo_total: float = 0.0
    avisos: list = field(default_factory=list)
    interrompido: bool = False

    def por_id(self, modelo_id: str) -> ResultadoModelo:
        for r in self.resultados:
            if r.id == modelo_id:
                return r
        raise KeyError(modelo_id)

    def validos(self) -> list:
        return [r for r in self.resultados if r.estado == OK]

    def falhas(self) -> list:
        return [r for r in self.resultados if r.estado in (FALHOU, INDISPONIVEL, INCOMPATIVEL, IGNORADO)]

    # ── tabela ──
    def colunas_metricas(self) -> list:
        if self.tipo == "regressao":
            return [("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE")]
        return [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]

    def tabela(self, ordenar_por: str = None, incluir_referencia: bool = True) -> pd.DataFrame:
        """Tabela comparativa. `ordenar_por`: nome interno da métrica ('rmse', 'f1', ...) ou 'tempo'."""
        cols = self.colunas_metricas()
        principal = "r2" if self.tipo == "regressao" else "f1"
        linhas = []
        itens = list(self.resultados) + ([self.referencia] if incluir_referencia and self.referencia else [])
        for r in itens:
            linha = {"Modelo": r.nome}
            for k, nome in cols:
                linha[nome] = r.metricas.get(k) if r.ok else None
            if self.cv_folds:
                linha[f"{ev.rotulo(self.tipo, principal)} (CV)"] = r.cv.get(principal, (None,))[0] if r.cv else None
            linha["Tempo de treino (s)"] = r.tempo_treino if r.ok else None
            linha["Estado"] = ROTULO_ESTADO[r.estado] + (f": {r.motivo}" if r.motivo and r.estado != OK else "")
            linha["_id"] = r.id
            linha["_ok"] = r.ok
            linhas.append(linha)
        df = pd.DataFrame(linhas)
        if ordenar_por and len(df):
            nome_col = "Tempo de treino (s)" if ordenar_por == "tempo" else ev.rotulo(self.tipo, ordenar_por)
            asc = True if ordenar_por == "tempo" else not ev.direccao(self.tipo, ordenar_por)
            df["_ordem_ok"] = ~df["_ok"].astype(bool)
            df = df.sort_values(["_ordem_ok", nome_col], ascending=[True, asc], na_position="last").drop(columns="_ordem_ok")
        return df.reset_index(drop=True)

    def tabela_visivel(self, ordenar_por: str = None) -> pd.DataFrame:
        return self.tabela(ordenar_por).drop(columns=["_id", "_ok"])


def comparar_modelos(ids: list, dados: DadosPreparados, params_por_modelo: dict = None, cv_folds: int = 3,
                     random_state: int = 42, max_modelos: int = None, orcamento_s: float = None,
                     progresso=None, cancelar=None, incluir_referencia: bool = True) -> ComparacaoResultado:
    """Treina os modelos um a um (sequencial: poupa memória) com os mesmos dados e regras.

    `progresso(i, n, nome)` é chamado antes de cada modelo; `cancelar()` devolve True para parar entre modelos;
    `orcamento_s` é um limite de tempo total (verificado entre modelos: um modelo já a treinar não é interrompido).
    """
    params_por_modelo = params_por_modelo or {}
    ids = [i for k, i in enumerate(ids) if i not in ids[:k]]
    t0 = time.perf_counter()
    comp = ComparacaoResultado(tipo=dados.tipo, dados=dados, resultados=[], cv_folds=cv_folds if cv_folds and cv_folds >= 2 else 0)
    n = len(ids)
    for i, mid in enumerate(ids):
        nome = mid
        try:
            nome = reg.obter(mid).nome
        except KeyError:
            pass
        if max_modelos is not None and i >= max_modelos:
            comp.resultados.append(ResultadoModelo(mid, nome, dados.tipo, IGNORADO, f"Limite de {max_modelos} modelos atingido."))
            continue
        if cancelar is not None and cancelar():
            comp.interrompido = True
            comp.resultados.append(ResultadoModelo(mid, nome, dados.tipo, IGNORADO, "Cancelado pelo utilizador."))
            continue
        if orcamento_s is not None and time.perf_counter() - t0 > orcamento_s:
            comp.interrompido = True
            comp.resultados.append(ResultadoModelo(mid, nome, dados.tipo, IGNORADO, f"Limite de tempo ({orcamento_s:.0f} s) atingido."))
            continue
        if progresso:
            progresso(i, n, nome)
        comp.resultados.append(treinar_modelo(mid, dados, params_por_modelo.get(mid), random_state, comp.cv_folds))
    if incluir_referencia:
        comp.referencia = treinar_referencia(dados, random_state)
        if comp.cv_folds and comp.referencia.ok:
            from modules.ml_engine.model_training import _validacao_cruzada
            try:
                comp.referencia.cv = _validacao_cruzada(comp.referencia.pipeline, dados, comp.cv_folds)[0] or {}
            except Exception:
                comp.referencia.cv = {}
    comp.tempo_total = time.perf_counter() - t0
    comp.avisos = list(dados.avisos)
    if comp.interrompido:
        comp.avisos.append("A comparação foi interrompida antes de avaliar todos os modelos.")
    if progresso:
        progresso(n, n, "")
    return comp


# ══════════════════════════════════════════════════════════════════════════════
# Recomendação
# ══════════════════════════════════════════════════════════════════════════════
def _valor(r: ResultadoModelo, criterio: str, usar_cv: bool):
    if usar_cv:
        v = r.cv.get(criterio)
        return None if v is None else v[0]
    return r.metricas.get(criterio)


def _fmt(v, criterio):
    return "n/d" if v is None else (f"{v:.1%}" if criterio in ("accuracy", "precision", "recall", "f1", "roc_auc") else f"{v:.4g}")


def _fmt_dp(v, criterio):
    return f"{v * 100:.1f} p.p." if criterio in ("accuracy", "precision", "recall", "f1", "roc_auc") else f"{v:.2g}"


def recomendar(comp: ComparacaoResultado, criterio: str = None) -> Recomendacao:
    """Recomenda um modelo segundo um critério explícito (por omissão RMSE em regressão e F1-score em classificação)."""
    tipo = comp.tipo
    criterio = criterio or ("rmse" if tipo == "regressao" else "f1")
    rec = Recomendacao(criterio=ev.rotulo(tipo, criterio))
    cands = comp.validos()
    if not cands:
        rec.justificacao = "Nenhum modelo foi treinado com sucesso, por isso não há recomendação."
        return rec
    maior = ev.direccao(tipo, criterio)
    # base da decisão: validação cruzada (só treino) quando existe para todos; senão o teste, com aviso
    usar_cv = all(r.cv.get(criterio) for r in cands) and bool(comp.cv_folds)
    com_valor = [r for r in cands if _valor(r, criterio, usar_cv) is not None]
    if not com_valor:
        rec.justificacao = f"Nenhum modelo tem «{rec.criterio}» calculável (por exemplo ROC-AUC exige probabilidades). Escolhe outro critério."
        return rec
    sinal = 1 if maior else -1
    melhor = max(com_valor, key=lambda r: sinal * _valor(r, criterio, usar_cv))
    vm = _valor(melhor, criterio, usar_cv)
    if usar_cv:
        tol = melhor.cv[criterio][1]                                   # um desvio-padrão da validação cruzada do melhor
        proximos = [r for r in com_valor if abs(_valor(r, criterio, True) - vm) <= tol]
        rec.base = f"validação cruzada ({comp.cv_folds} blocos, só nos dados de treino)"
    else:
        proximos = [melhor]
        rec.base = "conjunto de teste"
        rec.avisos.append("Sem validação cruzada, a escolha usa o conjunto de teste e fica mais exposta ao acaso. Activa a validação cruzada para uma recomendação mais fiável.")
    escolhido = min(proximos, key=lambda r: (r.complexidade, r.tempo_treino))
    ve = _valor(escolhido, criterio, usar_cv)
    rec.modelo_id, rec.nome = escolhido.id, escolhido.nome
    rec.alternativas = [(r.nome, _valor(r, criterio, usar_cv)) for r in sorted(com_valor, key=lambda r: -sinal * _valor(r, criterio, usar_cv))[:5]
                        if r.id != escolhido.id]

    unid = f", na unidade de «{comp.dados.alvo}»" if tipo == "regressao" and criterio in ("mae", "rmse") else ""
    partes = [f"**{escolhido.nome}** é o recomendado segundo o critério **{rec.criterio}** ({'maior' if maior else 'menor'} é melhor{unid}), "
              f"calculado por {rec.base}: {_fmt(ve, criterio)}"
              + (f" ± {_fmt_dp(escolhido.cv[criterio][1], criterio)}" if usar_cv else "") + "."]
    if usar_cv and len(proximos) > 1:
        partes.append(f"{len(proximos)} modelos ficaram dentro de um desvio-padrão do melhor ({melhor.nome}: {_fmt(vm, criterio)}); "
                      "entre eles escolheu-se o mais simples e rápido, porque a diferença está dentro do ruído.")
    elif escolhido.id == melhor.id and usar_cv:
        partes.append("Foi o melhor na validação cruzada e nenhum outro ficou dentro da margem de erro.")
    nota_teste = escolhido.metricas.get(criterio)
    if nota_teste is not None:
        partes.append(f"No conjunto de teste (reservado só para a avaliação final) obteve {_fmt(nota_teste, criterio)}.")
    partes.append(f"Tempo de treino: {escolhido.tempo_treino:.2f} s.")
    partes.append(f"É o melhor entre os {len(cands)} modelos avaliados, não necessariamente o melhor algoritmo possível.")
    rec.justificacao = " ".join(partes)

    # avisos de fiabilidade
    ref = comp.referencia
    if ref and ref.ok:
        vr = _valor(ref, criterio, bool(usar_cv and ref.cv.get(criterio)))
        if vr is not None and ve is not None and ((maior and ve <= vr) or (not maior and ve >= vr)):
            rec.avisos.append(f"Atenção: o modelo recomendado não supera a referência simples ({_fmt(vr, criterio)}). Os dados podem não ter sinal suficiente.")
    p = "r2" if tipo == "regressao" else "f1"
    tr, te = escolhido.metricas_treino.get(p), escolhido.metricas.get(p)
    if tr is not None and te is not None and tr - te > 0.15:
        rec.avisos.append(f"Possível sobreajuste: {ev.rotulo(tipo, p)} de {_fmt(tr, p)} no treino contra {_fmt(te, p)} no teste.")
    if comp.dados.n_teste < 30:
        rec.avisos.append(f"O teste tem só {comp.dados.n_teste} exemplos: as métricas finais são pouco fiáveis.")
    if tipo == "classificacao":
        d = ev.aviso_desequilibrio(comp.dados.y_train)
        if d:
            rec.avisos.append(d)
    if tipo == "regressao" and te is not None and te >= 0.95:
        rec.avisos.append("R² muito alto no teste não garante previsões correctas no futuro: confirma que nenhuma coluna revela o alvo e que os dados novos são parecidos com estes.")
    return rec
