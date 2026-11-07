"""Guardar, testar e descarregar os modelos das páginas de clustering, anomalias, redução, redes neurais (MLP) e reforço.

Cada página deixa o seu resultado em memória; aqui `de_*` transforma-o num objecto de `dataforge_runtime` (que só depende de
numpy, pandas e scikit-learn) mais os metadados. `exportar_modelo` faz o ZIP descarregável de QUALQUER modelo da plataforma.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from modules.ml_engine import ciclo_ml as cl
from modules.ml_engine import model_persistence as mp
from modules.ml_engine import prediction_service as ps
from modules.ml_engine.dataforge_runtime import (ACCOES_GRELHA, ModeloAnomalias, ModeloDummies, ModeloGrupos, ModeloLegado, ModeloReducao,  # noqa: F401
                                                 PoliticaDQN, PoliticaQ)
from modules.ml_engine.modelo_legado import ErroLegado, _f

PASTA_CODIGO = Path(cl.PASTA_CODIGO)
TIPOS_NAO_SUPERVISIONADOS = ("cluster", "anomalia", "reducao")
TIPOS_REFORCO = ("reforco_q", "reforco_dqn")
MAX_REFERENCIA = 3000


def _antigo():
    return ErroLegado("Este resultado é antigo e não tem tudo o que é preciso. Treina outra vez e depois guarda.")


def _meta_ns(tipo, nome, algoritmo, features, X_raw, metricas, parametros, origem) -> dict:
    X = pd.DataFrame(np.asarray(X_raw, dtype=float), columns=list(features))
    intervalos = {c: (float(X[c].min()), float(X[c].max()), float(X[c].median())) for c in X.columns if X[c].notna().any()}
    return mp._simples({"tipo": tipo, "nome": nome, "algoritmo_id": "legado", "algoritmo": algoritmo, "alvo": None, "features": list(features),
                        "colunas_data": [], "colunas_modelo": list(features), "num_cols": list(features), "cat_cols": [], "classes": None,
                        "categorias_conhecidas": {}, "intervalos_numericos": intervalos, "parametros": parametros or {}, "metricas_teste": metricas,
                        "cv": {}, "n_treino": int(len(X)), "origem": origem, "avisos": [], "pipeline_completo": True})


def _medianas(X_raw, features) -> dict:
    X = pd.DataFrame(np.asarray(X_raw, dtype=float), columns=list(features))
    return {c: float(X[c].median()) for c in X.columns if X[c].notna().any()}


# ── Clustering ────────────────────────────────────────────────────────────────
def de_clustering(res: dict, algoritmo: str, nome: str) -> tuple:
    try:
        modelo, feats, X_raw, X_sc = res["model"], list(res["features"]), res["X_raw"], res["X_scaled"]
    except KeyError:
        raise _antigo()
    vizinho = None
    if not hasattr(modelo, "predict"):          # DBSCAN, Agglomerative, Spectral...: não classificam casos novos, por isso usamos o vizinho mais próximo
        from sklearn.neighbors import KNeighborsClassifier
        idx = np.random.default_rng(42).permutation(len(X_sc))[:MAX_REFERENCIA]
        vizinho = KNeighborsClassifier(n_neighbors=1).fit(np.asarray(X_sc)[idx], np.asarray(res["labels"])[idx])
    obj = ModeloGrupos(feats, res.get("scaler"), modelo, _medianas(X_raw, feats), vizinho, algoritmo)
    metricas = {"grupos": res.get("n_clusters"), "ruido": res.get("n_ruido"), "silhouette": _f(res.get("sil")),
                "calinski_harabasz": _f(res.get("ch")), "davies_bouldin": _f(res.get("db"))}
    meta = _meta_ns("cluster", nome, algoritmo, feats, X_raw, metricas, {}, "página Clustering")
    if vizinho is not None:
        meta["avisos"] = ["Este algoritmo não classifica casos novos: cada caso novo recebe o grupo do caso de treino mais parecido."]
    return obj, meta


# ── Anomalias ─────────────────────────────────────────────────────────────────
def de_anomalias(res: dict, nome: str) -> tuple:
    try:
        modelo, feats, X_raw, X_sc = res["modelo"], list(res["features"]), res["X_raw"], res["X_sc"]
    except KeyError:
        raise _antigo()
    from sklearn.neighbors import LocalOutlierFactor
    if isinstance(modelo, LocalOutlierFactor) and not getattr(modelo, "novelty", False):
        modelo = LocalOutlierFactor(**{**modelo.get_params(), "novelty": True}).fit(X_sc)       # só assim o LOF classifica casos novos
    obj = ModeloAnomalias(feats, res.get("scaler"), modelo, _medianas(X_raw, feats))
    metricas = {"anomalias": res.get("n_anom"), "normais": res.get("n_norm"), "contaminacao": _f(res.get("contaminacao"))}
    return obj, _meta_ns("anomalia", nome, str(res.get("algo", "Deteção de anomalias")), feats, X_raw, metricas, {}, "página Anomalias")


# ── Redução de dimensionalidade ───────────────────────────────────────────────
def de_reducao(res: dict, nome: str) -> tuple:
    try:
        modelo, feats, X_raw = res["modelo_t"], list(res["features"]), res["X_raw"]
    except KeyError:
        raise _antigo()
    if modelo is None or not hasattr(modelo, "transform"):
        raise ErroLegado(f"{res.get('algo', 'Este método')} não consegue projectar casos novos (só trabalha com os dados do treino), por isso não dá para o guardar. "
                         "Usa PCA, Kernel PCA, ICA, Factor Analysis ou UMAP.")
    obj = ModeloReducao(feats, res.get("scaler"), modelo, _medianas(X_raw, feats))
    return obj, _meta_ns("reducao", nome, str(res.get("algo", "Redução")), feats, X_raw, {"componentes": res.get("n_comp")}, {}, "página Redução")


# ── Reforço ───────────────────────────────────────────────────────────────────
def _meta_rl(tipo, nome, algoritmo, features, metricas, parametros, origem) -> dict:
    return mp._simples({"tipo": tipo, "nome": nome, "algoritmo_id": "legado", "algoritmo": algoritmo, "alvo": None, "features": list(features),
                        "colunas_data": [], "num_cols": list(features), "cat_cols": [], "classes": None, "categorias_conhecidas": {},
                        "intervalos_numericos": {}, "parametros": parametros, "metricas_teste": metricas, "cv": {}, "origem": origem, "avisos": [],
                        "pipeline_completo": True})


def de_q_learning(res: dict, nome: str) -> tuple:
    try:
        Q, (n, obstaculos, armadilhas) = res["Q"], res["env_params"]
    except KeyError:
        raise _antigo()
    obj = PoliticaQ(Q, n, obstaculos, armadilhas)
    recompensas = list(res.get("rewards") or [])
    metricas = {"episodios": len(recompensas), "recompensa_media_final": _f(np.mean(recompensas[-100:])) if recompensas else None,
                "chegou_ao_objectivo": bool(res.get("reached_goal")), "grelha": int(n), "passos_do_caminho": len(res.get("path") or [])}
    return obj, _meta_rl("reforco_q", nome, "Q-Learning (mundo em grelha)", ["linha", "coluna"], metricas, res.get("setup") or {}, "página Q-Learning")


def de_dqn(res: dict, nome: str) -> tuple:
    if not res.get("rede"):
        raise _antigo()
    obj = PoliticaDQN(res["rede"])
    metricas = {"episodios": res.get("episodes"), "melhor_recompensa": _f(res.get("best_reward")), "media_ultimos_50": _f(res.get("avg_last_50"))}
    return obj, _meta_rl("reforco_dqn", nome, "DQN (CartPole)", obj.nomes_estado, metricas, res.get("setup") or {}, "página DQN")


# ── Usar um modelo guardado com dados novos ───────────────────────────────────
def aplicar_tabela(modelo, meta: dict, df: pd.DataFrame) -> tuple:
    """Para clustering, anomalias, redução e reforço. Devolve (tabela com resultados, relatório {erros, avisos})."""
    if len(df) > cl.MAX_LINHAS_PREVISAO:
        raise cl.ErroCiclo(f"O ficheiro tem {len(df)} linhas; o máximo é {cl.MAX_LINHAS_PREVISAO}.")
    if df.columns.duplicated().any():
        raise cl.ErroCiclo("O ficheiro tem colunas com o mesmo nome. Cada coluna deve aparecer só uma vez.")
    rel = ps.verificar_entrada(df, meta) if meta.get("tipo") in TIPOS_NAO_SUPERVISIONADOS else {"erros": [], "avisos": []}
    if rel["erros"]:
        raise cl.ErroCiclo(" ".join(rel["erros"]))
    try:
        return modelo.aplicar(df), rel
    except ValueError as e:
        raise cl.ErroCiclo(str(e))


# ── Descarregar QUALQUER modelo ───────────────────────────────────────────────
def _exemplo(meta: dict) -> dict:
    if meta.get("tipo") == "reforco_q":
        return {"linha": 0, "coluna": 0}
    if meta.get("tipo") == "reforco_dqn":
        return {c: 0.0 for c in meta.get("features") or []}
    return cl._exemplo_linha(meta)


def _leeme(meta: dict, executavel: bool) -> str:
    base = (f"Modelo «{meta.get('nome', '')}» exportado do DataForge EDU ({meta.get('algoritmo', '')}).\n\n"
            "modelo.joblib    O modelo treinado.\nmetadados.json   Algoritmo, parâmetros, métricas, colunas e versões das bibliotecas.\n")
    if executavel:
        base += ("dataforge_runtime.py   Pequeno ficheiro que abre o modelo (só precisa de numpy, pandas e scikit-learn).\n"
                 "usar_modelo.py   Exemplo: carrega o modelo e faz previsões.\n\nCorre:  pip install pandas scikit-learn joblib   e depois   python usar_modelo.py\n")
    else:
        base += ("\nEste modelo (série temporal) só se usa com o código do DataForge: abre a página «Testar Modelos» da plataforma. "
                 "O ficheiro serve para guardar uma cópia e para estudo.\n")
    return base + "Usa as mesmas versões das bibliotecas (ver metadados.json). Só abras ficheiros joblib em que confias: abrir um executa código.\n"


def exportar_modelo(objecto, meta: dict) -> bytes:
    """ZIP descarregável de qualquer modelo (classificação, regressão, séries, clustering, anomalias, redução, MLP, reforço, Pipelines do Laboratório)."""
    from sklearn.pipeline import Pipeline
    meta = {**meta, "versoes_bibliotecas": meta.get("versoes_bibliotecas") or mp._versoes()}
    if isinstance(objecto, Pipeline):
        return cl.exportar_pacote(objecto, meta)
    import joblib
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        runtime = hasattr(objecto, "para_pacote")
        pacote_buf = io.BytesIO()
        joblib.dump(objecto.para_pacote() if runtime else objecto, pacote_buf, compress=3)
        z.writestr("modelo.joblib", pacote_buf.getvalue())
        z.writestr("metadados.json", json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        if runtime:
            z.writestr("dataforge_runtime.py", (Path(__file__).with_name("dataforge_runtime.py")).read_text(encoding="utf-8"))
            texto = (PASTA_CODIGO / "uso_runtime.py.txt").read_text(encoding="utf-8")
            z.writestr("usar_modelo.py", texto.replace("@@NOME@@", str(meta.get("nome", "")).replace('"', "'"))
                       .replace("@@ALGORITMO@@", str(meta.get("algoritmo", ""))).replace("@@EXEMPLO@@", repr(_exemplo(meta))))
        z.writestr("LEEME.txt", _leeme(meta, runtime))
    return buf.getvalue()
