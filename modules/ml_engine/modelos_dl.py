"""Guardar e descarregar os modelos de NLP, Visão, geradores (Autoencoder, VAE, GAN) e séries com rede neuronal.

Segue o mesmo molde de `modelos_extra.py`: `de_*` transforma o resultado que a página deixou em memória em (objecto, metadados),
que `guardar_ui.bloco_guardar` guarda na conta ou descarrega com `exportar_modelo`. O objecto é um `PacoteGenerico`, que também sabe
fazer o teste (ver `modules/testar_modelos_dl.py`). Estes modelos precisam de PyTorch (excepto o texto TF-IDF) para serem abertos.
"""
from modules.ml_engine.modelo_legado import ErroLegado
from modules.ml_engine.pacote_generico import PacoteGenerico, construir


def _antigo():
    return ErroLegado("Este resultado é antigo e não tem tudo o que é preciso. Treina outra vez e depois guarda.")


def de_nlp_bow(res: dict, nome: str) -> tuple:
    try:
        obj = {"vectorizer": res["vectorizer"], "model": res["model"], "class_names": res["class_names"]}
        algo = f"{res['setup']['vec']} + {res['setup']['clf']}"
    except KeyError:
        raise _antigo()
    return construir("texto", obj, nome=nome, algoritmo=algo, origem="página NLP", metricas={"accuracy": res.get("acc"), "f1": res.get("f1")})


def de_nlp_lstm(res: dict, nome: str) -> tuple:
    if not res.get("_obj"):
        raise _antigo()
    return construir("texto_lstm", res["_obj"], nome=nome, algoritmo="LSTM", origem="página NLP", metricas={"accuracy": res.get("acc")})


def de_visao(res: dict, nome: str) -> tuple:
    try:
        obj = {"model": res["model"], "class_names": res["class_names"]}
    except KeyError:
        raise _antigo()
    return construir("imagem", obj, nome=nome, algoritmo=str((res.get("setup") or {}).get("arch", "CNN")), origem="página Visão",
                     metricas={"accuracy": res.get("acc")})


def de_generativo(res: dict, subtipo: str, nome: str) -> tuple:
    if not res.get("_obj"):
        raise _antigo()
    algoritmo = {"ae": "Autoencoder", "vae": "VAE", "gan": "GAN"}[subtipo]
    if subtipo == "gan":
        metricas = {"loss_g": res["g_losses"][-1], "loss_d": res["d_losses"][-1]}
    else:
        metricas = {"loss": res.get("loss_final")}
    return construir("generativo", res["_obj"], nome=nome, algoritmo=algoritmo, subtipo=subtipo, origem="página Modelos Generativos", metricas=metricas)


def de_serie_dl(res: dict, nome: str) -> tuple:
    if not res.get("_obj"):
        raise _antigo()
    return construir("serie_dl", res["_obj"], nome=nome, algoritmo=str((res.get("setup") or {}).get("modelo", "LSTM")), origem="página Séries Temporais",
                     metricas={"rmse": res.get("rmse"), "mae": res.get("mae")})
