"""Guardar, descarregar e testar os modelos de texto, imagens, geradores e séries com rede neuronal (PyTorch).
Clustering, anomalias, redução e reforço já vivem em `dataforge_runtime.py`.

`PacoteGenerico` junta o modelo com tudo o que é preciso para o usar depois (vocabulário, vectorizador, scaler, dimensões...)
e sabe fazer o teste mais natural de cada tipo. As bibliotecas pesadas (PyTorch) só são importadas quando se usa o modelo.
"""
import io
import re

import numpy as np
import pandas as pd

TIPOS = {
    "texto": "Texto (BoW / TF-IDF)",
    "texto_lstm": "Texto (LSTM)",
    "imagem": "Imagens (CNN)",
    "generativo": "Gerador de dados",
    "serie_dl": "Série temporal (rede neuronal)",
}


class ErroPacote(Exception):
    """Pedido impossível para este modelo (mensagem pronta a mostrar)."""


def _limpar_texto(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-záéíóúàâêôãõü\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class PacoteGenerico:
    def __init__(self, tipo: str, objetos: dict, subtipo: str = ""):
        if tipo not in TIPOS:
            raise ErroPacote("Tipo de modelo desconhecido.")
        self.tipo, self.objetos, self.subtipo = tipo, dict(objetos), subtipo

    # ── texto ─────────────────────────────────────────────────────────────
    def prever_texto(self, texto: str) -> tuple:
        """(classe, {classe: probabilidade ou None})."""
        o = self.objetos
        limpo = _limpar_texto(str(texto))
        if not limpo:
            raise ErroPacote("Escreve um texto com palavras.")
        nomes = [str(c) for c in o["class_names"]]
        if self.tipo == "texto":
            X = o["vectorizer"].transform([limpo])
            idx = int(o["model"].predict(X)[0])
            probas = None
            if hasattr(o["model"], "predict_proba"):
                try:
                    p = o["model"].predict_proba(X)[0]
                    probas = {nomes[i]: float(v) for i, v in enumerate(p)}
                except Exception:
                    probas = None
            return nomes[idx], probas
        import torch
        vocab, max_len = o["vocab"], int(o["max_len"])
        ids = [vocab.get(w, 0) for w in limpo.split()][:max_len]
        ids = ids + [0] * (max_len - len(ids))
        m = o["model"]
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(torch.tensor([ids], dtype=torch.long)), dim=1)[0].numpy()
        return nomes[int(p.argmax())], {nomes[i]: float(v) for i, v in enumerate(p)}

    # ── imagens ───────────────────────────────────────────────────────────
    def prever_imagem(self, imagem) -> tuple:
        """`imagem`: PIL.Image. Mesma preparação do treino: 64x64, RGB, valores entre -1 e 1."""
        import torch
        o = self.objetos
        arr = np.asarray(imagem.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        x = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)
        m = o["model"]
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(x), dim=1)[0].numpy()
        nomes = [str(c) for c in o["class_names"]]
        return nomes[int(p.argmax())], {nomes[i]: float(v) for i, v in enumerate(p)}

    # ── geradores ─────────────────────────────────────────────────────────
    def gerar(self, n: int = 200) -> np.ndarray:
        """Amostras novas (VAE e GAN). O autoencoder não gera: usa `reconstruir`."""
        import torch
        o = self.objetos
        n = int(max(1, min(n, 5000)))
        if self.subtipo == "vae":
            m = o["model"]
            m.eval()
            with torch.no_grad():
                return m.decode(torch.randn(n, int(o["latent_dim"]))).numpy()
        if self.subtipo == "gan":
            g = o["model"]
            g.eval()
            with torch.no_grad():
                return g(torch.randn(n, int(o["noise_dim"]))).numpy()
        raise ErroPacote("Este modelo não gera dados novos: usa «reconstruir».")

    def reconstruir(self, df: pd.DataFrame) -> pd.DataFrame:
        """Autoencoder: devolve a reconstrução e o erro de cada linha (erro alto = caso estranho)."""
        import torch
        if self.subtipo != "ae":
            raise ErroPacote("Só o autoencoder reconstrói dados.")
        d = int(self.objetos["input_dim"])
        X = df.apply(pd.to_numeric, errors="coerce")
        if X.shape[1] != d:
            raise ErroPacote(f"O modelo espera {d} colunas numéricas e a tabela tem {X.shape[1]}.")
        X = X.fillna(X.median()).to_numpy(dtype=np.float32)
        m = self.objetos["model"]
        m.eval()
        with torch.no_grad():
            rec, _ = m(torch.tensor(X))
        rec = rec.numpy()
        saida = pd.DataFrame(rec, columns=[f"rec_{i + 1}" for i in range(d)])
        saida["erro"] = ((rec - X) ** 2).mean(axis=1)
        return saida

    # ── séries ────────────────────────────────────────────────────────────
    def prever_serie(self, passos: int) -> np.ndarray:
        """Continua a série a partir do fim dos dados de treino."""
        import torch
        o = self.objetos
        passos = int(max(1, min(passos, 365)))
        seq = np.asarray(o["last_seq"], dtype=np.float32).copy()
        m = o["model"]
        m.eval()
        out = []
        with torch.no_grad():
            for _ in range(passos):
                p = m(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)).item()
                out.append(p)
                seq = np.append(seq[1:], p)
        return o["scaler"].inverse_transform(np.array(out).reshape(-1, 1)).flatten()


def construir(tipo: str, objetos: dict, *, nome: str, algoritmo: str, origem: str, subtipo: str = "",
              metricas: dict = None, extra: dict = None) -> tuple:
    """(pacote, meta) prontos para `model_persistence.guardar_modelo`."""
    pacote = PacoteGenerico(tipo, objetos, subtipo)
    meta = {"tipo": tipo, "subtipo": subtipo, "nome": nome, "algoritmo": str(algoritmo), "algoritmo_id": str(algoritmo),
            "alvo": "", "features": [], "colunas_data": [], "num_cols": [], "cat_cols": [], "classes": None,
            "categorias_conhecidas": {}, "intervalos_numericos": {}, "parametros": {}, "cv": {}, "origem": origem,
            "metricas_teste": {k: float(v) for k, v in (metricas or {}).items() if isinstance(v, (int, float, np.number))},
            "avisos": [], "pipeline_completo": True}
    if "class_names" in objetos:
        meta["classes"] = [str(c) for c in objetos["class_names"]]
    if "features" in objetos:
        meta["features"] = [str(c) for c in objetos["features"]]
    meta.update(extra or {})
    return pacote, meta


def bytes_para_descarregar(pacote: PacoteGenerico, meta: dict) -> bytes:
    import joblib
    buf = io.BytesIO()
    joblib.dump({"modelo": pacote, "meta": meta, "formato": "dataforge-pacote-v1"}, buf, compress=3)
    return buf.getvalue()
