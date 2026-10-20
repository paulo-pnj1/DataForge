"""Métricas de avaliação. Só se calcula o que é matematicamente válido; o resto fica `None`."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)

# nome interno -> (rótulo, maior_é_melhor)
METRICAS_REGRESSAO = {"r2": ("R²", True), "mae": ("MAE", False), "rmse": ("RMSE", False), "mse": ("MSE", False)}
METRICAS_CLASSIFICACAO = {"accuracy": ("Accuracy", True), "precision": ("Precision", True), "recall": ("Recall", True),
                          "f1": ("F1-score", True), "roc_auc": ("ROC-AUC", True), "balanced_accuracy": ("Accuracy balanceada", True)}
METRICAS_SERIE = {"mae": ("MAE", False), "rmse": ("RMSE", False), "mape": ("MAPE", False), "mase": ("MASE", False)}
CRITERIOS = {
    "regressao": ["rmse", "mae", "r2"],
    "classificacao": ["f1", "precision", "recall", "accuracy", "roc_auc"],
    "serie": ["mae", "rmse", "mase"],
}
AJUDA = {
    "r2": "R²: parte da variação explicada (1 = perfeito; 0 = não melhor que a média). Um R² alto não garante bons resultados no futuro.",
    "mae": "MAE: erro médio absoluto, na unidade do alvo. Fácil de interpretar.",
    "rmse": "RMSE: como o MAE mas castiga mais os erros grandes, na unidade do alvo.",
    "mse": "MSE: erro quadrático médio (unidade do alvo ao quadrado).",
    "mape": "MAPE: erro percentual médio. Só se calcula quando a série nunca é zero.",
    "mase": "MASE: erro em relação à previsão ingénua. Abaixo de 1 = melhor que a ingénua.",
    "accuracy": "Accuracy: percentagem de acertos. Engana quando as classes são muito desiguais.",
    "precision": "Precision (média por classe): quando prevê uma classe, quantas vezes acerta.",
    "recall": "Recall (média por classe): dos casos reais de uma classe, quantos encontra.",
    "f1": "F1-score (média por classe): equilíbrio entre precision e recall. Boa escolha com classes desiguais.",
    "roc_auc": "ROC-AUC: capacidade de separar as classes (0,5 = acaso; 1 = perfeito).",
}


def direccao(tipo: str, metrica: str) -> bool:
    """True se maior é melhor."""
    tabela = {"regressao": METRICAS_REGRESSAO, "classificacao": METRICAS_CLASSIFICACAO, "serie": METRICAS_SERIE}[tipo]
    return tabela[metrica][1]


def rotulo(tipo: str, metrica: str) -> str:
    tabela = {"regressao": METRICAS_REGRESSAO, "classificacao": METRICAS_CLASSIFICACAO, "serie": METRICAS_SERIE}[tipo]
    return tabela[metrica][0]


def _finitas(a) -> bool:
    return bool(np.all(np.isfinite(np.asarray(a, dtype=float))))


# ── Regressão ─────────────────────────────────────────────────────────────────
def metricas_regressao(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float).ravel()
    if not _finitas(y_pred):
        raise ValueError("O modelo devolveu previsões inválidas (NaN ou infinito).")
    mse = float(mean_squared_error(y_true, y_pred))
    return {"r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
            "mae": float(mean_absolute_error(y_true, y_pred)), "mse": mse, "rmse": float(np.sqrt(mse))}


def resumo_residuos(y_true, y_pred) -> dict:
    r = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float).ravel()
    return {"media": float(r.mean()), "desvio": float(r.std()), "max_abs": float(np.abs(r).max())}


# ── Classificação ─────────────────────────────────────────────────────────────
def pontuacao_classes(estimador, X):
    """Probabilidades (n×k) ou pontuação de decisão; None se o modelo não dá nenhuma."""
    try:
        if hasattr(estimador, "predict_proba"):
            return np.asarray(estimador.predict_proba(X))
        if hasattr(estimador, "decision_function"):
            return np.asarray(estimador.decision_function(X))
    except Exception:
        return None
    return None


def roc_auc_valido(y_true, score, n_classes: int):
    """ROC-AUC se for válido (todas as classes presentes no teste; probabilidades no multiclasse); senão None."""
    if score is None:
        return None
    y_true = np.asarray(y_true)
    presentes = np.unique(y_true)
    try:
        if n_classes == 2:
            if len(presentes) < 2:
                return None
            s = score[:, 1] if score.ndim == 2 else score
            return float(roc_auc_score(y_true, s))
        if score.ndim != 2 or len(presentes) < n_classes or not np.allclose(score.sum(axis=1), 1.0, atol=1e-3):
            return None
        return float(roc_auc_score(y_true, score, multi_class="ovr", average="macro", labels=list(range(n_classes))))
    except Exception:
        return None


def metricas_classificacao(y_true, y_pred, score, n_classes: int) -> dict:
    """Precision, recall e F1 são médias macro (cada classe pesa o mesmo): honestas com classes desiguais."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "roc_auc": roc_auc_valido(y_true, score, n_classes),
    }


def matriz_confusao(y_true, y_pred, n_classes: int):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def curvas_binarias(y_true, score) -> dict:
    """Curvas ROC e Precision-Recall (só para 2 classes e com pontuações)."""
    if score is None or len(np.unique(y_true)) < 2:
        return {}
    s = score[:, 1] if score.ndim == 2 else score
    fpr, tpr, _ = roc_curve(y_true, s)
    prec, rec, _ = precision_recall_curve(y_true, s)
    return {"roc": (fpr, tpr), "pr": (prec, rec)}


def aviso_desequilibrio(y, limite: float = 0.75):
    contagem = np.bincount(np.asarray(y, dtype=int))
    topo = contagem.max() / contagem.sum()
    if topo > limite:
        return (f"Classes desequilibradas ({topo:.0%} na maior): a accuracy engana. "
                "Usa F1-score, recall ou a matriz de confusão para escolher.")
    return None


# ── Séries temporais ──────────────────────────────────────────────────────────
def metricas_serie(y_true, y_pred, y_treino=None, periodo: int = 1) -> dict:
    """MAE, RMSE, MAPE (só se nunca houver zeros) e MASE (escala = erro da previsão ingénua no treino)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float).ravel()
    if not _finitas(y_pred):
        raise ValueError("O modelo devolveu previsões inválidas (NaN ou infinito).")
    erro = y_true - y_pred
    mae = float(np.mean(np.abs(erro)))
    out = {"mae": mae, "rmse": float(np.sqrt(np.mean(erro ** 2))), "mape": None, "mase": None}
    if np.all(np.abs(y_true) > 1e-8):
        out["mape"] = float(np.mean(np.abs(erro / y_true)) * 100)
    if y_treino is not None:
        yt = np.asarray(y_treino, dtype=float)
        m = int(periodo) if periodo and 1 <= int(periodo) < len(yt) else 1
        if len(yt) > m:
            escala = float(np.mean(np.abs(yt[m:] - yt[:-m])))
            if escala > 1e-12 and np.isfinite(escala):
                out["mase"] = mae / escala
    return out
