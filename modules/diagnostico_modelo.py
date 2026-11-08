"""
DataForge EDU - Secção «Diagnóstico do Modelo» (regressão e classificação)

Compara o desempenho do modelo nos dados de TREINO (que ele viu) com o desempenho nos dados de TESTE
(que nunca viu), para o estudante perceber se o modelo aprendeu a regra geral ou se está só a memorizar.

Regressão:     R² e MAE (treino e teste)        -> `diagnosticar_modelo`        / `mostrar_diagnostico_modelo`
Classificação: Accuracy e F1 (treino e teste)   -> `diagnosticar_classificacao` / `mostrar_diagnostico_classificacao`

Em ambos há a diferença entre treino e teste e o diagnóstico final: Overfitting, Underfitting ou Boa Generalização.

A lógica (`diagnosticar_*`) não depende do Streamlit, para se poder testar sozinha.
Os limites são os mesmos que o resto da plataforma usa (ver `ml_engine.ciclo_ml.LIMITE_SOBREAJUSTE`).
"""
import numpy as np
import streamlit as st

from modules.i18n import tr
from modules.utils import (C_ACCENT, C_AMBER, C_GREEN, C_RED, aviso_box, section_title, sucesso_box)

# ── Limites de decisão (fáceis de explicar a um estudante) ────────────────────
LIMITE_SOBREAJUSTE = 0.15     # R² de treino acima do de teste por mais do que isto -> overfitting
LIMITE_SOBREAJUSTE_MAE = 1.5  # ...ou, com diferença de R² já visível, erro de teste ≥ 1,5× o do treino
LIMITE_DIF_R2_MAE = 0.10      # diferença mínima de R² para a regra do MAE contar (um erro de treino minúsculo é normal em muitos modelos)
LIMITE_SUBAJUSTE = 0.50       # R² de treino abaixo disto -> o modelo nem nos exemplos que viu aprendeu bem
MIN_EXEMPLOS = 30             # abaixo disto a leitura é pouco fiável

OVERFITTING, UNDERFITTING, BOA, INCONCLUSIVO = "overfitting", "underfitting", "boa", "inconclusivo"
ROTULOS = {OVERFITTING: "Overfitting", UNDERFITTING: "Underfitting", BOA: "Boa Generalização",
           INCONCLUSIVO: "Inconclusivo"}


_SEM_DADOS = "Faltam métricas de treino ou de teste (ou têm valores inválidos), por isso não dá para comparar."


def _valido(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def metricas_treino_teste(y_treino, pred_treino, y_teste, pred_teste) -> dict:
    """R² e MAE no treino e no teste, a partir dos valores reais e previstos de cada conjunto."""
    from sklearn.metrics import mean_absolute_error, r2_score
    return {
        "r2_treino": float(r2_score(y_treino, pred_treino)), "r2_teste": float(r2_score(y_teste, pred_teste)),
        "mae_treino": float(mean_absolute_error(y_treino, pred_treino)),
        "mae_teste": float(mean_absolute_error(y_teste, pred_teste)),
        "n_treino": int(len(y_treino)), "n_teste": int(len(y_teste)),
    }


def diagnosticar_modelo(r2_treino, r2_teste, mae_treino, mae_teste, n_treino=None, n_teste=None) -> dict:
    """Devolve o diagnóstico (dicionário) a partir das quatro métricas.

    Regras, por esta ordem:
      1. Overfitting: o R² baixa mais de 0,15 do treino para o teste; ou baixa mais de 0,10 e, ao mesmo tempo,
         o MAE de teste é pelo menos 1,5× o de treino.
      2. Underfitting: o R² de treino é inferior a 0,50 (nem nos dados que viu o modelo aprendeu bem).
      3. Boa Generalização: todo o resto (treino e teste próximos e com R² aceitável).
    """
    if not all(_valido(v) for v in (r2_treino, r2_teste, mae_treino, mae_teste)):
        return {"estado": INCONCLUSIVO, "rotulo": ROTULOS[INCONCLUSIVO], "nivel": "aviso",
                "titulo": "Não foi possível diagnosticar",
                "explicacao": _SEM_DADOS, "partes": [_SEM_DADOS],
                "sugestoes": [], "avisos": [], "dif_r2": None, "dif_mae": None, "dif_mae_pct": None,
                "r2_treino": None, "r2_teste": None, "mae_treino": None, "mae_teste": None}

    r2_tr, r2_te, mae_tr, mae_te = float(r2_treino), float(r2_teste), float(mae_treino), float(mae_teste)
    dif_r2 = r2_tr - r2_te                       # positivo = pior no teste
    dif_mae = mae_te - mae_tr                    # positivo = erra mais no teste
    dif_mae_pct = (dif_mae / mae_tr) if mae_tr > 1e-12 else None
    razao_mae = (mae_te / mae_tr) if mae_tr > 1e-12 else None

    avisos = []
    if n_teste is not None and n_teste < MIN_EXEMPLOS:
        avisos.append(f"O teste tem só {n_teste} exemplos: pequenas diferenças podem ser pura sorte.")
    if n_treino is not None and n_treino < MIN_EXEMPLOS:
        avisos.append(f"O treino tem só {n_treino} exemplos: com tão poucos dados é difícil tirar conclusões firmes.")

    erro_sobe = dif_mae > 0 and dif_mae_pct is not None and dif_mae_pct > 0.10
    sobreajuste = dif_r2 > LIMITE_SOBREAJUSTE or (
        dif_r2 > LIMITE_DIF_R2_MAE and razao_mae is not None and razao_mae >= LIMITE_SOBREAJUSTE_MAE)

    if sobreajuste:
        estado, nivel = OVERFITTING, "aviso"
        titulo = "Overfitting: o modelo está a memorizar"
        partes = [f"O modelo acerta muito melhor nos dados de treino (R² = {r2_tr:.4f}) do que nos dados de teste "
                  f"(R² = {r2_te:.4f}): perde {dif_r2:.4f} de R²."]
        if erro_sobe:
            partes.append(f"O erro médio (MAE) também sobe de {mae_tr:.4f} no treino para {mae_te:.4f} no teste.")
        partes.append("Isto indica que decorou os exemplos de treino em vez de aprender a regra geral, e por isso falha "
                      "em dados novos.")
        sugestoes = ["Juntar mais exemplos de treino, se possível.",
                     "Usar um modelo mais simples ou limitar a sua complexidade (por exemplo, menor profundidade nas árvores).",
                     "Aumentar a regularização (Ridge, Lasso) ou retirar variáveis pouco úteis.",
                     "Confirmar com a validação cruzada: se as notas variarem muito entre folds, o modelo é instável."]
    elif r2_tr < LIMITE_SUBAJUSTE:
        estado, nivel = UNDERFITTING, "aviso"
        titulo = "Underfitting: o modelo ainda não aprendeu"
        partes = [f"O R² é baixo mesmo nos dados de treino (R² = {r2_tr:.4f}; no teste, {r2_te:.4f}).",
                  "O modelo nem nos exemplos que já viu consegue encontrar o padrão: é simples demais para estes dados, "
                  "ou as variáveis escolhidas explicam pouco o que se quer prever."]
        sugestoes = ["Experimentar um algoritmo mais flexível (por exemplo, Random Forest ou Gradient Boosting).",
                     "Acrescentar variáveis que expliquem melhor o alvo, ou criar novas a partir das existentes.",
                     "Reduzir a regularização, se estiveres a usar Ridge ou Lasso.",
                     "Verificar se os dados têm mesmo sinal: talvez as variáveis não expliquem o alvo."]
    else:
        estado, nivel = BOA, "ok"
        titulo = "Boa Generalização: o modelo aprendeu a regra geral"
        partes = [f"O desempenho no treino (R² = {r2_tr:.4f}) e no teste (R² = {r2_te:.4f}) é parecido: a diferença "
                  f"é de {dif_r2:.4f}.",
                  "O modelo aprendeu algo que também funciona em dados que nunca viu, em vez de decorar os exemplos."]
        if razao_mae is not None and razao_mae >= LIMITE_SOBREAJUSTE_MAE:
            partes.append(f"O erro médio (MAE) sobe de {mae_tr:.4f} no treino para {mae_te:.4f} no teste. Isto é comum em modelos "
                          "que se ajustam muito aos dados de treino (como as florestas aleatórias) e, com um R² de teste "
                          "bom, não chega para falar de memorização; mas convém acompanhar.")
        sugestoes = ["Confirmar com a validação cruzada para ter mais confiança.",
                     "Se quiseres melhorar, experimenta outros algoritmos e ajusta os hiperparâmetros."]

    return {"estado": estado, "rotulo": ROTULOS[estado], "nivel": nivel, "titulo": titulo,
            "explicacao": " ".join(partes), "partes": partes, "sugestoes": sugestoes, "avisos": avisos, "dif_r2": dif_r2, "dif_mae": dif_mae, "dif_mae_pct": dif_mae_pct,
            "r2_treino": r2_tr, "r2_teste": r2_te, "mae_treino": mae_tr, "mae_teste": mae_te}


def _barras_treino_teste(estado: str, paineis, formato: str = "{:.3f}", teto=None):
    """Barras treino × teste, um painel por métrica. `paineis` = [(título, valor_treino, valor_teste), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cor_teste = {OVERFITTING: C_AMBER, UNDERFITTING: C_RED}.get(estado, C_GREEN)
    fig, axes = plt.subplots(1, len(paineis), figsize=(8, 3.2), facecolor="#161B27")
    for ax, (titulo, tr, te) in zip(axes, paineis):
        ax.set_facecolor("#161B27")
        barras = ax.bar(["Treino", "Teste"], [tr, te], color=[C_ACCENT, cor_teste], alpha=0.9, edgecolor="#2A3347")
        ax.axhline(0, color="#2A3347", lw=1)
        if teto is not None:
            ax.set_ylim(0, teto)
        ax.set_title(titulo, color="#E8EBF0", fontsize=10)
        ax.tick_params(colors="#9BA3B2")
        for s_ in ax.spines.values():
            s_.set_edgecolor("#2A3347")
        for b, v in zip(barras, (tr, te)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), formato.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", color="#9BA3B2", fontsize=9)
    fig.tight_layout()
    return fig


def _grafico(d: dict):
    """Regressão: barras treino × teste de R² (à esquerda) e MAE (à direita)."""
    return _barras_treino_teste(d["estado"], (("R² (maior é melhor)", d["r2_treino"], d["r2_teste"]),
                                              ("MAE (menor é melhor)", d["mae_treino"], d["mae_teste"])))


def _grafico_clf(d: dict):
    """Classificação: barras treino × teste de Accuracy (à esquerda) e F1 (à direita)."""
    return _barras_treino_teste(d["estado"], (("Accuracy (maior é melhor)", d["acc_treino"], d["acc_teste"]),
                                              ("F1 (maior é melhor)", d["f1_treino"], d["f1_teste"])),
                                formato="{:.1%}", teto=1.12)


def mostrar_diagnostico_modelo(r2_treino, r2_teste, mae_treino, mae_teste, n_treino=None, n_teste=None,
                               alvo: str = "", titulo: bool = True) -> dict:
    """Desenha a secção «Diagnóstico do Modelo» (regressão) e devolve o diagnóstico (para guardar ou explicar).

    `titulo=False` omite o título e a legenda (para usar a secção dentro de um separador, como no Comparador)."""
    d = diagnosticar_modelo(r2_treino, r2_teste, mae_treino, mae_teste, n_treino, n_teste)

    if titulo:
        section_title("Diagnóstico do Modelo")
        st.caption("Compara a nota do modelo nos dados com que treinou e nos dados que nunca viu. "
                   "Assim vês se ele aprendeu a regra geral ou se está só a memorizar os exemplos.")

    if d["estado"] == INCONCLUSIVO:
        aviso_box("<br>".join(d["partes"]))
        return d

    unidade = f" (na unidade de «{alvo}»)" if alvo else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² do treino", f"{d['r2_treino']:.4f}",
              help="Nota nos dados que o modelo usou para aprender. 1 = perfeito, 0 = não melhor do que prever sempre a média.")
    c2.metric("R² do teste", f"{d['r2_teste']:.4f}",
              help="Nota nos dados que o modelo nunca viu. É a que mostra como ele se vai portar na vida real.")
    c3.metric("MAE do treino", f"{d['mae_treino']:.4f}",
              help=f"Erro médio nos dados de treino{unidade}. Quanto menor, melhor.")
    c4.metric("MAE do teste", f"{d['mae_teste']:.4f}",
              help=f"Erro médio nos dados de teste{unidade}. Se for muito maior do que no treino, o modelo não generaliza.")

    c5, c6, c7 = st.columns(3)
    c5.metric("Diferença de R² (treino − teste)", f"{d['dif_r2']:+.4f}",
              help="Quanto o R² baixa do treino para o teste. Perto de 0 = comportamento parecido; "
                   f"acima de {LIMITE_SOBREAJUSTE:.2f} é sinal de memorização.")
    pct = f" ({d['dif_mae_pct']:+.0%})" if d["dif_mae_pct"] is not None else ""
    c6.metric("Diferença de MAE (teste − treino)", f"{d['dif_mae']:+.4f}",
              delta=pct.strip(" ()") or None, delta_color="inverse",
              help="Quanto o erro médio aumenta no teste em relação ao treino. Positivo = o modelo erra mais em dados novos.")
    c7.metric("Diagnóstico", tr(d["rotulo"]),
              help="Overfitting = memoriza o treino. Underfitting = não aprendeu o padrão. Boa Generalização = aprendeu a regra geral.")

    # uma frase por linha: o tradutor trabalha frase a frase (ver modules/i18n.py)
    (sucesso_box if d["nivel"] == "ok" else aviso_box)(f"<b>{d['titulo']}</b><br>" + "<br>".join(d["partes"]))
    for a in d["avisos"]:
        st.caption(a)
    if d["sugestoes"]:
        with st.expander("O que posso fazer?"):
            st.markdown("\n".join(f"- {s}" for s in d["sugestoes"]))

    fig = _grafico(d)
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO: Accuracy e F1, treino × teste
# ══════════════════════════════════════════════════════════════════════════════
LIMITE_SOBREAJUSTE_CLF = 0.15  # Accuracy ou F1 de treino acima do de teste por mais do que isto -> overfitting (o mesmo limite do resto da plataforma)
LIMITE_SUBAJUSTE_CLF = 0.60    # Accuracy de treino abaixo disto -> o modelo nem nos exemplos que viu aprendeu bem
LIMITE_ACC_F1 = 0.10           # Accuracy de teste acima do F1 por mais do que isto -> aviso de classes desequilibradas


def metricas_treino_teste_clf(y_treino, pred_treino, y_teste, pred_teste) -> dict:
    """Accuracy e F1 (média ponderada, como no resto da página) no treino e no teste."""
    from sklearn.metrics import accuracy_score, f1_score

    def _f1(y, p):
        return float(f1_score(y, p, average="weighted", zero_division=0))
    return {
        "acc_treino": float(accuracy_score(y_treino, pred_treino)), "acc_teste": float(accuracy_score(y_teste, pred_teste)),
        "f1_treino": _f1(y_treino, pred_treino), "f1_teste": _f1(y_teste, pred_teste),
        "n_treino": int(len(y_treino)), "n_teste": int(len(y_teste)),
    }


def diagnosticar_classificacao(acc_treino, acc_teste, f1_treino, f1_teste, n_treino=None, n_teste=None) -> dict:
    """Devolve o diagnóstico (dicionário) a partir das quatro métricas (valores entre 0 e 1).

    Regras, por esta ordem:
      1. Overfitting: a Accuracy ou o F1 baixam mais de 0,15 (15 pontos percentuais) do treino para o teste.
      2. Underfitting: a Accuracy de treino é inferior a 0,60 (nem nos dados que viu o modelo aprendeu bem).
      3. Boa Generalização: todo o resto (treino e teste próximos e com Accuracy aceitável).
    """
    if not all(_valido(v) for v in (acc_treino, acc_teste, f1_treino, f1_teste)):
        return {"estado": INCONCLUSIVO, "rotulo": ROTULOS[INCONCLUSIVO], "nivel": "aviso",
                "titulo": "Não foi possível diagnosticar",
                "explicacao": _SEM_DADOS, "partes": [_SEM_DADOS],
                "sugestoes": [], "avisos": [], "dif_acc": None, "dif_f1": None,
                "acc_treino": None, "acc_teste": None, "f1_treino": None, "f1_teste": None}

    a_tr, a_te, f_tr, f_te = float(acc_treino), float(acc_teste), float(f1_treino), float(f1_teste)
    dif_acc = a_tr - a_te                       # positivo = pior no teste
    dif_f1 = f_tr - f_te

    avisos = []
    if n_teste is not None and n_teste < MIN_EXEMPLOS:
        avisos.append(f"O teste tem só {n_teste} exemplos: pequenas diferenças podem ser pura sorte.")
    if n_treino is not None and n_treino < MIN_EXEMPLOS:
        avisos.append(f"O treino tem só {n_treino} exemplos: com tão poucos dados é difícil tirar conclusões firmes.")
    if a_te - f_te > LIMITE_ACC_F1:
        avisos.append(f"A Accuracy ({a_te:.1%}) está bem acima do F1 ({f_te:.1%}): com classes desequilibradas a Accuracy engana, "
                      "por isso confia mais no F1.")

    if max(dif_acc, dif_f1) > LIMITE_SOBREAJUSTE_CLF:
        estado, nivel = OVERFITTING, "aviso"
        titulo = "Overfitting: o modelo está a memorizar"
        partes = [f"A Accuracy é de {a_tr:.1%} nos dados de treino e de {a_te:.1%} nos dados de teste: "
                  f"perde {dif_acc * 100:.1f} pontos percentuais."]
        if dif_f1 > 0.05:
            partes.append(f"O F1 também desce, de {f_tr:.1%} no treino para {f_te:.1%} no teste.")
        partes.append("Isto indica que decorou os exemplos de treino em vez de aprender a regra geral, e por isso falha "
                      "em dados novos.")
        sugestoes = ["Juntar mais exemplos de treino, se possível.",
                     "Usar um modelo mais simples ou limitar a sua complexidade (por exemplo, menor profundidade nas árvores).",
                     "Aumentar a regularização (por exemplo, diminuir o C na Regressão Logística ou no SVM) ou retirar variáveis pouco úteis.",
                     "Confirmar com a validação cruzada: se as notas variarem muito entre folds, o modelo é instável."]
    elif a_tr < LIMITE_SUBAJUSTE_CLF:
        estado, nivel = UNDERFITTING, "aviso"
        titulo = "Underfitting: o modelo ainda não aprendeu"
        partes = [f"A Accuracy é baixa mesmo nos dados de treino ({a_tr:.1%}; no teste, {a_te:.1%}).",
                  "O modelo nem nos exemplos que já viu consegue encontrar o padrão: é simples demais para estes dados, "
                  "ou as variáveis escolhidas explicam pouco o que se quer prever."]
        sugestoes = ["Experimentar um algoritmo mais flexível (por exemplo, Random Forest ou Gradient Boosting).",
                     "Acrescentar variáveis que expliquem melhor o alvo, ou criar novas a partir das existentes.",
                     "Reduzir a regularização (por exemplo, aumentar o C na Regressão Logística ou no SVM), se a estiveres a usar.",
                     "Verificar se os dados têm mesmo sinal: talvez as variáveis não expliquem o alvo."]
    else:
        estado, nivel = BOA, "ok"
        titulo = "Boa Generalização: o modelo aprendeu a regra geral"
        partes = [f"A Accuracy no treino ({a_tr:.1%}) e no teste ({a_te:.1%}) é parecida: a diferença é de "
                  f"{dif_acc * 100:.1f} pontos percentuais.",
                  "O modelo aprendeu algo que também funciona em dados que nunca viu, em vez de decorar os exemplos."]
        sugestoes = ["Confirmar com a validação cruzada para ter mais confiança.",
                     "Se quiseres melhorar, experimenta outros algoritmos e ajusta os hiperparâmetros."]

    return {"estado": estado, "rotulo": ROTULOS[estado], "nivel": nivel, "titulo": titulo,
            "explicacao": " ".join(partes), "partes": partes, "sugestoes": sugestoes, "avisos": avisos,
            "dif_acc": dif_acc, "dif_f1": dif_f1,
            "acc_treino": a_tr, "acc_teste": a_te, "f1_treino": f_tr, "f1_teste": f_te}


def mostrar_diagnostico_classificacao(acc_treino, acc_teste, f1_treino, f1_teste, n_treino=None, n_teste=None,
                                      titulo: bool = True) -> dict:
    """Desenha a secção «Diagnóstico do Modelo» (classificação) e devolve o diagnóstico (para guardar ou explicar).

    `titulo=False` omite o título e a legenda (para usar a secção dentro de um separador, como no Comparador)."""
    d = diagnosticar_classificacao(acc_treino, acc_teste, f1_treino, f1_teste, n_treino, n_teste)

    if titulo:
        section_title("Diagnóstico do Modelo")
        st.caption("Compara a nota do modelo nos dados com que treinou e nos dados que nunca viu. "
                   "Assim vês se ele aprendeu a regra geral ou se está só a memorizar os exemplos.")

    if d["estado"] == INCONCLUSIVO:
        aviso_box("<br>".join(d["partes"]))
        return d

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy do treino", f"{d['acc_treino']:.1%}",
              help="Percentagem de acertos nos dados que o modelo usou para aprender.")
    c2.metric("Accuracy do teste", f"{d['acc_teste']:.1%}",
              help="Percentagem de acertos nos dados que o modelo nunca viu. É a que mostra como ele se vai portar na vida real.")
    c3.metric("F1 do treino", f"{d['f1_treino']:.1%}",
              help="Nota que junta Precision e Recall, nos dados de treino. 100% = perfeito.")
    c4.metric("F1 do teste", f"{d['f1_teste']:.1%}",
              help="O F1 nos dados que o modelo nunca viu. Conta mais do que a Accuracy quando há classes muito desequilibradas.")

    c5, c6, c7 = st.columns(3)
    limite = f"{LIMITE_SOBREAJUSTE_CLF * 100:.0f}"
    c5.metric("Diferença de Accuracy (treino − teste)", f"{d['dif_acc'] * 100:+.1f} p.p.",
              help="Quanto a Accuracy baixa do treino para o teste, em pontos percentuais (p.p.). Perto de 0 = comportamento parecido; "
                   f"acima de {limite} p.p. é sinal de memorização.")
    c6.metric("Diferença de F1 (treino − teste)", f"{d['dif_f1'] * 100:+.1f} p.p.",
              help="Quanto o F1 baixa do treino para o teste, em pontos percentuais (p.p.). Positivo = o modelo é pior em dados novos.")
    c7.metric("Diagnóstico", tr(d["rotulo"]),
              help="Overfitting = memoriza o treino. Underfitting = não aprendeu o padrão. Boa Generalização = aprendeu a regra geral.")

    # uma frase por linha: o tradutor trabalha frase a frase (ver modules/i18n.py)
    (sucesso_box if d["nivel"] == "ok" else aviso_box)(f"<b>{d['titulo']}</b><br>" + "<br>".join(d["partes"]))
    for a in d["avisos"]:
        st.caption(a)
    if d["sugestoes"]:
        with st.expander("O que posso fazer?"):
            st.markdown("\n".join(f"- {s}" for s in d["sugestoes"]))

    fig = _grafico_clf(d)
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return d
