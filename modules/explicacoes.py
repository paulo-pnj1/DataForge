"""
DataForge EDU - Explicações em linguagem simples
Traduz métricas e resultados de ML para frases que qualquer pessoa entende.
"""
import numpy as np
import streamlit as st


# ══════════════════════════════════════════════════════
# FRASES CURTAS (reutilizáveis)
# ══════════════════════════════════════════════════════
def em_cada_10(valor: float) -> str:
    """0.87 -> '9 em cada 10'"""
    n = int(round(max(0.0, min(1.0, valor)) * 10))
    return f"{n} em cada 10"


def frase_acerto(acc: float) -> str:
    return f"O modelo acertou cerca de {em_cada_10(acc)} casos novos ({acc:.0%})."


def nivel_qualidade(acc: float) -> tuple:
    """(rótulo, cor) para a qualidade de uma accuracy."""
    if acc >= 0.90:
        return "Excelente", "green"
    if acc >= 0.75:
        return "Bom", "green"
    if acc >= 0.60:
        return "Razoável", "amber"
    return "Fraco", "red"


def nivel_r2(r2: float) -> tuple:
    if r2 >= 0.85:
        return "Excelente", "green"
    if r2 >= 0.65:
        return "Bom", "green"
    if r2 >= 0.40:
        return "Razoável", "amber"
    return "Fraco", "red"


# Textos de ajuda (tooltips) para st.metric(help=...)
AJUDA_METRICAS = {
    "Accuracy":  "De todos os casos de teste, quantos o modelo acertou. 90% = acerta 9 em cada 10.",
    "Precision": "Quando o modelo diz 'é desta classe', quantas vezes tem razão. Importante quando um falso alarme custa caro.",
    "Recall":    "De todos os casos que REALMENTE eram desta classe, quantos o modelo encontrou. Importante quando falhar um caso é grave.",
    "F1-Score":  "Uma nota única que junta Precision e Recall. Só é alta se as duas forem boas.",
    "ROC-AUC":   "Mede se o modelo consegue separar bem as classes. 1.0 = perfeito, 0.5 = tanto faz atirar uma moeda ao ar.",
    "R²":        "Quanto da variação dos dados o modelo consegue explicar. 1.0 = perfeito, 0 = não melhor do que prever sempre a média.",
    "RMSE":      "O erro típico do modelo, na MESMA unidade do que estás a prever. Quanto menor, melhor.",
    "MAE":       "Em média, quanto o modelo falha (em valor absoluto). Fácil de interpretar.",
    "MSE":       "Erro ao quadrado. Serve para o computador comparar modelos; para humanos o RMSE é mais fácil.",
    "Média CV":  "Nota média do modelo em várias 'provas' diferentes. Mais fiável do que uma só prova.",
    "Desvio Padrão": "Quanto a nota varia de prova para prova. Pequeno = modelo estável.",
}


# ══════════════════════════════════════════════════════
# CLASSIFICAÇÃO
# ══════════════════════════════════════════════════════
def explicar_classificacao(r: dict):
    """Painel 'Explica-me estes números' para resultados de classificação."""
    acc, prec, rec, f1 = r["acc"], r["prec"], r["rec"], r["f1"]
    cv = r["cv_scores"]
    rotulo, _ = nivel_qualidade(acc)
    from modules.tutor_ia import registar
    registar("Resultado de classificação", "\n".join([
        f"Accuracy (taxa de acerto): {acc:.1%}", f"Precision: {prec:.1%}", f"Recall: {rec:.1%}", f"F1: {f1:.1%}"]
        + ([f"ROC-AUC: {r['auc']:.3f}"] if r.get("auc") is not None else [])
        + [f"Validação cruzada ({len(cv)} provas): média {cv.mean():.1%}, variação ±{cv.std():.1%}", f"Avaliação geral: {rotulo}"]))

    with st.expander("Explica-me estes números (linguagem simples)", expanded=False):
        st.markdown(f"""
**Resumo:** resultado **{rotulo.lower()}**. {frase_acerto(acc)}

| Número | O que significa, sem complicar |
|---|---|
| **Accuracy {acc:.0%}** | Fizeste uma prova com casos que o modelo nunca viu. Acertou **{em_cada_10(acc)}**. |
| **Precision {prec:.0%}** | Quando o modelo aponta uma classe, tem razão em **{em_cada_10(prec)}** vezes. (Mede os *falsos alarmes*.) |
| **Recall {rec:.0%}** | Dos casos que realmente eram dessa classe, o modelo encontrou **{em_cada_10(rec)}**. (Mede os casos *que escaparam*.) |
| **F1 {f1:.0%}** | Nota que junta Precision e Recall numa só. |
""")
        if r.get("auc") is not None:
            auc = r["auc"]
            leitura = ("separa muito bem as classes" if auc >= 0.9 else
                       "separa razoavelmente as classes" if auc >= 0.75 else
                       "tem dificuldade em separar as classes")
            st.markdown(f"**ROC-AUC {auc:.2f}** - o modelo {leitura} (1.0 seria perfeito; 0.5 seria tirar à sorte).")

        st.markdown(f"""
**Validação cruzada ({len(cv)} provas):** o modelo foi testado {len(cv)} vezes com partes diferentes dos dados.
Nota média **{cv.mean():.0%}**, variando cerca de **±{cv.std():.0%}**.
{"As notas são parecidas entre si, por isso o modelo é **estável**." if cv.std() < 0.05 else "As notas variam bastante, por isso o modelo é **instável** - experimenta mais dados ou um modelo mais simples."}
""")

        st.markdown("""
**Como ler a Matriz de Confusão:** cada linha é a resposta *verdadeira* e cada coluna é o que o modelo *previu*.
A diagonal (de cima à esquerda para baixo à direita) são os **acertos**. Tudo fora da diagonal são **erros**,
e mostra que classes o modelo anda a confundir.
""")

        # O que fazer a seguir
        sugestoes = []
        if acc > 0.95 and cv.mean() < acc - 0.05:
            sugestoes.append("O modelo parece ter **decorado** os dados de treino (overfitting). Tenta um modelo mais simples ou limita a profundidade/complexidade.")
        elif cv.mean() < 0.6:
            sugestoes.append("O modelo está **a aprender pouco**. Experimenta um algoritmo mais forte (Random Forest, Gradient Boosting) ou acrescenta mais informação (colunas) aos dados.")
        if prec + 0.15 < rec:
            sugestoes.append("O modelo **dispara muitos falsos alarmes** (Precision baixa face ao Recall).")
        elif rec + 0.15 < prec:
            sugestoes.append("O modelo **deixa escapar casos** (Recall baixo face à Precision).")
        if not sugestoes:
            sugestoes.append("Está bem equilibrado. Para tentar melhorar: compara com outro algoritmo no **Comparador de Modelos**.")
        st.markdown("**O que fazer a seguir:**")
        for s in sugestoes:
            st.markdown(f"- {s}")


# ══════════════════════════════════════════════════════
# REGRESSÃO
# ══════════════════════════════════════════════════════
def explicar_regressao(y_test, y_pred, r2: float, rmse: float, mae: float,
                       cv_scores, target_nome: str = "o valor"):
    """Painel 'Explica-me estes números' para resultados de regressão."""
    y_test = np.asarray(y_test, dtype=float)
    media = float(np.mean(np.abs(y_test))) or 1.0
    erro_rel = mae / media
    rotulo, _ = nivel_r2(r2)
    from modules.tutor_ia import registar
    registar("Resultado de regressão", "\n".join([
        f"A prever: {target_nome}", f"R²: {r2:.3f}", f"MAE (erro médio): {mae:.4g}", f"RMSE: {rmse:.4g}",
        f"Erro médio em relação ao valor típico: {erro_rel:.0%}",
        f"Validação cruzada: média {float(np.mean(cv_scores)):.3f}, variação ±{float(np.std(cv_scores)):.3f}", f"Avaliação geral: {rotulo}"]))

    with st.expander("Explica-me estes números (linguagem simples)", expanded=False):
        st.markdown(f"""
**Resumo:** resultado **{rotulo.lower()}**. Ao prever **{target_nome}**, o modelo falha em média por **{mae:,.3g}**
(cerca de **{erro_rel:.0%}** do valor típico).

| Número | O que significa, sem complicar |
|---|---|
| **R² = {r2:.2f}** | O modelo consegue explicar **{max(r2, 0):.0%}** do que faz {target_nome} subir e descer. (0% = tão mau como chutar sempre a média; 100% = perfeito.) |
| **MAE = {mae:,.3g}** | Em média, a previsão está a **{mae:,.3g}** do valor real. |
| **RMSE = {rmse:,.3g}** | Parecido com o MAE, mas castiga mais os erros grandes. Quando o RMSE é muito maior do que o MAE, há alguns erros muito feios. |
""")
        cv = np.asarray(cv_scores)
        st.markdown(f"""
**Validação cruzada ({len(cv)} provas):** R² médio **{cv.mean():.2f}** (±{cv.std():.2f}).
{"Notas parecidas entre provas = modelo **estável**." if cv.std() < 0.08 else "As notas variam bastante entre provas = modelo **instável**."}
""")
        st.markdown("""
**Como ler o gráfico Real vs Previsto:** cada ponto é um caso de teste. Quanto mais os pontos
estiverem encostados à linha diagonal, melhor o modelo. Pontos longe da linha = erros grandes.
""")
        if r2 < 0:
            st.markdown("**A seguir:** o modelo é pior do que prever sempre a média. Verifica as colunas escolhidas e tenta outro algoritmo.")
        elif r2 < 0.4:
            st.markdown("**A seguir:** as colunas escolhidas explicam pouco. Tenta acrescentar mais informação ou usar Random Forest / Gradient Boosting.")
        elif r2 > 0.97:
            st.markdown("**A seguir:** nota quase perfeita é suspeita. Confirma que nenhuma coluna **contém já a resposta** (fuga de informação).")
        else:
            st.markdown("**A seguir:** bom resultado! Compara com outros algoritmos no **Comparador de Modelos**.")
