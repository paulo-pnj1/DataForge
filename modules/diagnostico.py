"""
DataForge EDU - Diagnóstico automático de um CSV
Olha para a tabela e explica, em linguagem simples, o que precisa de atenção antes de treinar.
"""
import numpy as np
import pandas as pd
import streamlit as st


def diagnosticar(df: pd.DataFrame) -> list:
    """Devolve lista de (nivel, texto): nivel em {'ok','aviso','erro','info'}."""
    out = []
    n, c = df.shape
    out.append(("info", f"A tabela tem **{n} linhas** (exemplos) e **{c} colunas**."))
    if n < 100:
        out.append(("aviso", "Poucos exemplos (menos de 100): os resultados vão mexer muito de execução para execução. Quantos mais exemplos, melhor."))

    dup = int(df.duplicated().sum())
    if dup:
        out.append(("aviso", f"Há **{dup} linhas repetidas**. Costuma ser boa ideia removê-las (senão o modelo vê o mesmo exemplo duas vezes e a prova fica enganada)."))

    falta = df.isna().sum()
    falta = falta[falta > 0].sort_values(ascending=False)
    if len(falta):
        top = ", ".join(f"**{k}** ({v}, {v / n:.0%})" for k, v in falta.head(4).items())
        out.append(("aviso", f"Valores em falta em {len(falta)} coluna(s): {top}. Preenche com média/mediana ou remove, antes de treinar."))
        for k, v in falta.items():
            if v / n > 0.5:
                out.append(("erro", f"A coluna **{k}** tem mais de metade dos valores em falta. Provavelmente é melhor não a usar."))
    else:
        out.append(("ok", "Não há valores em falta."))

    for col in df.columns:
        s = df[col]
        nun = s.nunique(dropna=True)
        if nun <= 1:
            out.append(("aviso", f"A coluna **{col}** tem sempre o mesmo valor: não ajuda o modelo em nada."))
        elif nun == n and (s.dtype == object or str(col).lower() in ("id", "index", "codigo", "código") or str(col).lower().endswith("_id")):
            out.append(("aviso", f"A coluna **{col}** parece um identificador (todos os valores são diferentes). Não a uses para prever."))
        elif s.dtype == object and nun > 20:
            out.append(("aviso", f"A coluna **{col}** tem {nun} categorias diferentes. Demasiadas categorias dificultam a aprendizagem; considera agrupá-las."))

    num = df.select_dtypes(include=[np.number])
    extremos = []
    for col in num.columns:
        s = num[col].dropna()
        if len(s) > 10 and s.std() > 0:
            z = ((s - s.mean()) / s.std()).abs()
            k = int((z > 4).sum())
            if k:
                extremos.append(f"**{col}** ({k})")
    if extremos:
        out.append(("aviso", f"Valores muito fora do normal (outliers) em: {', '.join(extremos[:4])}. Podem ser erros de escrita ou casos raros: investiga."))

    alvo = df.columns[-1]
    s = df[alvo]
    if s.dtype == object or s.nunique() <= 10:
        vc = s.value_counts(normalize=True)
        if len(vc) >= 2:
            out.append(("info", f"Última coluna (**{alvo}**) parece ser o alvo, com {len(vc)} classes. A mais comum tem {vc.iloc[0]:.0%} dos casos."))
            if vc.iloc[-1] < 0.10:
                out.append(("aviso", f"Classes desequilibradas: **{vc.index[-1]}** tem só {vc.iloc[-1]:.0%}. A accuracy engana; vigia Recall e F1."))
    else:
        out.append(("info", f"Última coluna (**{alvo}**) é numérica: parece um problema de **regressão** (prever um número)."))

    if num.shape[1] >= 2:
        corr = num.corr().abs().where(~np.eye(num.shape[1], dtype=bool))
        mx = corr.max().max()
        if pd.notna(mx) and mx > 0.95:
            a, b = corr.stack().idxmax()
            out.append(("aviso", f"As colunas **{a}** e **{b}** são quase iguais (correlação {mx:.2f}). Uma delas é redundante, ou uma delas contém a resposta (fuga de informação)."))
    return out


def mostrar_diagnostico(df: pd.DataFrame, expandido: bool = True):
    with st.expander("Diagnóstico automático dos teus dados", expanded=expandido):
        itens = diagnosticar(df)
        problemas = sum(1 for n, _ in itens if n in ("aviso", "erro"))
        st.caption("O sistema olhou para a tabela e diz-te o que merece atenção antes de treinar.")
        for nivel, txt in itens:
            {"ok": st.success, "aviso": st.warning, "erro": st.error, "info": st.info}[nivel](txt)
        if problemas == 0:
            st.markdown("Os dados parecem prontos a usar.")
