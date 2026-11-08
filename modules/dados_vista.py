"""Ver e descarregar os dados: dataset completo e divisão Treino / Teste.

Usado pelas páginas de dados (supervisionado, não supervisionado, redes neuronais, comparador), para o aluno ver o
dataset inteiro, descarregá-lo e saber exactamente com que linhas o modelo aprendeu e com que linhas foi avaliado.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

# Acima deste número de linhas a tabela Treino/Teste guarda só as linhas de Teste (as de Treino ficam só contadas)
LIMITE_TABELA_SPLIT = 100000
# Máximo de linhas desenhadas no ecrã de uma vez (o descarregar leva SEMPRE todas)
MAX_LINHAS_NO_ECRA = 100000


def _slug(txt: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(txt or "dados").strip().lower()).strip("_")
    return s[:40] or "dados"


def _csv(df: pd.DataFrame) -> bytes:
    """CSV com acentos certos no Excel (UTF-8 com BOM)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def ver_dataset(df: pd.DataFrame, chave: str, nome: str = "dataset", amostra: int = 8, expander: bool = False):
    """Mostra uma amostra, deixa ver o dataset completo e descarregá-lo em CSV.
    `expander=True` embrulha tudo numa caixa que se abre (usar quando a página ainda não está dentro de uma)."""
    if df is None or not hasattr(df, "shape"):
        return
    if expander:
        with st.expander("Ver e descarregar o dataset"):
            ver_dataset(df, chave, nome, amostra, expander=False)
        return
    n, m = df.shape
    ver_tudo = st.toggle("Ver o dataset completo", key=f"{chave}_ver_tudo")
    if ver_tudo:
        st.dataframe(df.head(MAX_LINHAS_NO_ECRA), width="stretch", height=420)
        if n > MAX_LINHAS_NO_ECRA:
            st.caption(f"O dataset tem {n} linhas × {m} colunas. No ecrã aparecem as primeiras {MAX_LINHAS_NO_ECRA}; "
                       "o ficheiro descarregado leva todas.")
        else:
            st.caption(f"A mostrar todas as {n} linhas × {m} colunas.")
    else:
        st.dataframe(df.head(amostra), width="stretch")
        st.caption(f"A mostrar as primeiras {min(amostra, n)} de {n} linhas × {m} colunas. "
                   "Liga «Ver o dataset completo» para ver todas.")
    st.download_button("Descarregar o dataset (CSV)", data=_csv(df), file_name=f"{_slug(nome)}.csv",
                       mime="text/csv", key=f"{chave}_dl_dataset")


def tabela_treino_teste(df_limpo: pd.DataFrame, idx_teste, alvo: str, previsto=None) -> pd.DataFrame:
    """Devolve as linhas do dataset (já sem nulos, com o texto original) com a coluna `Conjunto` (Treino/Teste).
    Nas linhas de Teste junta `Previsto` (o que o modelo respondeu) e, se o alvo for número, `Diferença`.
    Em datasets enormes (> LIMITE_TABELA_SPLIT linhas) devolve só as linhas de Teste e guarda em `attrs` quantas
    linhas de Treino ficaram de fora."""
    t = df_limpo.reset_index(drop=True).copy()
    t.insert(0, "Conjunto", "Treino")
    idx = np.asarray(idx_teste, dtype=int)
    t.loc[idx, "Conjunto"] = "Teste"
    if previsto is not None:
        t["Previsto"] = pd.Series([None] * len(t), dtype=object)
        t.loc[idx, "Previsto"] = list(np.asarray(previsto))
        if pd.api.types.is_numeric_dtype(t[alvo]):
            prev_num = pd.to_numeric(t["Previsto"], errors="coerce")
            t["Previsto"] = prev_num.round(3)
            t["Diferença"] = (prev_num - t[alvo]).round(3)
    if len(t) > LIMITE_TABELA_SPLIT:
        n_treino = int((t["Conjunto"] == "Treino").sum())
        t = t[t["Conjunto"] == "Teste"].copy()
        t.attrs["n_treino_fora"] = n_treino
    return t


def mostrar_treino_teste(tabela: pd.DataFrame, chave: str, nome: str = "dataset"):
    """Secção dos resultados: quantas linhas treinaram, quantas testaram, e a tabela para filtrar e descarregar."""
    if tabela is None or "Conjunto" not in tabela:
        return
    fora = int(tabela.attrs.get("n_treino_fora", 0))
    n_te = int((tabela["Conjunto"] == "Teste").sum())
    n_tr = int((tabela["Conjunto"] == "Treino").sum()) + fora
    st.markdown("---")
    st.subheader("Dados usados no treino e no teste")
    st.write(f"O modelo aprendeu com **{n_tr} linhas** (Treino) e foi avaliado em **{n_te} linhas** (Teste) "
             "que nunca tinha visto. Na tabela, a coluna **Conjunto** diz a que grupo pertence cada linha.")
    if fora:
        st.info(f"O dataset é muito grande. Para não pesar no navegador, a tabela só tem as {n_te} linhas de Teste; "
                f"as outras {fora} linhas de Treino não aparecem aqui.")
        vista = tabela
    else:
        escolha = st.radio("Mostrar", ["Todas as linhas", "Só Teste", "Só Treino"], horizontal=True,
                           key=f"{chave}_filtro_split")
        vista = tabela
        if escolha == "Só Teste":
            vista = tabela[tabela["Conjunto"] == "Teste"]
        elif escolha == "Só Treino":
            vista = tabela[tabela["Conjunto"] == "Treino"]
    st.dataframe(vista.head(MAX_LINHAS_NO_ECRA), width="stretch", height=380)
    st.caption(f"{len(vista)} linhas. As colunas «Previsto» e «Diferença» só têm valores nas linhas de Teste.")
    st.download_button("Descarregar a tabela Treino/Teste (CSV)", data=_csv(tabela),
                       file_name=f"{_slug(nome)}_treino_teste.csv", mime="text/csv", key=f"{chave}_dl_split")
