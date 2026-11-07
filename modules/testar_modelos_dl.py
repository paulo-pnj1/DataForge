"""Testar com dados novos os modelos de texto, imagens, geradores e séries com rede neuronal (guardados com `modelos_dl.py`)."""
import pandas as pd
import streamlit as st

from modules.ml_engine import pacote_generico as pg
from modules.ml_engine.data_loader import ErroDados, carregar_ficheiro


def testar_generico(modelo, meta: dict, p: str, mid: str):
    _modo_generico(modelo, meta, mid, p)


def _modo_generico(modelo, meta, mid, pref):
    """Texto, imagens, geradores, séries (rede neuronal), redução e reforço."""
    tipo = meta["tipo"]
    k = f"{pref}_{mid}"
    try:
        if tipo in ("texto", "texto_lstm"):
            texto = st.text_area("Escreve um texto para classificar", key=f"{k}_txt")
            if st.button("Classificar", type="primary", key=f"{k}_go"):
                classe, probas = modelo.prever_texto(texto)
                st.metric("Classe prevista", classe)
                if probas:
                    tab = pd.DataFrame({"Classe": list(probas), "Probabilidade": list(probas.values())})
                    st.bar_chart(tab.set_index("Classe"), height=180)
        elif tipo == "imagem":
            f = st.file_uploader("Imagem para classificar", type=["jpg", "jpeg", "png"], key=f"{k}_img")
            if f is not None:
                from PIL import Image
                img = Image.open(f)
                st.image(img, width=220)
                classe, probas = modelo.prever_imagem(img)
                st.metric("Classe prevista", classe)
                tab = pd.DataFrame({"Classe": list(probas), "Probabilidade": list(probas.values())})
                st.bar_chart(tab.set_index("Classe"), height=180)
        elif tipo == "generativo" and meta.get("subtipo") in ("vae", "gan"):
            n = int(st.slider("Quantas amostras novas gerar", 10, 2000, 200, key=f"{k}_n"))
            if st.button("Gerar", type="primary", key=f"{k}_go"):
                dados = pd.DataFrame(modelo.gerar(n), columns=None)
                dados.columns = [f"x{i + 1}" for i in range(dados.shape[1])]
                if dados.shape[1] >= 2:
                    st.scatter_chart(dados, x="x1", y="x2")
                st.dataframe(dados.head(200), width="stretch", hide_index=True)
                st.download_button("Descarregar amostras (CSV)", dados.to_csv(index=False).encode("utf-8"), "amostras.csv",
                                   "text/csv", key=f"{k}_dl")
        elif tipo == "generativo":
            f = st.file_uploader("Ficheiro com os casos (CSV ou Excel, só colunas numéricas)", type=["csv", "xlsx", "xlsm"], key=f"{k}_f")
            if f is not None:
                novo = carregar_ficheiro(f.name, f.getvalue())
                out = modelo.reconstruir(novo)
                st.caption("Erro alto = caso que o autoencoder reconstrói mal, ou seja, diferente do que viu no treino.")
                st.dataframe(out.head(500), width="stretch")
                st.download_button("Descarregar (CSV)", out.to_csv(index=False).encode("utf-8"), "reconstrucao.csv", "text/csv",
                                   key=f"{k}_dl")
        elif tipo == "serie_dl":
            passos = int(st.number_input("Passos a prever", 1, 365, 12, key=f"{k}_h"))
            prev = modelo.prever_serie(passos)
            tab = pd.DataFrame({"passo": range(1, passos + 1), "previsao": prev})
            st.line_chart(tab.set_index("passo"))
            st.dataframe(tab, width="stretch", hide_index=True)
            st.download_button("Descarregar (CSV)", tab.to_csv(index=False).encode("utf-8"), "previsoes.csv", "text/csv", key=f"{k}_dl")
    except (pg.ErroPacote, ErroDados) as e:
        st.error(str(e))
    except ImportError:
        st.error("Falta uma biblioteca para usar este modelo neste ambiente (por exemplo o PyTorch).")
    except Exception:
        st.error("O modelo não conseguiu processar estes dados. Verifica-os ou treina o modelo outra vez.")
