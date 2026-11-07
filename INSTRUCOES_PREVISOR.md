# ATUALIZAÇÃO: «Testar Modelo» (substitui o menu «Previsor de Alunos»)

- O menu agora chama-se **Testar Modelo** (`modules/testar_modelo.py`, chave `testar_modelo`) e prevê com **qualquer modelo guardado**:
  classificação, regressão, clustering, anomalias e séries. Testa um caso, uma tabela editável ou um ficheiro CSV/Excel.
- O KNN dos alunos continua na lista como «Exemplo do projecto».
- Em **Classificação, Regressão, MLP, Comparador, O Meu Primeiro Modelo, Clustering e Anomalias** há agora o bloco
  «Guardar este modelo» (`modules/guardar_ui.py`). Guardam-se também o scaler e os codificadores (`modules/ml_engine/pacote_modelo.py`).
- NLP, Visão, Generativos e Reforço ainda não se guardam (não usam tabelas).

---

# Previsor de Alunos (KNN) para o DataForge EDU

## O que vem aqui
- `modules/previsor_alunos.py`: lógica (carregar, validar, prever) e a página `render_previsor_alunos(username)`.
- `artifacts/knn_model.pkl` e `artifacts/knn_model.pkl.sha256`: o modelo e a sua impressão digital.
- `tests/test_previsor_alunos.py`: 7 testes (`pytest -q tests/test_previsor_alunos.py`).
- `previsor_demo.py`: demo autónoma, não precisa do resto do projeto.

## Ver já a funcionar (sem mexer no projeto)
    pip install streamlit scikit-learn joblib pandas
    streamlit run previsor_demo.py

## Pôr dentro do DataForge (3 passos)
1. Copia as pastas `modules/`, `artifacts/` e `tests/` para a raiz do projeto (juntam-se às que já existem).
2. Em `app.py`, no roteador, por baixo do ramo `pagina == "laboratorio"`, junta:

        elif pagina == "previsor_alunos":
            from modules.previsor_alunos import render_previsor_alunos
            render_previsor_alunos(username)

3. Em `modules/navegacao.py`, acrescenta a chave `"previsor_alunos"` ao menu do painel do aluno (no mesmo formato
   das outras entradas, por exemplo ao lado do Laboratório de ML) e à lista de `paginas_permitidas("aluno")`.
   Sem isto o menu não mostra a página e o roteador manda para o Início.

## Mudar o significado do "sim"
No topo de `modules/previsor_alunos.py` estão `TITULO_RESULTADO`, `TEXTO_SIM` e `TEXTO_NAO`.
Se o "sim" for, por exemplo, «Aprovado», muda ali e pronto.

## Se trocares o modelo
Gera outra vez o hash: `sha256sum artifacts/knn_model.pkl | awk '{print $1}' > artifacts/knn_model.pkl.sha256`
(sem o ficheiro .sha256 certo, a página recusa abrir o modelo, de propósito).
