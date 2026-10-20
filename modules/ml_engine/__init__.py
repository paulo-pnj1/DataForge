"""DataForge - Motor de Machine Learning (sem interface).

Tudo o que está aqui funciona sem Streamlit, por isso é testável e reutilizável:

    data_loader        ler CSV/Excel com segurança
    data_validation    validar dados e variável-alvo
    preprocessing      separar treino/teste e construir os pipelines (sem fuga de informação)
    model_registry     catálogo de algoritmos (regressão, classificação, séries)
    model_evaluation   métricas
    model_training     treinar UM modelo (com tratamento de erros)
    model_comparison   comparar vários modelos e recomendar um
    auto_ml            selecção automática em duas fases
    forecasting        séries temporais (validação cronológica)
    model_persistence  guardar/carregar modelos por utilizador, com assinatura
    prediction_service previsões com novos dados

A interface Streamlit está em `modules/laboratorio.py`.
"""
