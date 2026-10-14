"""
DataForge EDU - NLP (Processamento de Linguagem Natural)
BoW, TF-IDF, Word2Vec (sklearn), LSTM (PyTorch)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_SURFACE2, PALETTE, add_pontos
)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _DummyModule:
        Module = object

    nn = _DummyModule()

try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# Datasets de texto embutidos
DATASETS_NLP = {
    "Sentimento de Reviews (PT)": {
        "texts": [
            "Este produto é excelente! Adorei completamente.",
            "Muito bom, recomendo a todos os amigos.",
            "Qualidade superior, vale cada centavo.",
            "Fantástico, superou as minhas expectativas.",
            "Produto de alta qualidade, muito satisfeito.",
            "Bom produto, entrega rápida e sem problemas.",
            "Razoável, poderia ser melhor pelo preço.",
            "Produto péssimo, quebrou em dois dias.",
            "Muito dececionante, não recomendo de forma alguma.",
            "Horrível, perdi o meu dinheiro com isto.",
            "Qualidade muito fraca, não vale o preço.",
            "Terrível experiência, nunca mais compro aqui.",
            "Entrega demorou demasiado e produto chegou danificado.",
            "Produto ok, mas esperava mais pela descrição.",
            "Aceitável para o uso básico que precisava.",
            "Cumpre o prometido, sem grandes surpresas.",
            "Muito bom custo-benefício para o que oferece.",
            "Excelente relação qualidade-preço, muito contente.",
            "Superou expectativas, produto de primeira qualidade.",
            "Adorei! Já comprei segunda unidade para oferta.",
        ],
        "labels": ["positivo","positivo","positivo","positivo","positivo",
                   "neutro","neutro","negativo","negativo","negativo",
                   "negativo","negativo","negativo","neutro","neutro",
                   "neutro","positivo","positivo","positivo","positivo"],
        "desc": "20 reviews de produto em português com 3 classes: positivo, neutro, negativo."
    },
    "Spam vs Ham (EN)": {
        "texts": [
            "Win a free iPhone now! Click here to claim your prize!",
            "URGENT: Your account has been suspended. Verify now!",
            "Congratulations! You've been selected for a cash reward.",
            "Buy cheap medication online, no prescription needed!",
            "You won 1 million dollars in our lottery draw!",
            "Hey, are you coming to the meeting tomorrow at 3pm?",
            "The report is attached. Let me know if you need changes.",
            "Can we reschedule our lunch for next Friday?",
            "Your package has been shipped and will arrive tomorrow.",
            "Thanks for your email, I'll get back to you soon.",
            "FREE GIFT! Limited time offer, act now!",
            "Click to win big prizes in our exclusive promotion!",
            "Hi, just checking if you received my last message.",
            "The project deadline has been extended to next week.",
            "Please review the attached document before our call.",
        ],
        "labels": ["spam","spam","spam","spam","spam",
                   "ham","ham","ham","ham","ham",
                   "spam","spam","ham","ham","ham"],
        "desc": "15 mensagens em inglês classificadas como spam ou ham (legítimo)."
    },
    "Tópicos de Notícias (PT)": {
        "texts": [
            "O banco central anunciou nova taxa de juro para controlar a inflação.",
            "As bolsas mundiais registaram queda acentuada esta semana.",
            "O governo aprovou novo orçamento para o próximo ano fiscal.",
            "Empresa de tecnologia anuncia despedimento de milhares de trabalhadores.",
            "Fusão entre duas grandes empresas cria novo gigante do mercado.",
            "Selecção nacional venceu jogo de qualificação para o campeonato.",
            "Atleta angolano bate recorde nacional nos 100 metros rasos.",
            "Clube contrata avançado estrela por valor recorde na história.",
            "Final do campeonato nacional marca este fim de semana em Luanda.",
            "Nova geração de inteligência artificial supera capacidades humanas.",
            "Investigadores descobrem tratamento promissor para doença rara.",
            "Satélite angolano enviará dados sobre alterações climáticas.",
            "Novo smartphone com bateria de 7 dias chega ao mercado africano.",
            "Chuvas intensas causam inundações em várias províncias do país.",
            "Temperatura global atinge novo máximo histórico em 2025.",
        ],
        "labels": ["economia","economia","economia","economia","economia",
                   "desporto","desporto","desporto","desporto",
                   "tecnologia","ciencia","tecnologia","tecnologia",
                   "clima","clima"],
        "desc": "15 títulos de notícias em português com 5 categorias temáticas."
    }
}


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.classifier(h[-1])


def _preprocess(texts):
    def clean(t):
        t = t.lower()
        t = re.sub(r'[^a-záéíóúàâêôãõü\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    return [clean(t) for t in texts]


def _wordcloud_bar(vectorizer, feature_names, tfidf_matrix, labels, label_names):
    """Gráfico de barras com palavras mais importantes por classe"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    unique_labels = sorted(set(labels))
    n = len(unique_labels)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[f"Classe: {label_names[l] if hasattr(label_names,'__getitem__') else str(l)}"
                                       for l in unique_labels])

    for col_idx, lbl in enumerate(unique_labels):
        mask = np.array(labels) == lbl
        if mask.sum() == 0:
            continue
        mean_tfidf = tfidf_matrix[mask].mean(axis=0).A1
        top_idx    = mean_tfidf.argsort()[-10:][::-1]
        words      = [feature_names[i] for i in top_idx]
        scores     = [mean_tfidf[i] for i in top_idx]

        fig.add_trace(
            go.Bar(x=scores, y=words, orientation='h',
                   marker_color=PALETTE[col_idx % len(PALETTE)],
                   name=str(lbl), showlegend=False),
            row=1, col=col_idx + 1
        )

    fig.update_layout(
        title=dict(text="Palavras mais discriminativas por classe (TF-IDF)", font=dict(color="#FFFFFF", size=14)),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), height=350
    )
    for i in range(1, n + 1):
        fig.update_xaxes(color="#FFFFFF", gridcolor=C_BORDER, row=1, col=i)
        fig.update_yaxes(color="#FFFFFF", row=1, col=i)
    return fig


def render_nlp(username: str):
    inject_css()
    page_header("NLP - Processamento de Linguagem Natural",
                "Classifica texto com BoW, TF-IDF e LSTM", "")

    teoria_box("Como os computadores entendem texto?",
        "O texto precisa de ser convertido em números antes de poder ser processado. "
        "<strong>Bag of Words (BoW)</strong>: conta quantas vezes cada palavra aparece. "
        "<strong>TF-IDF</strong>: pondera palavras raras mais importante que palavras comuns. "
        "<strong>LSTM</strong>: rede neural que processa o texto sequencialmente, "
        "capturando o contexto e a ordem das palavras.")

    tab_classif, tab_lstm = st.tabs(["  BoW / TF-IDF  ", "  LSTM Neural  "])

    # ════════════════════════════════════════════════
    # BoW / TF-IDF
    # ════════════════════════════════════════════════
    with tab_classif:
        col_cfg, col_res = st.columns([1, 2])

        with col_cfg:
            section_title("Dados")
            fonte = st.radio("Fonte", ["Dataset embutido", "Upload CSV"], horizontal=True, key="nlp_fonte")

            if fonte == "Dataset embutido":
                ds_nome = st.selectbox("Dataset", list(DATASETS_NLP.keys()), key="nlp_ds")
                ds = DATASETS_NLP[ds_nome]
                texts  = ds["texts"]
                labels_raw = ds["labels"]
                st.markdown(f'<div style="font-size:13px;color:#D0D8F0;font-weight:600;">{ds["desc"]}</div>',
                            unsafe_allow_html=True)
            else:
                f = st.file_uploader("CSV com colunas 'text' e 'label'", type=["csv"], key="nlp_up")
                if f:
                    df_nlp = pd.read_csv(f)
                    text_col  = st.selectbox("Coluna de texto", df_nlp.columns.tolist(), key="nlp_tcol")
                    label_col = st.selectbox("Coluna de classe", df_nlp.columns.tolist(), key="nlp_lcol")
                    texts      = df_nlp[text_col].astype(str).tolist()
                    labels_raw = df_nlp[label_col].astype(str).tolist()
                else:
                    texts, labels_raw = None, None

            if texts:
                section_title("Vectorização")
                vec_tipo = st.selectbox("Método", ["TF-IDF", "Bag of Words"], key="nlp_vec")
                max_feats = st.slider("Vocabulário máximo", 100, 5000, 1000, 100, key="nlp_maxf")
                ngram_max = st.select_slider("N-gramas", [1, 2, 3], value=2, key="nlp_ng")

                section_title("Classificador")
                clf_nome = st.selectbox("Algoritmo", ["Logistic Regression", "Naive Bayes", "SVM Linear"],
                                        key="nlp_clf")
                test_size = st.slider("% Teste", 10, 40, 20, key="nlp_ts") / 100

                if st.button("Treinar Classificador", width='stretch', key="nlp_run"):
                    _train_bow(texts, labels_raw, vec_tipo, max_feats, ngram_max,
                               clf_nome, test_size, username)

        with col_res:
            res = st.session_state.get("nlp_bow_res", {})
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy",  f"{res['acc']:.1%}")
                c2.metric("F1 Macro",  f"{res['f1']:.1%}")
                c3.metric("Classes",   res['n_classes'])

                st.plotly_chart(res["fig_words"], width='stretch')

                # Confusão
                import plotly.graph_objects as go
                cm = res["cm"]
                cn = res["class_names"]
                fig_cm = go.Figure(go.Heatmap(
                    z=cm, x=cn, y=cn,
                    colorscale=[[0, C_SURFACE],[1, C_ACCENT]],
                    text=[[str(v) for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont=dict(color="#FFFFFF", size=13),
                    showscale=False
                ))
                fig_cm.update_layout(
                    title=dict(text="Matriz de Confusão", font=dict(color="#FFFFFF", size=14)),
                    xaxis=dict(title="Previsto", color="#FFFFFF"),
                    yaxis=dict(title="Real", color="#FFFFFF", autorange="reversed"),
                    plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                    font=dict(color="#FFFFFF")
                )
                st.plotly_chart(fig_cm, width='stretch')

                # Previsão interactiva
                section_title("Testar com novo texto")
                novo_texto = st.text_area("Escreve um texto para classificar:", key="nlp_novo",
                                          placeholder="Escreve aqui...")
                if st.button("Classificar", key="nlp_clf_btn", width='stretch'):
                    if novo_texto.strip():
                        texto_clean = _preprocess([novo_texto])
                        X_novo = res["vectorizer"].transform(texto_clean)
                        pred   = res["model"].predict(X_novo)[0]
                        try:
                            proba = res["model"].predict_proba(X_novo)[0]
                            cn    = res["class_names"]
                            sucesso_box(f"Classe prevista: <strong>{cn[pred] if isinstance(pred, int) else pred}</strong>")
                            fig_p = go.Figure(go.Bar(
                                x=[str(c) for c in cn],
                                y=proba.tolist(),
                                marker_color=PALETTE[:len(cn)],
                                text=[f"{p:.1%}" for p in proba],
                                textposition="auto", textfont=dict(color="#FFFFFF")
                            ))
                            fig_p.update_layout(
                                plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                                font=dict(color="#FFFFFF"),
                                xaxis=dict(color="#FFFFFF"), yaxis=dict(color="#FFFFFF")
                            )
                            st.plotly_chart(fig_p, width='stretch')
                        except Exception:
                            pred_label = pred if not isinstance(pred, int) else res["class_names"][pred]
                            sucesso_box(f"Classe prevista: <strong>{pred_label}</strong>")
            else:
                st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#128172;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura e clica em <strong>Treinar Classificador</strong></div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # LSTM
    # ════════════════════════════════════════════════
    with tab_lstm:
        if not HAS_TORCH:
            aviso_box("PyTorch necessário para LSTM. Adiciona <code>torch</code> ao requirements.txt")
        else:
            col_l1, col_l2 = st.columns([1, 2])
            with col_l1:
                section_title("Dados para LSTM")
                ds_lstm = st.selectbox("Dataset", list(DATASETS_NLP.keys()), key="lstm_ds")
                ds_l    = DATASETS_NLP[ds_lstm]

                section_title("Arquitectura LSTM")
                embed_dim  = st.slider("Dimensão do embedding", 16, 128, 32, key="lstm_emb")
                hidden_dim = st.slider("Neurónios LSTM", 32, 256, 64, key="lstm_hid")
                n_layers   = st.slider("Camadas LSTM", 1, 3, 1, key="lstm_nl")
                dropout    = st.slider("Dropout", 0.0, 0.5, 0.3, 0.1, key="lstm_drop")
                epochs_l   = st.slider("Épocas", 5, 50, 20, key="lstm_ep")
                lr_l       = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001, key="lstm_lr")

                if st.button("Treinar LSTM", width='stretch', key="lstm_run"):
                    _train_lstm(ds_l["texts"], ds_l["labels"], embed_dim, hidden_dim,
                                n_layers, dropout, epochs_l, lr_l, username)

            with col_l2:
                res_l = st.session_state.get("nlp_lstm_res", {})
                if res_l:
                    c1, c2 = st.columns(2)
                    c1.metric("Accuracy Final", f"{res_l['acc']:.1%}")
                    c2.metric("Épocas", res_l["epochs"])
                    st.plotly_chart(res_l["fig_loss"], width='stretch')
                    teoria_box("LSTM vs TF-IDF",
                        "O <strong>LSTM</strong> captura a ordem das palavras e contexto, "
                        "sendo melhor em textos longos e complexos. "
                        "O <strong>TF-IDF</strong> é mais rápido e muitas vezes suficiente "
                        "para classificação de texto curto como reviews e emails.")
                else:
                    st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#129504;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Configura e clica em <strong>Treinar LSTM</strong></div></div>""", unsafe_allow_html=True)


def _train_bow(texts, labels_raw, vec_tipo, max_feats, ngram_max, clf_nome, test_size, username):
    try:
        texts_clean = _preprocess(texts)
        le = LabelEncoder()
        y  = le.fit_transform(labels_raw)
        class_names = list(le.classes_)

        if vec_tipo == "TF-IDF":
            vec = TfidfVectorizer(max_features=max_feats, ngram_range=(1, ngram_max))
        else:
            vec = CountVectorizer(max_features=max_feats, ngram_range=(1, ngram_max))

        X = vec.fit_transform(texts_clean)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if clf_nome == "Logistic Regression":
            clf = LogisticRegression(max_iter=500, random_state=42)
        elif clf_nome == "Naive Bayes":
            from sklearn.preprocessing import MaxAbsScaler
            X_train = MaxAbsScaler().fit_transform(X_train)
            X_test  = MaxAbsScaler().fit_transform(X_test)
            clf = MultinomialNB()
        else:
            clf = LinearSVC(max_iter=1000)

        with st.spinner("A treinar..."):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
            from sklearn.metrics import confusion_matrix
            cm  = confusion_matrix(y_test, y_pred)

        fig_words = _wordcloud_bar(vec, vec.get_feature_names_out(), X, y, class_names)

        st.session_state.nlp_bow_res = {
            "acc": acc, "f1": f1, "cm": cm,
            "class_names": class_names, "n_classes": len(class_names),
            "vectorizer": vec, "model": clf,
            "fig_words": fig_words
        }
        add_pontos(username, 12, f"NLP {clf_nome}")
        sucesso_box(f"Treino concluído! Accuracy: {acc:.1%} · F1: {f1:.1%}")
    except Exception as e:
        erro_box(f"Erro: {e}")


def _train_lstm(texts, labels_raw, embed_dim, hidden_dim, n_layers, dropout, epochs, lr, username):
    try:
        texts_clean = _preprocess(texts)
        le = LabelEncoder()
        y  = le.fit_transform(labels_raw)
        n_classes = len(le.classes_)

        # Tokenização simples
        vocab = {"<PAD>": 0}
        for t in texts_clean:
            for w in t.split():
                if w not in vocab:
                    vocab[w] = len(vocab)

        max_len = max(len(t.split()) for t in texts_clean)
        def encode(t):
            ids = [vocab.get(w, 0) for w in t.split()]
            ids = ids[:max_len] + [0] * (max_len - len(ids))
            return ids

        X_enc = torch.tensor([encode(t) for t in texts_clean], dtype=torch.long)
        y_enc = torch.tensor(y, dtype=torch.long)

        model = LSTMClassifier(len(vocab), embed_dim, hidden_dim, n_classes, n_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        losses, accs = [], []
        progress = st.progress(0)
        status   = st.empty()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out  = model(X_enc)
            loss = criterion(out, y_enc)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            acc   = (preds == y_enc).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)

            progress.progress((epoch + 1) / epochs)
            status.markdown(
                f'<div style="color:#FFFFFF;font-weight:700;">Época {epoch+1}/{epochs} - '
                f'Loss: {loss.item():.4f} - Acc: {acc:.1%}</div>',
                unsafe_allow_html=True
            )

        fig_loss = go.Figure()
        ep_list  = list(range(1, epochs + 1))
        fig_loss.add_trace(go.Scatter(x=ep_list, y=losses, mode="lines",
                                      name="Loss", line=dict(color=C_RED, width=2)))
        fig_loss.add_trace(go.Scatter(x=ep_list, y=accs, mode="lines",
                                      name="Accuracy", line=dict(color=C_GREEN, width=2)))
        fig_loss.update_layout(
            title=dict(text="LSTM - Curvas de Treino", font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(title="Época", color="#FFFFFF", gridcolor=C_BORDER),
            yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
            plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
            font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
        )

        st.session_state.nlp_lstm_res = {
            "acc": accs[-1], "epochs": epochs, "fig_loss": fig_loss
        }
        add_pontos(username, 20, "NLP LSTM")
        sucesso_box(f"LSTM treinado! Accuracy final: {accs[-1]:.1%}")
    except Exception as e:
        erro_box(f"Erro LSTM: {e}")