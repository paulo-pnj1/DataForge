"""
DataForge EDU - Visão Computacional
CNN personalizada + Transfer Learning (ResNet18, MobileNetV2)
PyTorch - sem GPU necessária (CPU mode)
"""

import streamlit as st
import numpy as np
import io
import os
import zipfile
import tempfile
from pathlib import Path

from modules.utils import (
    inject_css, page_header, section_title, teoria_box,
    aviso_box, sucesso_box, erro_box, info_box, progresso_bar,
    C_ACCENT, C_GREEN, C_AMBER, C_RED, C_SURFACE, C_BORDER,
    C_TEXT, C_SURFACE2, PALETTE, add_pontos
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _DummyModule:
        """Placeholder usado quando o PyTorch não está instalado, para que as
        classes do módulo possam ser definidas sem rebentar com NameError."""
        Module = object

    class _DummyDataset:
        pass

    nn = _DummyModule()
    Dataset = _DummyDataset


# ── DATASET CUSTOMIZADO ──────────────────────────────
class ImageDatasetFromArrays(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── CNN SIMPLES ───────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, n_classes, n_conv=2, n_filters=32, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 3
        for i in range(n_conv):
            out_ch = n_filters * (2 ** i)
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_ch * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _get_transfer_model(nome, n_classes):
    if nome == "ResNet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif nome == "MobileNetV2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif nome == "EfficientNet-B0":
        try:
            m = models.efficientnet_b0(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        except AttributeError:
            m = models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def _loss_acc_plot(train_losses, train_accs, val_accs):
    fig = go.Figure()
    epochs = list(range(1, len(train_losses) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines",
                             name="Loss treino", line=dict(color=C_RED, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=train_accs, mode="lines",
                             name="Acc treino", line=dict(color=C_ACCENT, width=2)))
    if val_accs:
        fig.add_trace(go.Scatter(x=epochs, y=val_accs, mode="lines",
                                 name="Acc validação", line=dict(color=C_GREEN, width=2, dash="dash")))
    fig.update_layout(
        title=dict(text="Curvas de Aprendizagem", font=dict(color="#FFFFFF", size=15)),
        xaxis=dict(title="Época", color="#FFFFFF", gridcolor=C_BORDER),
        yaxis=dict(color="#FFFFFF", gridcolor=C_BORDER),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF"), legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig


def _confusion_plot(cm, classes):
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=cm, x=classes, y=classes,
        colorscale=[[0, C_SURFACE], [1, C_ACCENT]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(color="#FFFFFF", size=13),
        showscale=False
    ))
    fig.update_layout(
        title=dict(text="Matriz de Confusão", font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(title="Previsto", color="#FFFFFF"),
        yaxis=dict(title="Real", color="#FFFFFF", autorange="reversed"),
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color="#FFFFFF")
    )
    return fig


def _load_images_from_zip(zip_file) -> dict:
    """Carrega imagens de um ZIP com estrutura classe/imagem.jpg"""
    classes = {}
    with zipfile.ZipFile(zip_file) as z:
        for name in z.namelist():
            parts = Path(name).parts
            if len(parts) < 2:
                continue
            classe = parts[-2]
            if name.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                with z.open(name) as f:
                    try:
                        img = Image.open(f).convert("RGB").resize((64, 64))
                        if classe not in classes:
                            classes[classe] = []
                        classes[classe].append(np.array(img))
                    except Exception:
                        pass
    return classes


def render_visao(username: str):
    inject_css()
    page_header("Visão Computacional",
                "Classifica imagens com CNN e Transfer Learning", "")

    if not HAS_TORCH:
        aviso_box("PyTorch não está instalado. Adiciona <code>torch</code> e <code>torchvision</code> ao requirements.txt")
        return

    teoria_box("Como funcionam as CNNs?",
        "Uma <strong>Rede Neural Convolucional (CNN)</strong> aprende a detectar padrões visuais em camadas: "
        "as primeiras camadas detectam bordas e cores, as do meio detectam formas, "
        "as últimas reconhecem objectos completos. "
        "<strong>Transfer Learning</strong> reutiliza uma rede já treinada em milhões de imagens "
        "e adapta-a ao teu problema com muito menos dados.")

    tab_treino, tab_inferencia = st.tabs(["  Treinar Modelo  ", "  Classificar Imagem  "])

    with tab_treino:
        col_cfg, col_res = st.columns([1, 2])

        with col_cfg:
            section_title("Dataset")
            st.markdown(f"""<div style="background:{C_SURFACE2};border:2px solid {C_BORDER};border-radius:10px; padding:1rem;margin-bottom:1rem;font-size:14px;color:#D0D8F0;font-weight:600;">Estrutura do ZIP esperada:<br><code>dataset.zip/<br>&nbsp;&nbsp;classe_A/img1.jpg<br>&nbsp;&nbsp;classe_A/img2.jpg<br>&nbsp;&nbsp;classe_B/img1.jpg</code></div>""", unsafe_allow_html=True)

            zip_file = st.file_uploader("ZIP com imagens por classe", type=["zip"], key="vis_zip")

            section_title("Arquitectura")
            arch = st.selectbox("Modelo", ["CNN Simples", "ResNet18", "MobileNetV2", "EfficientNet-B0"], key="vis_arch")

            if arch == "CNN Simples":
                n_conv    = st.slider("Camadas convolucionais", 1, 4, 2, key="vis_nconv")
                n_filters = st.select_slider("Filtros iniciais", [16, 32, 64, 128], value=32, key="vis_filt")
                dropout   = st.slider("Dropout", 0.0, 0.5, 0.3, 0.1, key="vis_drop")
            else:
                n_conv, n_filters, dropout = 2, 32, 0.3
                info_box(f"{arch} - Transfer Learning. Arquitectura pré-definida, fine-tuning nas últimas camadas.")

            section_title("Treino")
            epochs    = st.slider("Épocas", 3, 30, 10, key="vis_ep")
            lr        = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001, key="vis_lr")
            batch_sz  = st.select_slider("Batch Size", [8, 16, 32], value=16, key="vis_bs")
            val_split = st.slider("% Validação", 10, 30, 20, key="vis_vs") / 100

            if st.button("Treinar CNN", width='stretch', key="vis_run"):
                if zip_file is None:
                    aviso_box("Carrega um ZIP com imagens primeiro.")
                else:
                    with st.spinner("A carregar imagens..."):
                        classes_data = _load_images_from_zip(zip_file)
                    if len(classes_data) < 2:
                        aviso_box("O ZIP precisa ter pelo menos 2 pastas (classes).")
                    else:
                        _train_cnn(classes_data, arch, n_conv, n_filters, dropout,
                                   epochs, lr, batch_sz, val_split, username)

        with col_res:
            res = st.session_state.get("vis_result", {})
            if res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy Teste", f"{res['acc']:.1%}")
                c2.metric("Classes",        res['n_classes'])
                c3.metric("Épocas",         res['epochs_run'])

                st.plotly_chart(
                    _loss_acc_plot(res["train_losses"], res["train_accs"], res["val_accs"]),
                    width='stretch'
                )
                if res.get("cm") is not None and res['n_classes'] <= 10:
                    st.plotly_chart(
                        _confusion_plot(res["cm"], res["class_names"]),
                        width='stretch'
                    )

                buf = io.BytesIO()
                torch.save(res["model"].state_dict(), buf)
                st.download_button("Descarregar Modelo (.pt)", buf.getvalue(),
                                   "cnn_model.pt", width='stretch')

                teoria_box("Interpretar as curvas",
                    "Se a <strong>loss desce</strong> e a <strong>accuracy sobe</strong>, o modelo está a aprender. "
                    "Se a accuracy de validação fica muito abaixo da de treino, há <strong>overfitting</strong> "
                    "- experimenta mais dropout ou menos épocas.")
            else:
                st.markdown(f"""<div style="text-align:center;padding:5rem;color:#7A8BA8; border:2px dashed {C_BORDER};border-radius:16px;margin-top:1rem;"><div style="font-size:48px;margin-bottom:1rem;">&#128247;</div><div style="font-size:16px;font-weight:700;color:#FFFFFF;">Carrega um ZIP e clica em <strong>Treinar CNN</strong></div></div>""", unsafe_allow_html=True)

    with tab_inferencia:
        section_title("Classificar nova imagem")
        res = st.session_state.get("vis_result", {})
        if not res:
            aviso_box("Treina um modelo primeiro na aba 'Treinar Modelo'.")
        else:
            img_file = st.file_uploader("Imagem para classificar", type=["jpg","jpeg","png"], key="vis_inf")
            if img_file:
                img = Image.open(img_file).convert("RGB")
                st.image(img, caption="Imagem carregada", width=200)
                if st.button("Classificar", key="vis_clf", width='stretch'):
                    transform = transforms.Compose([
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                    ])
                    tensor = transform(img).unsqueeze(0)
                    model  = res["model"]
                    model.eval()
                    with torch.no_grad():
                        out   = model(tensor)
                        probs = torch.softmax(out, dim=1)[0].numpy()
                        pred  = int(np.argmax(probs))
                    class_names = res["class_names"]
                    sucesso_box(f"Classe prevista: <strong>{class_names[pred]}</strong> ({probs[pred]:.1%} confiança)")

                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        x=class_names, y=probs.tolist(),
                        marker_color=[C_GREEN if i == pred else C_ACCENT for i in range(len(class_names))],
                        text=[f"{p:.1%}" for p in probs],
                        textposition="auto", textfont=dict(color="#FFFFFF", size=13)
                    ))
                    fig.update_layout(
                        title=dict(text="Probabilidades por Classe", font=dict(color="#FFFFFF")),
                        xaxis=dict(color="#FFFFFF"), yaxis=dict(color="#FFFFFF"),
                        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
                        font=dict(color="#FFFFFF")
                    )
                    st.plotly_chart(fig, width='stretch')


def _train_cnn(classes_data, arch, n_conv, n_filters, dropout,
               epochs, lr, batch_sz, val_split, username):
    try:
        class_names = sorted(classes_data.keys())
        n_classes   = len(class_names)
        label_map   = {c: i for i, c in enumerate(class_names)}

        all_imgs, all_labels = [], []
        for cls, imgs in classes_data.items():
            for img in imgs:
                all_imgs.append(img)
                all_labels.append(label_map[cls])

        all_imgs   = np.array(all_imgs, dtype=np.float32) / 255.0
        all_labels = np.array(all_labels, dtype=np.int64)

        n_total = len(all_imgs)
        n_val   = max(1, int(n_total * val_split))
        idx     = np.random.permutation(n_total)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        class NpDataset(torch.utils.data.Dataset):
            def __init__(self, imgs, labels, transform):
                self.imgs = imgs; self.labels = labels; self.transform = transform
            def __len__(self): return len(self.imgs)
            def __getitem__(self, i):
                img = Image.fromarray((self.imgs[i]*255).astype(np.uint8))
                return self.transform(img), self.labels[i]

        train_ds = NpDataset(all_imgs[train_idx], all_labels[train_idx], transform)
        val_ds   = NpDataset(all_imgs[val_idx],   all_labels[val_idx],   transform)
        train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_sz)

        if arch == "CNN Simples":
            model = SimpleCNN(n_classes, n_conv, n_filters, dropout)
        else:
            model = _get_transfer_model(arch, n_classes)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses, train_accs, val_accs = [], [], []
        progress_bar = st.progress(0)
        status_txt   = st.empty()

        model.train()
        for epoch in range(epochs):
            running_loss, correct, total = 0.0, 0, 0
            for imgs_b, labels_b in train_dl:
                optimizer.zero_grad()
                out  = model(imgs_b)
                loss = criterion(out, labels_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                preds  = out.argmax(dim=1)
                correct += (preds == labels_b).sum().item()
                total   += labels_b.size(0)

            epoch_loss = running_loss / len(train_dl)
            epoch_acc  = correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for imgs_b, labels_b in val_dl:
                    out = model(imgs_b)
                    preds = out.argmax(dim=1)
                    val_correct += (preds == labels_b).sum().item()
                    val_total   += labels_b.size(0)
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_accs.append(val_acc)
            model.train()

            progress_bar.progress((epoch + 1) / epochs)
            status_txt.markdown(
                f'<div style="color:#FFFFFF;font-weight:700;">Época {epoch+1}/{epochs} - '
                f'Loss: {epoch_loss:.4f} - Acc treino: {epoch_acc:.1%} - Acc val: {val_acc:.1%}</div>',
                unsafe_allow_html=True
            )

        # Matriz de confusão
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs_b, labels_b in val_dl:
                out = model(imgs_b)
                all_preds.extend(out.argmax(dim=1).numpy())
                all_true.extend(labels_b.numpy())

        from sklearn.metrics import confusion_matrix, accuracy_score
        cm  = confusion_matrix(all_true, all_preds)
        acc = accuracy_score(all_true, all_preds)

        st.session_state.vis_result = {
            "model": model, "class_names": class_names, "n_classes": n_classes,
            "train_losses": train_losses, "train_accs": train_accs, "val_accs": val_accs,
            "cm": cm, "acc": acc, "epochs_run": epochs
        }
        add_pontos(username, 25, f"CNN {arch} treinada")
        sucesso_box(f"Treino concluído! Accuracy validação: {val_accs[-1]:.1%}")
    except Exception as e:
        erro_box(f"Erro durante o treino: {e}")