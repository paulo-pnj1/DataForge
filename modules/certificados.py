"""
DataForge EDU - Certificados em PDF com QR code e página pública de verificação.
"""
import datetime
import io
import secrets

import streamlit as st

from modules import db
from modules.settings import app_url
from modules.utils import page_header, get_user_progress, C_SURFACE, C_BORDER, C_GREEN, C_TEXT, C_TEXT_SEC, C_TEXT_MUTE


def _criterios_fundamentos(progress: dict):
    from modules.escola import LICOES
    feitas = len(progress.get("licoes", []))
    pend = []
    if feitas < len(LICOES):
        pend.append(f"Concluir todas as lições da Escola de ML ({feitas}/{len(LICOES)})")
    if "primeiro_modelo" not in progress.get("modulos", {}):
        pend.append("Treinar o teu primeiro modelo")
    return not pend, pend


def _criterios_projeto(progress: dict):
    marcos = progress.get("marcos", {})
    pend = []
    if "projeto" not in marcos:
        pend.append("Terminar o Projecto Final")
    if "comparador" not in marcos:
        pend.append("Fazer pelo menos uma comparação no Comparador de Modelos")
    return not pend, pend


TIPOS = {
    "fundamentos": {"titulo": "Fundamentos de Machine Learning",
                    "texto": "concluiu todas as lições da Escola de ML do DataForge EDU e treinou o seu primeiro modelo.",
                    "criterio": _criterios_fundamentos},
    "projeto": {"titulo": "Projecto Aplicado de Machine Learning",
                "texto": "concluiu um projecto completo de Machine Learning, desde os dados até à comparação e avaliação de modelos.",
                "criterio": _criterios_projeto},
}


def _l1(t) -> str:
    return str(t).encode("latin-1", "replace").decode("latin-1")


def emitir(username: str, nome: str, tipo: str) -> dict:
    """Devolve o certificado existente ou cria um novo."""
    for c in db.list_certs(username):
        if c.get("tipo") == tipo:
            return c
    cert = {"codigo": "DF-" + secrets.token_hex(4).upper(), "username": username, "nome": nome, "tipo": tipo,
            "emitido_em": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    db.save_cert(cert)
    return cert


def url_verificacao(codigo: str) -> str:
    base = app_url()
    return f"{base}/?verificar={codigo}" if base else f"?verificar={codigo}"


def gerar_pdf(cert: dict) -> bytes:
    from fpdf import FPDF
    import qrcode
    info = TIPOS[cert["tipo"]]
    data = str(cert.get("emitido_em", ""))[:10]
    try:
        data = datetime.date.fromisoformat(data).strftime("%d/%m/%Y")
    except Exception:
        pass

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(False)
    pdf.add_page()
    pdf.set_draw_color(30, 90, 170)
    pdf.set_line_width(1.6)
    pdf.rect(8, 8, 281, 194)
    pdf.set_line_width(0.4)
    pdf.rect(12, 12, 273, 186)

    pdf.set_text_color(30, 90, 170)
    pdf.set_font("Helvetica", "B", 34)
    pdf.set_xy(0, 28)
    pdf.cell(297, 16, "CERTIFICADO", align="C")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(90, 100, 120)
    pdf.set_xy(0, 46)
    pdf.cell(297, 8, "DataForge EDU - Plataforma de Machine Learning", align="C")

    pdf.set_text_color(40, 40, 50)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_xy(0, 70)
    pdf.cell(297, 8, "Certifica-se que", align="C")
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(15, 25, 60)
    pdf.set_xy(0, 82)
    pdf.cell(297, 16, _l1(cert["nome"]), align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(40, 40, 50)
    pdf.set_xy(30, 106)
    pdf.multi_cell(237, 8, _l1(info["texto"]), align="C")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 90, 170)
    pdf.set_xy(0, 130)
    pdf.cell(297, 10, _l1(info["titulo"]), align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 100, 120)
    pdf.set_xy(20, 168)
    pdf.cell(120, 6, f"Emitido em {data}")
    pdf.set_xy(20, 175)
    pdf.cell(120, 6, f"Codigo de verificacao: {cert['codigo']}")
    pdf.set_xy(20, 182)
    pdf.cell(150, 6, _l1("Verifica a autenticidade em: " + url_verificacao(cert["codigo"])[:70]))

    buf = io.BytesIO()
    qrcode.make(url_verificacao(cert["codigo"])).save(buf, format="PNG")
    buf.seek(0)
    pdf.image(buf, x=245, y=150, w=36)
    return bytes(pdf.output())


def render_certificados(username: str, nome: str):
    page_header("Certificados", "Ganha um certificado em PDF com QR code verificável")
    progress = get_user_progress(username)
    st.caption("Os certificados guardam o nome com que te registaste. Confirma-o em Perfil antes de emitir.")
    for tipo, info in TIPOS.items():
        ok, pend = info["criterio"](progress)
        st.markdown(f"### {info['titulo']}")
        if not ok:
            st.info("Ainda falta:")
            for p in pend:
                st.markdown(f"- {p}")
            continue
        cert = emitir(username, nome, tipo)
        st.success("Parabéns, cumpriste todos os requisitos!")
        st.download_button("Descarregar certificado (PDF)", gerar_pdf(cert), file_name=f"certificado_{tipo}_{username}.pdf",
                           mime="application/pdf", key=f"cert_{tipo}")
        st.caption(f"Código: {cert['codigo']} · qualquer pessoa pode confirmar em {url_verificacao(cert['codigo'])}")


def render_verificacao(codigo: str):
    """Página pública: ?verificar=CODIGO (não precisa de login)."""
    st.markdown("## Verificação de certificado")
    codigo = (codigo or "").strip().upper()
    try:
        cert = db.get_cert(codigo)
    except Exception:
        st.error("Não foi possível consultar a base de dados agora. Tenta mais tarde.")
        return
    if not cert:
        st.error(f"Não existe nenhum certificado com o código **{codigo}**.")
        return
    info = TIPOS.get(cert["tipo"], {"titulo": cert["tipo"]})
    st.success("Certificado **válido**.")
    st.markdown(f"**Nome:** {cert['nome']}  \n**Certificado:** {info['titulo']}  \n"
                f"**Emitido em:** {str(cert.get('emitido_em', ''))[:10]}  \n**Código:** {cert['codigo']}")
