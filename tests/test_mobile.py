"""No telemóvel a barra lateral tem de ser uma gaveta (não pode ser forçada aberta) e o botão ☰ tem de ficar visível."""
import re

from modules import utils


def _css():
    return utils._css_global()


def _fora_de_media(css):
    """O CSS sem os blocos @media (para ver o que vale em TODOS os ecrãs)."""
    out, i = [], 0
    while i < len(css):
        m = css.find("@media", i)
        if m < 0:
            out.append(css[i:]); break
        out.append(css[i:m])
        j = css.index("{", m); depth = 0
        for k in range(j, len(css)):
            depth += css[k] == "{"; depth -= css[k] == "}"
            if depth == 0:
                i = k + 1; break
    return "".join(out)


def test_nenhuma_regra_global_forca_a_barra_aberta():
    base = _fora_de_media(_css())
    blocos = re.findall(r'\[data-testid="stSidebar"\]\s*\{[^}]*\}', base)
    assert blocos
    for b in blocos:
        assert "transform: none" not in b and "visibility: visible" not in b and "min-width: 290px" not in b


def test_a_barra_so_e_fixada_no_computador():
    css = _css()
    assert "@media (min-width: 769px)" in css
    i = css.index("@media (min-width: 769px)")
    assert "transform: none" in css[i:i + 600]


def test_botao_para_abrir_a_barra_fica_visivel():
    css = _css()
    assert 'stExpandSidebarButton"] *' in css and "visibility: visible" in css
