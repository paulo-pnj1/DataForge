"""DataForge EDU - Segmentação de texto para tradução.

Partilhado pelo extractor (que recolhe as frases do código) e pelo runtime
(que traduz o que vai ser mostrado), para as chaves coincidirem sempre.
"""
import re

_SPLIT = re.compile(r"(<style[\s\S]*?</style>|<script[\s\S]*?</script>|<[^>]*>|\n|\|)", re.IGNORECASE)
_PREFIX = re.compile(r"^(?:[\s#>*_`\-+]|&nbsp;|&middot;|&bull;)*(?:\d+[.)]\s+)?")
_SUFFIX = re.compile(r"(?:[\s*_`]|&nbsp;)*$")
_HAS_LETTER = re.compile(r"[A-Za-zÀ-ÿ]")


def split_pieces(text: str):
    """Parte o texto em (é_texto, pedaço): tags HTML, quebras de linha e '|' ficam intactos."""
    out = []
    for i, p in enumerate(_SPLIT.split(text)):
        if p == "":
            continue
        out.append((i % 2 == 0, p))
    return out


def split_core(piece: str):
    """Devolve (prefixo, núcleo, sufixo) de um pedaço de texto."""
    m = _PREFIX.match(piece)
    pre = m.group(0)
    rest = piece[len(pre):]
    s = _SUFFIX.search(rest)
    suf = s.group(0)
    core = rest[: len(rest) - len(suf)] if suf else rest
    return pre, core, suf


def cores(text: str):
    """Núcleos traduzíveis de uma string (um por pedaço de texto)."""
    res = []
    for is_text, p in split_pieces(text):
        if not is_text:
            continue
        _, core, _ = split_core(p)
        if core and _HAS_LETTER.search(core):
            res.append(core)
    return res
