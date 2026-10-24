"""Lista as frases em português do código que ainda NÃO têm tradução em modules/locales/en.json.

Uso:  python tools/extrair_i18n.py            -> resumo + ficheiro tools/i18n_em_falta.json
      python tools/extrair_i18n.py --ver 60   -> mostra 60 frases em falta
"""
import ast
import json
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from modules.i18n_seg import split_pieces, split_core   # noqa: E402

_FUNCS_IA = {"_descrever_para_tutor", "_setup_para_tutor", "_loss_para_tutor", "_grupos_automaticos", "_registar_gen"}
_CHAMADAS_IA = {"registar_ecra", "acrescentar_ecra", "_registar_gen", "registar"}
_FICHEIROS_IA = {"i18n.py", "i18n_seg.py", "i18n_patch.py", "tutor_ia.py", "email_service.py"}
_LETRA = re.compile(r"[A-Za-zÀ-ÿ]")
_ACENTO = re.compile(r"[À-ÿ]")
# palavras muito comuns em português que quase não existem em inglês
_PT = re.compile(r"\b(o|a|os|as|um|uma|de|do|da|dos|das|em|no|na|nos|nas|para|por|com|sem|que|se|não|nao|é|e|ou|ao|aos|mais|como|já|"
                 r"teu|tua|teus|tuas|seu|sua|este|esta|estes|estas|isto|isso|quando|onde|todos|todas|cada|pode|podes|dados|modelo|treino|"
                 r"resultado|erro|valor|escolhe|escolher|treinar|gráfico|aluno|professor|turma|página|ficheiro|pergunta|resposta)\b", re.I)
_IDENTIFICADOR = re.compile(r"[\w./:#%\-\[\]{}()=,;@]+")     # uma só «palavra» sem espaços: chave, caminho, nome de coluna...


def _texto_de(no):
    """String de um nó (Constant ou f-string, com {} no lugar dos valores)."""
    if isinstance(no, ast.Constant) and isinstance(no.value, str):
        return no.value
    if isinstance(no, ast.JoinedStr):
        partes = []
        for v in no.values:
            if isinstance(v, ast.Constant):
                partes.append(str(v.value))
            else:
                partes.append("{}")
        return "".join(partes)
    return None


def _parece_pt(core: str) -> bool:
    if not _LETRA.search(core) or len(core) < 3:
        return False
    if _ACENTO.search(core):
        return True
    return len(_PT.findall(core)) >= 1 and len(core.split()) >= 2


def candidatos():
    """{núcleo: [ficheiro:linha, ...]} para todas as strings do código que parecem português."""
    res = {}
    for base, _, ficheiros in os.walk(RAIZ):
        if any(p in base for p in ("/tests", "/tools", "__pycache__", "/.git")):
            continue
        for f in ficheiros:
            if not f.endswith(".py"):
                continue
            caminho = os.path.join(base, f)
            try:
                arv = ast.parse(open(caminho, encoding="utf8").read())
            except Exception:
                continue
            docstrings = set()
            for n in ast.walk(arv):
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body \
                        and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                    docstrings.add(id(n.body[0].value))
            # texto que só vai para a IA (contexto do tutor, prompts): não é mostrado ao aluno
            para_ia = set()
            for n in ast.walk(arv):
                if isinstance(n, ast.FunctionDef) and n.name in _FUNCS_IA:
                    para_ia.update(id(x) for x in ast.walk(n))
                if isinstance(n, ast.Call):
                    nome = getattr(n.func, "id", getattr(n.func, "attr", ""))
                    if nome in _CHAMADAS_IA:
                        for a in n.args[1:] + [k.value for k in n.keywords]:    # o 1.º argumento é o título (esse aparece no ecrã)
                            para_ia.update(id(x) for x in ast.walk(a))
            if f in _FICHEIROS_IA:
                continue
            internos = set()      # partes constantes dentro de f-strings: já vêm no JoinedStr
            for n in ast.walk(arv):
                if isinstance(n, ast.JoinedStr):
                    for v in n.values:
                        internos.add(id(v))
            for n in ast.walk(arv):
                if id(n) in docstrings or id(n) in internos or id(n) in para_ia:
                    continue
                t = _texto_de(n)
                if not t:
                    continue
                rel = os.path.relpath(caminho, RAIZ)
                for e_texto, p in split_pieces(t):
                    if not e_texto:
                        continue
                    _, core, _ = split_core(p)
                    if core and _LETRA.search(core) and _parece_pt(core) and not _IDENTIFICADOR.fullmatch(core):
                        res.setdefault(core, []).append(f"{rel}:{getattr(n, 'lineno', 0)}")
    return res


def em_falta():
    from modules import i18n
    i18n._traduzir_en.cache_clear()
    falta = {}
    sys.path.insert(0, os.path.join(RAIZ, "tools"))
    from scan_en import _ACENTO as _AC, _PT as _PTW
    for core, locais in candidatos().items():
        en = i18n._traduzir_en(core)
        # em falta = ficou igual OU ficou misturado (ainda com palavras portuguesas)
        if en == core or _AC.search(en) or _PTW.search(en):
            falta[core] = locais
    return falta


if __name__ == "__main__":
    f = em_falta()
    por_ficheiro = {}
    for core, locais in f.items():
        por_ficheiro.setdefault(locais[0].split(":")[0], []).append(core)
    print(f"{len(f)} frases em falta")
    for k, v in sorted(por_ficheiro.items(), key=lambda x: -len(x[1]))[:40]:
        print(f"  {len(v):4d}  {k}")
    json.dump(f, open(os.path.join(RAIZ, "tools", "i18n_em_falta.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    if "--ver" in sys.argv:
        n = int(sys.argv[sys.argv.index("--ver") + 1])
        for core in list(f)[:n]:
            print("-", core[:140].replace("\n", " "), "   <", f[core][0])
