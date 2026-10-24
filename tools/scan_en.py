"""Percorre as páginas com o idioma em inglês e lista o texto que continua em português.

Uso:  python tools/scan_en.py [pagina ...]      (sem argumentos: todas as páginas de aluno)
"""
import os
import re
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(tempfile.mkdtemp())
for v in ("SUPABASE_URL", "SUPABASE_KEY"):
    os.environ.pop(v, None)

from streamlit.testing.v1 import AppTest   # noqa: E402
from modules.i18n_seg import split_pieces, split_core   # noqa: E402

PAGINAS = ["dashboard", "escola", "comparador", "reforco", "dicionario", "perfil", "revisao", "primeiro_modelo", "classificacao",
           "regressao", "clustering", "reducao", "anomalias", "mlp", "visao", "nlp", "series", "generativo", "laboratorio",
           "projeto", "conquistas", "certificados", "historico", "assistente", "business"]
TIPOS = ["markdown", "caption", "text", "header", "subheader", "title", "info", "success", "warning", "error", "metric", "button",
         "radio", "selectbox", "multiselect", "slider", "checkbox", "toggle", "text_input", "text_area", "number_input", "expander",
         "tab", "file_uploader", "select_slider", "date_input"]
_ACENTO = re.compile(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]")
_PT = re.compile(r"\b(de|da|dos|das|em|para|por|sem|que|não|ou|ao|aos|mais|como|já|teu|tua|teus|tuas|seu|sua|este|esta|isto|"
                 r"quando|onde|todos|todas|cada|podes|dados|modelo|treino|treinar|resultado|escolhe|gráfico|aluno|turma|página|ficheiro|"
                 r"pergunta|resposta|lição|lições|exercício|exercícios|projecto|percurso|conquistas|desafio|desafios|sair|voltar|guardar|"
                 r"gerar|criar|abrir|fechar|mostrar|nome|idade|previsão|previsões|série|séries|ajuda|aprender|aprende|começar|continuar|"
                 r"concluído|concluída|pontos|nível|você|uma|um|nos|nas|na|teste|erros|valores|colunas|coluna|linhas|escolher|escolhe|"
                 r"tenta|usa|vê|faz|sobre|entre|ainda|também|só|aqui|agora|depois|antes)\b", re.I)


def _textos(el):
    out = []
    for atr in ("value", "label", "body", "placeholder", "help"):
        v = getattr(el, atr, None)
        if isinstance(v, str):
            out.append(v)
    for atr in ("options",):
        v = getattr(el, atr, None)
        if isinstance(v, (list, tuple)):
            out += [x for x in v if isinstance(x, str)]
    return out


def ler(at):
    res = []
    for t in TIPOS:
        try:
            els = at.get(t)
        except Exception:
            continue
        for e in els:
            for s in _textos(e):
                res.append((t, s))
    return res


def suspeitos(textos):
    vistos, out = set(), []
    for tipo, s in textos:
        if tipo == "markdown" and ("<style" in s[:20] or s.lstrip().startswith("<style")):
            s = re.sub(r"<style[\s\S]*?</style>", " ", s)
        for e_texto, p in split_pieces(s):
            if not e_texto:
                continue
            _, core, _ = split_core(p)
            if not core or core in vistos:
                continue
            vistos.add(core)
            if core in ("Português", "English"):
                continue
            if _ACENTO.search(core) or (len(core) > 2 and _PT.search(core)):
                out.append((tipo, core))
    return out


def correr(pagina, user="aluno"):
    at = AppTest.from_file(os.path.join(RAIZ, "app.py"), default_timeout=180)
    at.session_state["authentication_status"] = True
    at.session_state["username"] = user
    at.session_state["name"] = user.title()
    at.session_state["pagina"] = pagina
    at.session_state["lang"] = "en"
    at.run()
    if at.exception:
        return None, [e.value for e in at.exception]
    return suspeitos(ler(at)), None


if __name__ == "__main__":
    paginas = [a for a in sys.argv[1:] if not a.startswith("--")] or PAGINAS
    total = 0
    for p in paginas:
        sus, err = correr(p)
        if err:
            print(f"## {p}: ERRO {err[0][:100]}")
            continue
        print(f"## {p}: {len(sus)}")
        for tipo, core in sus:
            print(f"   [{tipo}] {core[:130]}")
        total += len(sus)
    print("TOTAL", total)
