"""Junta as traduções de tools/traducoes/*.json ao modules/locales/en.json.

Valida que cada frase tem os mesmos «{}» (e %s/%d) no português e no inglês, para o tradutor de padrões funcionar.
Não apaga nem altera traduções que já existem (a não ser com --substituir).
Uso: python tools/aplicar_traducoes.py [--substituir]
"""
import glob
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN = os.path.join(RAIZ, "modules", "locales", "en.json")


def main(substituir=False):
    dados = json.load(open(EN, encoding="utf8"))
    t = dados["t"]
    novas, trocadas, erros = 0, 0, []
    for f in sorted(glob.glob(os.path.join(RAIZ, "tools", "traducoes", "[!_]*.json"))):
        for pt, en in json.load(open(f, encoding="utf8")).items():
            if not en.strip():
                erros.append(f"vazio: {pt[:60]}")
                continue
            if pt.count("{}") != en.count("{}") or pt.count("%s") != en.count("%s") or pt.count("%d") != en.count("%d"):
                erros.append(f"{{}} diferentes: {pt[:70]}")
                continue
            if pt in t:
                if t[pt] != en and substituir:
                    t[pt] = en; trocadas += 1
                continue
            t[pt] = en; novas += 1
    extra = os.path.join(RAIZ, "tools", "traducoes", "_inline.txt")      # palavras soltas que também se traduzem DENTRO de frases
    if os.path.exists(extra):
        for linha in open(extra, encoding="utf8").read().split("\n"):
            linha = linha.strip()
            if linha and linha not in dados["inline"]:
                dados["inline"].append(linha)
    if erros:
        print("ERROS:\n  " + "\n  ".join(erros))
        sys.exit(1)
    dados["t"] = dict(sorted(t.items()))
    with open(EN, "w", encoding="utf8") as fh:
        json.dump(dados, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"{novas} novas, {trocadas} substituídas; total {len(t)}")


if __name__ == "__main__":
    main("--substituir" in sys.argv)
