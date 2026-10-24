"""Garante que nenhuma frase em português fica sem tradução quando o idioma é inglês.

Se este teste falhar depois de escreveres texto novo na interface:
  1. python tools/extrair_i18n.py --ver 200      (mostra o que falta)
  2. escreve as traduções em tools/traducoes/<nome>.json  ({"português": "english"})
  3. python tools/aplicar_traducoes.py
Texto que NUNCA aparece ao aluno (prompts da IA, dados de exemplo em português, logs) vai para tools/i18n_ignorar.json.
"""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))


def test_nao_ha_frases_novas_sem_traducao():
    import extrair_i18n
    ignorar = set(json.load(open(os.path.join(RAIZ, "tools", "i18n_ignorar.json"), encoding="utf8")))
    novas = {k: v[0] for k, v in extrair_i18n.em_falta().items() if k not in ignorar}
    assert not novas, "Frases sem tradução para inglês:\n" + "\n".join(f"  {v}  {k[:100]!r}" for k, v in list(novas.items())[:30])


def test_traducoes_tem_os_mesmos_marcadores():
    d = json.load(open(os.path.join(RAIZ, "modules", "locales", "en.json"), encoding="utf8"))["t"]
    maus = [k for k, v in d.items() if k.count("{}") != v.count("{}") and "{}" in k]
    assert not maus, maus[:5]
