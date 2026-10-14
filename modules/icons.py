"""
DataForge EDU - Sistema de Ícones SVG Profissionais
Ícones Lucide-style: limpos, consistentes, sem emojis
"""

# Tamanho padrão: 20px. Usar size= para ajustar.
# Cor padrão: currentColor (herda do CSS pai)


def icon(name: str, size: int = 20, color: str = "currentColor", stroke_width: float = 1.75) -> str:
    """Devolve SVG inline de um ícone profissional."""
    s = size
    sw = stroke_width
    c = color

    base = f'width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;"'

    ICONS = {
        # ── Navegação / UI ──────────────────────────────────────────────
        "home":         f'<svg {base}><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
        "menu":         f'<svg {base}><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="18" y2="18"/></svg>',
        "chevron_right":f'<svg {base}><polyline points="9 18 15 12 9 6"/></svg>',
        "chevron_down": f'<svg {base}><polyline points="6 9 12 15 18 9"/></svg>',
        "arrow_right":  f'<svg {base}><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
        "logout":       f'<svg {base}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>',
        "settings":     f'<svg {base}><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/><path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>',
        "x":            f'<svg {base}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
        "check":        f'<svg {base}><polyline points="20 6 9 17 4 12"/></svg>',
        "lock":         f'<svg {base}><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
        "play":         f'<svg {base}><polygon points="5 3 19 12 5 21 5 3"/></svg>',
        "info":         f'<svg {base}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        "alert":        f'<svg {base}><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "star":         f'<svg {base}><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
        "filter":       f'<svg {base}><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "search":       f'<svg {base}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "eye":          f'<svg {base}><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
        "download":     f'<svg {base}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "upload":       f'<svg {base}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "refresh":      f'<svg {base}><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>',
        "copy":         f'<svg {base}><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>',
        "trash":        f'<svg {base}><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
        "edit":         f'<svg {base}><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
        "save":         f'<svg {base}><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
        "user":         f'<svg {base}><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "users":        f'<svg {base}><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "key":          f'<svg {base}><circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/></svg>',

        # ── ML / Dados ───────────────────────────────────────────────────
        "target":       f'<svg {base}><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "trending_up":  f'<svg {base}><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
        "bar_chart":    f'<svg {base}><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
        "scatter":      f'<svg {base}><circle cx="7" cy="7" r="2"/><circle cx="17" cy="7" r="2"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/><circle cx="12" cy="12" r="2"/></svg>',
        "layers":       f'<svg {base}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "cpu":          f'<svg {base}><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "network":      f'<svg {base}><circle cx="12" cy="5" r="3"/><circle cx="4" cy="19" r="3"/><circle cx="20" cy="19" r="3"/><line x1="12" y1="8" x2="4" y2="16"/><line x1="12" y1="8" x2="20" y2="16"/></svg>',
        "git_branch":   f'<svg {base}><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
        "database":     f'<svg {base}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
        "table":        f'<svg {base}><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/></svg>',
        "file_data":    f'<svg {base}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "chart_line":   f'<svg {base}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "sigma":        f'<svg {base}><path d="M18 7V4H6l6 8-6 8h12v-3"/></svg>',
        "function":     f'<svg {base}><path d="M15.5 2H8.6L4 6.6V20l.6.5H11v-1H5V7.4L8.4 4H14v4h5v11h-6v1h6.6l.4-.5V7.4L15.5 2z"/><path d="M8 16v-4l4 4 4-4v4"/></svg>',
        "minimize":     f'<svg {base}><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>',
        "maximize":     f'<svg {base}><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>',
        "compass":      f'<svg {base}><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
        "map":          f'<svg {base}><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21 3 6"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>',
        "clock":        f'<svg {base}><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "sliders":      f'<svg {base}><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
        "zap":          f'<svg {base}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "box":          f'<svg {base}><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
        "package":      f'<svg {base}><line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
        "grid":         f'<svg {base}><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
        "activity":     f'<svg {base}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "award":        f'<svg {base}><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
        "book":         f'<svg {base}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
        "book_open":    f'<svg {base}><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
        "flask":        f'<svg {base}><path d="M9 3h6"/><path d="M6.7 21h10.6a2 2 0 0 0 1.7-3.07L14 9V3H10v6L5.3 17.93A2 2 0 0 0 6.7 21z"/><path d="M6.7 17h10.6"/></svg>',
        "microscope":   f'<svg {base}><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>',
        "alert_triangle":f'<svg {base}><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "radar":        f'<svg {base}><path d="M19.07 4.93A10 10 0 0 0 6.99 3.34"/><path d="M4 6h.01"/><path d="M2.29 9.62A10 10 0 1 0 21.31 8.35"/><path d="M16.24 7.76A6 6 0 1 0 8.23 16.67"/><path d="M12 18h.01"/><path d="M17.99 11.66A6 6 0 0 1 15.77 16.67"/><circle cx="12" cy="12" r="2"/><path d="m13.41 10.59 5.66-5.66"/></svg>',
        "shuffle":      f'<svg {base}><polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/></svg>',
        "play_circle":  f'<svg {base}><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
        "code":         f'<svg {base}><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
        "terminal":     f'<svg {base}><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
        "link":         f'<svg {base}><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
    }

    return ICONS.get(name, f'<svg {base}><circle cx="12" cy="12" r="10"/></svg>')


def icon_html(name: str, size: int = 18, color: str = "currentColor",
              stroke_width: float = 1.75, style: str = "") -> str:
    """Ícone pronto para usar em st.markdown com unsafe_allow_html=True."""
    svg = icon(name, size, color, stroke_width)
    if style:
        svg = svg.replace('style="', f'style="{style}; ')
    return svg


# ── Mapeamento semântico para módulos ML ─────────────────────────────
MODULE_ICONS = {
    # Supervisionado
    "supervisionado_classificacao": "target",
    "supervisionado_regressao":     "trending_up",
    # Não supervisionado
    "nao_supervisionado_clustering":"grid",
    "nao_supervisionado_reducao":   "minimize",
    "nao_supervisionado_anomalias": "radar",
    # Redes neurais
    "redes_neurais_mlp":            "network",
    # Deep Learning
    "deep_learning_visao":          "eye",
    "deep_learning_nlp":            "book_open",
    "deep_learning_series":         "chart_line",
    "deep_learning_generativo":     "shuffle",
    # Reforço
    "reforco":                      "compass",
}

CATEGORY_ICONS = {
    "Supervisionado":       "target",
    "Não Supervisionado":   "layers",
    "Redes Neurais":        "network",
    "Deep Learning":        "cpu",
    "Reforço":              "compass",
}

NAV_ICONS = {
    "Dashboard":            "home",
    "Classificação":        "target",
    "Regressão":            "trending_up",
    "Clustering":           "grid",
    "Redução Dimensionalidade": "minimize",
    "Detecção Anomalias":   "radar",
    "Redes Neurais (MLP)":  "network",
    "Visão Computacional":  "eye",
    "NLP":                  "book_open",
    "Séries Temporais":     "chart_line",
    "Modelos Generativos":  "shuffle",
    "Aprendizagem Reforço": "compass",
    "Comparador":           "sliders",
    "Configurações":        "settings",
}
