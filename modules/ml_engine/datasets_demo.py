"""Datasets fictícios para experimentar o Laboratório de ML.

Todos os dados são gerados por código com semente fixa: nada vem de pessoas, empresas ou medições reais.
Têm relações reais entre as colunas (para os modelos terem algo para aprender), ruído, categorias e alguns
valores em falta, para se parecerem com dados do dia-a-dia. Sem Streamlit, por isso são testáveis.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetDemo:
    id: str
    nome: str
    tipo: str            # "regressao" | "classificacao" | "serie"
    alvo: str            # coluna a prever
    descricao: str
    gerar: Callable[[int, int], pd.DataFrame]   # (n_linhas, semente) -> DataFrame
    n_padrao: int = 1000


def _faltas(df: pd.DataFrame, colunas, frac: float, rng) -> pd.DataFrame:
    for c in colunas:
        df.loc[rng.random(len(df)) < frac, c] = np.nan
    return df


def _imoveis(n, seed):
    rng = np.random.default_rng(seed)
    zona = rng.choice(["Centro", "Subúrbio", "Periferia", "Litoral"], n, p=[.25, .35, .25, .15])
    area = np.clip(rng.normal(95, 35, n), 25, 300).round(0)
    quartos = np.clip((area / 30 + rng.normal(0, .6, n)).round(), 1, 6).astype(int)
    idade = rng.integers(0, 60, n)
    garagem = rng.choice(["Sim", "Não"], n, p=[.55, .45])
    base = {"Centro": 2300, "Subúrbio": 1700, "Periferia": 1200, "Litoral": 2900}
    preco_m2 = np.array([base[z] for z in zona]) - idade * 9 + (garagem == "Sim") * 150
    preco = area * preco_m2 + quartos * 4000 + rng.normal(0, 12000, n)
    df = pd.DataFrame({"zona": zona, "area_m2": area, "quartos": quartos, "idade_anos": idade,
                       "garagem": garagem, "preco": preco.round(0)})
    return _faltas(df, ["idade_anos"], .04, rng)


def _salarios(n, seed):
    rng = np.random.default_rng(seed)
    anos = np.clip(rng.gamma(2.2, 3.5, n), 0, 35).round(1)
    educ = rng.choice(["Secundário", "Licenciatura", "Mestrado", "Doutoramento"], n, p=[.3, .45, .2, .05])
    area = rng.choice(["Tecnologia", "Saúde", "Educação", "Finanças", "Comércio"], n)
    nivel = {"Secundário": 0, "Licenciatura": 350, "Mestrado": 700, "Doutoramento": 1100}
    setor = {"Tecnologia": 500, "Saúde": 300, "Educação": 0, "Finanças": 600, "Comércio": -100}
    sal = (900 + anos * 65 + np.array([nivel[e] for e in educ]) + np.array([setor[a] for a in area])
           + rng.normal(0, 150, n))
    df = pd.DataFrame({"anos_experiencia": anos, "escolaridade": educ, "area": area,
                       "horas_semana": rng.choice([35, 40, 45], n, p=[.2, .65, .15]),
                       "salario_mensal": sal.round(0)})
    return _faltas(df, ["anos_experiencia"], .03, rng)


def _consumo_carros(n, seed):
    rng = np.random.default_rng(seed)
    cil = rng.choice([1.0, 1.4, 1.6, 2.0, 2.5, 3.0], n, p=[.1, .25, .25, .22, .1, .08])
    peso = np.clip(780 + cil * 260 + rng.normal(0, 90, n), 750, 2200).round(0)
    comb = rng.choice(["Gasolina", "Diesel", "Híbrido"], n, p=[.5, .35, .15])
    ajuste = {"Gasolina": 0, "Diesel": -1.0, "Híbrido": -2.3}
    l100 = 2.2 + cil * 1.1 + peso / 700 + np.array([ajuste[c] for c in comb]) + rng.normal(0, .5, n)
    df = pd.DataFrame({"cilindrada_l": cil, "peso_kg": peso, "combustivel": comb,
                       "ano": rng.integers(2008, 2025, n), "consumo_l_100km": np.clip(l100, 2.5, None).round(2)})
    return df


def _churn(n, seed):
    rng = np.random.default_rng(seed)
    meses = rng.integers(1, 72, n)
    contrato = rng.choice(["Mensal", "Anual", "Bienal"], n, p=[.55, .3, .15])
    mensal = np.clip(rng.normal(45, 18, n), 15, 120).round(2)
    tickets = rng.poisson(1.2, n)
    pagamento = rng.choice(["Cartão", "Transferência", "Multibanco"], n)
    z = (-0.6 - 0.035 * meses + 0.022 * (mensal - 45) + 0.45 * tickets
         + np.select([contrato == "Mensal", contrato == "Anual"], [1.1, 0.1], -0.9))
    sai = rng.random(n) < 1 / (1 + np.exp(-z))
    df = pd.DataFrame({"meses_cliente": meses, "tipo_contrato": contrato, "mensalidade": mensal,
                       "tickets_suporte": tickets, "metodo_pagamento": pagamento,
                       "abandonou": np.where(sai, "Sim", "Não")})
    return _faltas(df, ["mensalidade"], .03, rng)


def _credito(n, seed):
    rng = np.random.default_rng(seed)
    idade = rng.integers(19, 70, n)
    rend = np.clip(rng.lognormal(7.2, .45, n), 450, 9000).round(0)
    divida = np.clip(rng.beta(2, 5, n) * 1.2, 0, 1).round(3)
    historico = rng.choice(["Bom", "Médio", "Mau"], n, p=[.55, .3, .15])
    emprego = rng.choice(["Efectivo", "Contrato", "Independente", "Desempregado"], n, p=[.5, .25, .18, .07])
    z = (-1.2 + 0.0006 * (rend - 1500) - 3.2 * divida + 0.012 * (idade - 40)
         + np.select([historico == "Bom", historico == "Médio"], [1.4, 0.3], -1.5)
         + np.select([emprego == "Efectivo", emprego == "Desempregado"], [0.6, -1.5], 0)) + 1.0
    ok = rng.random(n) < 1 / (1 + np.exp(-z))
    df = pd.DataFrame({"idade": idade, "rendimento_mensal": rend, "rácio_divida": divida,
                       "historico_pagamentos": historico, "situacao_emprego": emprego,
                       "credito_aprovado": np.where(ok, "Sim", "Não")})
    return _faltas(df, ["rendimento_mensal"], .05, rng)


def _fraude(n, seed):
    rng = np.random.default_rng(seed)
    valor = np.clip(rng.lognormal(3.6, 1.1, n), 1, 8000).round(2)
    hora = rng.integers(0, 24, n)
    estrang = rng.random(n) < .08
    novo_disp = rng.random(n) < .12
    tentativas = rng.poisson(.3, n)
    z = (-5.6 + 0.0006 * valor + 1.1 * estrang + 1.4 * novo_disp + 0.7 * tentativas
         + 0.9 * ((hora < 5) | (hora > 22)))
    fraude = rng.random(n) < 1 / (1 + np.exp(-z))
    df = pd.DataFrame({"valor": valor, "hora_dia": hora,
                       "pais_estrangeiro": np.where(estrang, "Sim", "Não"),
                       "dispositivo_novo": np.where(novo_disp, "Sim", "Não"),
                       "tentativas_falhadas": tentativas,
                       "fraude": np.where(fraude, "Sim", "Não")})
    return df


def _flores(n, seed):
    rng = np.random.default_rng(seed)
    esp = rng.choice(["Aurea", "Boreal", "Cinerea"], n)
    medias = {"Aurea": (5.0, 3.4, 1.5, .25), "Boreal": (5.9, 2.8, 4.3, 1.3), "Cinerea": (6.6, 3.0, 5.6, 2.0)}
    m = np.array([medias[e] for e in esp])
    x = m + rng.normal(0, [.4, .3, .45, .2], (n, 4))
    return pd.DataFrame({"comprimento_sepala": x[:, 0].round(2), "largura_sepala": x[:, 1].round(2),
                         "comprimento_petala": x[:, 2].round(2), "largura_petala": x[:, 3].round(2),
                         "especie": esp})


def _serie_vendas(n, seed):
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2018-01-01", periods=n, freq="MS")
    t = np.arange(n)
    v = 1000 + 9 * t + 140 * np.sin(2 * np.pi * (t % 12) / 12 - 1.2) + rng.normal(0, 45, n)
    v += np.where(datas.month == 12, 220, 0)
    return pd.DataFrame({"data": datas.strftime("%Y-%m-%d"), "vendas": v.round(0)})


def _serie_visitas(n, seed):
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    semana = np.array([1.0, 1.05, 1.05, 1.0, .95, .7, .6])[datas.dayofweek]
    v = (800 + 1.4 * t) * semana + rng.normal(0, 40, n)
    picos = rng.random(n) < .015
    v = v + picos * rng.uniform(250, 600, n)
    return pd.DataFrame({"data": datas.strftime("%Y-%m-%d"), "visitas_site": v.round(0)})


def _serie_energia(n, seed):
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n)
    temp = 17 + 9 * np.sin(2 * np.pi * (t - 110) / 365) + rng.normal(0, 2, n)
    kwh = 310 + 4.2 * np.abs(temp - 19) + np.where(datas.dayofweek >= 5, 35, 0) + 0.03 * t + rng.normal(0, 12, n)
    return pd.DataFrame({"data": datas.strftime("%Y-%m-%d"), "temperatura_c": temp.round(1),
                         "consumo_kwh": kwh.round(1)})


DATASETS: dict = {d.id: d for d in [
    DatasetDemo("imoveis", "Preço de imóveis", "regressao", "preco",
                "Prever o preço de casas a partir da zona, área, quartos, idade e garagem.", _imoveis, 1200),
    DatasetDemo("salarios", "Salários", "regressao", "salario_mensal",
                "Prever o salário mensal a partir da experiência, escolaridade e área profissional.", _salarios, 1000),
    DatasetDemo("consumo_carros", "Consumo de automóveis", "regressao", "consumo_l_100km",
                "Prever os litros aos 100 km a partir da cilindrada, peso e combustível.", _consumo_carros, 800),
    DatasetDemo("churn", "Abandono de clientes (churn)", "classificacao", "abandonou",
                "Prever se um cliente cancela o serviço. Duas classes, ligeiramente desequilibradas.", _churn, 1500),
    DatasetDemo("credito", "Aprovação de crédito", "classificacao", "credito_aprovado",
                "Prever se um pedido de crédito é aprovado. Tem valores em falta e várias categorias.", _credito, 1500),
    DatasetDemo("fraude", "Deteção de fraude", "classificacao", "fraude",
                "Prever transacções fraudulentas. Muito desequilibrado: poucos casos de fraude. Bom para ver "
                "porque a exactidão sozinha engana.", _fraude, 4000),
    DatasetDemo("flores", "Espécies de flores (3 classes)", "classificacao", "especie",
                "Classificação com 3 categorias a partir de medidas das pétalas e sépalas. Fácil, bom para começar.",
                _flores, 450),
    DatasetDemo("serie_vendas", "Vendas mensais (série)", "serie", "vendas",
                "Série mensal com tendência, sazonalidade anual e pico em Dezembro.", _serie_vendas, 72),
    DatasetDemo("serie_visitas", "Visitas diárias a um site (série)", "serie", "visitas_site",
                "Série diária com tendência, ciclo semanal e picos ocasionais.", _serie_visitas, 540),
    DatasetDemo("serie_energia", "Consumo diário de energia (série)", "serie", "consumo_kwh",
                "Série diária com ciclo anual ligado à temperatura e fins-de-semana mais altos.", _serie_energia, 730),
]}

NOMES_TIPO = {"regressao": "Regressão", "classificacao": "Classificação", "serie": "Série temporal"}


def listar(tipo: str = None) -> list:
    return [d for d in DATASETS.values() if tipo in (None, d.tipo)]


def gerar(id_: str, n: int = None, semente: int = 42) -> pd.DataFrame:
    if id_ not in DATASETS:
        raise KeyError(f"Dataset de exemplo desconhecido: {id_}")
    d = DATASETS[id_]
    n = int(n or d.n_padrao)
    if not 30 <= n <= 50000:
        raise ValueError("O número de linhas tem de estar entre 30 e 50 000.")
    return d.gerar(n, int(semente))


def para_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
