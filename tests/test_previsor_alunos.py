import shutil

import pytest

from modules import previsor_alunos as pa

BOM = {"horas_estudo_semana": 10, "faltas": 2, "nota_anterior": 12, "trabalha": 0, "tem_internet": 1}
MAU = {"horas_estudo_semana": 2, "faltas": 10, "nota_anterior": 7, "trabalha": 1, "tem_internet": 0}


@pytest.fixture(scope="module")
def pacote():
    return pa.carregar_modelo()


def test_carrega_e_tem_as_cinco_variaveis(pacote):
    assert pacote["features"] == list(pa.CAMPOS)
    assert pacote["model"].n_features_in_ == 5


def test_previsao_devolve_classe_e_probabilidades(pacote):
    r = pa.prever(pacote, BOM)
    assert r["classe"] in ("sim", "nao")
    assert abs(sum(r["probabilidades"].values()) - 1) < 1e-9
    assert len(r["vizinhos"]) == pacote["model"].n_neighbors


def test_exemplos_conhecidos(pacote):
    assert pa.prever(pacote, BOM)["classe"] == "sim"
    assert pa.prever(pacote, MAU)["classe"] == "nao"


def test_faltam_variaveis(pacote):
    with pytest.raises(ValueError):
        pa.prever(pacote, {"faltas": 1})


def test_aviso_fora_do_intervalo(pacote):
    assert pa.fora_do_intervalo(pacote, BOM) == []
    avisos = pa.fora_do_intervalo(pacote, {**BOM, "faltas": 99})
    assert [a[0] for a in avisos] == ["faltas"]


def test_recusa_ficheiro_alterado(tmp_path):
    copia = tmp_path / "knn_model.pkl"
    shutil.copy(pa.CAMINHO_MODELO, copia)
    shutil.copy(pa.CAMINHO_HASH, str(copia) + ".sha256")
    with open(copia, "ab") as f:
        f.write(b"x")                      # basta um byte para o hash deixar de bater
    with pytest.raises(pa.ModeloInvalido):
        pa.carregar_modelo(str(copia), str(copia) + ".sha256")


def test_recusa_sem_ficheiro_de_hash(tmp_path):
    copia = tmp_path / "knn_model.pkl"
    shutil.copy(pa.CAMINHO_MODELO, copia)
    with pytest.raises(pa.ModeloInvalido):
        pa.carregar_modelo(str(copia), str(copia) + ".sha256")
