"""
Tests de las curvas típicas cargadas dentro del cálculo de factores.

Una curva cargada es una medida con código RPM reservado (CT-<id>-D suma a la barra destino, CT-<id>-O resta a la barra de origen,
con factor -1). Al calcular con "ignorar negativos" esas filas NO deben descartarse aunque queden negativas: si se descartaran,
la barra de origen conservaría la carga que ya se trasladó y el total del sistema se contaría dos veces.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.calculos_service import _procesar_medidas_con_factores  # noqa: E402


def _medida(codigo_rpm, flujo, valor, fecha='24-11-2026', barra='ORIGEN'):
    fila = {'babarra': barra, 'mefecha': fecha, 'mecodigo_rpm': codigo_rpm, 'meflujo': flujo}
    fila.update({f'mep{i}': float(valor) for i in range(1, 25)})
    return fila


def _factor(codigo_rpm, flujo, factor):
    return {'codigo_rpm': codigo_rpm, 'flujo': flujo, 'factor': str(factor), 'dividir_por_1000': False, 'valor_absoluto': False}


def test_la_curva_cargada_que_resta_no_se_descarta_con_ignorar_negativos():
    medidas = [_medida('RPM_REAL', 'AE', 100.0), _medida('CT-7-O', 'AE', 8.0)]
    factores = [_factor('RPM_REAL', 'AE', 1), _factor('CT-7-O', 'AE', -1)]

    r = _procesar_medidas_con_factores(medidas, factores, ignorar_negativos=True)

    assert len(r) == 1
    assert r[0]['barra'] == 'ORIGEN'
    assert r[0]['periodos']['p1'] == 92.0   # 100 - 8: la porción trasladada sí se resta
    assert r[0]['periodos']['p24'] == 92.0


def test_igual_resultado_con_y_sin_ignorar_negativos_cuando_no_hay_otros_negativos():
    medidas = [_medida('RPM_REAL', 'AE', 100.0), _medida('CT-7-O', 'AE', 8.0)]
    factores = [_factor('RPM_REAL', 'AE', 1), _factor('CT-7-O', 'AE', -1)]

    con = _procesar_medidas_con_factores(medidas, factores, ignorar_negativos=True)
    sin = _procesar_medidas_con_factores(medidas, factores, ignorar_negativos=False)

    assert con == sin


def test_una_medida_real_negativa_sigue_descartandose_con_ignorar_negativos():
    medidas = [_medida('RPM_REAL', 'AE', 100.0), _medida('RPM_MALA', 'AE', 5.0)]
    factores = [_factor('RPM_REAL', 'AE', 1), _factor('RPM_MALA', 'AE', -1)]  # el factor deja negativa a la medida real

    r = _procesar_medidas_con_factores(medidas, factores, ignorar_negativos=True)

    assert r[0]['periodos']['p1'] == 100.0   # la fila negativa de un código real se descarta, como siempre


def test_el_codigo_reservado_es_exacto():
    # Un código que solo se parece no recibe el trato especial.
    medidas = [_medida('RPM_REAL', 'AE', 100.0), _medida('CT-7', 'AE', 8.0), _medida('xCT-7-O', 'AE', 8.0)]
    factores = [_factor('RPM_REAL', 'AE', 1), _factor('CT-7', 'AE', -1), _factor('xCT-7-O', 'AE', -1)]

    r = _procesar_medidas_con_factores(medidas, factores, ignorar_negativos=True)

    assert r[0]['periodos']['p1'] == 100.0   # ambos parecidos se descartan como cualquier medida negativa


def test_con_ignorar_negativos_apagado_todo_se_suma():
    medidas = [_medida('RPM_REAL', 'AE', 100.0), _medida('CT-7-D', 'AE', 3.0, barra='ORIGEN')]
    factores = [_factor('RPM_REAL', 'AE', 1), _factor('CT-7-D', 'AE', 1)]

    r = _procesar_medidas_con_factores(medidas, factores, ignorar_negativos=False)

    assert r[0]['periodos']['p10'] == 103.0
