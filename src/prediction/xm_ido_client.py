"""
Cliente para el portal IDO de XM (https://ido.xm.com.co) — Demanda No
Atendida (DNA), programada y no programada, acumulada y limitación de
suministro.

La página usa Azure B2C (Client Credentials) vía un endpoint propio del
mismo sitio (`/api/auth/client-token`, sin credenciales de usuario) para
obtener un token, y luego consulta un servicio REST separado
(`serviciossistemareportes.xm.com.co`) con ese token. No requiere headless
browser: son dos llamadas HTTP simples (`requests`), verificadas
directamente contra el sitio real.
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class XMIdoClient:
    """
    Cliente para obtener eventos de Demanda No Atendida (DNA) del portal
    IDO de XM.

    El token de Azure B2C dura 1h (`expires_in`); se cachea en memoria por
    instancia y se renueva automáticamente cuando expira.
    """

    TOKEN_URL = "https://ido.xm.com.co/api/auth/client-token"
    SERVICE_BASE_URL = "https://serviciossistemareportes.xm.com.co/eventos/XmService.svc"

    def __init__(
        self,
        token_url: Optional[str] = None,
        service_base_url: Optional[str] = None,
        timeout: int = 15,
    ):
        self.token_url = token_url or self.TOKEN_URL
        self.service_base_url = service_base_url or self.SERVICE_BASE_URL
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expira_en: float = 0.0

    def _obtener_token(self) -> str:
        """Obtiene (o reutiliza del caché en memoria) el Bearer token."""
        ahora = time.time()
        if self._token and ahora < self._token_expira_en:
            return self._token

        try:
            response = requests.post(
                self.token_url,
                json={},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            if not token:
                raise ValueError(f"Respuesta sin access_token: {data}")

            # Margen de 60s antes de la expiración real reportada por el servidor
            expira_en = int(data.get("expires_in", 3600))
            self._token = token
            self._token_expira_en = ahora + max(expira_en - 60, 60)
            return token
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo token de XM IDO: {e}")
            raise

    def _get(self, path: str) -> dict:
        token = self._obtener_token()
        url = f"{self.service_base_url}/{path}"
        try:
            logger.debug(f"[XM_IDO] Request → {url}")
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout consultando XM IDO: {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición HTTP a XM IDO ({url}): {e}")
            raise

    def get_demanda_no_atendida(
        self, fecha_inicio: str, fecha_fin: str, desde: int = 0, hasta: int = 500
    ) -> List[Dict]:
        """
        DNA no programada. fecha_inicio/fecha_fin en formato YYYY-MM-DD.
        Cada evento trae: area, descripcion, energia (MWh), fechaini,
        fechafin, municipio, subestacion.
        """
        data = self._get(
            f"demandanoprogramada/{fecha_inicio}/{fecha_fin}/{desde}/{hasta}"
        )
        eventos = data.get("registrodemandanoprogramada", []) or []
        for e in eventos:
            e["tipo"] = "no programada"
        return eventos

    def get_demanda_programada(
        self, fecha_inicio: str, fecha_fin: str, desde: int = 0, hasta: int = 500
    ) -> List[Dict]:
        """DNA programada — misma forma que get_demanda_no_atendida."""
        data = self._get(
            f"demandaprogramada/{fecha_inicio}/{fecha_fin}/{desde}/{hasta}"
        )
        eventos = data.get("registrodemandaprogramada", []) or []
        for e in eventos:
            e["tipo"] = "programada"
        return eventos

    def get_demanda_acumulada(self) -> List[Dict]:
        """DNA acumulada por área (sin rango de fechas — snapshot histórico)."""
        data = self._get("demandaacumulada")
        return data.get("registrodemandaacumuladanoprogramada", []) or []

    def get_limitacion_suministro(self, fecha_inicio: str, fecha_fin: str) -> List[Dict]:
        """Limitación de suministro. Cada evento trae: area, descripcion,
        energia (MWh), fechaini, fechafin (sin municipio/subestacion)."""
        data = self._get(f"limitacion/{fecha_inicio}/{fecha_fin}")
        return data.get("registrolimitacion", []) or []

    def get_eventos_dna_para_fecha(self, fecha: str) -> List[Dict]:
        """
        Trae DNA no programada + programada para UN día puntual (fecha_inicio
        = fecha_fin = fecha) — pensado para el caso de uso de
        analyze_deviation_with_openai: ¿hubo DNA ese día específico?
        """
        eventos: List[Dict] = []
        for metodo in (self.get_demanda_no_atendida, self.get_demanda_programada):
            try:
                eventos.extend(metodo(fecha, fecha))
            except requests.exceptions.RequestException:
                # Un tipo de evento fallando (timeout, 5xx, etc.) no debe
                # tumbar el otro — se reporta lo que sí se pudo traer.
                continue
        return eventos


def formatear_eventos_dna(eventos: List[Dict]) -> str:
    """Texto plano legible para inyectar en un prompt de LLM o mostrar en un
    informe, a partir de la lista que devuelve get_eventos_dna_para_fecha."""
    if not eventos:
        return "No se encontraron eventos de Demanda No Atendida (DNA) en XM IDO para esta fecha."

    lineas = []
    for e in eventos:
        lineas.append(
            f"- [{e.get('tipo', '')}] {e.get('area', '')}: {e.get('energia', 0)} MWh no atendidos "
            f"({e.get('fechaini', '')} → {e.get('fechafin', '')}) "
            f"| {e.get('descripcion', '')} | Subestación: {e.get('subestacion', '')} "
            f"| Municipio: {e.get('municipio', '')}"
        )
    return "\n".join(lineas)
