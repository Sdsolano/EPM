"""
Sistema de Desagregación Horaria de Demanda Energética

Este módulo contiene el sistema de clustering y desagregación horaria
que convierte pronósticos diarios totales en distribuciones horarias (24 períodos).
"""

from .disaggregation_engine import HourlyDisaggregationEngine
from .calendar_utils import CalendarClassifier

__all__ = ['HourlyDisaggregationEngine', 'CalendarClassifier']
