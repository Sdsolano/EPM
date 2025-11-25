"""
Módulo de Predicción EPM
========================

Pipeline automático para predicción de demanda energética
"""

from .predict_next_30_days import ForecastPipeline

__all__ = ['ForecastPipeline']
