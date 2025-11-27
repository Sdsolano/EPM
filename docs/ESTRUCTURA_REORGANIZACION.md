# Reorganización del Repositorio EPM

## Estructura Propuesta

```
EPM/
├── README.md                           # Documentación principal
├── requirements.txt                    # Dependencias del proyecto
├── setup.py                           # Instalación del paquete
├── .gitignore                         # Archivos ignorados por git
│
├── docs/                              # 📚 DOCUMENTACIÓN
│   ├── FASE1_COMPLETADA.md
│   ├── FASE2_MODELOS_IMPLEMENTADOS.md
│   ├── proyecto_especificaciones.pdf
│   └── arquitectura_sistema.md
│
├── config/                            # ⚙️ CONFIGURACIÓN
│   ├── __init__.py
│   ├── settings.py                    # Configuración principal
│   ├── festivos.json                  # Calendario de festivos
│   └── logging.yaml                   # Configuración de logs
│
├── src/                               # 💻 CÓDIGO FUENTE PRINCIPAL
│   ├── __init__.py
│   │
│   ├── pipeline/                      # Pipeline de datos (Fase 1)
│   │   ├── __init__.py
│   │   ├── connectors.py
│   │   ├── cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── monitoring.py
│   │   └── orchestrator.py
│   │
│   ├── models/                        # Modelos ML (Fase 2)
│   │   ├── __init__.py
│   │   ├── base_models.py
│   │   ├── metrics.py
│   │   ├── trainer.py
│   │   └── registry.py
│   │
│   ├── prediction/                    # Sistema de predicción
│   │   ├── __init__.py
│   │   ├── forecaster.py
│   │   └── disaggregation.py
│   │
│   ├── api/                           # API Gateway (Fase 4)
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app
│   │   ├── endpoints/
│   │   │   ├── predict.py
│   │   │   ├── metrics.py
│   │   │   ├── health.py
│   │   │   └── retrain.py
│   │   └── schemas.py                 # Pydantic models
│   │
│   ├── monitoring/                    # Sistema de monitoreo (Fase 3-4)
│   │   ├── __init__.py
│   │   ├── performance_monitor.py
│   │   ├── retraining_trigger.py
│   │   └── alerts.py
│   │
│   └── utils/                         # Utilidades generales
│       ├── __init__.py
│       ├── datetime_utils.py
│       └── validators.py
│
├── scripts/                           # 🔧 SCRIPTS EJECUTABLES
│   ├── run_pipeline.py                # Ejecuta pipeline completo
│   ├── train_models.py                # Entrena modelos
│   ├── predict_30_days.py             # Predicción de 30 días
│   ├── setup_environment.py           # Setup inicial
│   └── migration_scripts/             # Scripts de migración
│
├── tests/                             # 🧪 TESTS
│   ├── __init__.py
│   ├── test_pipeline.py
│   ├── test_models.py
│   ├── test_api.py
│   └── fixtures/
│
├── notebooks/                         # 📊 JUPYTER NOTEBOOKS
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_analisis_features.ipynb
│   ├── 03_evaluacion_modelos.ipynb
│   └── README.md
│
├── data/                              # 📁 DATOS (NO EN GIT)
│   ├── raw/                           # Datos crudos originales
│   │   ├── datos.csv
│   │   └── weather_raw.csv
│   │
│   ├── processed/                     # Datos procesados
│   │   ├── power_clean_*.csv
│   │   └── weather_clean_*.csv
│   │
│   ├── features/                      # Datos con features
│   │   ├── data_with_features_latest.csv
│   │   └── data_with_features_*.csv
│   │
│   └── predictions/                   # Predicciones generadas
│       └── predictions_*.csv
│
├── models/                            # 🤖 MODELOS ENTRENADOS (NO EN GIT)
│   ├── trained/                       # Modelos históricos
│   │   └── *.joblib
│   │
│   └── registry/                      # Registry de modelos
│       ├── champion_model.joblib
│       └── registry_metadata.json
│
├── logs/                              # 📝 LOGS (NO EN GIT)
│   ├── pipeline/
│   ├── training/
│   ├── api/
│   └── monitoring/
│
├── dashboards/                        # 📈 DASHBOARDS Y VISUALIZACIÓN
│   ├── streamlit_dashboard.py
│   └── monitoring_dashboard.py
│
└── deployment/                        # 🚀 DESPLIEGUE
    ├── Dockerfile
    ├── docker-compose.yml
    ├── kubernetes/
    └── README.md
```

## Cambios Principales

### 1. Código Fuente en `src/`
- Todo el código principal está bajo `src/` (estructura estándar Python)
- Módulos bien organizados por funcionalidad
- Fácil instalación con `pip install -e .`

### 2. Scripts Separados
- Scripts ejecutables en `scripts/` en lugar de raíz
- Nombres descriptivos y claros
- Separados del código fuente

### 3. Documentación Centralizada
- Toda la documentación en `docs/`
- Fácil de encontrar y mantener

### 4. Configuración Centralizada
- Archivos de configuración en `config/`
- Separados del código

### 5. Tests Organizados
- Todos los tests en `tests/`
- Estructura paralela al código fuente

### 6. Datos y Modelos Fuera de Git
- `.gitignore` actualizado
- Solo estructura de carpetas en git, no contenido

## Archivos a Mover

### De raíz → src/
- `config.py` → `src/config/settings.py`
- `pipeline/` → `src/pipeline/`
- `models/` → `src/models/`
- `prediction/` → `src/prediction/`

### De raíz → scripts/
- `train_models.py` → `scripts/train_models.py`
- `test_pipeline.py` → `tests/test_pipeline.py`

### De raíz → docs/
- `FASE1_COMPLETADA.md` → `docs/`
- `FASE2_MODELOS_IMPLEMENTADOS.md` → `docs/`
- PDF → `docs/proyecto_especificaciones.pdf`

### De raíz → data/raw/
- `datos.csv` → `data/raw/`
- `data_cleaned_weather.csv` → `data/raw/` (si existe)

### A eliminar (obsoletos/duplicados)
- `main.py` (obsoleto)
- `read.py` (obsoleto)
- `graphs.py` (mover a notebooks o eliminar)
- `cluster.py` (mover a notebooks o eliminar)
- `dias.py` (mover a notebooks o eliminar)
- `dashboard_week2.py` → `dashboards/`
- `a50f5d9785250195ea4ef2cb78efad38.csv` (archivo temporal?)

## Ventajas de Esta Estructura

✅ **Profesional**: Sigue estándares de Python (PEP 8, PEP 518)
✅ **Escalable**: Fácil agregar nuevos módulos
✅ **Mantenible**: Código organizado por funcionalidad
✅ **Instalable**: Se puede instalar con pip
✅ **Testeable**: Tests bien organizados
✅ **Documentado**: Documentación centralizada
✅ **Deployable**: Carpeta de deployment lista para producción

## Próximos Pasos

1. ✅ Revisar y aprobar estructura
2. Crear carpetas necesarias
3. Mover archivos a nuevas ubicaciones
4. Actualizar imports en archivos Python
5. Crear setup.py
6. Actualizar .gitignore
7. Probar que todo funciona
8. Commit de reorganización
