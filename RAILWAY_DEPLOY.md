# 🚂 Guía de Despliegue en Railway

Esta guía te ayudará a desplegar el servidor FastAPI de EPM en Railway.

## 📋 Requisitos Previos

1. Cuenta en [Railway](https://railway.app)
2. Repositorio Git (GitHub, GitLab, o Bitbucket)
3. Variables de entorno configuradas

## 🚀 Pasos para Desplegar

### 1. Preparar el Repositorio

Asegúrate de que tu código esté en un repositorio Git y que todos los archivos necesarios estén commitados:

```bash
git add .
git commit -m "Preparar para despliegue en Railway"
git push origin main
```

### 2. Crear Proyecto en Railway

1. Ve a [Railway Dashboard](https://railway.app/dashboard)
2. Haz clic en **"New Project"**
3. Selecciona **"Deploy from GitHub repo"** (o tu proveedor Git)
4. Conecta tu repositorio y selecciona el proyecto EPM

### 3. Configurar Variables de Entorno

En el dashboard de Railway, ve a tu servicio y luego a la pestaña **"Variables"**:

#### Variables Requeridas

- **`PORT`**: Railway lo proporciona automáticamente (no necesitas configurarlo)

#### Variables Opcionales

- **`API_KEY`**: Tu API key de OpenAI (para análisis de errores y eventos futuros)
  - Si no la configuras, estas funcionalidades estarán deshabilitadas
  - Obtén tu key en: https://platform.openai.com/api-keys

- **`LOG_LEVEL`**: Nivel de logging (default: `INFO`)
  - Opciones: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

### 4. Configuración Automática

Railway detectará automáticamente:
- ✅ **Python 3.11** desde `runtime.txt` y `nixpacks.toml`
- ✅ **requirements.txt** para dependencias
- ✅ **nixpacks.toml** para configuración de build
- ✅ **Comando de inicio** desde `nixpacks.toml`

**Nota importante**: Si el build falla con "pip: command not found", asegúrate de que:
1. `runtime.txt` existe con `python-3.11`
2. `nixpacks.toml` tiene `[providers] python = "3.11"`
3. No hay fases de install personalizadas que sobrescriban la instalación de Python

### 5. Verificar el Despliegue

Una vez desplegado, Railway te proporcionará una URL como:
```
https://tu-proyecto.up.railway.app
```

#### Endpoints Disponibles

- **Documentación Swagger**: `https://tu-proyecto.up.railway.app/docs`
- **Documentación ReDoc**: `https://tu-proyecto.up.railway.app/redoc`
- **Health Check**: `https://tu-proyecto.up.railway.app/health`
- **API Root**: `https://tu-proyecto.up.railway.app/`

### 6. Probar el Despliegue

```bash
# Health check
curl https://tu-proyecto.up.railway.app/health

# Listar modelos disponibles
curl https://tu-proyecto.up.railway.app/models

# Ejemplo de predicción (ajusta los parámetros)
curl -X POST "https://tu-proyecto.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ucp": "Atlantico",
    "n_days": 30,
    "force_retrain": false
  }'
```

## 📁 Archivos de Configuración

### `nixpacks.toml`
Configuración de build para Railway:
- Dependencias del sistema (libgomp1 para LightGBM/XGBoost)
- Comando de inicio del servidor

### `railway.json`
Configuración opcional de Railway:
- Políticas de reinicio
- Comandos de build personalizados

### `.env.example`
Template de variables de entorno (no se despliega, solo referencia)

## 🔧 Solución de Problemas

### El servidor no inicia

1. **Verifica los logs** en Railway Dashboard → Service → Deployments → Logs
2. **Revisa que el puerto sea dinámico**: El código usa `$PORT` automáticamente
3. **Verifica dependencias**: Asegúrate de que `requirements.txt` esté actualizado

### Error: "Module not found"

- Verifica que todos los módulos estén en `src/`
- Asegúrate de que `requirements.txt` incluya todas las dependencias

### Error: "Port already in use"

- Railway maneja el puerto automáticamente
- No configures `PORT` manualmente en Railway (déjalo que Railway lo asigne)

### Modelos no encontrados

- Los modelos deben estar en `models/{UCP}/registry/champion_model.joblib`
- Considera usar **Railway Volumes** para persistir modelos entre despliegues
- O sube los modelos al repositorio (si no son muy grandes)

## 💾 Persistencia de Datos

### Opción 1: Railway Volumes (Recomendado)

Para persistir modelos y datos entre despliegues:

1. En Railway Dashboard → Service → **Volumes**
2. Crea un volumen y monta:
   - `/models` → Para modelos entrenados
   - `/data` → Para datos históricos (opcional)

### Opción 2: Storage Externo

- Usa S3, Google Cloud Storage, o similar
- Modifica el código para cargar modelos desde storage externo

## 🔄 Actualizaciones

Railway despliega automáticamente cuando haces push a la rama conectada:

```bash
git add .
git commit -m "Actualización"
git push origin main
```

Railway detectará el cambio y desplegará automáticamente.

## 📊 Monitoreo

### Logs en Tiempo Real

Railway Dashboard → Service → **Logs** muestra logs en tiempo real.

### Métricas

Railway Dashboard → Service → **Metrics** muestra:
- CPU usage
- Memory usage
- Network traffic

## 🔐 Seguridad

### Variables Sensibles

- ✅ **NUNCA** commitees `.env` o archivos con API keys
- ✅ Usa **Railway Variables** para secretos
- ✅ `.env.example` está en `.gitignore`

### HTTPS

Railway proporciona HTTPS automáticamente en todas las URLs.

## 📝 Notas Importantes

1. **Primera ejecución**: El primer despliegue puede tardar varios minutos (instalación de dependencias)
2. **Cold starts**: Si el servicio está inactivo, puede tardar ~30s en responder
3. **Límites de Railway**: Revisa los límites de tu plan (CPU, RAM, storage)
4. **Modelos grandes**: Si los modelos son >100MB, considera usar Volumes o storage externo

## 🆘 Soporte

Si tienes problemas:
1. Revisa los logs en Railway Dashboard
2. Verifica que todas las variables de entorno estén configuradas
3. Prueba localmente primero: `uvicorn src.api.main:app --reload`

---

**¡Listo!** Tu API debería estar funcionando en producción. 🎉

