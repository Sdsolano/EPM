# A4 - Sub-plan (graficas en curvas y lenguaje de negocio en vistas existentes)
Zona: mesa_ajustes/components/{paso,resumen,analogos}/*, DetallePaso.tsx, ResumenSemanal.tsx, BuscadorAnalogos.tsx (+ css). No tocar A3 (MesaAjustes, PanelSeleccion, StepperPasos, resumen ejecutivo, src/demo).
Pasos:
1. components/paso/guiaPaso.ts: guia por paso (que hace / que debes hacer, variantes por estado) + textos de parametros con nombre de negocio.
2. GraficaPaso: semana = CurvaSemana168 (anterior vs resultado, real, banda solo paso control si hay insights), dia = CurvaDia24 modo "paso"; selector y tabla accesible (la del chart) se conservan. Cero Bar/BarChart.
3. EnergiaSemana: nuevo CurvaEnergiaSemana (CurvaEnergiaDiaria) + brecha por dia en ResumenSemanal.
4. GraficaAnalogos: lineas suaves grises + promedio/objetivo resaltados.
5. Lenguaje: EncabezadoPaso, AlertasPaso, Diagnosticos, AccionesPaso, EditorCurvas, AlertasVigentes, Aprobacion, Historial, Filtros/Panel/Lista analogos, DetallePaso (estados). traducirAlerta en vez de codigos; AyudaTermino junto a terminos.
6. Verificacion: tsc diff, eslint, vite build, build:demo, navegador mock con capturas 1280/768/375, better-interface.
Estado vivo: A4/REPORTE.md al final.
