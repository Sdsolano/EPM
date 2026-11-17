-- Versionado, concurrencia optimista, una aprobada vigente por semana y pasos por nombre.
-- Se aplica sobre el esquema de 001_init.sql. Idempotente (scripts/apply_adjustments_schema.py).

-- ---------------------------------------------------------------------------
-- corrida: versión, extras, estado `reemplazada`
-- ---------------------------------------------------------------------------
ALTER TABLE ajustes.corrida ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;
ALTER TABLE ajustes.corrida ADD COLUMN IF NOT EXISTS extras jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE ajustes.corrida ADD COLUMN IF NOT EXISTS reemplazada_por uuid
    REFERENCES ajustes.corrida (id);
ALTER TABLE ajustes.corrida ADD COLUMN IF NOT EXISTS reemplazada_at timestamptz;

ALTER TABLE ajustes.corrida DROP CONSTRAINT IF EXISTS corrida_estado_check;
ALTER TABLE ajustes.corrida ADD CONSTRAINT corrida_estado_check
    CHECK (estado IN ('borrador', 'aprobada', 'reemplazada'));

ALTER TABLE ajustes.corrida DROP CONSTRAINT IF EXISTS corrida_version_check;
ALTER TABLE ajustes.corrida ADD CONSTRAINT corrida_version_check CHECK (version >= 1);

-- Datos previos a 002: si hubiera varias aprobadas para la misma (ucp, semana), la más
-- reciente queda vigente y las demás pasan a `reemplazada` (necesario para el índice único).
WITH ranked AS (
    SELECT id, ucp, semana_inicio,
           first_value(id) OVER w AS vigente,
           row_number() OVER w AS rn
    FROM ajustes.corrida
    WHERE estado = 'aprobada'
    WINDOW w AS (PARTITION BY ucp, semana_inicio
                 ORDER BY approved_at DESC NULLS LAST, created_at DESC)
)
UPDATE ajustes.corrida c
SET estado = 'reemplazada', reemplazada_por = r.vigente, reemplazada_at = now()
FROM ranked r
WHERE c.id = r.id AND r.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS corrida_aprobada_unica_idx
    ON ajustes.corrida (ucp, semana_inicio) WHERE estado = 'aprobada';

-- ---------------------------------------------------------------------------
-- corrida_paso: PK por nombre de paso; `orden` queda como columna informativa
-- ---------------------------------------------------------------------------
ALTER TABLE ajustes.corrida_paso ADD COLUMN IF NOT EXISTS curvas_calculadas jsonb
    NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.key_column_usage
        WHERE table_schema = 'ajustes' AND table_name = 'corrida_paso'
          AND constraint_name = 'corrida_paso_pkey' AND column_name = 'paso'
    ) THEN
        ALTER TABLE ajustes.corrida_paso DROP CONSTRAINT IF EXISTS corrida_paso_pkey;
        ALTER TABLE ajustes.corrida_paso ADD CONSTRAINT corrida_paso_pkey PRIMARY KEY (corrida_id, paso);
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- corrida_evento.paso: de orden (smallint) a nombre del paso (text)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'ajustes' AND table_name = 'corrida_evento'
          AND column_name = 'paso' AND data_type = 'smallint'
    ) THEN
        ALTER TABLE ajustes.corrida_evento ALTER COLUMN paso TYPE text USING (
            CASE paso
                WHEN 0 THEN 'base' WHEN 1 THEN 'referencia' WHEN 2 THEN 'nivel'
                WHEN 3 THEN 'forma' WHEN 4 THEN 'control'
                ELSE paso::text
            END
        );
    END IF;
END $$;

ALTER TABLE ajustes.corrida_evento ADD COLUMN IF NOT EXISTS version integer;

-- ---------------------------------------------------------------------------
-- corrida_version: snapshot completo de los pasos en cada versión (historial)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ajustes.corrida_version (
    corrida_id  uuid        NOT NULL REFERENCES ajustes.corrida (id) ON DELETE CASCADE,
    version     integer     NOT NULL CHECK (version >= 1),
    records     jsonb       NOT NULL,
    creado_por  text,
    motivo      text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (corrida_id, version)
);

-- Corridas creadas antes de 002: su estado actual queda como snapshot de su versión vigente.
INSERT INTO ajustes.corrida_version (corrida_id, version, records, creado_por, motivo)
SELECT c.id, c.version,
       COALESCE((
           SELECT jsonb_agg(jsonb_build_object(
                      'orden', p.orden, 'paso', p.paso, 'estado', p.estado, 'params', p.params,
                      'result', jsonb_build_object(
                          'curves', p.curvas, 'diagnostics', p.diagnostico, 'alerts', p.alertas,
                          'curvas_calculadas', p.curvas_calculadas))
                      ORDER BY p.orden)
           FROM ajustes.corrida_paso p WHERE p.corrida_id = c.id
       ), '[]'::jsonb),
       c.creado_por, 'migracion 002'
FROM ajustes.corrida c
WHERE NOT EXISTS (SELECT 1 FROM ajustes.corrida_version v WHERE v.corrida_id = c.id);
