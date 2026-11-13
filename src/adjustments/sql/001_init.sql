-- Esquema de persistencia del servicio de ajustes por pasos (Fase 1).
-- Idempotente: se puede aplicar varias veces (scripts/apply_adjustments_schema.py).
-- Los ids (uuid) se generan en Python (uuid4); el DEFAULT con gen_random_uuid()
-- (nativo desde Postgres 13) es solo respaldo para inserciones manuales.

CREATE SCHEMA IF NOT EXISTS ajustes;

CREATE TABLE IF NOT EXISTS ajustes.corrida (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ucp             text        NOT NULL,
    semana_inicio   date        NOT NULL,
    fecha_corte     date        NOT NULL,
    fecha_emision   date        NOT NULL,
    modelo_version  text,
    params          jsonb       NOT NULL DEFAULT '{}'::jsonb,
    estado          text        NOT NULL DEFAULT 'borrador'
                    CHECK (estado IN ('borrador', 'aprobada')),
    creado_por      text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    approved_at     timestamptz,
    approved_by     text
);

CREATE INDEX IF NOT EXISTS corrida_ucp_semana_idx
    ON ajustes.corrida (ucp, semana_inicio DESC);
CREATE INDEX IF NOT EXISTS corrida_estado_idx
    ON ajustes.corrida (estado);

CREATE TABLE IF NOT EXISTS ajustes.corrida_paso (
    corrida_id   uuid        NOT NULL REFERENCES ajustes.corrida (id) ON DELETE CASCADE,
    orden        smallint    NOT NULL CHECK (orden >= 0),
    paso         text        NOT NULL,
    estado       text        NOT NULL CHECK (estado IN ('aplicado', 'omitido', 'editado')),
    params       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    curvas       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    diagnostico  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    alertas      jsonb       NOT NULL DEFAULT '[]'::jsonb,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (corrida_id, orden)
);

CREATE TABLE IF NOT EXISTS ajustes.corrida_evento (
    id          bigserial   PRIMARY KEY,
    corrida_id  uuid        NOT NULL REFERENCES ajustes.corrida (id) ON DELETE CASCADE,
    usuario     text,
    accion      text        NOT NULL,
    paso        smallint,
    motivo      text,
    detalle     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    ts          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS corrida_evento_corrida_ts_idx
    ON ajustes.corrida_evento (corrida_id, ts);
