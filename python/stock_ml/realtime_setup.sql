-- Run on every shard after the stock_engine collector has been installed.
-- v2 is independent of the earlier experimental Python and SQL baselines.
begin;
create index if not exists stock_prices_received
  on stock_engine.prices(received_at);
create table if not exists stock_engine.ml_v2_models (
  stock_id integer not null references stock_engine.stocks(stock_id),
  style text not null check(style in ('short','long')),
  analysis_minutes integer not null,
  horizon_minutes integer not null,
  trained_at timestamptz not null default now(),
  payload jsonb not null,
  primary key(stock_id,style,analysis_minutes,horizon_minutes)
);
create table if not exists stock_engine.ml_v2_latest (
  stock_id integer not null references stock_engine.stocks(stock_id),
  style text not null check(style in ('short','long')),
  analysis_minutes integer not null,
  horizon_minutes integer not null,
  origin_at timestamptz not null,
  computed_at timestamptz not null default now(),
  payload jsonb not null,
  primary key(stock_id,style,analysis_minutes,horizon_minutes)
);
create table if not exists stock_engine.ml_v2_checks (
  stock_id integer not null references stock_engine.stocks(stock_id),
  style text not null check(style in ('short','long')),
  analysis_minutes integer not null,
  horizon_minutes integer not null,
  origin_at timestamptz not null,
  target_at timestamptz not null,
  issued_at timestamptz not null default now(),
  origin_price numeric not null check(origin_price>0),
  raw_return double precision not null,
  predicted_return double precision not null,
  model_id text not null,
  actual_return double precision,
  checked_at timestamptz,
  primary key(stock_id,style,analysis_minutes,horizon_minutes,origin_at),
  check(target_at>origin_at)
);
create index if not exists stock_ml_v2_due on stock_engine.ml_v2_checks(target_at)
  where checked_at is null;
create table if not exists stock_engine.ml_v2_calibration (
  stock_id integer not null references stock_engine.stocks(stock_id),
  style text not null check(style in ('short','long')),
  analysis_minutes integer not null,
  horizon_minutes integer not null,
  calibrated_at timestamptz not null default now(),
  payload jsonb not null,
  primary key(stock_id,style,analysis_minutes,horizon_minutes)
);
create table if not exists stock_engine.ml_v2_health (
  singleton boolean primary key default true check(singleton),
  heartbeat timestamptz not null default now(),
  last_training timestamptz,
  last_calibration timestamptz,
  processing_ms double precision,
  fresh_combinations integer not null default 0,
  training_ms double precision,
  missed_latency_target boolean not null default false
);
insert into stock_engine.ml_v2_health(singleton) values(true) on conflict do nothing;
alter table stock_engine.ml_v2_models enable row level security;
alter table stock_engine.ml_v2_latest enable row level security;
alter table stock_engine.ml_v2_checks enable row level security;
alter table stock_engine.ml_v2_calibration enable row level security;
alter table stock_engine.ml_v2_health enable row level security;
revoke all on stock_engine.ml_v2_models,stock_engine.ml_v2_latest,
  stock_engine.ml_v2_checks,stock_engine.ml_v2_calibration,
  stock_engine.ml_v2_health from public,anon,authenticated;
commit;
