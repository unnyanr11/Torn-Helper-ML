-- Install in each stock shard. This is additive and does not alter RLS on existing tables.
begin;
create table if not exists stock_engine.ml_backtest_runs (
  run_id uuid primary key,
  stock_id integer not null references stock_engine.stocks(stock_id),
  created_at timestamptz not null default now(),
  status text not null check(status in ('running','completed','insufficient_history','failed')),
  config jsonb not null,
  metrics jsonb,
  source_commit text
);
create table if not exists stock_engine.ml_backtest_predictions (
  run_id uuid not null references stock_engine.ml_backtest_runs(run_id) on delete cascade,
  origin_at timestamptz not null,
  target_at timestamptz not null,
  origin_price numeric not null,
  actual_price numeric not null,
  raw_return double precision not null,
  predicted_return double precision not null,
  actual_return double precision not null,
  signal text not null,
  model_id text,
  primary key(run_id,origin_at)
);
create table if not exists stock_engine.ml_backtest_trades (
  run_id uuid not null references stock_engine.ml_backtest_runs(run_id) on delete cascade,
  origin_at timestamptz not null,
  exit_at timestamptz not null,
  side text not null check(side in ('long','short')),
  entry_price numeric not null,
  exit_price numeric not null,
  gross_return double precision not null,
  net_return double precision not null,
  capital_before numeric not null,
  capital_after numeric not null,
  primary key(run_id,origin_at)
);
create table if not exists stock_engine.ml_backtest_calibration_candidates (
  run_id uuid primary key references stock_engine.ml_backtest_runs(run_id) on delete cascade,
  created_at timestamptz not null default now(),
  candidate jsonb not null,
  promoted boolean not null default false,
  promotion_reason text
);
create index if not exists ml_backtest_runs_stock_created on stock_engine.ml_backtest_runs(stock_id,created_at desc);
alter table stock_engine.ml_backtest_runs enable row level security;
alter table stock_engine.ml_backtest_predictions enable row level security;
alter table stock_engine.ml_backtest_trades enable row level security;
alter table stock_engine.ml_backtest_calibration_candidates enable row level security;
revoke all on stock_engine.ml_backtest_runs,stock_engine.ml_backtest_predictions,
 stock_engine.ml_backtest_trades,stock_engine.ml_backtest_calibration_candidates from public,anon,authenticated;
commit;
