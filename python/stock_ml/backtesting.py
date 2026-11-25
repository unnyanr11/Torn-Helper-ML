"""Leakage-safe walk-forward backtesting for Torn-Helper stock forecasts.

Each fold trains only on observations at or before its decision timestamp.
A trade is entered at the decision close and exited at the exact horizon price.
This is a close-to-close research assumption, not a claim of executable fills.
"""
from __future__ import annotations
import math
import numpy as np
from realtime_model import exact, feature_matrix, make_result, prepare_series, train_range, validate_series

def _metrics(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return dict(count=0,mean=0.0,mae=0.0,rmse=0.0)
    return dict(count=int(len(a)),mean=float(np.mean(a)),mae=float(np.mean(np.abs(a))),rmse=float(np.sqrt(np.mean(a*a))))

def run_backtest(times,prices,*,window_minutes:int,horizon_minutes:int,style='short',
                 fee_bps=10.0,slippage_bps=5.0,initial_capital=10000.0,
                 max_folds=120,min_train_rows=200,trade_on='signal'):
    times,prices=validate_series(times,prices)
    if style not in ('short','long') or window_minutes<15 or horizon_minutes<15:
        raise ValueError('Invalid model style, window or horizon')
    if not 0<=fee_bps<=500 or not 0<=slippage_bps<=500 or initial_capital<=0:
        raise ValueError('Invalid cost or capital assumption')
    stride=15 if style=='short' else 60
    # The first candidate is deliberately after enough history; train_range itself
    # requires >=200 matured examples and applies a horizon-sized purge.
    candidates=times[(times%stride)==0]
    candidates=candidates[candidates>=times[0]+window_minutes+horizon_minutes+min_train_rows*stride]
    if len(candidates)>max_folds:
        candidates=candidates[np.linspace(0,len(candidates)-1,max_folds).astype(int)]
    folds=[]; predictions=[]; trades=[]; equity=float(initial_capital); peak=equity; max_dd=0.0
    cost=(fee_bps+slippage_bps)*2/10000.0
    for origin in candidates:
        # Strict prefix: future observations are not even passed into model fitting.
        end=int(np.searchsorted(times,origin,side='right'))
        train_t,train_p=times[:end],prices[:end]
        if len(train_t)<min_train_rows: continue
        prepared=prepare_series(train_t,train_p)
        fitted=train_range(train_t,train_p,style,window_minutes,[horizon_minutes],prepared).get(horizon_minutes,{})
        if fitted.get('status')!='ready': continue
        x,valid=feature_matrix(train_t,train_p,[int(origin)],window_minutes,prepared)
        if not bool(valid[0]): continue
        result=make_result(fitted,None,x[0],True)
        future=float(exact(times,prices,[int(origin)+horizon_minutes])[0])
        current=float(exact(times,prices,[int(origin)])[0])
        if not math.isfinite(future) or not math.isfinite(current): continue
        actual=future/current-1.0
        predictions.append(dict(origin_minute=int(origin),target_minute=int(origin)+horizon_minutes,
            origin_price=current,actual_price=future,predicted_return=float(result.get('predicted_return',0.0)),
            raw_return=float(result.get('raw_return',0.0)),actual_return=actual,signal=result['signal'],
            model_id=result.get('model_id'),validation_mae=fitted['metrics']['mae']))
        # Signal policy: long-only Buy, otherwise cash. No shorting or compounding
        # on a forecast that is not validated by the model's chronological holdout.
        side=1.0 if result['signal']=='Buy' and fitted['metrics']['approved'] else 0.0
        if trade_on=='long_short' and result['signal']=='Sell' and fitted['metrics']['approved']: side=-1.0
        if side:
            gross=side*actual; net=gross-cost
            before=equity; equity*=max(0.0,1.0+net); peak=max(peak,equity)
            drawdown=equity/peak-1.0; max_dd=min(max_dd,drawdown)
            trades.append(dict(origin_minute=int(origin),exit_minute=int(origin)+horizon_minutes,
                side='long' if side>0 else 'short',entry_price=current,exit_price=future,
                gross_return=gross,net_return=net,capital_before=before,capital_after=equity,
                signal=result['signal']))
    actual=[p['actual_return'] for p in predictions]; errors=[p['actual_return']-p['predicted_return'] for p in predictions]
    trade_returns=[t['net_return'] for t in trades]
    wins=sum(r>0 for r in trade_returns); losses=sum(r<0 for r in trade_returns)
    total_return=equity/initial_capital-1.0
    # Sharpe uses trade returns; annualization is intentionally omitted because folds
    # may be irregular and the backtest uses non-overlapping horizon spacing only
    # when max_folds happens to preserve it. Report a non-annualized ratio.
    sharpe=float(np.mean(trade_returns)/np.std(trade_returns,ddof=1)) if len(trade_returns)>1 and np.std(trade_returns,ddof=1)>0 else None
    return dict(status='completed' if predictions else 'insufficient_history',
        config=dict(style=style,window_minutes=window_minutes,horizon_minutes=horizon_minutes,
            fee_bps=fee_bps,slippage_bps=slippage_bps,initial_capital=initial_capital,
            execution='decision-close to exact-horizon close',trade_on=trade_on),
        metrics=dict(folds=len(predictions),trades=len(trades),total_return=total_return,
            final_capital=equity,win_rate=wins/len(trades) if trades else None,
            wins=wins,losses=losses,max_drawdown=max_dd,sharpe_non_annualized=sharpe,
            prediction_error=_metrics(errors),actual_return=_metrics(actual),
            profit_factor=(sum(r for r in trade_returns if r>0)/abs(sum(r for r in trade_returns if r<0))) if losses else None),
        predictions=predictions,trades=trades)

def calibration_from_backtest(predictions,*,minimum_samples=30):
    """Return a candidate only; production promotion requires a separate forward holdout."""
    if len(predictions)<minimum_samples:
        return dict(active=False,reason='insufficient_backtest_samples',samples=len(predictions),scale=1.0,bias=0.0)
    raw=np.asarray([p['raw_return'] for p in predictions],dtype=float)
    actual=np.asarray([p['actual_return'] for p in predictions],dtype=float)
    split=int(len(raw)*.7)
    if len(raw)-split<10: return dict(active=False,reason='insufficient_holdout',samples=len(raw),scale=1.0,bias=0.0)
    denom=float(np.dot(raw[:split],raw[:split]))
    scale=float(np.clip(np.dot(raw[:split],actual[:split])/(denom+1e-12),0.0,1.5))
    bias=float(np.clip(np.median(actual[:split]-scale*raw[:split]),-.02,.02))
    baseline=float(np.mean(np.abs(actual[split:]-raw[split:])))
    corrected=float(np.mean(np.abs(actual[split:]-(scale*raw[split:]+bias))))
    approved=corrected<baseline
    return dict(active=approved,reason='holdout_improved_mae' if approved else 'holdout_not_improved',
        samples=len(raw),scale=scale if approved else 1.0,bias=bias if approved else 0.0,
        holdout_mae_raw=baseline,holdout_mae_corrected=corrected,holdout_start=int(predictions[split]['origin_minute']))
