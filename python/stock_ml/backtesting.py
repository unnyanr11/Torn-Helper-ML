"""Leakage-safe walk-forward comparison of forecasting model families."""
from __future__ import annotations
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from realtime_model import exact, feature_matrix, prepare_series, validate_series, train_range, make_result

MODEL_FAMILIES=('ridge','elastic_net','random_forest','extra_trees','hist_gradient_boosting','ensemble')
BASE_FAMILIES=MODEL_FAMILIES[:-1]

def _fit_predict(family, x_train, y_train, x_eval):
    if family in ('ridge','elastic_net'):
        scaler=StandardScaler().fit(x_train)
        z_train=scaler.transform(x_train); z_eval=scaler.transform(x_eval)
        if family=='ridge':
            model=Ridge(alpha=100.0)
        else:
            model=ElasticNet(alpha=0.0001,l1_ratio=.25,max_iter=3000)
        model.fit(z_train,y_train)
        return model.predict(z_eval)
    if family=='random_forest':
        model=RandomForestRegressor(n_estimators=60,max_depth=6,min_samples_leaf=8,
            max_features=.8,n_jobs=1,random_state=17)
    elif family=='extra_trees':
        model=ExtraTreesRegressor(n_estimators=60,max_depth=6,min_samples_leaf=8,
            max_features=.8,n_jobs=1,random_state=17)
    elif family=='hist_gradient_boosting':
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,
            l2_regularization=2.0,learning_rate=.05,random_state=17)
    else:
        raise ValueError('Unknown model family: '+str(family))
    model.fit(x_train,y_train)
    return model.predict(x_eval)

def _family_predict(family,x_train,y_train,x_eval):
    if family!='ensemble':
        return _fit_predict(family,x_train,y_train,x_eval)
    predictions=np.vstack([_fit_predict(name,x_train,y_train,x_eval) for name in BASE_FAMILIES])
    return np.median(predictions,axis=0)

def _metrics(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return dict(count=0,mean=0.0,mae=0.0,rmse=0.0)
    return dict(count=int(len(a)),mean=float(np.mean(a)),mae=float(np.mean(np.abs(a))),
                rmse=float(np.sqrt(np.mean(a*a))))

PROFIT_THRESHOLD_GRID=(0.0,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02)

def calibrate_profit_threshold(predictions,*,estimated_trading_cost,
                               thresholds=PROFIT_THRESHOLD_GRID,
                               minimum_calibration_samples=20,
                               minimum_calibration_trades=5,
                               minimum_holdout_samples=6):
    """Choose a minimum expected NET edge on an earlier chronological slice.

    The final chronological slice is never used to select the threshold. It is
    used once to report untouched holdout performance. Thresholds are minimum
    net-edge hurdles; there is deliberately no maximum-profit ceiling.
    """
    rows=sorted(predictions,key=lambda p:int(p['origin_minute']))
    split=int(len(rows)*0.70)
    calibration,holdout=rows[:split],rows[split:]
    base=dict(method='chronological_70_30_threshold_calibration',
        thresholds_tested=list(thresholds),calibration_samples=len(calibration),
        holdout_samples=len(holdout),estimated_trading_cost=float(estimated_trading_cost),
        selected_minimum_net_return=None,active=False)
    if len(calibration)<minimum_calibration_samples or len(holdout)<minimum_holdout_samples:
        return dict(**base,status='insufficient_samples',
            reason='Need more chronological out-of-sample predictions to calibrate and independently test a threshold')
    candidates=[]
    cost=float(estimated_trading_cost)
    for threshold in thresholds:
        selected=[]
        for row in calibration:
            if not row.get('approved',False):
                continue
            edge=abs(float(row['predicted_return']))-float(row.get('uncertainty',0.0))-cost
            if edge >= float(threshold):
                direction=1.0 if float(row['predicted_return'])>0 else -1.0
                if direction<0 and row.get('trade_on','signal')!='long_short':
                    continue
                selected.append(direction*float(row['actual_return'])-cost)
        if len(selected)<minimum_calibration_trades:
            continue
        values=np.asarray(selected,dtype=float)
        wins=values[values>0]; losses=values[values<0]
        pf=float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else (1000.0 if len(wins) else 0.0)
        equity=np.cumprod(1.0+values)
        peak=np.maximum.accumulate(np.r_[1.0,equity])[1:]
        drawdown=float(np.min(equity/peak-1.0)) if len(equity) else 0.0
        mean=float(values.mean())
        candidates.append(dict(minimum_net_return=float(threshold),trades=len(values),
            mean_net_return=mean,win_rate=float(np.mean(values>0)),profit_factor=pf,
            max_drawdown=drawdown,total_return=float(equity[-1]-1.0),
            score=(mean/max(abs(drawdown),0.005)),eligible=bool(mean>0 and pf>1.0)))
    eligible=[x for x in candidates if x['eligible']]
    if not eligible:
        return dict(**base,status='no_positive_calibration_edge',
            reason='No tested threshold produced sufficient positive after-cost expectancy',
            calibration_candidates=candidates)
    best=max(eligible,key=lambda x:(x['score'],x['mean_net_return'],x['trades']))
    threshold=best['minimum_net_return']
    test_values=[]
    for row in holdout:
        if not row.get('approved',False):
            continue
        edge=abs(float(row['predicted_return']))-float(row.get('uncertainty',0.0))-cost
        if edge>=threshold:
            direction=1.0 if float(row['predicted_return'])>0 else -1.0
            if direction<0 and row.get('trade_on','signal')!='long_short':
                continue
            test_values.append(direction*float(row['actual_return'])-cost)
    if len(test_values):
        values=np.asarray(test_values,dtype=float)
        wins=values[values>0]; losses=values[values<0]
        pf=float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else (1000.0 if len(wins) else 0.0)
        equity=np.cumprod(1.0+values)
        peak=np.maximum.accumulate(np.r_[1.0,equity])[1:]
        test_metrics=dict(trades=len(values),mean_net_return=float(values.mean()),
            win_rate=float(np.mean(values>0)),profit_factor=pf,
            max_drawdown=float(np.min(equity/peak-1.0)),total_return=float(equity[-1]-1.0))
    else:
        test_metrics=dict(trades=0,mean_net_return=None,win_rate=None,profit_factor=None,
            max_drawdown=0.0,total_return=0.0)
    return dict(**base,status='calibrated',active=True,
        selected_minimum_net_return=threshold,calibration_metrics=best,
        calibration_candidates=candidates,untouched_holdout=test_metrics,
        holdout_start=int(holdout[0]['origin_minute']))

def run_backtest(times,prices,*,window_minutes:int,horizon_minutes:int,style='short',
                 fee_bps=10.0,slippage_bps=5.0,initial_capital=10000.0,
                 max_folds=120,min_train_rows=200,trade_on='signal',model_family='ridge',evaluation_start_minute=None,
                 minimum_net_return=None):
    times,prices=validate_series(times,prices)
    if style != 'short' or window_minutes<15 or horizon_minutes<15 or model_family not in MODEL_FAMILIES:
        raise ValueError('Invalid model style, window, horizon or model family')
    if not 0<=fee_bps<=500 or not 0<=slippage_bps<=500 or initial_capital<=0:
        raise ValueError('Invalid cost or capital assumption')
    stride=15
    candidates=times[(times%stride)==0]
    candidates=candidates[candidates>=times[0]+window_minutes+horizon_minutes+min_train_rows*stride]
    if evaluation_start_minute is not None:
        candidates=candidates[candidates>=int(evaluation_start_minute)]
    if len(candidates)>max_folds:
        candidates=candidates[np.linspace(0,len(candidates)-1,max_folds).astype(int)]
    non_overlapping=[]; last_exit=-1
    for candidate in candidates:
        if int(candidate)>=last_exit:
            non_overlapping.append(int(candidate)); last_exit=int(candidate)+horizon_minutes
    predictions=[]; trades=[]; equity=float(initial_capital); peak=equity; max_dd=0.0
    cost=(fee_bps + 2.0*slippage_bps)/10000.0
    # minimum_net_return is retained for API compatibility only. Research calibrates
    # its operating threshold from chronological predictions instead of imposing it.
    configured_live_threshold = 0.002 if minimum_net_return is None else float(minimum_net_return)
    if not math.isfinite(configured_live_threshold) or configured_live_threshold < 0:
        raise ValueError('minimum_net_return must be finite and non-negative')
    for origin in non_overlapping:
        end=int(np.searchsorted(times,origin,side='right'))
        train_t,train_p=times[:end],prices[:end]
        if len(train_t)<min_train_rows: continue
        prepared=prepare_series(train_t,train_p)
        x,valid=feature_matrix(train_t,train_p,[origin],window_minutes,prepared)
        if not bool(valid[0]): continue
        if model_family=='ridge':
            fitted=train_range(train_t,train_p,style,window_minutes,[horizon_minutes],prepared).get(horizon_minutes,{})
            if fitted.get('status')!='ready': continue
            result=make_result(fitted,None,x[0],True,minimum_net_return=0.0,estimated_trading_cost=cost)
            validation_mae=float(fitted['metrics']['mae'])
            baseline=float(fitted['metrics']['baseline_mae'])
            approved=bool(fitted['metrics']['approved'])
            final_prediction=float(result.get('predicted_return',0.0))
            uncertainty=float(result.get('error_q90',validation_mae) or validation_mae)
        else:
            stride_origins=train_t[(train_t%stride)==0]
            features,feature_ok=feature_matrix(train_t,train_p,stride_origins,window_minutes,prepared)
            targets=exact(train_t,train_p,stride_origins+horizon_minutes)
            matured=feature_ok & np.isfinite(targets)
            origins=stride_origins[matured]; features=features[matured]
            if len(origins)<240: continue
            y=targets[matured]/exact(train_t,train_p,origins)-1.0
            boundary=origins[int(len(origins)*.75)]
            fit_mask=origins+horizon_minutes<boundary
            val_mask=origins>=boundary
            if fit_mask.sum()<100 or val_mask.sum()<30: continue
            val_y=y[val_mask]
            val_pred=_family_predict(model_family,features[fit_mask],y[fit_mask],features[val_mask])
            validation_mae=float(np.mean(np.abs(val_y-val_pred)))
            baseline=float(np.mean(np.abs(val_y)))
            approved=bool(baseline>0 and validation_mae<baseline*.98)
            final_prediction=float(_family_predict(model_family,features,y,x)[0])
            uncertainty=validation_mae
        future=float(exact(times,prices,[origin+horizon_minutes])[0])
        current=float(exact(times,prices,[origin])[0])
        if not math.isfinite(future) or not math.isfinite(current): continue
        actual=future/current-1.0
        # Record direction/confidence without a preset profit hurdle. The threshold
        # calibration below decides which predictions qualify for trading.
        signal=('Buy' if approved and final_prediction>uncertainty else
                'Sell' if approved and final_prediction < -uncertainty else 'Wait')
        predictions.append(dict(origin_minute=origin,target_minute=origin+horizon_minutes,
            origin_price=current,actual_price=future,predicted_return=final_prediction,
            raw_return=final_prediction,actual_return=actual,
            predicted_price=current*(1.0+final_prediction),signal=signal,
            model_id=model_family,model_family=model_family,validation_mae=validation_mae,
            validation_baseline_mae=baseline,uncertainty=uncertainty,approved=approved,
            trade_on=trade_on))
    threshold_calibration=calibrate_profit_threshold(predictions,estimated_trading_cost=cost)
    selected_threshold=threshold_calibration.get('selected_minimum_net_return')
    if threshold_calibration.get('active'):
        holdout_start=int(threshold_calibration['holdout_start'])
        holdout=[p for p in predictions if int(p['origin_minute'])>=holdout_start]
        equity=float(initial_capital); peak=equity; max_dd=0.0
        for row in holdout:
            edge=abs(float(row['predicted_return']))-float(row['uncertainty'])-cost
            if edge < selected_threshold: continue
            side=1.0 if row['predicted_return']>0 else (-1.0 if trade_on=='long_short' else 0.0)
            if not side: continue
            net=side*float(row['actual_return'])-cost; before=equity
            equity*=max(0.0,1.0+net); peak=max(peak,equity); max_dd=min(max_dd,equity/peak-1.0)
            trades.append(dict(origin_minute=row['origin_minute'],exit_minute=row['target_minute'],
                side='long' if side>0 else 'short',entry_price=row['origin_price'],
                exit_price=row['actual_price'],gross_return=side*row['actual_return'],
                net_return=net,capital_before=before,capital_after=equity))
    else:
        equity=float(initial_capital); max_dd=0.0
    actuals=[p['actual_return'] for p in predictions]
    errors=[p['actual_return']-p['predicted_return'] for p in predictions]
    trade_returns=[t['net_return'] for t in trades]
    price_errors=[p['predicted_price']-p['actual_price'] for p in predictions]
    directional=[(p['predicted_return']>=0)==(p['actual_return']>=0) for p in predictions]
    wins=sum(r>0 for r in trade_returns); losses=sum(r<0 for r in trade_returns)
    sharpe=float(np.mean(trade_returns)/np.std(trade_returns,ddof=1)) if len(trade_returns)>1 and np.std(trade_returns,ddof=1)>0 else None
    return dict(status='completed' if predictions else 'insufficient_history',
        config=dict(style=style,window_minutes=window_minutes,horizon_minutes=horizon_minutes,
            model_family=model_family,evaluation_start_minute=(int(evaluation_start_minute) if evaluation_start_minute is not None else None),
            fee_bps=fee_bps,slippage_bps=slippage_bps,
            configured_live_minimum_net_return=configured_live_threshold,
            threshold_calibration=threshold_calibration,
            initial_capital=initial_capital,execution='decision-close to exact-horizon close',trade_on=trade_on),
        metrics=dict(folds=len(predictions),trades=len(trades),total_return=equity/initial_capital-1.0,
            final_capital=equity,win_rate=wins/len(trades) if trades else None,wins=wins,losses=losses,
            max_drawdown=max_dd,sharpe_non_annualized=sharpe,prediction_error=_metrics(errors),
            actual_return=_metrics(actuals),profit_factor=(sum(r for r in trade_returns if r>0)/
            abs(sum(r for r in trade_returns if r<0))) if losses else (1000.0 if wins else None)),
        predictions=predictions,trades=trades)

def calibration_from_backtest(predictions,*,minimum_samples=30):
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
        holdout_mae_raw=baseline,holdout_mae_corrected=corrected,
        holdout_start=int(predictions[split]['origin_minute']))
