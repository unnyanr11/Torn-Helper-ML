"""Vectorized causal features and nested walk-forward Ridge model selection."""
import hashlib
import json
import math
import os
import numpy as np

VERSION = 'minute-adaptive-linear-v1'
SHORT_RANGES = (15, 30, 60, 120, 180, 360, 720, 1080, 1440, 4320, 10080, 43200)
SHORT_HORIZONS = tuple(h * 60 for h in (1, 3, 6, 12, 18, 24))
# This repository owns short-term analysis only. Long-term analysis lives in
# the dedicated LongTerm-Torn-Helper-ML repository.
COMBINATIONS = tuple(('short', r, h)
                     for r in SHORT_RANGES for h in SHORT_HORIZONS)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
# Short-term defaults: 0.2% minimum NET return after configured costs.
# Environment overrides configure worker/backtest defaults; no profit ceiling exists.
ESTIMATED_TRADING_COST = float(os.getenv('SHORT_ESTIMATED_TRADING_COST', '0.002'))
MIN_EXPECTED_NET_RETURN = float(os.getenv('SHORT_MIN_EXPECTED_NET_RETURN', '0.002'))
MIN_PREDICTED_GROSS_RETURN = ESTIMATED_TRADING_COST + MIN_EXPECTED_NET_RETURN

if (not math.isfinite(ESTIMATED_TRADING_COST) or not math.isfinite(MIN_EXPECTED_NET_RETURN)
        or ESTIMATED_TRADING_COST < 0 or MIN_EXPECTED_NET_RETURN < 0):
    raise ValueError('Short-term trading cost and minimum net return must be non-negative')

def validate_series(times, prices):
    times=np.asarray(times,dtype=np.int64); prices=np.asarray(prices,dtype=np.float64)
    if len(times)!=len(prices) or not len(times) or np.any(np.diff(times)<=0) or np.any(prices<=0) or not np.all(np.isfinite(prices)):
        raise ValueError('Invalid minute observations')
    return times,prices

def exact(times,prices,queries):
    queries=np.asarray(queries,dtype=np.int64); idx=np.searchsorted(times,queries); clipped=np.minimum(idx,len(times)-1)
    return np.where((idx<len(times))&(times[clipped]==queries),prices[clipped],np.nan)

def prepare_series(times,prices):
    times,prices=validate_series(times,prices); logs=np.log(prices)
    adjacent=np.r_[0.,np.where(np.diff(times)==1,np.diff(logs)**2,0.)]; count=np.r_[0,(np.diff(times)==1).astype(int)]
    return dict(times=times,prices=prices,logs=logs,sums=np.r_[0.,np.cumsum(logs)],squares=np.r_[0.,np.cumsum(logs*logs)],rv=np.r_[0.,np.cumsum(adjacent)],rc=np.r_[0,np.cumsum(count)])

def feature_matrix(times,prices,origins,window,prepared=None):
    data=prepared if prepared is not None else prepare_series(times,prices); times,prices=data['times'],data['prices']
    origins=np.asarray(origins,dtype=np.int64); left=np.searchsorted(times,origins-window,side='right'); right=np.searchsorted(times,origins,side='right'); count=right-left
    sums,squares=data['sums'],data['squares']; mean=(sums[right]-sums[left])/np.maximum(count,1)
    variance=np.maximum(0,(squares[right]-squares[left])/np.maximum(count,1)-mean*mean); current=exact(times,prices,origins)
    lags=[min(window-1,lag) for lag in (1,5,15,60,240,1440)]; changes=[current/exact(times,prices,origins-lag)-1 for lag in lags]
    first_return=np.minimum(left+1,right); rv,rc=data['rv'],data['rc']; volatility=np.sqrt((rv[right]-rv[first_return])/np.maximum(1,rc[right]-rc[first_return]))
    x=np.column_stack([*changes,np.log(current)-mean,np.sqrt(variance),volatility,count/window])
    first=times[np.minimum(left,len(times)-1)]; valid=(count>=np.ceil(window*.95))&(first<=origins-window+max(1,window//100))&np.all(np.isfinite(x),axis=1)
    return x,valid

def infer(parameters,x):
    return ((np.asarray(x)-np.asarray(parameters['mean']))/np.asarray(parameters['scale']))@np.asarray(parameters['coef'])+parameters['intercept']

def train_range(times, prices, style, window, horizons, prepared=None, observed_mask=None):
    """Select Ridge, ElasticNet or their linear blend on chronological validation.

    Export remains one standardized linear model, preserving the browser/Python
    inference contract and the existing forecast response shape.
    """
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    times, prices = validate_series(times, prices)
    if style != 'short':
        raise ValueError('Only short-term analysis is supported in this repository')
    stride = 15
    origins = times[times % stride == 0]
    if observed_mask is not None:
        observed_mask = np.asarray(observed_mask, dtype=bool)
        if len(observed_mask) != len(times):
            raise ValueError('Observed-price mask must match the minute series')
        origins = origins[observed_mask[np.searchsorted(times, origins)]]
    if len(origins) > 3000:
        origins = origins[np.linspace(0, len(origins) - 1, 3000).astype(int)]
    x, complete = feature_matrix(times, prices, origins, window, prepared)
    origins, x = origins[complete], x[complete]
    result = {}

    def fit_candidate(family, alpha, design, target):
        if family == 'elastic_net':
            estimator = ElasticNet(alpha=alpha, l1_ratio=.25, max_iter=5000,
                                   selection='cyclic', tol=1e-5)
        else:
            estimator = Ridge(alpha=alpha)
        estimator.fit(design, target)
        return estimator

    for horizon in horizons:
        future = exact(times, prices, origins + horizon)
        matured = np.isfinite(future)
        if observed_mask is not None:
            target_idx = np.searchsorted(times, origins + horizon)
            target_exists = target_idx < len(times)
            target_real = np.zeros(len(target_idx), dtype=bool)
            target_real[target_exists] = observed_mask[target_idx[target_exists]]
            matured &= target_real
        t, design = origins[matured], x[matured]
        y = future[matured] / exact(times, prices, t) - 1
        if len(t) < 240:
            result[horizon] = {'status': 'insufficient_history',
                               'reason': 'Need 240 matured examples for nested model selection'}
            continue
        boundary_select = t[int(len(t) * .60)]
        boundary_test = t[int(len(t) * .75)]
        train = t + horizon < boundary_select
        select = (t >= boundary_select) & (t + horizon < boundary_test)
        test = t >= boundary_test
        if train.sum() < 100 or select.sum() < 30 or test.sum() < 50:
            result[horizon] = {'status': 'insufficient_history',
                               'reason': 'Not enough rows after nested horizon purges'}
            continue

        selection_scaler = StandardScaler().fit(design[train])
        z_train = selection_scaler.transform(design[train])
        z_select = selection_scaler.transform(design[select])
        candidates = [('ridge', alpha) for alpha in RIDGE_ALPHAS]
        candidates += [('elastic_net', alpha) for alpha in (.00005, .0001, .0005, .001)]
        fitted, scores = {}, {}
        for family, alpha in candidates:
            key = (family, alpha)
            fitted[key] = fit_candidate(family, alpha, z_train, y[train])
            scores[key] = float(np.mean(np.abs(y[select] - fitted[key].predict(z_select))))

        best_ridge = min((k for k in fitted if k[0] == 'ridge'),
                         key=lambda k: (scores[k], k[1]))
        best_elastic = min((k for k in fitted if k[0] == 'elastic_net'),
                           key=lambda k: (scores[k], k[1]))
        blend_key = ('ridge_elastic_blend', 0.0)
        scores[blend_key] = float(np.mean(np.abs(y[select] -
            (fitted[best_ridge].predict(z_select) + fitted[best_elastic].predict(z_select)) / 2.0)))
        _, selected_family, selected_alpha = min(
            ((score, key[0], key[1]) for key, score in scores.items()),
            key=lambda item: (item[0], item[1], item[2]))

        fit_before_test = t + horizon < boundary_test
        final_scaler = StandardScaler().fit(design[fit_before_test])
        z_fit = final_scaler.transform(design[fit_before_test])
        z_test = final_scaler.transform(design[test])
        if selected_family == 'ridge_elastic_blend':
            ridge_model = fit_candidate('ridge', best_ridge[1], z_fit, y[fit_before_test])
            elastic_model = fit_candidate('elastic_net', best_elastic[1], z_fit, y[fit_before_test])
            test_prediction = (ridge_model.predict(z_test) + elastic_model.predict(z_test)) / 2.0
        else:
            final_model = fit_candidate(selected_family, selected_alpha, z_fit, y[fit_before_test])
            test_prediction = final_model.predict(z_test)
        error = y[test] - test_prediction
        mae, baseline = float(np.mean(np.abs(error))), float(np.mean(np.abs(y[test])))
        q90 = float(np.quantile(np.abs(error), .9))
        metrics = dict(mae=mae, baseline_mae=baseline, error_q90=q90,
                       training_rows=int(train.sum()), selection_rows=int(select.sum()),
                       validation_rows=int(test.sum()), selected_family=selected_family,
                       selected_alpha=float(selected_alpha), selection_mae=float(min(scores.values())),
                       last_training_target=int((t[fit_before_test] + horizon).max()),
                       first_validation_origin=int(boundary_test),
                       approved=bool(baseline > 0 and mae < baseline * .98))

        live_scaler = StandardScaler().fit(design)
        z_all = live_scaler.transform(design)
        if selected_family == 'ridge_elastic_blend':
            ridge_live = fit_candidate('ridge', best_ridge[1], z_all, y)
            elastic_live = fit_candidate('elastic_net', best_elastic[1], z_all, y)
            coef = (ridge_live.coef_ + elastic_live.coef_) / 2.0
            intercept = float((ridge_live.intercept_ + elastic_live.intercept_) / 2.0)
        else:
            live_model = fit_candidate(selected_family, selected_alpha, z_all, y)
            coef, intercept = live_model.coef_, float(live_model.intercept_)
        parameters = dict(mean=live_scaler.mean_.tolist(), scale=live_scaler.scale_.tolist(),
                          coef=np.asarray(coef).tolist(), intercept=intercept)
        model_id = hashlib.sha256(json.dumps(
            dict(parameters=parameters, family=selected_family, alpha=selected_alpha),
            sort_keys=True).encode()).hexdigest()[:20]
        result[horizon] = dict(status='ready', parameters=parameters, metrics=metrics,
                               model_id=model_id, trained_through=int(times[-1]),
                               model_family=selected_family)
    return result

def baseline_forecast(times, prices, origin, window, horizon):
    """Unvalidated trend fallback: forecasts from the selected real-price window only."""
    times, prices = validate_series(times, prices)
    left = np.searchsorted(times, origin - window, side='right')
    right = np.searchsorted(times, origin, side='right')
    sample_t, sample_p = times[left:right], prices[left:right]
    required = int(np.ceil(window * .95))
    if (len(sample_t) < required or not len(sample_t) or
            sample_t[0] > origin - window + max(1, window // 100)):
        return dict(status='insufficient_history', signal='Wait',
                    reason='Selected input window is incomplete')
    x = (sample_t - origin).astype(np.float64)
    y = np.log(sample_p)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return dict(status='unavailable', signal='Wait',
                    reason='Not enough distinct timestamps for baseline forecast')
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    residual_q90 = float(np.quantile(np.abs(residuals), .90))
    predicted_return = float(np.expm1(np.clip(slope * horizon, -0.405465, 0.405465)))
    uncertainty = float(min(.5, max(.003, residual_q90 * np.sqrt(1 + horizon / max(window, 1)))))
    if not np.isfinite(predicted_return) or not np.isfinite(uncertainty):
        return dict(status='unavailable', signal='Wait',
                    reason='Baseline forecast is numerically unstable')
    return dict(status='ready', signal='Wait', predicted_return=predicted_return,
                raw_return=predicted_return, error_q90=uncertainty,
                forecast_method='unvalidated_linear_trend',
                model_id=None, calibrated=False, calibration_samples=0,
                training_status='insufficient_history',
                training_reason='A validated ML model is not available; showing a trend-only fallback',
                validation=None)

def calibrate(rows):
    if len(rows)<20: return dict(scale=1.,bias=0.,error_q90=None,samples=len(rows),active=False)
    data=np.asarray(rows,dtype=float); raw,actual=data[:,0],data[:,1]; split=int(len(data)*.7)
    if len(data)-split<6: return dict(scale=1.,bias=0.,error_q90=None,samples=len(rows),active=False)
    scale=float(np.clip(np.dot(raw[:split],actual[:split])/(np.dot(raw[:split],raw[:split])+1e-9),0,1))
    bias=float(np.clip(np.median(actual[:split]-scale*raw[:split]),-.02,.02)); corrected=raw[split:]*scale+bias
    approved=bool(np.mean(abs(actual[split:]-corrected))<np.mean(abs(actual[split:]-raw[split:])))
    q90=float(np.quantile(abs(actual[split:]-corrected),.9))
    return dict(scale=scale if approved else 1.,bias=bias if approved else 0.,error_q90=q90 if approved else None,samples=len(rows),active=approved)

def make_result(model,calibration,x,coverage_ok,*,minimum_net_return=None,estimated_trading_cost=None):
    if not coverage_ok: return dict(status='insufficient_history',signal='Wait',reason='Selected input window is incomplete')
    if not model or model.get('status')!='ready': return dict(status=model.get('status','training') if model else 'training',signal='Wait',reason=model.get('reason','Initial model is training') if model else 'Initial model is training')
    raw=float(infer(model['parameters'],x)); cal=calibration or dict(scale=1.,bias=0.,error_q90=None,active=False,samples=0); value=raw*cal['scale']+cal['bias']
    if not np.isfinite(value) or abs(value)>.5: return dict(status='unavailable',signal='Wait',reason='Prediction outside supported bounds')
    uncertainty=max(model['metrics']['error_q90'],cal.get('error_q90') or 0.); approved=model['metrics']['approved']
    # Require the net target after costs and the model's uncertainty margin.
    minimum_net = MIN_EXPECTED_NET_RETURN if minimum_net_return is None else float(minimum_net_return)
    estimated_cost = ESTIMATED_TRADING_COST if estimated_trading_cost is None else float(estimated_trading_cost)
    if (not math.isfinite(minimum_net) or not math.isfinite(estimated_cost)
            or minimum_net < 0 or estimated_cost < 0):
        raise ValueError('Minimum net return and estimated trading cost must be finite and non-negative')
    required_edge = estimated_cost + minimum_net
    signal='Buy' if approved and value-uncertainty>required_edge else 'Sell' if approved and value+uncertainty< -required_edge else 'Wait'
    return dict(status='ready',signal=signal,predicted_return=value,raw_return=raw,error_q90=uncertainty,model_id=model['model_id'],validation=model['metrics'],calibrated=cal.get('active',False),calibration_samples=cal.get('samples',0),minimum_expected_net_return=minimum_net,estimated_trading_cost=estimated_cost,minimum_predicted_gross_return=required_edge)
