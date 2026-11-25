"""Vectorized causal features and nested walk-forward Ridge model selection."""
import hashlib
import json
import numpy as np

VERSION = 'minute-ridge-v3'
SHORT_RANGES = (15, 30, 60, 120, 180, 360, 720, 1080, 1440, 4320, 10080, 43200)
LONG_RANGES = tuple(d * 1440 for d in (7, 15, 30, 60, 90, 120, 150, 180))
SHORT_HORIZONS = tuple(h * 60 for h in (1, 3, 6, 12, 18, 24))
LONG_HORIZONS = tuple(h * 60 for h in (168, 336, 504, 672))
COMBINATIONS = tuple((style, r, h) for style, ranges, horizons in
                     (('short', SHORT_RANGES, SHORT_HORIZONS),
                      ('long', LONG_RANGES, LONG_HORIZONS)) for r in ranges for h in horizons)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)

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

def train_range(times,prices,style,window,horizons,prepared=None):
    """Nested chronological selection, purged by forecast horizon; final holdout is untouched."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    times,prices=validate_series(times,prices); stride=15 if style=='short' else 60; origins=times[times%stride==0]
    if len(origins)>3000: origins=origins[np.linspace(0,len(origins)-1,3000).astype(int)]
    x,complete=feature_matrix(times,prices,origins,window,prepared); origins,x=origins[complete],x[complete]; result={}
    for horizon in horizons:
        future=exact(times,prices,origins+horizon); matured=np.isfinite(future); t,design=origins[matured],x[matured]; y=future[matured]/exact(times,prices,t)-1
        if len(t)<240:
            result[horizon]={'status':'insufficient_history','reason':'Need 240 matured examples for nested validation'}; continue
        boundary_select=t[int(len(t)*.60)]; boundary_test=t[int(len(t)*.75)]
        train=t+horizon<boundary_select
        select=(t>=boundary_select)&(t+horizon<boundary_test)
        test=t>=boundary_test
        if train.sum()<100 or select.sum()<30 or test.sum()<50:
            result[horizon]={'status':'insufficient_history','reason':'Not enough rows after nested horizon purges'}; continue
        scaler=StandardScaler().fit(design[train]); z_train=scaler.transform(design[train]); z_select=scaler.transform(design[select])
        alpha_scores=[]
        for alpha in RIDGE_ALPHAS:
            candidate=Ridge(alpha=alpha).fit(z_train,y[train])
            score=float(np.mean(np.abs(y[select]-candidate.predict(z_select))))
            alpha_scores.append((score,alpha))
        selected_alpha=min(alpha_scores,key=lambda item:(item[0],item[1]))[1]
        # Final untouched test evaluates the selected alpha, never selects it.
        fit_before_test=t+horizon<boundary_test
        final_scaler=StandardScaler().fit(design[fit_before_test])
        final_model=Ridge(alpha=selected_alpha).fit(final_scaler.transform(design[fit_before_test]),y[fit_before_test])
        test_prediction=final_model.predict(final_scaler.transform(design[test]))
        error=y[test]-test_prediction
        mae,baseline=float(np.mean(abs(error))),float(np.mean(abs(y[test])))
        q90=float(np.quantile(abs(error),.9))
        metrics=dict(mae=mae,baseline_mae=baseline,error_q90=q90,
            training_rows=int(train.sum()),selection_rows=int(select.sum()),validation_rows=int(test.sum()),
            selected_alpha=selected_alpha,alpha_selection_mae=float(min(alpha_scores)[0]),
            last_training_target=int((t[fit_before_test]+horizon).max()),
            first_validation_origin=int(boundary_test),approved=bool(baseline>0 and mae<baseline*.98))
        # Refit selected model on every matured example for live inference.
        scaler=StandardScaler().fit(design); model=Ridge(alpha=selected_alpha).fit(scaler.transform(design),y)
        parameters=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_))
        model_id=hashlib.sha256(json.dumps(dict(parameters=parameters,alpha=selected_alpha),sort_keys=True).encode()).hexdigest()[:20]
        result[horizon]=dict(status='ready',parameters=parameters,metrics=metrics,model_id=model_id,trained_through=int(times[-1]))
    return result

def calibrate(rows):
    if len(rows)<20: return dict(scale=1.,bias=0.,error_q90=None,samples=len(rows),active=False)
    data=np.asarray(rows,dtype=float); raw,actual=data[:,0],data[:,1]; split=int(len(data)*.7)
    if len(data)-split<6: return dict(scale=1.,bias=0.,error_q90=None,samples=len(rows),active=False)
    scale=float(np.clip(np.dot(raw[:split],actual[:split])/(np.dot(raw[:split],raw[:split])+1e-9),0,1))
    bias=float(np.clip(np.median(actual[:split]-scale*raw[:split]),-.02,.02)); corrected=raw[split:]*scale+bias
    approved=bool(np.mean(abs(actual[split:]-corrected))<np.mean(abs(actual[split:]-raw[split:])))
    q90=float(np.quantile(abs(actual[split:]-corrected),.9))
    return dict(scale=scale if approved else 1.,bias=bias if approved else 0.,error_q90=q90 if approved else None,samples=len(rows),active=approved)

def make_result(model,calibration,x,coverage_ok):
    if not coverage_ok: return dict(status='insufficient_history',signal='Wait',reason='Selected input window is incomplete')
    if not model or model.get('status')!='ready': return dict(status=model.get('status','training') if model else 'training',signal='Wait',reason=model.get('reason','Initial model is training') if model else 'Initial model is training')
    raw=float(infer(model['parameters'],x)); cal=calibration or dict(scale=1.,bias=0.,error_q90=None,active=False,samples=0); value=raw*cal['scale']+cal['bias']
    if not np.isfinite(value) or abs(value)>.5: return dict(status='unavailable',signal='Wait',reason='Prediction outside supported bounds')
    uncertainty=max(model['metrics']['error_q90'],cal.get('error_q90') or 0.); approved=model['metrics']['approved']
    signal='Buy' if approved and value-uncertainty>.003 else 'Sell' if approved and value+uncertainty<-.003 else 'Wait'
    return dict(status='ready',signal=signal,predicted_return=value,raw_return=raw,error_q90=uncertainty,model_id=model['model_id'],validation=model['metrics'],calibrated=cal.get('active',False),calibration_samples=cal.get('samples',0))
