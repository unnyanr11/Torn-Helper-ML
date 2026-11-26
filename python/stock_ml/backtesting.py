"""Leakage-safe walk-forward comparison of forecasting model families."""
from __future__ import annotations
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from realtime_model import exact, feature_matrix, prepare_series, validate_series, train_range, infer
from algorithm_registry import algorithm_id_for_family

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

def _prepare_backtest_folds(times, prices, *, window_minutes, horizon_minutes,
                            evaluation_start_minute, max_folds, min_train_rows):
    """Build each walk-forward fold once; model families share these immutable arrays."""
    stride = 15
    candidates = times[(times % stride) == 0]
    candidates = candidates[
        candidates >= times[0] + window_minutes + horizon_minutes + min_train_rows * stride
    ]
    if evaluation_start_minute is not None:
        candidates = candidates[candidates >= int(evaluation_start_minute)]
    if len(candidates) > max_folds:
        candidates = candidates[np.linspace(0, len(candidates) - 1, max_folds).astype(int)]

    non_overlapping = []
    last_exit = -1
    for candidate in candidates:
        if int(candidate) >= last_exit:
            non_overlapping.append(int(candidate))
            last_exit = int(candidate) + horizon_minutes

    folds = []
    for origin in non_overlapping:
        end = int(np.searchsorted(times, origin, side='right'))
        train_t, train_p = times[:end], prices[:end]
        if len(train_t) < min_train_rows:
            continue

        # The expensive time-series preparation and feature extraction are
        # independent of model family, so do them once for all six competitors.
        prepared = prepare_series(train_t, train_p)
        stride_origins = train_t[train_t % stride == 0]
        features_all, feature_ok_all = feature_matrix(
            train_t, train_p, stride_origins, window_minutes, prepared
        )
        # The current forecast origin is itself one of the 15-minute feature
        # origins. Reuse that row rather than calculating its features twice.
        origin_index = int(np.searchsorted(stride_origins, origin))
        if (origin_index >= len(stride_origins)
                or int(stride_origins[origin_index]) != origin
                or not bool(feature_ok_all[origin_index])):
            continue
        x = features_all[origin_index:origin_index + 1]

        targets = exact(train_t, train_p, stride_origins + horizon_minutes)
        current_prices = exact(train_t, train_p, stride_origins)
        matured = feature_ok_all & np.isfinite(targets) & np.isfinite(current_prices) & (current_prices > 0)
        origins = stride_origins[matured]
        features = features_all[matured]
        if len(origins) < 240:
            continue

        y = targets[matured] / current_prices[matured] - 1.0
        boundary = origins[int(len(origins) * .75)]
        fit_mask = origins + horizon_minutes < boundary
        val_mask = origins >= boundary
        if fit_mask.sum() < 100 or val_mask.sum() < 30:
            continue

        future = float(exact(times, prices, [origin + horizon_minutes])[0])
        current = float(exact(times, prices, [origin])[0])
        if not math.isfinite(future) or not math.isfinite(current) or current <= 0:
            continue

        folds.append(dict(
            origin=origin,
            x_eval=x,
            x_fit=features[fit_mask],
            y_fit=y[fit_mask],
            x_validation=features[val_mask],
            y_validation=y[val_mask],
            x_full=features,
            y_full=y,
            current=current,
            future=future,
            actual=future / current - 1.0,
        ))
    return folds


def _results_from_folds(folds, *, families, window_minutes, horizon_minutes,
                         evaluation_start_minute):
    """Fit each base estimator once per fold and reuse its predictions for the ensemble."""
    families = tuple(families)
    fit_families = BASE_FAMILIES if 'ensemble' in families else families
    predictions_by_family = {family: [] for family in families}

    for fold in folds:
        validation_predictions = {
            family: _fit_predict(
                family, fold['x_fit'], fold['y_fit'], fold['x_validation']
            )
            for family in fit_families
        }
        final_predictions = {
            family: _fit_predict(
                family, fold['x_full'], fold['y_full'], fold['x_eval']
            )
            for family in fit_families
        }

        if 'ensemble' in families:
            validation_predictions['ensemble'] = np.median(
                np.vstack([validation_predictions[name] for name in BASE_FAMILIES]), axis=0
            )
            final_predictions['ensemble'] = np.median(
                np.vstack([final_predictions[name] for name in BASE_FAMILIES]), axis=0
            )

        baseline = float(np.mean(np.abs(fold['y_validation'])))
        for family in families:
            val_pred = validation_predictions[family]
            validation_mae = float(np.mean(np.abs(fold['y_validation'] - val_pred)))
            approved = bool(baseline > 0 and validation_mae < baseline * .98)
            final_prediction = float(final_predictions[family][0])
            signal = ('Buy' if approved and final_prediction > 0 else
                      'Sell' if approved and final_prediction < 0 else 'Wait')
            predictions_by_family[family].append(dict(
                origin_minute=fold['origin'],
                target_minute=fold['origin'] + horizon_minutes,
                origin_price=fold['current'],
                actual_price=fold['future'],
                predicted_return=final_prediction,
                raw_return=final_prediction,
                actual_return=fold['actual'],
                predicted_price=fold['current'] * (1.0 + final_prediction),
                signal=signal,
                model_id=family,
                model_family=family,
                validation_mae=validation_mae,
                validation_baseline_mae=baseline,
                approved=approved,
            ))

    results = {}
    for family, predictions in predictions_by_family.items():
        actuals = [p['actual_return'] for p in predictions]
        errors = [p['actual_return'] - p['predicted_return'] for p in predictions]
        directional = [
            (p['predicted_return'] >= 0) == (p['actual_return'] >= 0)
            for p in predictions
        ]
        results[family] = dict(
            status='completed' if predictions else 'insufficient_history',
            config=dict(
                style='short',
                window_minutes=window_minutes,
                horizon_minutes=horizon_minutes,
                model_family=family,
                evaluation_start_minute=(
                    int(evaluation_start_minute)
                    if evaluation_start_minute is not None else None
                ),
                evaluation_method='walk_forward_prediction_vs_actual_price',
                signal_rule='forecast_direction_only; no trading-cost threshold',
            ),
            metrics=dict(
                folds=len(predictions),
                prediction_error=_metrics(errors),
                actual_return=_metrics(actuals),
                directional_accuracy=(
                    float(np.mean(directional)) if directional else None
                ),
                directionally_correct=int(sum(directional)),
                directionally_incorrect=int(len(directional) - sum(directional)),
            ),
            predictions=predictions,
        )
    return results


def run_backtest_competition(times, prices, *, window_minutes: int, horizon_minutes: int,
                             style='short', max_folds=120, min_train_rows=200,
                             model_families=MODEL_FAMILIES, evaluation_start_minute=None):
    """Evaluate model families on shared, precomputed walk-forward feature folds."""
    times, prices = validate_series(times, prices)
    families = tuple(model_families)
    if (style != 'short' or window_minutes < 15 or horizon_minutes < 15
            or not families or any(family not in MODEL_FAMILIES for family in families)):
        raise ValueError('Invalid model style, window, horizon or model family')
    if max_folds < 1 or min_train_rows < 1:
        raise ValueError('Invalid fold or training-row limit')

    folds = _prepare_backtest_folds(
        times, prices,
        window_minutes=window_minutes,
        horizon_minutes=horizon_minutes,
        evaluation_start_minute=evaluation_start_minute,
        max_folds=max_folds,
        min_train_rows=min_train_rows,
    )
    return _results_from_folds(
        folds,
        families=families,
        window_minutes=window_minutes,
        horizon_minutes=horizon_minutes,
        evaluation_start_minute=evaluation_start_minute,
    )


def run_backtest(times, prices, *, window_minutes: int, horizon_minutes: int, style='short',
                 max_folds=120, min_train_rows=200, model_family='ridge',
                 evaluation_start_minute=None):
    """Backward-compatible single-family entry point."""
    return run_backtest_competition(
        times, prices,
        window_minutes=window_minutes,
        horizon_minutes=horizon_minutes,
        style=style,
        max_folds=max_folds,
        min_train_rows=min_train_rows,
        model_families=(model_family,),
        evaluation_start_minute=evaluation_start_minute,
    )[model_family]


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
    approved=baseline>0 and corrected<=baseline*.98
    return dict(active=approved,reason='holdout_mae_improved_by_at_least_2pct' if approved else 'holdout_improvement_below_2pct',
        samples=len(raw),scale=scale if approved else 1.0,bias=bias if approved else 0.0,
        holdout_mae_raw=baseline,holdout_mae_corrected=corrected,
        holdout_error_q90=float(np.quantile(np.abs(actual[split:]-(scale*raw[split:]+bias)),.90)) if approved else None,
        holdout_start=int(predictions[split]['origin_minute']))


def _export_tree(estimator):
    tree=estimator.tree_
    return {
        "left":tree.children_left.astype(int).tolist(),
        "right":tree.children_right.astype(int).tolist(),
        "feature":tree.feature.astype(int).tolist(),
        "threshold":tree.threshold.astype(float).tolist(),
        "value":tree.value[:,0,0].astype(float).tolist(),
    }


def export_estimator(estimator, family):
    """Export a fitted sklearn estimator to a compact, browser-safe JSON payload."""
    if family in ("ridge","elastic_net"):
        scaler=estimator.named_steps["standardscaler"]
        model=estimator.named_steps[list(estimator.named_steps)[-1]]
        return {"kind":"linear","mean":scaler.mean_.astype(float).tolist(),
                "scale":scaler.scale_.astype(float).tolist(),
                "coef":model.coef_.astype(float).tolist(),"intercept":float(model.intercept_)}
    if family in ("random_forest","extra_trees"):
        return {"kind":"forest","trees":[_export_tree(t) for t in estimator.estimators_],
                "aggregation":"mean"}
    if family=="hist_gradient_boosting":
        predictors=[]
        for iteration in estimator._predictors:
            predictor=iteration[0]
            nodes=predictor.nodes
            predictors.append({
                "left":nodes["left"].astype(int).tolist(),
                "right":nodes["right"].astype(int).tolist(),
                "feature":nodes["feature_idx"].astype(int).tolist(),
                "threshold":nodes["num_threshold"].astype(float).tolist(),
                "value":nodes["value"].astype(float).tolist(),
                "is_leaf":nodes["is_leaf"].astype(bool).tolist(),
            })
        return {"kind":"hist_gradient_boosting","baseline":float(estimator._baseline_prediction[0,0]),
                "learning_rate":float(estimator.learning_rate),"trees":predictors}
    raise ValueError("Unsupported exported model family: "+str(family))


def fit_live_model(times, prices, *, window_minutes, horizon_minutes, model_family, observed_mask=None):
    """Fit the selected backtest winner on all matured, genuine observations."""
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                                  HistGradientBoostingRegressor)
    times,prices=validate_series(times,prices)
    prepared=prepare_series(times,prices)
    origins=times[times%15==0]
    if observed_mask is not None:
        observed_mask=np.asarray(observed_mask,dtype=bool)
        if len(observed_mask)!=len(times):
            raise ValueError("Observed-price mask must match the minute series")
        origins=origins[observed_mask[np.searchsorted(times,origins)]]
    features,valid=feature_matrix(times,prices,origins,window_minutes,prepared)
    targets=exact(times,prices,origins+horizon_minutes)
    current=exact(times,prices,origins)
    valid &= np.isfinite(targets) & np.isfinite(current) & (current>0)
    if observed_mask is not None:
        target_idx=np.searchsorted(times,origins+horizon_minutes)
        target_exists=target_idx<len(times)
        target_real=np.zeros(len(target_idx),dtype=bool)
        target_real[target_exists]=observed_mask[target_idx[target_exists]]
        valid &= target_real
    x=features[valid]; y=targets[valid]/current[valid]-1.0
    if len(y)<240:
        raise ValueError("Insufficient matured observations to fit the selected model")
    def fit_family(family, design, target):
        if family=="ridge":
            estimator=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
        elif family=="elastic_net":
            estimator=make_pipeline(StandardScaler(),ElasticNet(alpha=.0001,l1_ratio=.25,max_iter=3000))
        elif family=="random_forest":
            estimator=RandomForestRegressor(n_estimators=60,max_depth=6,min_samples_leaf=8,max_features=.8,n_jobs=1,random_state=17)
        elif family=="extra_trees":
            estimator=ExtraTreesRegressor(n_estimators=60,max_depth=6,min_samples_leaf=8,max_features=.8,n_jobs=1,random_state=17)
        elif family=="hist_gradient_boosting":
            estimator=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,l2_regularization=2.0,learning_rate=.05,random_state=17)
        else:
            raise ValueError("Unknown model family: "+str(family))
        estimator.fit(design,target)
        return estimator
    if model_family=="ensemble":
        members={name:fit_family(name,x,y) for name in BASE_FAMILIES}
        exported={name:export_estimator(est,name) for name,est in members.items()}
        payload={"kind":"ensemble","members":exported,"aggregation":"median"}
    else:
        estimator=fit_family(model_family,x,y)
        payload=export_estimator(estimator,model_family)
    return {"status":"ready","algorithm_id":algorithm_id_for_family(model_family),
            "model_family":model_family,"inference_format":"tree-json-v1",
            "parameters":payload,"trained_through":int(times[-1]),
            "training_rows":int(len(y))}


