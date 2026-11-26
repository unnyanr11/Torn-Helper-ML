"""Bounded GitHub Actions job. Python owns all fitting, calibration and inference."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
from realtime_model import (COMBINATIONS,SHORT_RANGES,SHORT_HORIZONS,
    prepare_series,feature_matrix,train_range,make_result,baseline_forecast,calibrate,validate_series)
from diagnostics import safe_error,project_connection,ConnectionStageError
from realtime_store import connect,load_payloads,save_payloads,save_results,read_prices,check_outcomes,health
from clean_series import reconstruct_short_gaps
from backtesting import fit_live_model
from algorithm_registry import algorithm_id_for_family

CACHE = Path(os.environ.get('STOCK_ML_PRICE_CACHE','.stock-ml-price-cache'))


def cache_load(project, stock):
    try:
        with np.load(CACHE/f'{project}-{stock}.npz',allow_pickle=False) as data:
            times,prices=validate_series(data['times'],data['prices'])
            stamp=datetime.fromisoformat(str(data['watermark'].item()))
            if stamp.tzinfo is None or stamp.timestamp()>time.time()+60 or (datetime.now(timezone.utc)-stamp).total_seconds()>86400:
                raise ValueError('Expired cache')
            if int(data['project'])!=project or int(data['stock'])!=stock or len(times)>259202:
                raise ValueError('Invalid cache')
            if times[-1] > time.time()/60+1:
                raise ValueError('Future cache')
            return (times,prices),stamp
    except (OSError,ValueError,KeyError,EOFError):
        return (np.array([],dtype=np.int64),np.array([],dtype=float)),None


def cache_save(project,stock,prices,watermark):
    CACHE.mkdir(parents=True,exist_ok=True)
    # Public market prices only. Never cache credentials, models or user data.
    temporary=CACHE/f'{project}-{stock}.tmp.npz'
    np.savez_compressed(temporary,times=prices[0],prices=prices[1],
        watermark=watermark.isoformat(),project=project,stock=stock)
    temporary.replace(CACHE/f'{project}-{stock}.npz')


def is_due(previous, seconds, now=None):
    return previous is None or ((now or datetime.now(timezone.utc))-previous).total_seconds()>=seconds


def process(project,dsn,force_train=False,force_calibrate=False):
    with project_connection(connect,dsn) as conn:
        if not conn.execute('select pg_try_advisory_lock(%s,%s)',(74183000,project)).fetchone()[0]:
            # Scheduled cycles may overlap. Another run already owns this shard,
            # so skip it cleanly rather than failing the whole workflow.
            print(json.dumps(dict(event='project_skipped',project=project,
                reason='another_prediction_run_owns_project_lease')),flush=True)
            return
        assignment=conn.execute('select project_number,enabled from stock_engine.settings where singleton').fetchone()
        if not assignment or assignment[0]!=project:
            raise ValueError('Project assignment mismatch')
        if not assignment[1]:
            print(json.dumps(dict(event='disabled',project=project)),flush=True)
            return
        state=conn.execute('select last_training,last_calibration from stock_engine.ml_v2_health where singleton').fetchone()
        if state is None:
            raise ValueError('Install realtime_setup.sql first')
        train_due=force_train or is_due(state[0],600)
        calibrate_due=force_calibrate or is_due(state[1],10800)
        models=load_payloads(conn,'models')
        corrections=load_payloads(conn,'calibration')
        stocks=[r[0] for r in conn.execute('select stock_id from stock_engine.stocks order by stock_id').fetchall()]
        if not stocks or any(s not in (project*2-1,project*2) or s>35 for s in stocks):
            raise ValueError('Invalid stock assignment')
        # Turn off the legacy SQL baseline in this stock project.
        conn.execute("select cron.unschedule(jobid) from cron.job where jobname='stock-engine-analysis'")
        # Processing responses stores received data; this does not predict prices.
        conn.execute('select stock_engine.collect_responses()')
        check_outcomes(conn)
        if calibrate_due:
            checks=conn.execute('''select stock_id,style,analysis_minutes,horizon_minutes,raw_return,actual_return
                from stock_engine.ml_v2_checks where checked_at is not null and target_at<=now()
                and origin_at>=now()-interval '180 days' order by target_at''').fetchall()
            groups={}
            for row in checks:
                groups.setdefault(tuple(row[:4]),[]).append(tuple(row[4:]))
            corrections={key:calibrate(groups.get(key,[])[-200:])
                         for stock in stocks for key in [(stock,*combo) for combo in COMBINATIONS]}
            save_payloads(conn,'calibration',corrections)
            conn.execute('update stock_engine.ml_v2_health set last_calibration=now() where singleton')
            conn.execute("delete from stock_engine.ml_v2_checks where origin_at<now()-interval '180 days'")
        started=time.monotonic()
        results={}
        for stock in stocks:
            old,watermark=cache_load(project,stock)
            (times,values),watermark=read_prices(conn,stock,old,watermark)
            cache_save(project,stock,(times,values),watermark)
            if not len(times):
                continue
            clean_times,clean_values,synthetic_mask,_=reconstruct_short_gaps(times,values)
            observed_mask=~synthetic_mask
            prepared=prepare_series(clean_times,clean_values)
            if train_due:
                for style,ranges,horizons in [('short',SHORT_RANGES,SHORT_HORIZONS)]:
                    for window in ranges:
                        fitted=train_range(clean_times,clean_values,style,window,horizons,prepared,observed_mask)
                        for model in fitted.values():
                            model.setdefault('trained_through',int(times[-1]))
                        mapping={}
                        for horizon, candidate in fitted.items():
                            key=(stock,style,window,horizon)
                            existing=models.get(key)
                            # The backtest-selected family remains authoritative. Once its
                            # fitted parameters are older than a day, refresh that same
                            # family on current genuine observations rather than falling
                            # back to a different algorithm or a baseline forecast.
                            if existing and existing.get('source')=='walk_forward_backtest_winner':
                                if int(times[-1])-int(existing.get('trained_through',0))>1440:
                                    family=existing.get('model_family')
                                    try:
                                        refreshed=fit_live_model(
                                            clean_times,clean_values,
                                            window_minutes=window,horizon_minutes=horizon,
                                            model_family=family,observed_mask=observed_mask,
                                        )
                                    except ValueError as exc:
                                        print(json.dumps(dict(event='selected_model_refresh_skipped',
                                            stock_id=stock,analysis_minutes=window,
                                            horizon_minutes=horizon,algorithm_id=existing.get('algorithm_id'),
                                            reason=str(exc))),flush=True)
                                        continue
                                    trained_through=int(refreshed['trained_through'])
                                    refreshed.update(
                                        stock_id=stock,style=style,analysis_minutes=window,
                                        horizon_minutes=horizon,algorithm_id=existing.get('algorithm_id') or algorithm_id_for_family(family),
                                        model_family=family,source='walk_forward_backtest_winner',
                                        source_backtest_run=existing.get('source_backtest_run'),
                                        metrics=existing.get('metrics',{}),
                                        model_id=hashlib.sha256(json.dumps({
                                            'previous_model_id':existing.get('model_id'),
                                            'stock_id':stock,'style':style,'window':window,
                                            'horizon':horizon,'algorithm_id':existing.get('algorithm_id'),
                                            'trained_through':trained_through,
                                        },sort_keys=True).encode()).hexdigest()[:24],
                                    )
                                    mapping[key]=refreshed
                                continue
                            mapping[key]=candidate
                        if mapping:
                            save_payloads(conn,'models',mapping)
                            models.update(mapping)
            origin=int(times[-1])
            if time.time()/60-origin>3:
                print(json.dumps(dict(event='stale_prices',project=project,stock=stock)),flush=True)
                continue
            features={}
            for style,window,horizon in COMBINATIONS:
                if window not in features:
                    x,valid=feature_matrix(clean_times,clean_values,[origin],window,prepared)
                    features[window]=(x[0],bool(valid[0]))
                x,valid=features[window]
                key=(stock,style,window,horizon)
                model=models.get(key)
                if model and model.get('status')=='ready' and origin-model.get('trained_through',0)>(1440 if model.get('source')=='walk_forward_backtest_winner' else 60):
                    model=dict(status='training',reason='Waiting for a recently trained Python model')
                result=make_result(model,corrections.get(key),x,valid)
                if result['status'] != 'ready':
                    result=baseline_forecast(clean_times,clean_values,origin,window,horizon)
                result.update(stock_id=stock,style=style,analysis_minutes=window,horizon_minutes=horizon,
                    origin_minute=origin,origin_price=float(values[-1]),computed_at=time.time(),
                    max_age_minutes=15,prediction_engine='python',execution_mode='github-actions')
                if result['status']=='ready':
                    result['predicted_price']=float(values[-1])*(1+result['predicted_return'])
                    result['lower_price']=max(.000001,float(values[-1])*(1+result['predicted_return']-result['error_q90']))
                    result['upper_price']=float(values[-1])*(1+result['predicted_return']+result['error_q90'])
                results[key]=result
        if results:
            save_results(conn,results)
        if train_due:
            conn.execute('update stock_engine.ml_v2_health set last_training=now(),training_ms=%s where singleton',
                ((time.monotonic()-started)*1000,))
        elapsed=time.monotonic()-started
        health(conn,elapsed,sum(r['status']=='ready' for r in results.values()),0.)
        print(json.dumps(dict(event='actions_cycle',project=project,trained=train_due,calibrated=calibrate_due,
            combinations=len(results),ready=sum(r['status']=='ready' for r in results.values()),seconds=round(elapsed,3))),flush=True)


def cache_output():
    """Save the public-price cache at most hourly to bound storage churn."""
    marker=CACHE/'cache-saved-at.txt'
    try:
        previous=float(marker.read_text())
    except (OSError,ValueError):
        previous=0.
    due=time.time()-previous>=3600 and any(CACHE.glob('*.npz'))
    if due:
        marker.write_text(str(time.time()))
    output=os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output,'a') as file:
            file.write(f'cache_save={str(due).lower()}\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--force-train',action='store_true',default=os.getenv('ML_FORCE_TRAIN')=='true')
    parser.add_argument('--force-calibrate',action='store_true',default=os.getenv('ML_FORCE_CALIBRATE')=='true')
    args=parser.parse_args()
    try:
        config=json.loads(os.environ.get('STOCK_ML_PROJECTS_JSON','{}'))
        if not config or any(not 1<=int(n)<=18 for n in config):
            raise ValueError()
    except (ValueError,TypeError):
        parser.error('Set STOCK_ML_PROJECTS_JSON to a mapping of project numbers to PostgreSQL URLs')
    failed=False
    def execute(entry):
        number,dsn=entry
        try:
            process(int(number),dsn,args.force_train,args.force_calibrate)
            return True
        except Exception as exc:
            # Do not log raw exception text: connection errors can contain passwords.
            original=exc.original if isinstance(exc,ConnectionStageError) else exc
            details=safe_error(original,'connect' if isinstance(exc,ConnectionStageError) else 'database_work')
            print(json.dumps(dict(event='project_failed',project=int(number),error=type(original).__name__,**details)),flush=True)
            return False
    with ThreadPoolExecutor(max_workers=4) as executor:
        failed=not all(list(executor.map(execute,config.items())))
    cache_output()
    return int(failed)


if __name__=='__main__':
    sys.exit(main())
