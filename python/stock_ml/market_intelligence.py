"""Causal market-regime and trade-opportunity intelligence.

Only close prices are consumed: this module deliberately does not invent volume,
order-book or intrabar information. Probability calibration uses a chronological
holdout and is disabled unless it improves Brier score.
"""
from __future__ import annotations

import math
import numpy as np

DEFAULT_MIN_NET_RETURN = 0.002


def classify_market_regime(prices, *, short_window=20, long_window=60, volatility_window=20):
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < max(long_window + 1, volatility_window + 2) or np.any(~np.isfinite(p)) or np.any(p <= 0):
        return {"regime": "unknown", "confidence": 0.0, "trend": 0.0, "volatility": None, "volatility_percentile": None}
    returns = np.diff(np.log(p))
    short = float(p[-1] / p[-1-short_window] - 1.0)
    long = float(p[-1] / p[-1-long_window] - 1.0)
    recent_vol = float(np.std(returns[-volatility_window:], ddof=1))
    rolling_vol = np.asarray([np.std(returns[max(0, i-volatility_window):i], ddof=1)
                              for i in range(volatility_window, len(returns)+1)])
    rolling_vol = rolling_vol[np.isfinite(rolling_vol)]
    percentile = float(np.mean(rolling_vol <= recent_vol)) if len(rolling_vol) else 0.5
    trend = 0.6 * short + 0.4 * long
    if percentile >= 0.85:
        regime = "high_volatility"
    elif trend >= 0.015 and short > 0:
        regime = "uptrend"
    elif trend <= -0.015 and short < 0:
        regime = "downtrend"
    else:
        regime = "sideways"
    confidence = min(1.0, abs(trend) / 0.04) if regime in ("uptrend", "downtrend") else abs(percentile - 0.5) * 2
    return {"regime": regime, "confidence": float(confidence), "trend": trend,
            "short_return": short, "long_return": long, "volatility": recent_vol,
            "volatility_percentile": percentile}


def multi_timeframe_confirmation(short_prices, medium_prices, long_prices):
    """Return trend agreement from independently sampled completed close series."""
    series = [np.asarray(x, dtype=float) for x in (short_prices, medium_prices, long_prices)]
    changes = []
    for values in series:
        if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0):
            changes.append(None)
        else:
            changes.append(float(values[-1] / values[max(0, len(values)-6)] - 1.0))
    valid = [x for x in changes if x is not None]
    if len(valid) < 2:
        return {"status": "insufficient_timeframes", "direction": "unknown", "agreement": 0.0, "returns": changes}
    positives = sum(x > 0 for x in valid)
    negatives = sum(x < 0 for x in valid)
    direction = "up" if positives > negatives else "down" if negatives > positives else "mixed"
    agreement = max(positives, negatives) / len(valid)
    return {"status": "confirmed" if agreement >= 2/3 else "mixed",
            "direction": direction, "agreement": float(agreement), "returns": changes}


def calibrate_directional_probabilities(probabilities, outcomes, *, minimum_samples=40):
    """Chronological power calibration; never fit and score on the same holdout."""
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if probs.ndim != 1 or y.shape != probs.shape or len(y) < minimum_samples:
        return {"status": "insufficient_samples", "samples": int(len(y))}
    if np.any(~np.isfinite(probs)) or np.any((probs <= 0) | (probs >= 1)) or np.any(~np.isin(y, [0, 1])):
        return {"status": "invalid_samples", "samples": int(len(y))}
    split = max(20, int(len(y) * 0.7))
    if len(y) - split < 12:
        return {"status": "insufficient_holdout", "samples": int(len(y))}
    candidates = np.linspace(0.5, 2.0, 61)
    best = None
    for power in candidates:
        train_p = probs[:split] ** power / (probs[:split] ** power + (1-probs[:split]) ** power)
        hold_p = probs[split:] ** power / (probs[split:] ** power + (1-probs[split:]) ** power)
        train_brier = float(np.mean((train_p-y[:split])**2))
        hold_brier = float(np.mean((hold_p-y[split:])**2))
        item = (hold_brier, abs(power-1), float(power), train_brier)
        if best is None or item < best:
            best = item
    raw_brier = float(np.mean((probs[split:]-y[split:])**2))
    power = best[2]
    improved = best[0] < raw_brier * 0.98
    return {"status": "accepted" if improved else "rejected", "power": power if improved else 1.0,
            "samples": int(len(y)), "training_samples": int(split),
            "holdout_samples": int(len(y)-split), "raw_holdout_brier": raw_brier,
            "calibrated_holdout_brier": float(best[0]),
            "holdout_improvement": float((raw_brier-best[0])/raw_brier) if raw_brier else 0.0}


def apply_probability_calibration(probability, calibration):
    p = float(probability)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability must be finite and between zero and one")
    if not calibration or calibration.get("status") not in ("accepted", "active"):
        return p
    power = float(calibration.get("power", 1.0))
    if not math.isfinite(power) or power <= 0:
        return p
    if p in (0.0, 1.0):
        return p
    a, b = p ** power, (1-p) ** power
    return float(a/(a+b))


def adaptive_exit_plan(entry_price, trailing_prices, regime, *, risk_budget=0.005,
                       max_position_weight=0.15, trailing_peak=None):
    """Volatility-scaled stop/target and trailing stop using only observed closes."""
    p = np.asarray(trailing_prices, dtype=float)
    if not math.isfinite(entry_price) or entry_price <= 0 or p.ndim != 1 or len(p) < 10 or np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("entry_price and at least ten positive trailing closes are required")
    returns = np.diff(np.log(p))
    vol = float(np.std(returns[-min(20, len(returns)):], ddof=1))
    multiplier = {"uptrend": 1.8, "sideways": 1.3, "downtrend": 1.0, "high_volatility": 2.0}.get(regime, 1.3)
    stop_pct = max(0.003, multiplier * vol)
    target_pct = max(2.0 * stop_pct, 0.002 + 2.0 * stop_pct)
    peak = max(float(np.max(p)), float(trailing_peak or 0.0), float(entry_price))
    trailing_stop = peak * (1.0 - max(0.003, 1.25 * vol))
    return {"stop_price": float(entry_price*(1-stop_pct)),
            "target_price": float(entry_price*(1+target_pct)),
            "trailing_stop_price": float(max(entry_price*(1-stop_pct), trailing_stop)),
            "stop_pct": stop_pct, "target_pct": target_pct,
            "position_weight": float(min(max_position_weight, risk_budget/stop_pct)),
            "volatility": vol, "regime": regime}


def score_opportunity(*, expected_gross_return, probability_up, uncertainty,
                      round_trip_cost=0.002, minimum_net_return=DEFAULT_MIN_NET_RETURN,
                      regime_confidence=0.0, timeframe_agreement=0.0,
                      strategy_eligible=False):
    """Transparent 0-100 ranking score; a score never overrides the net-return gate."""
    values = [expected_gross_return, probability_up, uncertainty, round_trip_cost,
              minimum_net_return, regime_confidence, timeframe_agreement]
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("opportunity inputs must be finite")
    if not 0 <= probability_up <= 1 or uncertainty < 0 or round_trip_cost < 0 or minimum_net_return < 0:
        raise ValueError("invalid probability, uncertainty or cost")
    net = float(expected_gross_return-round_trip_cost)
    edge = max(0.0, min(1.0, (net-minimum_net_return)/0.05))
    probability = max(0.0, min(1.0, (probability_up-0.5)/0.35))
    certainty = 1.0/(1.0+max(0.0, uncertainty)/0.02)
    regime = max(0.0, min(1.0, regime_confidence))
    timeframe = max(0.0, min(1.0, timeframe_agreement))
    score = 100*(0.35*edge+0.25*probability+0.15*certainty+0.10*regime+0.10*timeframe+0.05*float(bool(strategy_eligible)))
    eligible = bool(net >= minimum_net_return and probability_up >= 0.5 and strategy_eligible)
    return {"score": float(score), "expected_net_return": net,
            "eligible": eligible, "status": "candidate" if eligible else "watch_only",
            "components": {"edge": edge, "probability": probability,
                           "certainty": certainty, "regime": regime,
                           "timeframe": timeframe, "strategy": float(bool(strategy_eligible))}}


def select_strategy_for_regime(strategy_results, regime):
    """Select only out-of-sample eligible strategies, preferring regime-matched families."""
    groups = {
        "uptrend": ("trend_ema", "trend_macd", "momentum_roc", "momentum_dual", "breakout_donchian", "bollinger_squeeze"),
        "downtrend": ("mean_reversion_bollinger", "mean_reversion_rsi"),
        "sideways": ("mean_reversion_bollinger", "mean_reversion_rsi"),
        "high_volatility": ("volatility_breakout", "breakout_donchian"),
    }
    preferred = groups.get(regime, ())
    candidates = []
    for name, item in (strategy_results or {}).items():
        metrics = item.get("metrics", {})
        if name not in preferred or metrics.get("eligible") is not True:
            continue
        ret = float(metrics.get("total_return", 0.0) or 0.0)
        pf = float(metrics.get("profit_factor", 0.0) or 0.0)
        dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
        candidates.append((ret, pf, -dd, name))
    candidates.sort(reverse=True)
    return candidates[0][3] if candidates else None
