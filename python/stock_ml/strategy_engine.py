"""Leakage-safe technical strategy competition with close-only execution assumptions.

All signals use information available at the decision close. Entries/exits execute
at the next observed close. Stop/target checks use closes (not unavailable intrabar
high/low data). This is research only; it never places orders.
"""
from __future__ import annotations

import math
import numpy as np

STRATEGY_FAMILIES = (
    "trend_ema", "trend_macd", "momentum_roc", "momentum_dual",
    "mean_reversion_bollinger", "mean_reversion_rsi",
    "breakout_donchian", "volatility_breakout", "bollinger_squeeze",
)
MIN_TRADES = 5
DEFAULT_MIN_NET_RETURN = 0.002
DEFAULT_ROUND_TRIP_COST = 0.002


def _rolling_mean(values, period):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    sums = np.r_[0.0, np.cumsum(x)]
    out[period - 1:] = (sums[period:] - sums[:-period]) / period
    return out


def _rolling_std(values, period):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    sums = np.r_[0.0, np.cumsum(x)]
    squares = np.r_[0.0, np.cumsum(x * x)]
    mean = (sums[period:] - sums[:-period]) / period
    variance = np.maximum(0.0, (squares[period:] - squares[:-period]) / period - mean * mean)
    out[period - 1:] = np.sqrt(variance)
    return out


def _ema(values, period):
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(x[:period]))
    for i in range(period, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(prices, period=14):
    p = np.asarray(prices, dtype=float)
    delta = np.diff(p, prepend=np.nan)
    gains = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    losses = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)
    avg_gain = np.full(len(p), np.nan)
    avg_loss = np.full(len(p), np.nan)
    if len(p) <= period:
        return avg_gain
    avg_gain[period] = float(np.nanmean(gains[1:period + 1]))
    avg_loss[period] = float(np.nanmean(losses[1:period + 1]))
    for i in range(period + 1, len(p)):
        avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gains[i]) / period
        avg_loss[i] = ((period - 1) * avg_loss[i - 1] + losses[i]) / period
    ratio = np.divide(avg_gain, avg_loss, out=np.full(len(p), np.inf), where=avg_loss > 0)
    result = 100.0 - 100.0 / (1.0 + ratio)
    result[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return result


def _signals(prices):
    p = np.asarray(prices, dtype=float)
    n = len(p)
    buy = {name: np.zeros(n, dtype=bool) for name in STRATEGY_FAMILIES}
    sell = {name: np.zeros(n, dtype=bool) for name in STRATEGY_FAMILIES}
    ema8, ema12, ema20, ema21, ema26 = (_ema(p, k) for k in (8, 12, 20, 21, 26))
    sma20, std20 = _rolling_mean(p, 20), _rolling_std(p, 20)
    rsi = _rsi(p)
    roc5 = np.r_[np.full(5, np.nan), p[5:] / p[:-5] - 1.0]
    roc10 = np.r_[np.full(10, np.nan), p[10:] / p[:-10] - 1.0]
    roc20 = np.r_[np.full(20, np.nan), p[20:] / p[:-20] - 1.0]
    returns = np.r_[0.0, np.diff(np.log(p))]
    vol14, vol50 = _rolling_std(returns, 14), _rolling_std(returns, 50)
    macd = ema12 - ema26
    macd_signal = _ema(np.nan_to_num(macd, nan=0.0), 9)
    prior_high20 = np.r_[np.full(20, np.nan), np.maximum.reduce([p[i:n - 20 + i] for i in range(20)])] if n > 20 else np.full(n, np.nan)
    prior_low10 = np.r_[np.full(10, np.nan), np.minimum.reduce([p[i:n - 10 + i] for i in range(10)])] if n > 10 else np.full(n, np.nan)
    # Bollinger squeeze: narrow band width relative to its own trailing regime.
    width = np.divide(4.0 * std20, sma20, out=np.full(n, np.nan), where=sma20 > 0)
    width_mean = _rolling_mean(np.nan_to_num(width, nan=0.0), 50)

    buy["trend_ema"] = (ema8 > ema21) & (p > ema21)
    sell["trend_ema"] = ema8 < ema21
    buy["trend_macd"] = (macd > macd_signal) & (macd > 0)
    sell["trend_macd"] = (macd < macd_signal)
    buy["momentum_roc"] = (roc10 > 0.005) & (rsi >= 50) & (rsi < 75)
    sell["momentum_roc"] = (roc10 <= 0) | (rsi >= 80)
    buy["momentum_dual"] = (roc5 > 0) & (roc20 > 0.01) & (p > ema20)
    sell["momentum_dual"] = (roc5 < 0) | (p < ema20)
    buy["mean_reversion_bollinger"] = (p < sma20 - 2.0 * std20) & (rsi < 35)
    sell["mean_reversion_bollinger"] = (p >= sma20) | (rsi > 55)
    buy["mean_reversion_rsi"] = (rsi < 25) & (p < sma20)
    sell["mean_reversion_rsi"] = (rsi > 55) | (p > sma20)
    buy["breakout_donchian"] = (p > prior_high20) & (p > ema20)
    sell["breakout_donchian"] = p < prior_low10
    buy["volatility_breakout"] = (roc5 > 0) & (vol14 > 1.2 * vol50) & (p > ema20)
    sell["volatility_breakout"] = (p < ema20) | (roc5 < 0)
    buy["bollinger_squeeze"] = (width < 0.75 * width_mean) & (p > sma20 + std20)
    sell["bollinger_squeeze"] = (p < sma20)
    return buy, sell, vol14


def _empty_metrics():
    return dict(trades=0, wins=0, losses=0, win_rate=None, mean_net_return=None,
                total_return=0.0, profit_factor=None, max_drawdown=0.0,
                sharpe_trade=None, exposure=0.0, eligible=False)


def run_strategy_competition(times, prices, *, horizon_steps, bar_seconds,
                             fee_bps=10.0, slippage_bps=5.0,
                             minimum_net_return=DEFAULT_MIN_NET_RETURN,
                             round_trip_cost=None, initial_capital=10000.0,
                             max_position_weight=0.15, risk_budget=0.005,
                             max_drawdown=0.15, min_trades=MIN_TRADES,
                             evaluation_start=None):
    """Evaluate all registered strategies using expanding historical information.

    A strategy signal at close i is executed at close i+1. Positions are long-only,
    one at a time per strategy, risk-sized from trailing close-to-close volatility,
    capped by max_position_weight, and exited by close-based stop, target, exit
    signal, or exact horizon close. Missing target bars / large gaps are skipped.
    """
    t = np.asarray(times, dtype=np.int64)
    p = np.asarray(prices, dtype=float)
    if len(t) != len(p) or len(t) < 80 or np.any(np.diff(t) <= 0) or np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("times/prices must be aligned, increasing, positive and contain at least 80 observations")
    if not isinstance(horizon_steps, int) or horizon_steps < 1 or not isinstance(bar_seconds, int) or bar_seconds < 1:
        raise ValueError("horizon_steps and bar_seconds must be positive integers")
    if not 0 <= fee_bps <= 500 or not 0 <= slippage_bps <= 500:
        raise ValueError("fee and slippage must be between 0 and 500 bps")
    if not 0 < max_position_weight <= 1 or not 0 < risk_budget <= 0.05:
        raise ValueError("invalid position or risk budget")
    if not 0 <= minimum_net_return or not math.isfinite(minimum_net_return):
        raise ValueError("minimum_net_return must be finite and non-negative")
    if initial_capital <= 0 or min_trades < 1 or max_drawdown <= 0:
        raise ValueError("invalid capital, trade count or drawdown limit")
    cost = (fee_bps + 2.0 * slippage_bps) / 10000.0
    if round_trip_cost is not None:
        cost = float(round_trip_cost)
    if not math.isfinite(cost) or not 0 <= cost < 1:
        raise ValueError("round_trip_cost must be finite and between 0 and 1")

    buy, sell, volatility = _signals(p)
    start = int(np.searchsorted(t, evaluation_start)) if evaluation_start is not None else 0
    results = {}
    for name in STRATEGY_FAMILIES:
        equity = peak = float(initial_capital)
        max_dd = 0.0
        trades = []
        i = max(start, 55)
        exposure_bars = 0
        while i + 1 < len(p):
            if not buy[name][i] or t[i + 1] - t[i] > int(bar_seconds * 1.5):
                i += 1
                continue
            entry_i = i + 1
            target_time = int(t[entry_i] + horizon_steps * bar_seconds)
            target_i = int(np.searchsorted(t, target_time))
            if target_i >= len(t) or t[target_i] != target_time:
                i += 1
                continue
            if np.any(np.diff(t[entry_i:target_i + 1]) > int(bar_seconds * 1.5)):
                i += 1
                continue
            entry = float(p[entry_i])
            vol = float(volatility[i]) if np.isfinite(volatility[i]) else 0.0
            stop_pct = max(1.5 * vol, 0.003)
            target_pct = 2.0 * stop_pct
            stop_price, target_price = entry * (1.0 - stop_pct), entry * (1.0 + target_pct)
            position_weight = min(max_position_weight, risk_budget / stop_pct)
            exit_i, reason = target_i, "time"
            for j in range(entry_i + 1, target_i + 1):
                if p[j] <= stop_price:
                    exit_i, reason = j, "stop"
                    break
                if p[j] >= target_price:
                    exit_i, reason = j, "target"
                    break
                if sell[name][j - 1]:
                    exit_i, reason = j, "signal"
                    break
            exit_price = float(p[exit_i])
            gross = exit_price / entry - 1.0
            net = (1.0 + gross) * (1.0 - cost) - 1.0
            before = equity
            equity *= max(0.0, 1.0 + position_weight * net)
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
            exposure_bars += exit_i - entry_i
            trades.append(dict(entry_index=entry_i, exit_index=exit_i,
                origin_time=int(t[i]), entry_time=int(t[entry_i]), exit_time=int(t[exit_i]),
                entry_price=entry, exit_price=exit_price, gross_return=gross,
                net_return=net, position_weight=position_weight,
                capital_before=before, capital_after=equity, exit_reason=reason))
            i = exit_i
        net_returns = np.asarray([x["net_return"] for x in trades], dtype=float)
        wins = int(np.sum(net_returns > 0))
        losses = int(np.sum(net_returns < 0))
        gross_wins = float(np.sum(net_returns[net_returns > 0]))
        gross_losses = float(abs(np.sum(net_returns[net_returns < 0])))
        pf = gross_wins / gross_losses if gross_losses > 0 else (1000.0 if wins else None)
        mean_net = float(np.mean(net_returns)) if len(net_returns) else None
        sharpe = (float(np.mean(net_returns) / np.std(net_returns, ddof=1))
                  if len(net_returns) > 1 and np.std(net_returns, ddof=1) > 0 else None)
        metrics = dict(trades=len(trades), wins=wins, losses=losses,
            win_rate=wins / len(trades) if trades else None,
            mean_net_return=mean_net, total_return=equity / initial_capital - 1.0,
            profit_factor=pf, max_drawdown=max_dd, sharpe_trade=sharpe,
            exposure=(exposure_bars / max(1, len(p) - start)),
            eligible=(len(trades) >= min_trades and mean_net is not None
                and mean_net >= minimum_net_return and equity > initial_capital
                and pf is not None and pf > 1.0 and max_dd >= -max_drawdown))
        results[name] = dict(status="eligible" if metrics["eligible"] else
            ("insufficient_trades" if len(trades) < min_trades else "rejected"),
            metrics=metrics, trades=trades)

    # Explicit reference baselines prevent strategies from winning by comparison
    # with a weak candidate set. Buy-and-hold includes one round-trip cost.
    first = int(np.searchsorted(t, t[start] + bar_seconds)) if start < len(t) else len(t)
    last = len(t) - 1
    baseline_return = ((p[last] / p[first]) * (1.0 - cost) - 1.0) if first < last else 0.0
    results["baseline_buy_hold"] = dict(status="benchmark", metrics=dict(
        trades=1 if first < last else 0, total_return=baseline_return,
        max_drawdown=None, eligible=False), trades=[])
    results["baseline_cash"] = dict(status="benchmark", metrics=dict(
        trades=0, total_return=0.0, max_drawdown=0.0, eligible=False), trades=[])
    return dict(status="completed", config=dict(horizon_steps=horizon_steps,
        bar_seconds=bar_seconds, fee_bps=fee_bps, slippage_bps=slippage_bps,
        round_trip_cost=cost, minimum_net_return=minimum_net_return,
        max_position_weight=max_position_weight, risk_budget=risk_budget,
        max_drawdown=max_drawdown, execution="next_observed_close",
        stops="close_only_atr_proxy", long_only=True),
        strategies=results,
        best_strategy=next((name for name, item in sorted(results.items(),
            key=lambda pair: (-(pair[1]["metrics"].get("total_return") or 0.0),
                              -(pair[1]["metrics"].get("profit_factor") or 0.0)))
            if item["metrics"].get("eligible")), None))
