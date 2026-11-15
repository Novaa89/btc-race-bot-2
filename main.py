"""Deep-washout cbBTC race bot for agent-fun/bars-replay/v1.

Strategy (selective mean reversion, low turnover): the bot watches the
1-minute bars of CBBTC and buys only after a deep washout, then lets a
trailing stop decide the exit.

  - Flat: buy with the full cash balance when the latest bar close sits
    at least 200 bps below the highest close of the trailing 90 bars
    (the current bar included), and at least 240 bars have passed since
    the last exit. If no round trip has completed by batch 1000, the
    depth requirement relaxes to 120 bps, so the bot still takes part
    in a calm race window instead of sitting out the whole race.
  - Long: sell the whole position on the first of (a) a 150 bps stop
    loss measured from the position cost the account reports, (b) an
    80 bps trailing stop measured from the highest close seen since the
    entry decision, (c) a 240-bar time stop. There is no fixed
    take-profit: the trailing stop cuts losers short and lets winners
    run, which is what survived backtesting on the dev window.
  - Every decision uses integer basis-point arithmetic on bar closes
    and on the account the harness reports, so two runs on the same
    data send the same orders. No clock, no randomness, no network,
    no files outside the repository.

The wire protocol is the kit shim's: one hello, one batch per minute
carrying the latest bar, one reply per batch echoing its seq, and the
orders decided on a bar filling at the next bar's open.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

BPS = 10_000
LOOKBACK_BARS = 90
MIN_CLOSES = LOOKBACK_BARS + 1
ENTRY_DEPTH_BPS = 200
FALLBACK_SEQ = 1000
FALLBACK_DEPTH_BPS = 120
STOP_LOSS_BPS = 150
TRAIL_BPS = 80
TIME_STOP_BARS = 240
COOLDOWN_BARS = 240
MAX_SLIPPAGE_BPS = 100
NEVER = -(10**9)


class WashoutBot:
    """The strategy state: bar history plus the open position's anchors."""

    def __init__(self):
        self.closes = []
        self.peak_close = 0
        self.entry_seq = NEVER
        self.pending_entry = None
        self.pending_exit_seq = None
        self.last_exit_seq = NEVER
        self.round_trips = 0
        self.had_position = False

    def on_batch(self, state, batch):
        symbol = next(iter(state.symbols))
        bar_close = None
        for bar in batch["bars"]:
            if bar["symbol"] == symbol:
                bar_close = int(bar["c"])
        if bar_close is None:
            return []
        self.closes.append(bar_close)
        if state.warming_up:
            return []

        holding = state.positions.get(symbol)
        has_position = holding is not None and holding.qty > 0

        if has_position and not self.had_position:
            # The buy decided on an earlier bar has filled: anchor the
            # trailing stop at the entry decision's close.
            if self.pending_entry is not None:
                self.peak_close = max(self.pending_entry[0], bar_close)
                self.entry_seq = self.pending_entry[1]
            else:
                self.peak_close = bar_close
                self.entry_seq = state.seq
            self.pending_entry = None
        elif not has_position and self.had_position:
            # The sell decided on an earlier bar has filled: start the
            # cooldown from the exit decision's batch.
            if self.pending_exit_seq is not None:
                self.last_exit_seq = self.pending_exit_seq
            else:
                self.last_exit_seq = state.seq
            self.pending_exit_seq = None
            self.round_trips += 1
        self.had_position = has_position

        if has_position:
            return self._exit_orders(state, symbol, holding, bar_close)
        return self._entry_orders(state, symbol, bar_close)

    def _exit_orders(self, state, symbol, holding, close):
        if close > self.peak_close:
            self.peak_close = close
        value = state.value(symbol, holding.qty, close)
        gain_bps = (value - holding.cost) * BPS // holding.cost
        drawdown_bps = (self.peak_close - close) * BPS // self.peak_close
        held_bars = state.seq - self.entry_seq
        if (
            gain_bps <= -STOP_LOSS_BPS
            or drawdown_bps >= TRAIL_BPS
            or held_bars >= TIME_STOP_BARS
        ):
            self.pending_exit_seq = state.seq
            return [
                {
                    "side": "sell",
                    "symbol": symbol,
                    "qty": holding.qty,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }
            ]
        return []

    def _entry_orders(self, state, symbol, close):
        if len(self.closes) < MIN_CLOSES:
            return []
        highest = max(self.closes[-LOOKBACK_BARS:])
        depth_bps = (highest - close) * BPS // highest
        threshold = ENTRY_DEPTH_BPS
        if state.seq >= FALLBACK_SEQ and self.round_trips == 0:
            threshold = FALLBACK_DEPTH_BPS
        if depth_bps < threshold:
            return []
        if state.seq - self.last_exit_seq < COOLDOWN_BARS:
            return []
        amount = min(state.cash, state.weight_room(symbol))
        fees = state.hello.get("fees", {})
        min_notional = int(fees.get("min_notional_quote_units", 1_000_000))
        if amount < min_notional:
            return []
        self.pending_entry = (close, state.seq)
        return [
            {
                "side": "buy",
                "symbol": symbol,
                "notional_quote_units": amount,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }
        ]


def main():
    bot = WashoutBot()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
