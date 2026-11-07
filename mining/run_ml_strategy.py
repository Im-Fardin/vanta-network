import os
import sys
import time
import json
import logging
import requests
import datetime
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_objects.log import logger

logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

from vanta_model.predictor import VantaMLPredictor
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair, TradePairCategory


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (TradePair, OrderType, ExecutionType)):
            return obj.__json__()
        if isinstance(obj, TradePairCategory):
            return obj.value
        return super().default(obj)


class VantaHybridTrader:
    def __init__(self, miner_api_url="http://127.0.0.1:8088", api_key="xxxx", max_leverage=0.49):
        self.submit_order_url = f"{miner_api_url}/api/submit-order"
        self.headers = {'Content-Type': 'application/json', 'Authorization': api_key}
        self.api_key = api_key
        
        # سقف مجاز اهرم کریپتو در ساب‌نت ۸ شبکه Vanta
        self.max_leverage = min(float(max_leverage), 0.49)
        self.current_position = 0  # 0: FLAT / Cash, 1: LONG
        self.entry_price = 0.0
        self.current_pos_size = 0.0

        # سیستم‌های حفاظتی دراداون (Circuit Breaker & Cooldown)
        self.daily_pnl = 0.0
        self.current_day = datetime.datetime.now(datetime.timezone.utc).date()
        self.cooldown_bars = 0
        self.MAX_DAILY_LOSS = -0.030      # افت روزانه بیش از ۳٪ = خروج و توقف
        
        # ضرایب حد ضرر و حد سود منطبق با کالیبراسیون انسمبل
        self.SL_MULT = 2.2
        self.TP_MULT = 3.2
        self.current_atr = 0.0035
        self.median_atr = 0.0035

        self.predictor = VantaMLPredictor()

        # فایل گزارش‌گیری زنده عملکرد
        self.log_filepath = os.path.join(REPO_ROOT, "paper_trade_log.csv")
        if not os.path.exists(self.log_filepath):
            pd.DataFrame(columns=[
                'timestamp', 'close_price', 'consensus_prob', 'cutoff',
                'signal', 'position', 'leverage', 'pnl', 'status_msg'
            ]).to_csv(self.log_filepath, index=False)

    def log_state(self, timestamp, close_p, prob, cutoff, sig, lev, pnl, msg):
        row = pd.DataFrame([{
            'timestamp': timestamp,
            'close_price': close_p,
            'consensus_prob': round(prob, 4),
            'cutoff': round(cutoff, 4),
            'signal': sig,
            'position': self.current_position,
            'leverage': round(lev, 2),
            'pnl': round(pnl, 4),
            'status_msg': msg
        }])
        row.to_csv(self.log_filepath, mode='a', header=False, index=False)

    def fetch_binance_15m_klines(self, limit=120):
        """دریافت کندل‌های ۱۵ دقیقه‌ای از صرافی بایننس شامل حجم کل و حجم خریدار مهاجم"""
        url = 'https://api.binance.com/api/v3/klines'
        params = {'symbol': 'BTCUSDT', 'interval': '15m', 'limit': limit}
        resp = requests.get(url, params=params, timeout=10).json()
        current_time_ms = int(time.time() * 1000)

        candles = []
        for k in resp:
            candles.append({
                'timestamp': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'taker_buy_volume': float(k[9]),
                'is_closed': bool(k[6] < current_time_ms)
            })
        return candles

    def send_order(self, order_type: OrderType, leverage: float = 0.0):
        """ارسال سیگنال سفارش به سرور ماینر (Port 8088)"""
        data = {
            'api_key': self.api_key,
            'execution_type': ExecutionType.MARKET,
            'trade_pair': TradePair.BTCUSD,
            'order_type': order_type,
            'leverage': leverage
        }
        json_data = json.dumps(data, cls=CustomEncoder)
        try:
            logger.info(f"Submitting order payload -> {order_type.name} (Lev: {leverage}x)")
            res = requests.post(self.submit_order_url, data=json_data, headers=self.headers, timeout=10)
            if res.status_code == 200:
                logger.info(f"Order accepted by Miner API: {res.json()}")
                return True
            else:
                logger.error(f"Order rejected ({res.status_code}): {res.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot reach Miner API at {self.submit_order_url}: {e}")
            return False

    def warm_up(self):
        """بارگذاری اولیه بافر کندل‌ها برای جلوگیری از تأخیر Cold-start"""
        logger.info("Hydrating in-memory buffer with closed 15m bars from Binance...")
        candles = self.fetch_binance_15m_klines(limit=130)
        closed_candles = [c for c in candles if c['is_closed']]
        if not closed_candles:
            closed_candles = candles[:-1]

        for c in closed_candles:
            self.predictor.add_candle(
                timestamp=c['timestamp'],
                open_p=c['open'],
                high_p=c['high'],
                low_p=c['low'],
                close_p=c['close'],
                volume=c['volume'],
                taker_buy_vol=c['taker_buy_volume']
            )
        logger.info(f"Hydration complete: {len(self.predictor.candle_buffer)} closed bars in buffer.")

    def run(self):
        self.warm_up()
        logger.info(f"4-Way Hybrid Strategy Active | Max Lev: {self.max_leverage}x | Circuit Breakers: ON")

        last_processed_timestamp = None
        last_heartbeat_minute = -1

        while True:
            try:
                now = datetime.datetime.now(datetime.timezone.utc)

                # ریست آمار روزانه در 00:00 UTC
                if now.date() != self.current_day:
                    self.current_day = now.date()
                    self.daily_pnl = 0.0
                    logger.info("New UTC Day: Daily PnL tracker reset to 0.0.")

                # ارزیابی کندل‌های ۱۵ دقیقه‌ای (:00, :15, :30, :45)
                if now.minute % 15 == 0 and 15 <= now.second <= 45:
                    is_hourly = (now.minute == 0)
                    
                    candles = self.fetch_binance_15m_klines(limit=3)
                    closed_candidate = candles[-2]
                    
                    if closed_candidate['timestamp'] == last_processed_timestamp:
                        time.sleep(5)
                        continue

                    # گارد تأخیر داده
                    candle_dt = closed_candidate['timestamp'].to_pydatetime()
                    if candle_dt.tzinfo is None:
                        candle_dt = candle_dt.replace(tzinfo=datetime.timezone.utc)
                    if (now - candle_dt).total_seconds() > 1200:
                        logger.warning("Data feed lag detected. Skipping bar for safety.")
                        time.sleep(35)
                        continue

                    self.predictor.add_candle(
                        timestamp=closed_candidate['timestamp'],
                        open_p=closed_candidate['open'],
                        high_p=closed_candidate['high'],
                        low_p=closed_candidate['low'],
                        close_p=closed_candidate['close'],
                        volume=closed_candidate['volume'],
                        taker_buy_vol=closed_candidate['taker_buy_volume']
                    )

                    last_processed_timestamp = closed_candidate['timestamp']
                    signal, prob, cutoff, latest_atr = self.predictor.predict(is_hourly_step=is_hourly)
                    self.current_atr = latest_atr
                    close_price = closed_candidate['close']
                    trade_pnl = 0.0
                    status_msg = "MONITORING"

                    # ارزیابی حد ضرر و حد سود محلی در هر کندل ۱۵ دقیقه‌ای
                    if self.current_position == 1:
                        current_unrealized = (close_price - self.entry_price) / self.entry_price
                        
                        # ۱. اصابت به حد ضرر
                        if current_unrealized <= (-self.SL_MULT * self.current_atr):
                            trade_pnl = current_unrealized
                            self.daily_pnl += trade_pnl
                            if self.send_order(OrderType.FLAT, 0.0):
                                self.current_position = 0
                                self.cooldown_bars = 2  # استراحت ۲ ساعته
                                status_msg = f"STOP_LOSS_HIT (PnL: {trade_pnl * 100:.2f}%)"

                        # ۲. برخورد با حد سود
                        elif current_unrealized >= (self.TP_MULT * self.current_atr):
                            trade_pnl = current_unrealized
                            self.daily_pnl += trade_pnl
                            if self.send_order(OrderType.FLAT, 0.0):
                                self.current_position = 0
                                status_msg = f"TAKE_PROFIT_HIT (PnL: {trade_pnl * 100:.2f}%)"

                    # تصمیم‌گیری‌های تغییر وضعیت پوزیشن در پایان ساعت (:00)
                    if is_hourly:
                        # مدار قطع اضطراری
                        is_circuit_active = (self.daily_pnl <= self.MAX_DAILY_LOSS)

                        if self.cooldown_bars > 0:
                            self.cooldown_bars -= 1

                        if is_circuit_active and self.current_position == 1:
                            trade_pnl = (close_price - self.entry_price) / self.entry_price
                            self.daily_pnl += trade_pnl
                            if self.send_order(OrderType.FLAT, 0.0):
                                self.current_position = 0
                                status_msg = "FORCE_EXIT_DAILY_DRAWDOWN_LIMIT"

                        # خروج عادی استراتژی در صورت عدم تایید ادامه موج
                        elif signal == 0 and self.current_position == 1:
                            trade_pnl = (close_price - self.entry_price) / self.entry_price
                            self.daily_pnl += trade_pnl
                            if self.send_order(OrderType.FLAT, 0.0):
                                self.current_position = 0
                                status_msg = f"NORMAL_EXIT_HOURLY (PnL: {trade_pnl * 100:.2f}%)"

                        # ورود جدید به پوزیشن خرید
                        elif signal == 1 and self.current_position == 0 and not is_circuit_active and self.cooldown_bars == 0:
                            # محاسبه اهرم تطبیقی بر مبنای نوسان (Volatility-Adjusted Sizing)
                            vol_scale = min(1.0, (self.median_atr / (self.current_atr + 1e-6)))
                            effective_leverage = round(min(self.max_leverage, self.max_leverage * vol_scale), 2)
                            
                            if self.send_order(OrderType.LONG, effective_leverage):
                                self.current_position = 1
                                self.entry_price = close_price
                                self.current_pos_size = effective_leverage
                                status_msg = f"ENTRY_LONG @ {close_price} (Lev: {effective_leverage}x)"

                    self.log_state(closed_candidate['timestamp'], close_price, prob, cutoff, signal, self.current_pos_size, trade_pnl, status_msg)
                    logger.info(f"[{closed_candidate['timestamp']}] Prob: {prob:.4f} | Cutoff: {cutoff:.4f} | Pos: {self.current_position} | Status: {status_msg}")

                    time.sleep(35)

                else:
                    if now.minute != last_heartbeat_minute:
                        mins_to_next = 15 - (now.minute % 15)
                        logger.info(f"Heartbeat: Subnet 8 Miner healthy at {now.strftime('%H:%M:%S')} UTC. Next check in ~{mins_to_next} min(s)...")
                        last_heartbeat_minute = now.minute

                time.sleep(2)

            except KeyboardInterrupt:
                logger.info("Trader stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in runner loop: {e}", exc_info=True)
                time.sleep(10)


if __name__ == "__main__":
    trader = VantaHybridTrader()
    trader.run()