import os
import json
import joblib
import numpy as np
import pandas as pd

class VantaMLPredictor:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            artifacts_dir = os.path.join(base_dir, 'artifacts')

        # ۱. بارگذاری کانفیگ انسمبل ۴ گانه
        config_path = os.path.join(artifacts_dir, 'model_config.json')
        if not os.path.exists(config_path):
            config_path = os.path.join(artifacts_dir, 'config.json')
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # اوزان بهینه ۴ مدله هیبریدی
        self.weights = self.config.get('weights', {
            'cat': 0.35,
            'xgb': 0.30,
            'rf': 0.20,
            'linear': 0.15
        })

        # ۱۶ متغیر نخبه با چینش قطعی نوت‌بوک
        self.feature_cols = self.config.get('elite_features', [
            'atr_14', 'volatility_4h', 'volatility_1h',
            'price_to_sma_4h', 'price_to_sma_1h',
            'return_24h', 'return_15m', 'return_1h',
            'rsi_14', 'macd', 'macd_signal',
            'cvd_slope_1h',
            'daily_vol_3d', 'day_sin', 'hour_cos', 'hour_sin'
        ])

        # سطوح آستانه پویا
        pct_tiers = self.config.get('percentile_tiers', {'trend_pct': 0.95, 'flat_pct': 0.98})
        self.pct_trend = float(pct_tiers.get('trend_pct', 0.95))
        self.pct_flat  = float(pct_tiers.get('flat_pct', 0.98))
        self.rolling_window = int(self.config.get('rolling_window_hours', 336))

        # ۲. بارگذاری ۴ مدل آموزش‌دیده
        self.model_cat = joblib.load(os.path.join(artifacts_dir, 'best_cat.joblib'))
        self.model_xgb = joblib.load(os.path.join(artifacts_dir, 'best_xgb.joblib'))
        self.model_rf  = joblib.load(os.path.join(artifacts_dir, 'best_rf.joblib'))
        self.model_lin = joblib.load(os.path.join(artifacts_dir, 'best_linear.joblib'))

        # بافر احتمالات تاریخی جهت محاسبه Percentile
        warmup_path = os.path.join(artifacts_dir, 'warmup_buffer.json')
        if os.path.exists(warmup_path):
            with open(warmup_path, 'r') as f:
                self.prob_history = json.load(f)
        else:
            self.prob_history = [0.35] * 72

        self.candle_buffer = pd.DataFrame()
        self.max_buffer_len = 250
        self.median_atr = 0.0035  # مقدار نرمال برای مدیریت ریسک

    def add_candle(self, timestamp, open_p, high_p, low_p, close_p, volume, taker_buy_vol, daily_vol_3d=0.04):
        """افزودن کندل ۱۵ دقیقه‌ای بسته شده به همراه تفکیک حجم خریدار و حجم فروشنده"""
        taker_bv = float(taker_buy_vol)
        tot_vol  = float(volume)
        vol_delta = (2.0 * taker_bv) - tot_vol
        taker_ratio = taker_bv / (tot_vol + 1e-12)

        row = pd.DataFrame([{
            'open': float(open_p),
            'high': float(high_p),
            'low': float(low_p),
            'close': float(close_p),
            'volume': tot_vol,
            'taker_buy_volume': taker_bv,
            'volume_delta': vol_delta,
            'taker_buy_ratio': taker_ratio,
            'daily_vol_3d': float(daily_vol_3d)
        }], index=[pd.to_datetime(timestamp)])

        self.candle_buffer = pd.concat([self.candle_buffer, row])
        self.candle_buffer = self.candle_buffer[~self.candle_buffer.index.duplicated(keep='last')]
        
        if len(self.candle_buffer) > self.max_buffer_len:
            self.candle_buffer = self.candle_buffer.iloc[-self.max_buffer_len:].copy()

    def _extract_features(self):
        df_win = self.candle_buffer.copy()
        feat = {}

        # الف) بازده‌های چندافقه لگاریتمی
        log_close = np.log(df_win['close'])
        feat['return_15m'] = log_close.diff(1)
        feat['return_1h']  = log_close.diff(4)
        feat['return_24h'] = log_close.diff(96)

        # ب) نوسان محقق‌شده
        feat['volatility_1h'] = feat['return_15m'].rolling(window=4, min_periods=4).std()
        feat['volatility_4h'] = feat['return_15m'].rolling(window=16, min_periods=16).std()

        # ج) انحراف لگاریتمی از میانگین‌های متحرک
        sma_1h = df_win['close'].rolling(window=4, min_periods=4).mean()
        sma_4h = df_win['close'].rolling(window=16, min_periods=16).mean()
        feat['price_to_sma_1h'] = np.log(df_win['close'] / sma_1h)
        feat['price_to_sma_4h'] = np.log(df_win['close'] / sma_4h)

        # د) Wilder's RSI (14)
        delta = df_win['close'].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        feat['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

        # ه) MACD
        exp12 = df_win['close'].ewm(span=12, adjust=False).mean()
        exp26 = df_win['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
        feat['macd']        = macd_line / df_win['close']
        feat['macd_signal'] = macd_sig / df_win['close']

        # و) Normalized ATR
        prev_c = df_win['close'].shift(1)
        tr0 = df_win['high'] - df_win['low']
        tr1 = (df_win['high'] - prev_c).abs()
        tr2 = (df_win['low'] - prev_c).abs()
        true_range = np.maximum(tr0, np.maximum(tr1, tr2))
        atr_series = true_range.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        feat['atr_14'] = atr_series / df_win['close']

        # ز) جریان سفارشات نقدینگی (CVD Slope 1h)
        df_win['cvd'] = df_win['volume_delta'].cumsum()
        rolling_vol_1h = df_win['volume'].rolling(window=4, min_periods=4).sum()
        feat['cvd_slope_1h'] = (df_win['cvd'] - df_win['cvd'].shift(4)) / (rolling_vol_1h + 1e-12)

        # ح) ویژگی‌های چرخه‌ای و کلان
        latest_idx = df_win.index
        feat['hour_sin'] = np.sin(2.0 * np.pi * latest_idx.hour / 24.0)
        feat['hour_cos'] = np.cos(2.0 * np.pi * latest_idx.hour / 24.0)
        feat['day_sin']  = np.sin(2.0 * np.pi * latest_idx.dayofweek / 7.0)
        feat['daily_vol_3d'] = df_win['daily_vol_3d']

        calc_df = pd.DataFrame(feat, index=latest_idx).astype(np.float32)
        return calc_df[self.feature_cols].iloc[[-1]]

    def predict(self, is_hourly_step=True):
        """
        اجرای اینفرانس انسمبل ۴ گانه و تعیین حد آستانه انطباقی
        """
        if len(self.candle_buffer) < 96:
            return 0, 0.0, 0.50, 0.0035

        X_latest = self._extract_features()
        latest_atr = float(X_latest['atr_14'].iloc[0])
        latest_sma = float(X_latest['price_to_sma_4h'].iloc[0])
        latest_cvd = float(X_latest['cvd_slope_1h'].iloc[0])

        # احتمالات هر ۴ مدل
        p_cat = float(self.model_cat.predict_proba(X_latest)[:, 1][0])
        p_xgb = float(self.model_xgb.predict_proba(X_latest)[:, 1][0])
        p_rf  = float(self.model_rf.predict_proba(X_latest)[:, 1][0])
        p_lin = float(self.model_lin.predict_proba(X_latest)[:, 1][0])

        p_ens = float(
            self.weights['cat']    * p_cat +
            self.weights['xgb']    * p_xgb +
            self.weights['rf']     * p_rf +
            self.weights['linear'] * p_lin
        )

        # محاسبه کات‌آف بر مبنای بافر غلتان (بدون نشت دیتای جاری)
        hist_slice = self.prob_history[-self.rolling_window:] if len(self.prob_history) >= 72 else [0.35]
        cutoff_98 = float(np.percentile(hist_slice, self.pct_flat * 100))
        cutoff_95 = float(np.percentile(hist_slice, self.pct_trend * 100))

        # فیلتر روند کلان و تایید مومنتوم خریدار
        is_trend_strong = (latest_sma > -0.001) and (latest_cvd > 0.03)
        current_cutoff = cutoff_95 if is_trend_strong else cutoff_98

        signal = 1 if (is_hourly_step and p_ens >= current_cutoff) else 0

        # به‌روزرسانی تاریخچه
        self.prob_history.append(p_ens)
        if len(self.prob_history) > self.rolling_window:
            self.prob_history.pop(0)

        return signal, p_ens, current_cutoff, latest_atr