"""퀀트 데이터 수집 - CoinGlass, Alternative.me 등 공개 API.

퀀트 매니저들이 차트 예측에 활용하는 핵심 데이터:
- 미결제약정 (Open Interest): 시장 참여도, 추세 확인
- 청산 데이터 (Liquidations): 강제 청산 쏠림 → 반전 신호
- 롱숏비율 (Long/Short Ratio): 군중 포지션 → 역추세
- 공포탐욕지수 (Fear & Greed): 극단 감정 → 역추세
- 고래 추적 (Whale Activity): 대량 주문 감지
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import requests

from config.logging_config import get_logger

logger = get_logger(__name__)


class QuantDataCollector:
    """공개 API에서 퀀트 데이터 수집."""

    def __init__(self, coinglass_api_key: str = ""):
        self.coinglass_key = coinglass_api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ChartPredictor/1.0"})

    def get_fear_greed_index(self) -> dict:
        """Alternative.me 공포탐욕지수.
        0 = Extreme Fear, 100 = Extreme Greed
        극단값(<=20 또는 >=80)은 강력한 역추세 신호.
        """
        try:
            resp = self.session.get(
                "https://api.alternative.me/fng/?limit=30&format=json",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                return {"value": 50, "label": "neutral", "history": []}

            current = data[0]
            history = [
                {"value": int(d["value"]), "date": d["timestamp"]}
                for d in data[:30]
            ]

            return {
                "value": int(current["value"]),
                "label": current.get("value_classification", "neutral"),
                "timestamp": current.get("timestamp", ""),
                "history": history,
            }
        except Exception as e:
            logger.warning("fear_greed_fetch_failed", error=str(e))
            return {"value": 50, "label": "neutral", "history": []}

    def get_open_interest(self, symbol: str = "BTC") -> dict:
        """미결제약정 데이터 (CoinGlass 또는 Bybit 직접).
        OI 증가 + 가격 상승 → 추세 강화
        OI 증가 + 가격 하락 → 하락 강화
        OI 감소 → 추세 약화/포지션 청산
        """
        try:
            # Bybit 공개 API에서 직접 가져오기
            resp = self.session.get(
                f"https://api.bybit.com/v5/market/open-interest",
                params={
                    "category": "linear",
                    "symbol": f"{symbol}USDT",
                    "intervalTime": "1h",
                    "limit": 48,
                },
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})
            entries = result.get("list", [])

            if not entries:
                return {"current": 0, "change_24h_pct": 0, "trend": "unknown"}

            values = [float(e.get("openInterest", 0)) for e in entries]
            current = values[0]
            prev_24h = values[-1] if len(values) >= 24 else values[-1]
            change = (current - prev_24h) / prev_24h if prev_24h > 0 else 0

            return {
                "current": current,
                "change_24h_pct": round(change * 100, 2),
                "trend": "increasing" if change > 0.05 else "decreasing" if change < -0.05 else "stable",
                "history_count": len(values),
            }
        except Exception as e:
            logger.warning("open_interest_fetch_failed", error=str(e))
            return {"current": 0, "change_24h_pct": 0, "trend": "unknown"}

    def get_long_short_ratio(self, symbol: str = "BTC") -> dict:
        """롱숏비율 (Bybit 공개 API).
        비율 > 1: 롱 포지션 우세 → 과매수 경고
        비율 < 1: 숏 포지션 우세 → 과매도 경고
        극단값은 역추세 신호.
        """
        try:
            resp = self.session.get(
                f"https://api.bybit.com/v5/market/account-ratio",
                params={
                    "category": "linear",
                    "symbol": f"{symbol}USDT",
                    "period": "1h",
                    "limit": 48,
                },
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})
            entries = result.get("list", [])

            if not entries:
                return {"ratio": 1.0, "long_pct": 50, "short_pct": 50}

            latest = entries[0]
            buy_ratio = float(latest.get("buyRatio", 0.5))
            sell_ratio = float(latest.get("sellRatio", 0.5))
            ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0

            return {
                "ratio": round(ratio, 4),
                "long_pct": round(buy_ratio * 100, 1),
                "short_pct": round(sell_ratio * 100, 1),
                "extreme": ratio > 2.0 or ratio < 0.5,
                "contrarian_signal": "short" if ratio > 2.0 else "long" if ratio < 0.5 else "neutral",
            }
        except Exception as e:
            logger.warning("long_short_ratio_failed", error=str(e))
            return {"ratio": 1.0, "long_pct": 50, "short_pct": 50}

    def get_recent_liquidations(self, symbol: str = "BTC") -> dict:
        """최근 청산 데이터 요약.
        대량 롱 청산 → 매도 압력 완화, 반등 가능
        대량 숏 청산 → 매수 압력 완화, 하락 가능
        """
        try:
            # Bybit doesn't have a direct liquidation API in v5, use ticker info
            resp = self.session.get(
                f"https://api.bybit.com/v5/market/tickers",
                params={"category": "linear", "symbol": f"{symbol}USDT"},
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})
            tickers = result.get("list", [])

            if not tickers:
                return {"available": False}

            ticker = tickers[0]
            return {
                "available": True,
                "turnover_24h": float(ticker.get("turnover24h", 0)),
                "volume_24h": float(ticker.get("volume24h", 0)),
                "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
                "high_24h": float(ticker.get("highPrice24h", 0)),
                "low_24h": float(ticker.get("lowPrice24h", 0)),
            }
        except Exception as e:
            logger.warning("liquidation_fetch_failed", error=str(e))
            return {"available": False}

    def get_orderbook_depth(self, symbol: str = "BTC", depth: int = 50) -> dict:
        """오더북 깊이 분석.
        매수벽 > 매도벽: 지지 강함
        매도벽 > 매수벽: 저항 강함
        """
        try:
            resp = self.session.get(
                f"https://api.bybit.com/v5/market/orderbook",
                params={
                    "category": "linear",
                    "symbol": f"{symbol}USDT",
                    "limit": depth,
                },
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})
            bids = result.get("b", [])
            asks = result.get("a", [])

            bid_volume = sum(float(b[1]) for b in bids) if bids else 0
            ask_volume = sum(float(a[1]) for a in asks) if asks else 0
            total = bid_volume + ask_volume

            imbalance = (bid_volume - ask_volume) / total if total > 0 else 0

            return {
                "bid_volume": round(bid_volume, 4),
                "ask_volume": round(ask_volume, 4),
                "imbalance": round(imbalance, 4),  # +: 매수우세, -: 매도우세
                "bid_wall": round(bid_volume / total * 100, 1) if total > 0 else 50,
                "depth_levels": depth,
            }
        except Exception as e:
            logger.warning("orderbook_fetch_failed", error=str(e))
            return {"bid_volume": 0, "ask_volume": 0, "imbalance": 0}

    def collect_all(self, symbol: str = "BTC") -> dict:
        """모든 퀀트 데이터 한번에 수집."""
        data = {
            "fear_greed": self.get_fear_greed_index(),
            "open_interest": self.get_open_interest(symbol),
            "long_short_ratio": self.get_long_short_ratio(symbol),
            "liquidations": self.get_recent_liquidations(symbol),
            "orderbook": self.get_orderbook_depth(symbol),
            "collected_at": datetime.now().isoformat(),
        }
        logger.info("quant_data_collected", symbol=symbol,
                     fear_greed=data["fear_greed"]["value"],
                     oi_change=data["open_interest"]["change_24h_pct"])
        return data
