"""
투자 서적 & 퀀트 전략 지식베이스
=================================

본 시스템에 반영된 저명한 투자자/퀀트의 전략 출처:

1. Larry Williams - "Long-Term Secrets to Short-Term Trading"
   - Williams %R 오실레이터 (과매수/과매도)
   - Large Range Day (대폭 변동일) 감지
   - 가격 패턴 + 시간 사이클
   → 구현: strategy/technical/williams.py

2. Alexander Elder - "Trading for a Living" / "The New Trading for a Living"
   - Triple Screen 시스템 (3중 필터)
   - Force Index (힘 지수)
   - Elder Ray (Bull Power / Bear Power)
   - 2-bar 모멘텀, Impulse System
   → 구현: strategy/technical/elder.py

3. 일목산인(細田悟一) - 일목균형표 (Ichimoku Kinko Hyo)
   - 전환선/기준선 교차
   - 구름대(쿠모) 지지/저항
   - 후행스팬 확인
   - 삼역호전/삼역역전 (강력 매수/매도)
   → 구현: strategy/technical/ichimoku.py

4. John J. Murphy - "Technical Analysis of the Financial Markets"
   - 지지/저항선 자동 감지
   - 피보나치 되돌림 (38.2%, 50%, 61.8%)
   - 다중 타임프레임 분석 원칙
   - 추세선 분석
   → 구현: strategy/technical/market_structure.py

5. Mark Minervini - "Trade Like a Stock Market Wizard"
   Stan Weinstein - "Secrets for Profiting in Bull and Bear Markets"
   Richard Wyckoff - Wyckoff Method
   - VCP (Volatility Contraction Pattern)
   - 4단계 스테이지 분석 (축적→상승→분배→하락)
   - Wyckoff 축적/분배 감지
   - 거래량 확인 원칙
   → 구현: strategy/technical/patterns.py

6. Van K. Tharp - "Trade Your Way to Financial Freedom"
   - R-Multiple 기반 포지션 사이징
   - 기대값(Expectancy) 계산
   - 이미 risk/position_sizer.py의 Half-Kelly에 반영

7. Jesse Livermore - "Reminiscences of a Stock Operator"
   - 피봇 포인트 (Pivotal Points)
   - 추세 추종 + 피라미딩
   - "시장이 항상 옳다" 원칙 → 킬스위치 로직에 반영

퀀트 데이터 소스:
- CoinGlass: 미결제약정(OI), 청산맵, 롱숏비율
- Alternative.me: 공포탐욕지수
- Bybit API: 오더북 깊이, 거래량 프로필
- 온체인: 거래소 입출금, 고래 추적
→ 구현: strategy/quant/, data/quant_collector.py
"""

BOOK_STRATEGIES = {
    "larry_williams": {
        "book": "Long-Term Secrets to Short-Term Trading",
        "indicators": ["williams_r", "large_range_day"],
        "module": "strategy.technical.williams",
    },
    "alexander_elder": {
        "book": "Trading for a Living",
        "indicators": ["triple_screen", "force_index", "elder_ray", "impulse_system"],
        "module": "strategy.technical.elder",
    },
    "ichimoku": {
        "book": "일목균형표 (Ichimoku Kinko Hyo)",
        "indicators": ["tenkan_kijun_cross", "kumo_breakout", "chikou_confirmation"],
        "module": "strategy.technical.ichimoku",
    },
    "john_murphy": {
        "book": "Technical Analysis of the Financial Markets",
        "indicators": ["support_resistance", "fibonacci", "multi_timeframe"],
        "module": "strategy.technical.market_structure",
    },
    "minervini_weinstein_wyckoff": {
        "book": "Trade Like a Stock Market Wizard / Secrets for Profiting",
        "indicators": ["vcp", "stage_analysis", "wyckoff"],
        "module": "strategy.technical.patterns",
    },
}

QUANT_DATA_SOURCES = {
    "coinglass": {
        "url": "https://open-api-v3.coinglass.com",
        "data": ["open_interest", "liquidations", "long_short_ratio", "funding_rate"],
    },
    "alternative_me": {
        "url": "https://api.alternative.me/fng/",
        "data": ["fear_greed_index"],
    },
    "bybit": {
        "data": ["orderbook_depth", "recent_trades", "ticker"],
    },
}
