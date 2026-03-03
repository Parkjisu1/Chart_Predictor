# Chart Predictor

AI 기반 코인 선물(Bybit) 자동매매 시스템. 멀티에이전트 아키텍처 + 자가학습 피드백 루프.

## Overview

| 항목 | 내용 |
|------|------|
| 시장 | Bybit 코인 선물 (BTC/USDT, ETH/USDT) |
| 초기 자본 | 10만원 (~$75) |
| AI 엔진 | Claude CLI subprocess (API 비용 없음) |
| 언어 | Python 3.13 |
| 목표 | 자가학습으로 승률 90% 달성 후 실전 가동 |

## Architecture

5개 에이전트 파이프라인:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Technical  │───▶│  Sentiment   │───▶│    Risk      │───▶│    Final     │───▶│  Supervisor  │
│  Analyst    │    │  Analyst     │    │  Reviewer    │    │  Decider     │    │  (Kill SW)   │
│  (정량분석)  │    │ (Claude CLI) │    │  (교차검증)   │    │ (Half-Kelly) │    │  (일간리뷰)   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Technical Analyst** - RSI + 다이버전스, 볼린저밴드 %B, MACD/ADX, OBV/VWAP, GARCH(1,1)
2. **Sentiment Analyst** - 펀딩비 분석 + Claude CLI 매크로/센티먼트
3. **Risk Reviewer** - Decision Matrix 교차검증, 상관관계 체크
4. **Final Decider** - Modified Half-Kelly 포지션 사이징
5. **Supervisor** - 킬스위치 권한 + Claude CLI 일간 리뷰

## Self-Learning Loop

```
백테스트 실행 → 손실 분류(8가지) → Claude 인사이트 → 파라미터 조정 → 수렴 체크 → 반복
```

4단계 검증: In-Sample → Out-of-Sample → Walk-Forward → Monte Carlo

- 목표 승률: 90% (in-sample), OOS 85% 이상
- 최대 50회 반복, 5회 정체 시 랜덤 섭동

### 8가지 실패 모드 분류

| 모드 | 설명 | 조정 방향 |
|------|------|-----------|
| Early Entry | 진입 너무 빠름 | 스톱로스 확대, 신호 강도 상향 |
| Late Entry | 진입 너무 늦음 | 신호 임계값 하향 |
| Wrong Direction | 방향 잘못 | 강한 신호 임계값 상향 |
| Overleveraged | 과도한 레버리지 | 포지션 사이징 축소 |
| No Stop Loss | 스톱로스 미설정 | 스톱로스 강제 |
| Premature Exit | 조기 청산 | 익절 기준 확대 |
| Trend Reversal | 추세 전환 | ADX 필터 강화 |
| High Volatility | 높은 변동성 | GARCH 가중치 증가 |

## Setup

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 Bybit API 키 입력
```

## Usage

```bash
# 1. 과거 데이터 수집 (2021~2024)
python main.py collect

# 2. 백테스트 실행
python main.py backtest --symbol "BTC/USDT:USDT" --timeframe 1h

# 3. 자가학습 루프 실행
python main.py learn --symbol "BTC/USDT:USDT" --max-iter 50

# 3-1. Claude CLI 없이 규칙 기반만으로 학습
python main.py learn --symbol "BTC/USDT:USDT" --no-claude

# 4. 실전 매매 (검증 통과 후)
python main.py live --symbol "BTC/USDT:USDT" --dry-run
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
D:/Bit/
├── main.py                              # CLI 진입점 (collect, backtest, learn, live)
├── config/
│   ├── settings.py                      # Pydantic 설정 (.env 로드)
│   ├── constants.py                     # 열거형, 수수료, 리스크 상수
│   └── logging_config.py               # structlog 구조화 로깅
├── data/
│   ├── database.py                      # SQLite WAL 모드 DB 매니저
│   ├── models.py                        # 테이블 스키마 + 데이터클래스
│   ├── collector.py                     # ccxt OHLCV 수집
│   ├── funding_rates.py                 # 펀딩비 수집
│   └── live_feed.py                     # Bybit WebSocket 실시간
├── agents/
│   ├── base.py                          # AgentBase + ClaudeCLIRunner
│   ├── technical_analyst.py             # 정량 기술분석 (LLM 미사용)
│   ├── sentiment_analyst.py             # Claude CLI 센티먼트
│   ├── risk_reviewer.py                 # Decision Matrix 교차검증
│   ├── final_decider.py                 # Half-Kelly 포지션 사이징
│   └── supervisor.py                    # 감독관 (킬스위치)
├── strategy/
│   ├── signals.py                       # StrategyParameters + SignalOutput
│   ├── technical/
│   │   ├── rsi.py                       # RSI + 다이버전스
│   │   ├── bollinger.py                 # 볼린저밴드 %B + 스퀴즈
│   │   ├── volume.py                    # OBV, VWAP, 거래량 스파이크
│   │   ├── garch.py                     # GARCH(1,1) 변동성 체제
│   │   ├── momentum.py                  # MACD, ADX, +DI/-DI
│   │   └── composite.py                # 가중 합산 신호
│   └── sentiment/
│       ├── funding_analysis.py          # 펀딩비 센티먼트
│       ├── onchain_prompts.py           # Claude 시장 분석 프롬프트
│       └── macro_prompts.py             # 매크로/일간리뷰 프롬프트
├── risk/
│   ├── position_sizer.py               # 포지션 사이징
│   ├── kill_switch.py                   # 긴급 킬스위치
│   ├── correlation.py                   # 상관관계 분석
│   ├── cvar.py                          # CVaR (조건부 VaR)
│   ├── slippage.py                      # 슬리피지 추정
│   └── limits.py                        # 거래 제한 체크
├── backtest/
│   ├── engine.py                        # 이벤트 기반 백테스트
│   ├── cost_model.py                    # 수수료+슬리피지+펀딩비
│   ├── metrics.py                       # Sharpe, Sortino, Calmar, MDD
│   ├── data_splitter.py                 # Walk-forward 분할
│   ├── monte_carlo.py                   # 몬테카를로 시뮬레이션
│   └── report.py                        # 결과 리포트
├── learning/
│   ├── feedback_loop.py                 # 자가학습 오케스트레이터
│   ├── trade_analyzer.py               # 손실 분류 (8가지 실패 모드)
│   ├── claude_insights.py              # Claude CLI 인사이트
│   ├── parameter_tuner.py              # 파라미터 조정 (경계값 내)
│   └── iteration_tracker.py            # 반복 이력 + 수렴 감지
├── execution/
│   ├── order_manager.py                 # ccxt 주문 관리
│   ├── position_tracker.py             # 포지션 추적
│   ├── fill_handler.py                  # 체결 처리
│   └── rate_limiter.py                  # 토큰 버킷 제한
├── monitoring/
│   ├── telegram_bot.py                  # 텔레그램 알림
│   ├── dashboard.py                     # 콘솔 대시보드
│   ├── health_check.py                  # 헬스체크
│   └── logger.py                        # 거래 로깅
└── tests/
    ├── test_strategy.py                 # 기술지표 테스트 (26개)
    ├── test_agents.py                   # 에이전트 테스트 (11개)
    ├── test_backtest.py                 # 백테스트 테스트 (12개)
    ├── test_risk.py                     # 리스크 테스트 (12개)
    ├── test_learning.py                 # 학습 테스트 (11개)
    └── test_database.py                 # DB 테스트 (8개)
```

## Risk Management

| 제한 | 값 | 설명 |
|------|-----|------|
| 포지션 최대 | 25% | 자본 대비 단일 포지션 |
| 총 노출 최대 | 50% | 전체 포지션 합계 |
| 일일 손실 | 5% | 초과 시 킬스위치 발동 |
| 최대 낙폭 | 15% | 초과 시 킬스위치 발동 |
| 상관관계 | 0.7 | 초과 시 동시 포지션 차단 |
| 최대 레버리지 | 5x | 신뢰도 기반 조절 |

## Tech Stack

- **언어**: Python 3.13
- **거래소**: ccxt (Bybit Linear Futures)
- **데이터**: pandas, numpy, SQLite WAL
- **변동성**: arch (GARCH 모델)
- **AI**: Claude CLI subprocess
- **실시간**: Bybit WebSocket (websockets)
- **알림**: python-telegram-bot
- **테스트**: pytest (73개 테스트 통과)

## Claude CLI Integration

3곳에서 사용:
- **센티먼트 분석가**: 실시간 시장 감성 분석
- **학습 루프**: 백테스트 후 전략 조정 제안
- **감독관**: 일간 리뷰 및 위험 평가

```
명령어: claude -p "prompt" --output-format text
타임아웃: 120초
실패 시: NEUTRAL 폴백 (규칙 기반)
백테스트 중: 비활성화 (속도 + 사후확증 편향 방지)
```

## Contributors

| 기여자 | 역할 |
|--------|------|
| [@Parkjisu1](https://github.com/Parkjisu1) | 프로젝트 기획, 아키텍처 설계, 요구사항 정의 |
| **Claude (Anthropic)** | 전체 코드 구현, 테스트 작성, 문서화 |

---

Built with [Claude Code](https://claude.ai/claude-code)
