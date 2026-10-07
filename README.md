# claude_invest_bot

BTC 自動交易研究用嘅程式。而家只做**研究同模擬**：攞歷史數據、跑策略、回測。冇真錢交易，唔使 API 金鑰。

數據來源：GMO Coin 公開 API（唔使開戶）。

## 安裝

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. 下載歷史數據

```bash
# BTC 現貨（日圓報價），1 小時 K 線
python -m btcbot.data --start 2023-01-01 --interval 1hour
# 存去 data/BTC_1hour.csv
```

- `--symbol BTC` 係現貨（預設），`BTC_JPY` 係槓桿。
- `--interval` 可以係 `1min 5min 10min 15min 30min 1hour 4hour 8hour 12hour 1day 1week 1month`。
- 1 小時或以下嘅 K 線每日一個請求，一年大約 365 個請求，要幾分鐘。

## 2. 策略（`btcbot/strategy.py`）

只做多（現貨唔可以沽空）。

| 部分 | 做咩 |
|---|---|
| 訊號 | 快均線（24 條 K）高過慢均線（96 條 K）就想持倉 |
| 時間窗口 | 只喺指定日本時間開新倉，平倉任何時間都得 |
| 波動率過濾 | 近期波動率要喺自己歷史嘅 20% 至 90% 百分位之間先開倉 |
| 貝葉斯更新 | 用 Beta 分佈估訊號勝率，每完成一單訊號交易就更新（就算冇落注都會學） |
| Kelly 注碼 | f* = p − (1 − p) / b，用四分一 Kelly，最多用 50% 本金 |

一開始假設勝率 50%、賺蝕比 1:1，即係冇優勢，所以 Kelly = 0，唔會落注。要等數據證明訊號有用先會開始落錢。

## 3. 回測（`btcbot/backtest.py`）

```bash
python -m btcbot.backtest --csv data/BTC_1hour.csv
python -m btcbot.backtest --csv data/BTC_1hour.csv --hours 9-17        # 只喺日本時間 9 點至 17 點開倉
python -m btcbot.backtest --csv data/BTC_1hour.csv --trades-out trades.csv
python -m btcbot.backtest --synthetic                                   # 用隨機數據試程式
```

規則：喺第 t 條 K 線收市時決定，第 t+1 條開市價成交（冇偷睇未來）。每次成交扣 0.05% 手續費同 0.02% 滑價。

報告包括：總回報、年化回報、最大回撤、Sharpe、已付手續費、交易次數、勝率，同埋直接持有 BTC 嘅回報同回撤做比較。

## 測試

```bash
python -m pytest -q
```

## 之後嘅步驟

4. 模擬盤：接即時報價，用假錢跑幾個星期
5. 小額實盤：API 金鑰只俾交易權限，唔好俾提款權限
6. 搵部長開嘅機持續運行
