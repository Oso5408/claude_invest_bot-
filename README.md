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

## ADA 2 倍槓桿均值回歸（`btcbot/meanrev.py`）

用 GMO Coin 暗號資產 FX（槓桿）嘅 ADA/JPY，本金預設 30,000 日圓（大約 200 美元）。只係回測，唔會落真單。

```bash
# 1. 睇 ADA_JPY 嘅落單規則（最少落單量、單位）
python -m btcbot.data --symbol ADA_JPY --rules
# 2. 下載數據（ADA/JPY 槓桿 2024 年 5 月 25 日先開始）
python -m btcbot.data --symbol ADA_JPY --start 2024-05-25
# 3. 回測
python -m btcbot.meanrev --csv data/ADA_JPY_1hour.csv
```

| 部分 | 做咩 |
|---|---|
| 訊號 | 價格偏離 48 小時平均超過 2 個標準差就反向入市（跌得太多買、升得太多沽） |
| 大趨勢過濾 | 4 小時 K 線 200 EMA：價格喺 EMA 之上只准做多，之下只准做空（只用已經收市嘅 4 小時 K 線） |
| 平倉 | 回到平均、偏離超過 4 個標準差（止蝕）、或者揸咗 72 小時 |
| 波動率過濾 | 波動率喺最高 10% 嗰陣唔開新倉 |
| 注碼（RiskController） | 頭 30 單固定 0.5 倍槓桿。之後用四分一 Kelly：f* = (p − (1 − p) / b) / 平均虧損，只有正期望值先會落注 |
| 2 倍上限 | `btcbot/risk.py` 入面寫死 `MAX_LEVERAGE = 2.0`。每張單都會被截到 2 倍以內，帳戶喺每次成交再檢查，超過就直接報錯停低 |
| 槓桿費 | 每日 06:00 仍然持倉，收倉位價值 0.04% |
| 斬倉 | 保證金維持率跌到 75% 即時斬倉 |
| 交易費 | ADA_JPY taker 0.03%（maker 0%），另外每次成交計 0.05% 滑價。最少 10 ADA，每 10 ADA 一個單位 |

## 策略 B：突破 + 移動止損（`btcbot/breakout.py`）

```bash
python -m btcbot.breakout --csv data/ADA_JPY_1hour.csv                  # 1 小時 K 線
python -m btcbot.breakout --csv data/ADA_JPY_1hour.csv --timeframe 4h   # 合併成 4 小時 K 線
python -m btcbot.breakout --csv data/ADA_JPY_1hour.csv --no-short       # 只做多
python -m btcbot.breakout --csv data/ADA_JPY_1hour.csv --timeframe 4h --no-short --max-leverage 1   # Alpaca 現貨：只做多、1 倍
```

| 部分 | 做咩 |
|---|---|
| 做多 | 收市價突破之前 20 條 K 線最高價，而且成交量大過之前 20 條平均嘅 1.2 倍 |
| 做空 | 收市價跌穿之前 10 條 K 線最低價 |
| 平倉 | 只用移動止損：入市價 ∓ 2 × ATR(14)，之後跟住最好價格移動，只會收緊唔會放鬆 |
| 注碼 | RiskController 預設勝率 35%、盈虧比 2.5、平均虧損 2%（當做已經有 30 單），第一單起用四分一 Kelly（約 1.1 倍），之後按真實交易更新 |
| 風控 | 同均值回歸一樣：2 倍上限、手續費、槓桿費、75% 斬倉 |

## 中性網格（`btcbot/grid.py`）

```bash
python -m btcbot.grid --csv data/ADA_JPY_1hour.csv
python -m btcbot.grid --csv data/ADA_JPY_1hour.csv --levels 20 --spacing 0.02
```

- 以建網格嗰陣嘅價格做中心，上下各 10 條線，每條相隔 1%（即 ±10%）。
- 價格每跌穿一條線就加一份多單（或者平一份空單），每升穿一條線就加一份空單（或者平一份多單）。
- 每份大小：價格去到區間邊位嗰陣，總倉位啱啱係 2 倍槓桿。
- 價格走出區間：全部即市平倉，再用新價格做中心重新起網格。
- 網格單係限價單，用 maker 手續費（ADA_JPY 係 0%）；走出區間平倉用 taker 手續費加滑價。

## 多時間框架帶量突破（`btcbot/breakout.py --mtf`）

```bash
python -m btcbot.data --symbol ADA_JPY --interval 15min --start 2024-05-25
python -m btcbot.breakout --csv data/ADA_JPY_15min.csv --mtf
```

- 大趨勢：4 小時 K 線 EMA50，價格喺上面只准做多，喺下面只准做空。
- 入市：15 分鐘 K 線收市突破之前 20 條最高價（做多）或者跌穿之前 20 條最低價（做空），而且成交量大過之前 20 條平均嘅 1.5 倍。
- 平倉同注碼：同策略 B 一樣（2 × ATR 移動止損，RiskController 預設勝率 35%、盈虧比 2.5）。

## 恐懼與貪婪指數過濾（`--fng-csv`）

```bash
python -m btcbot.data --fng      # alternative.me 免費每日數據，存去 data/fear_greed.csv
python -m btcbot.breakout --csv data/ADA_JPY_1hour.csv --timeframe 4h --fng-csv data/fear_greed.csv --fng-short 0 50
```

- `--fng-long MIN MAX`、`--fng-short MIN MAX`：指數喺呢個範圍入面先准做多／做空。預設 0 至 100，即係唔過濾。
- 每日嘅數值第二日先用，避免用到當時未公佈嘅數字。
- 模擬盤用 `--fng-short-max 50`：指數 50 或以下先准做空。攞唔到指數嗰次就唔做空。

## 模擬盤（`btcbot/paper.py`）

用 GMO 即時公開報價，用假錢跑 4 小時突破策略。唔使開帳戶，亦唔使 API 金鑰。喺 GCP 東京機房設定嘅步驟見 [docs/gcp-setup.md](docs/gcp-setup.md)。

```bash
python -m btcbot.paper step --folder data/paper-2x                                   # 每 5 分鐘行一次
python -m btcbot.paper step --folder data/paper-1x-long --no-short --max-leverage 1
python -m btcbot.paper status --folder data/paper-2x
```

## Claude 參與決定（`btcbot/advisor.py`）

- `python -m btcbot.paper step ... --claude`：每次開倉前問 Claude（go／half／skip）。Claude 只可以減細或者取消，2 倍上限照樣鎖死；連唔到 Claude 就照規則做。決定記喺 `claude.csv`。
- `python -m btcbot.advisor review`：Claude 睇晒所有 `data/paper-*` 帳戶，寫繁體中文總結去 `data/reviews/`。
- 兩樣都要 `ANTHROPIC_API_KEY`。設定步驟見 [docs/gcp-setup.md](docs/gcp-setup.md) 第 9 節。
- 注意：Claude 嘅判斷冇辦法用歷史數據回測，只可以同其他模擬帳戶並排比較。

## 測試

```bash
python -m pytest -q
```

## 之後嘅步驟

4. 模擬盤：接即時報價，用假錢跑幾個星期
5. 小額實盤：API 金鑰只俾交易權限，唔好俾提款權限
6. 搵部長開嘅機持續運行
