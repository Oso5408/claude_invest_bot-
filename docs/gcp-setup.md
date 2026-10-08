# 喺 GCP 東京機房行模擬盤

模擬盤用 GMO Coin 嘅即時公開報價，用假錢跑 4 小時突破策略。唔使開 GMO 帳戶，亦唔使 API 金鑰，唔會落任何真單。

部機每 5 分鐘行一次 `python -m btcbot.paper step`。每次會做呢幾樣嘢：
- 檢查移動止損同斬倉
- 每日朝早 6 點收槓桿費
- 每當有一條新嘅 4 小時 K 線收市，就更新止損，或者喺有突破訊號嗰陣入市

## 1. 開 GCP 帳戶同項目

1. 去 <https://console.cloud.google.com>，用 Google 帳戶登入。
2. 新用戶要加信用卡開 billing。東京機房唔包喺免費方案入面，會收少少錢。
3. 頂部揀 **Select a project → New Project**，改個名，例如 `invest-bot`，然後 **Create**。
4. 建議設定預算提醒：左邊選單 **Billing → Budgets & alerts → Create budget**，例如每月 10 美元，超過就 email 通知你。

## 2. 開一部細機

1. 左邊選單揀 **Compute Engine → VM instances**。第一次會叫你 enable API，撳 **Enable**。
2. 撳 **Create instance**，然後填：
   - **Name：** `invest-bot`
   - **Region：** `asia-northeast1 (Tokyo)`；**Zone** 隨便揀
   - **Machine type：** `e2-micro`
   - **Boot disk：** 撳 **Change**，揀 `Debian GNU/Linux 12`，大小 10 GB
   - **Firewall：** 唔使剔任何選項
3. 右邊會顯示每月大概收費，睇清楚之後先撳 **Create**。

## 3. 連入部機

喺 VM instances 列表，撳 `invest-bot` 嗰行嘅 **SSH** 掣。瀏覽器會開一個黑色終端機視窗。之後嘅指令全部喺呢個視窗度打。

## 4. 安裝

逐行複製貼上：

```bash
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/Oso5408/claude_invest_bot-.git
cd claude_invest_bot-
git checkout claude/project-thread-vdvfwn   # PR merge 咗之後可以唔使打呢行
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

試吓連唔連到 GMO：

```bash
.venv/bin/python -m btcbot.data --symbol ADA_JPY --rules
```

見到 `{'symbol': 'ADA_JPY', 'minOrderSize': '10', ...}` 就得。

## 5. 手動行一次

```bash
.venv/bin/python -m btcbot.paper step --folder data/paper-2x
```

第一次會見到 `started: waiting for the next 4h candle to close`。佢要等下一條 4 小時 K 線收市先會開始交易，避免用舊訊號入市。

## 6. 設定每 5 分鐘自動行

我建議同時行兩個模擬帳戶，各有 30,000 日圓，比較兩個版本：
- `paper-2x`：做多同做空，最多 2 倍槓桿，即係你原本想要嘅設定。
- `paper-1x-long`：只做多，最多 1 倍。回測入面呢個版本回報更高、回撤更細。

```bash
crontab -e
```

第一次會問你揀編輯器，揀 `1`（nano）。去到檔案最尾，貼上以下兩行：

```
*/5 * * * * cd $HOME/claude_invest_bot- && .venv/bin/python -m btcbot.paper step --folder data/paper-2x >> data/paper-2x.log 2>&1
*/5 * * * * cd $HOME/claude_invest_bot- && .venv/bin/python -m btcbot.paper step --folder data/paper-1x-long --no-short --max-leverage 1 >> data/paper-1x-long.log 2>&1
```

撳 `Ctrl+O`、`Enter` 儲存，再撳 `Ctrl+X` 離開。

## 7. 睇結果

```bash
cd ~/claude_invest_bot-
.venv/bin/python -m btcbot.paper status --folder data/paper-2x
tail -5 data/paper-2x/equity.csv     # 最近幾次嘅價格、結餘同倉位
cat data/paper-2x/trades.csv         # 每單交易紀錄
tail -20 data/paper-2x.log           # 出錯嘅話睇呢度
```

如果想我幫你分析，將 `trades.csv` 同 `equity.csv` 嘅內容貼返嚟就得。

## 8. 更新程式

我改咗程式之後，喺部機度打：

```bash
cd ~/claude_invest_bot- && git pull
```

模擬帳戶嘅紀錄放喺 `data/paper-*` 入面，`git pull` 唔會刪走佢哋。

## 9. 停止

- **暫停：** 打 `crontab -e`，喺嗰兩行前面加 `#`。
- **完全唔用：** 喺 VM instances 撳 **Stop** 或者 **Delete**。停咗機就唔會再收機器嘅錢，但係硬碟仍然會收少少錢，delete 咗就全部唔收。
