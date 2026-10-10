"""Telegram alerts for the monitor. Never raises: a failed alert is printed to the log instead.

Needs two environment variables (keep them in ~/.telegram_env on the VM, never in git):
    TELEGRAM_BOT_TOKEN   from @BotFather
    TELEGRAM_CHAT_ID     your chat with the bot; `python -m btcbot.alerts chat-id` prints it

    python -m btcbot.alerts chat-id     # after sending your bot any message
    python -m btcbot.alerts test        # sends a test message
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4000  # Telegram's limit is 4096 characters


def configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send(text: str) -> bool:
    """Send one message. Returns False (and prints the text) when Telegram is not set up or fails."""
    print(text, flush=True)
    if not configured():
        print("[alert not sent: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set]", flush=True)
        return False
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN - 20] + "\n...(truncated)"
    try:
        resp = requests.post(API.format(token=os.environ["TELEGRAM_BOT_TOKEN"], method="sendMessage"),
                             json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text,
                                   "disable_web_page_preview": True}, timeout=15)
        ok = resp.ok and resp.json().get("ok", False)
        if not ok:
            print(f"[alert failed: HTTP {resp.status_code} {resp.text[:200]}]", flush=True)
        return ok
    except Exception as e:  # network down must not crash the watchdog
        print(f"[alert failed: {e}]", flush=True)
        return False


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Telegram alert helper")
    p.add_argument("command", choices=["test", "chat-id"])
    a = p.parse_args(argv)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN is not set (run: . ~/.telegram_env)")
    if a.command == "chat-id":
        rows = requests.get(API.format(token=token, method="getUpdates"), timeout=15).json().get("result", [])
        chats = {r["message"]["chat"]["id"]: r["message"]["chat"].get("first_name", "") for r in rows if "message" in r}
        if not chats:
            sys.exit("No messages yet: open your bot in Telegram, send it 'hi', then run this again.")
        for cid, name in chats.items():
            print(f"TELEGRAM_CHAT_ID={cid}   ({name})")
        return
    sys.exit(0 if send("✅ 監察系統測試訊息：Telegram 設定成功。") else 1)


if __name__ == "__main__":
    main()
