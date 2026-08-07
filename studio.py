import os
import time
import asyncio
import requests
import html
from pyrogram import Client
from translator_engine import (
    process_phase1_engine,
    process_phase2_engine,
    process_phase2_from_local,
)

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")
TASK_TYPE = os.getenv("TASK_TYPE")
VIDEO_ID = os.getenv("VIDEO_ID")
SUB_ID = os.getenv("SUB_ID")
CHAT_ID = int(os.getenv("CHAT_ID"))
USER_ID = int(os.getenv("USER_ID"))
TRIGGER_MSG_ID = int(os.getenv("TRIGGER_MSG_ID"))
MODE = os.getenv("MODE", "normal")

WORK_DIR = "workspace"
os.makedirs(WORK_DIR, exist_ok=True)

WAIT_FOR_TXT_SECONDS = 560
POLL_INTERVAL_SECONDS = 8
MAX_TELEGRAM_DOC_MB = 1950

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _cancel_button(task_id):
    return {"inline_keyboard": [[{"text": "🛑 Cancel Task", "callback_data": f"cancel:{task_id}"}]]}


def update_status_http(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": CHAT_ID, "message_id": TRIGGER_MSG_ID, "text": text, "parse_mode": "HTML",
        "reply_markup": _cancel_button(TRIGGER_MSG_ID),
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


def too_large(path):
    try:
        return (os.path.getsize(path) / (1024 * 1024)) > MAX_TELEGRAM_DOC_MB
    except OSError:
        return False


def _bar(percent, width=20):
    filled = max(0, min(int(percent / 100 * width), width))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


class ThrottledReporter:
    def __init__(self, min_interval=8):
        self.min_interval = min_interval
        self.last = 0
        self.start = 0

    def reset(self):
        self.last = 0
        self.start = time.time()

    def maybe_report(self, label, current, total, unit="pages"):
        now = time.time()
        if now - self.last < self.min_interval and current != total:
            return
        self.last = now
        percent = (current / total * 100) if total else 0
        update_status_http(f"{label}\n{_bar(percent)} [{current}/{total} {unit}] ({percent:.0f}%)")


def make_transfer_progress(label):
    state = {"start": 0, "last": 0}

    def _cb(current, total):
        now = time.time()
        if state["start"] == 0:
            state["start"] = now
            state["last"] = now
            return
        if now - state["last"] < 8 and current != total:
            return
        state["last"] = now
        elapsed = now - state["start"]
        speed_mb = (current / elapsed / 1024 / 1024) if elapsed > 0 else 0
        percent = (current / total * 100) if total else 0
        update_status_http(
            f"{label}\n{_bar(percent)} [{percent:.1f}%]\n"
            f"🚀 {speed_mb:.2f} MB/s\n📦 {current / 1048576:.1f}MB / {total / 1048576:.1f}MB"
        )

    return _cb


async def download_telegram_file(app, file_link, output_name, label="📥 <b>Downloading...</b>"):
    try:
        msg_id = int(file_link.split("/")[-1])
        msg = await app.get_messages(CHAT_ID, msg_id)
        if msg and (msg.document or msg.video or msg.photo):
            target = msg.document or msg.video or msg.photo
            ext = os.path.splitext(target.file_name)[1] if hasattr(target, "file_name") and target.file_name else ".bin"
            out_path = os.path.join(WORK_DIR, output_name + ext)
            await msg.download(file_name=out_path, progress=make_transfer_progress(label))
            return out_path
    except Exception as e:
        print(f"Error downloading file: {e}")
    return None


async def wait_for_translated_txt(app, since_ts):
    """Poll USER's private chat for the .txt reply in a timezone-robust way."""
    from datetime import datetime, timezone
    since_dt = datetime.fromtimestamp(since_ts, timezone.utc)
    
    deadline = since_ts + WAIT_FOR_TXT_SECONDS
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            async for msg in app.get_chat_history(USER_ID, limit=15):
                if not msg.date:
                    continue
                
                # Align both times as UTC aware datetimes
                msg_date = msg.date
                if msg_date.tzinfo is None:
                    msg_date = msg_date.replace(tzinfo=timezone.utc)
                else:
                    msg_date = msg_date.astimezone(timezone.utc)
                
                # Allow a small clock drift grace window (5 seconds)
                if (since_dt - msg_date).total_seconds() > 5:
                    continue
                if not msg.from_user or msg.from_user.id != USER_ID:
                    continue
                if msg.document and msg.document.file_name and msg.document.file_name.lower().endswith(".txt"):
                    out_path = os.path.join(WORK_DIR, "translated_text.txt")
                    await msg.download(file_name=out_path)
                    return out_path
        except Exception as e:
            print(f"Polling error (will retry): {e}")
    return None


async def main():
    app = Client("worker_engine", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, no_updates=True)
    await app.start()

    # Pre-resolve peer cache to avoid PeerIdInvalid errors during polling
    try:
        await app.get_chat(USER_ID)
    except Exception as e:
        print(f"Pre-resolving USER_ID failed: {e}")

    page_reporter = ThrottledReporter()

    try:
        if TASK_TYPE == "process":
            doc_path = await download_telegram_file(app, VIDEO_ID, "manga_raw", "📥 <b>Downloading Manga Assets...</b>")
            if not doc_path:
                raise Exception("Failed to download input Manga asset.")

            page_reporter.reset()

            def page_progress(done, total):
                page_reporter.maybe_report("⚙️ <b>Detecting Bubbles & Preparing Pages...</b>", done, total)

            backup_zip, txt_path, clean_dir, translation_map = await asyncio.to_thread(
                process_phase1_engine, doc_path, WORK_DIR, MODE, page_progress, font_path=FONT_PATH
            )

            if too_large(txt_path) or too_large(backup_zip):
                raise Exception("Generated files exceed the upload limit for this manga — try a smaller batch.")

            update_status_http("📤 <b>Sending Backup Pack & Translation Template...</b>")
            await app.send_document(
                chat_id=USER_ID, document=txt_path,
                caption=f"📝 <b>Translation Template File</b>\nMode: <code>{MODE.upper()}</code>\nEdit this file and send it back to translate.",
            )
            await app.send_document(
                chat_id=USER_ID, document=backup_zip,
                caption="📦 <b>Manga_Backup.zip</b>\nContains clean and numbered reference pages. Keep this safe for `/repeat` mode if you're busy or the wait times out.",
            )

            wait_since = time.time()
            update_status_http(
                "⏳ <b>Waiting for your translated .txt file...</b>\n"
                "Reply within 10 minutes, or use <code>/repeat</code> anytime later with the Backup ZIP above."
            )

            translated_txt_path = await wait_for_translated_txt(app, wait_since)

            if not translated_txt_path:
                update_status_http(
                    "⌛ <b>Timeout!</b>\nNo translation received in time.\n"
                    "Use <code>/repeat</code> anytime with the Backup ZIP + your translated .txt to finish this."
                )
                await app.stop()
                return

            page_reporter.reset()

            def render_progress(done, total):
                page_reporter.maybe_report("⚙️ <b>Rendering Translation Dialogues onto Canvas...</b>", done, total)

            final_zip = await asyncio.to_thread(
                process_phase2_from_local, clean_dir, translation_map, translated_txt_path, WORK_DIR, render_progress, font_path=FONT_PATH
            )

            if too_large(final_zip):
                raise Exception("Final output exceeds the upload limit — try a smaller batch or fewer pages.")

            await app.send_document(
                chat_id=USER_ID, document=final_zip,
                caption="✅ <b>Compilation Completed!</b>\nHere is your translated manga book.",
                progress=make_transfer_progress("📤 <b>Sending final Translated Document...</b>"),
            )
            await app.delete_messages(CHAT_ID, TRIGGER_MSG_ID)

        elif TASK_TYPE == "repeat":
            backup_zip_path = await download_telegram_file(app, VIDEO_ID, "backup_pack", "📥 <b>Repeat Mode: Loading Backup ZIP...</b>")
            txt_path = await download_telegram_file(app, SUB_ID, "translated_text", "📥 <b>Repeat Mode: Loading Translation...</b>")

            if not backup_zip_path or not txt_path:
                raise Exception("Assets missing. Ensure you uploaded the correct Backup ZIP and .txt files.")

            page_reporter.reset()

            def render_progress(done, total):
                page_reporter.maybe_report("⚙️ <b>Rendering via Repeat Parser Engine...</b>", done, total)

            final_zip = await asyncio.to_thread(
                process_phase2_engine, backup_zip_path, txt_path, WORK_DIR, render_progress, font_path=FONT_PATH
            )

            if too_large(final_zip):
                raise Exception("Final output exceeds the upload limit — try a smaller batch or fewer pages.")

            await app.send_document(
                chat_id=USER_ID, document=final_zip,
                caption="✅ <b>Repeat Compilation Completed!</b>\nProcessed successfully from your backup ZIP.",
                progress=make_transfer_progress("📤 <b>Delivering final clean pages...</b>"),
            )
            await app.delete_messages(CHAT_ID, TRIGGER_MSG_ID)

        else:
            raise Exception(f"Unknown TASK_TYPE: {TASK_TYPE}")

    except Exception as e:
        update_status_http(f"❌ <b>Process Interrupted:</b>\n<code>{html.escape(str(e))}</code>")

    await app.stop()


if __name__ == "__main__":
    asyncio.run(main())
