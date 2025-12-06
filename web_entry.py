# web_entry.py
from threading import Thread
from flask import Flask
import os
import bot_hybrid_smart_early as bot_module  # if your file is bot_hybrid_smart_early.py

app = Flask("bot_service")

@app.route("/healthz")
def health():
    return "ok", 200

def run_bot():
    bot_module.main()  # assumes main() runs the loop

if __name__ == "__main__":
    t = Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
