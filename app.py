"""
================================================================================
 Concept Science Academy (CWL) - WhatsApp Hybrid AI Chatbot
================================================================================
Stack: Python Flask + WhatsApp Cloud API (Meta) + Google Gemini 1.5 Flash
Hosting: Render.com (Free Tier)

Yeh chatbot student queries ka jawab deta hai (Roman Urdu + English mein),
aur agar student "human se baat" mangay ya bot ko jawab na aaye, to chat
ko "Human Handover" state mein daal deta hai — matlab bot us number par
auto-reply band kar deta hai taake Admin manually chat sambhal sake.
================================================================================
"""

import os
import json
import logging
import threading
from datetime import datetime

import requests
from flask import Flask, request, jsonify
import google.generativeai as genai

# ------------------------------------------------------------------------
# 1. CONFIGURATION -- Environment Variables se load ho raha hai
# ------------------------------------------------------------------------
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
WHATSAPP_TOKEN   = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID  = os.environ.get("PHONE_NUMBER_ID")
VERIFY_TOKEN     = os.environ.get("VERIFY_TOKEN", "csa_verify_token")
ADMIN_NUMBER     = os.environ.get("ADMIN_NUMBER", "923006498489")  # bina '+' aur bina space ke

# Basic sanity check (Render logs mein dikhayega agar koi variable missing hai)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("csa-bot")

if not all([GEMINI_API_KEY, WHATSAPP_TOKEN, PHONE_NUMBER_ID]):
    log.warning("⚠️  Kuch environment variables missing hain. Render Dashboard mein check karein.")

# Gemini configure karo
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

WHATSAPP_API_URL = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"

# ------------------------------------------------------------------------
# 2. HANDOVER STATE -- kis number ka bot "paused" hai
# ------------------------------------------------------------------------
# Render Free tier ka dyno restart ho sakta hai, is liye hum ek JSON file
# mein bhi state save kar rahe hain (best-effort persistence). Agar restart
# ho bhi jaye, memory dictionary fori (immediate) source of truth hai.
STATE_FILE = "handover_state.json"
_state_lock = threading.Lock()

# Structure: { "923001234567": {"paused": True, "since": "2026-07-31T10:00:00", "reason": "..."} }
HANDOVER_STATE = {}

# Chat history bhi thora sa yaad rakhte hain (last few messages) taake Gemini
# ko context mile. Structure: { "923001234567": [{"role": "user"/"model", "text": "..."}] }
CHAT_HISTORY = {}
MAX_HISTORY_TURNS = 8  # per user, last N messages yaad rakho


def _load_state():
    """Startup par purani state file se load karo (agar mojood ho)."""
    global HANDOVER_STATE
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                HANDOVER_STATE = json.load(f)
                log.info(f"Loaded handover state for {len(HANDOVER_STATE)} users.")
    except Exception as e:
        log.error(f"State load nahi hui: {e}")


def _save_state():
    """State ko disk par save karo (best-effort, thread-safe)."""
    try:
        with _state_lock:
            with open(STATE_FILE, "w") as f:
                json.dump(HANDOVER_STATE, f, indent=2)
    except Exception as e:
        log.error(f"State save nahi hui: {e}")


def is_paused(phone_number: str) -> bool:
    return HANDOVER_STATE.get(phone_number, {}).get("paused", False)


def pause_bot(phone_number: str, reason: str = "handover"):
    HANDOVER_STATE[phone_number] = {
        "paused": True,
        "since": datetime.utcnow().isoformat(),
        "reason": reason,
    }
    _save_state()
    log.info(f"⏸️  Bot paused for {phone_number} (reason: {reason})")


def resume_bot(phone_number: str):
    if phone_number in HANDOVER_STATE:
        HANDOVER_STATE[phone_number]["paused"] = False
        _save_state()
    log.info(f"▶️  Bot resumed for {phone_number}")


_load_state()

# ------------------------------------------------------------------------
# 3. ACADEMY KNOWLEDGE BASE / GEMINI SYSTEM PROMPT
# ------------------------------------------------------------------------
# NOTE: Fee structure exact amounts yahan nahi dali gayin kyunke aapne wo
# provide nahi ki. Neeche "[FEE STRUCTURE YAHAN DALEIN]" jagah par apni
# actual fees likh dein. Jab tak sahi info na ho, bot honestly "confirm
# karwata hoon" bolega aur handover trigger karega -- yeh intentional hai
# taake galat info student ko na jaye.

SYSTEM_PROMPT = """
Tum "Concept Science Academy (CWL)" ke official WhatsApp AI Assistant ho.
Tumhara kaam hai students aur parents ke sawalat ka friendly, polite aur
concise andaz mein jawab dena — Roman Urdu ya English mein, jis language
mein student baat kare usi mein reply karo (natural mix bhi chalay ga,
jaise log Pakistan mein WhatsApp par likhte hain).

=== ACADEMY INFORMATION (Sirf yehi authoritative data hai) ===
- Naam: Concept Science Academy (CWL)
- Website: csakwl.com
- Admin / WhatsApp Contact: +92-300-649-8489
- Location: Khanewal, Punjab, Pakistan

Courses / Offerings:
1. Metric (9th & 10th) - Science aur Arts groups
2. Intermediate (11th & 12th) - FSc Pre-Medical, FSc Pre-Engineering, ICS
3. Entry Test Preparation - MDCAT aur ECAT

Fee Structure & Exact Class Timings:
[FEE STRUCTURE YAHAN DALEIN — abhi available nahi hai]
Agar student fee ya exact timings pooche, to unhe batao ke exact detail
confirm karne ke liye Admin se connect kara rahe ho.

=== JAWAB DENE KA STYLE (professional support jaisa) ===
- Jab bhi mumkin ho, jawab ko structured aur clear rakho: agar steps
  batane hain to numbered list ya line-breaks use karo (jaise koi
  professional support agent likhta hai), sirf ek lambi paragraph mein
  mat thoons do.
- Course/admission info detail se do — group, duration, kis exam ke
  liye prepare karata hai — na ke sirf ek line mein.
- Agar student ka koi pichla reference ho (naam, course already bataya
  ho), to usay dobara mention kar ke continuity dikhao, jaise:
  "Jee [Naam], jaisa aapne bataya ke aap FSc Pre-Medical mein dilchaspi
  rakhte hain..."
- Message ke aakhir mein chota sa professional closing rakho jab
  koi substantial jawab de rahe ho, jaise:
  "Koi aur sawal ho to zaroor poochein. 🙌
  Regards,
  Concept Science Academy (CWL) Support"
  (Chhoti greetings ya haan/na jawabon par yeh closing zaroori nahi.)

=== TUMHARE RULES (bohat zaroori) ===
1. SIRF upar diye gaye information ka istemal karo. Kabhi bhi khud se
   fees, dates, ya seats invent (bana kar) mat batana.
2. Agar student:
   a) directly Admin/insaan se baat karna chahay (e.g. "admin se baat
      karni hai", "talk to human", "call me", "kisi se baat karwao",
      "agent chahiye"), YA
   b) koi aisa sawal poochay jiska jawab upar ki information mein
      mojood NAHI hai (e.g. exact fee amount, exact class timing,
      specific teacher ka naam, refund policy, waghera),
   to tumhe reply ke bilkul AAKHIR mein is exact tag ko add karna hai
   (naya line par, kuch aur nahi likhna is line par):
   [[HANDOVER]]
   Is tag se pehle yeh professional message likho (student ka context
   agar maloom ho to short mein acknowledge kar ke):
   "Aapki request hamare live agent ko forward kar di gayi hai, wo jald
   hi is chat mein shamil ho kar aapki madad karein gay."
3. Admission ke baare mein poochne par politely student ka NAAM aur
   INTENDED COURSE zaroor poochlo (agar pehle se maloom na ho), aur
   phir unhe us course ki detail (duration, kis exam ke liye hai)
   professionally batao.
4. Tone hamesha polite, encouraging aur clear rakho. Detail dena mana
   nahi hai (jab genuinely madadgar ho), lekin WhatsApp jaisi
   readable chunks mein — bade block paragraphs se bacho.
5. Kabhi bhi apne aap ko "Google" ya "Gemini" mat kaho — tum "Concept
   Science Academy Support" ho.
6. Agar student sirf greeting kare ("assalam o alaikum", "hi"), to
   warmly welcome karo aur pucho kis cheez mein madad chahiye.
"""

HANDOVER_TAG = "[[HANDOVER]]"

# Keywords jo directly human-handover trigger karte hain (fast-path,
# Gemini call se pehle hi check ho jata hai)
HUMAN_REQUEST_KEYWORDS = [
    "admin se baat", "baat karni hai", "talk to human", "talk to admin",
    "call me", "human chahiye", "agent chahiye", "insan se baat",
    "kisi se baat", "real person", "speak to someone", "admin number",
    "contact admin", "human se baat", "representative",
]

FALLBACK_MESSAGE = (
    "Aapki request hamare live agent ko forward kar di gayi hai, wo jald "
    "hi is chat mein shamil ho kar aapki madad karein gay. 🙏\n\n"
    "Aap chahein to seedha bhi contact kar sakte hain: +92-300-649-8489\n\n"
    "Regards,\nConcept Science Academy (CWL) Support"
)


# ------------------------------------------------------------------------
# 4. GEMINI CALL
# ------------------------------------------------------------------------
def ask_gemini(phone_number: str, user_message: str) -> str:
    """Gemini 1.5 Flash ko system prompt + history + naya message bhejo."""
    model = genai.GenerativeModel(
        model_name="gemini-3.5-flash-lite",
        system_instruction=SYSTEM_PROMPT,
    )

    history = CHAT_HISTORY.get(phone_number, [])

    # Gemini chat format mein history convert karo
    gemini_history = [
        {"role": h["role"], "parts": [h["text"]]} for h in history
    ]

    try:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_message)
        reply_text = response.text.strip()
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        # Agar Gemini fail ho jaye to safe fallback -> handover
        return f"Mujhe abhi thodi dikkat ho rahi hai jawab dene mein. {FALLBACK_MESSAGE} {HANDOVER_TAG}"

    # History update karo (max N turns rakho)
    history.append({"role": "user", "text": user_message})
    history.append({"role": "model", "text": reply_text})
    CHAT_HISTORY[phone_number] = history[-(MAX_HISTORY_TURNS * 2):]

    return reply_text


def contains_human_request(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in HUMAN_REQUEST_KEYWORDS)


# ------------------------------------------------------------------------
# 5. WHATSAPP SEND MESSAGE FUNCTION
# ------------------------------------------------------------------------
def send_whatsapp_message(to_number: str, message_body: str):
    """Meta WhatsApp Cloud API ke zariye text message bhejta hai."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_body},
    }
    try:
        resp = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            log.error(f"WhatsApp send failed [{resp.status_code}]: {resp.text}")
        return resp.json()
    except Exception as e:
        log.error(f"WhatsApp send exception: {e}")
        return None


def send_whatsapp_template(to_number: str, template_name: str, params: list):
    """
    Approved WhatsApp Template message bhejta hai. Yeh business-initiated
    messages (24-hour window ke bahar) ke liye zaroori hai -- plain text
    is case mein kaam nahi karta, sirf approved template chalta hai.
    """
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": "en_US"},
            "components": [
                {
                    "type": "body",
                    "parameters": [{"type": "text", "text": p} for p in params],
                }
            ],
        },
    }
    try:
        resp = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=15)
        if resp.status_code != 200:
            log.error(f"WhatsApp template send failed [{resp.status_code}]: {resp.text}")
        return resp.json()
    except Exception as e:
        log.error(f"WhatsApp template send exception: {e}")
        return None


def notify_admin(student_number: str, last_message: str):
    """Admin ko inform karo ke kisi student ko personal attention chahiye.
    Yeh 'handover_alert' naam ka approved template use karta hai, kyunke
    Admin ne agar pichle 24 ghante mein bot ko message na kiya ho, to plain
    text WhatsApp API se reject ho jata hai (business-initiated message rule).
    """
    # Message ko chota rakho (WhatsApp template variables mein newline allowed nahi)
    short_message = last_message.replace("\n", " ").strip()[:200]
    send_whatsapp_template(
        ADMIN_NUMBER,
        template_name="handover_alert",
        params=[f"+{student_number}", short_message],
    )


# ------------------------------------------------------------------------
# 6. WEBHOOK -- GET (Meta Verification)
# ------------------------------------------------------------------------
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        log.info("✅ Webhook verified successfully.")
        return challenge, 200
    else:
        log.warning("❌ Webhook verification failed.")
        return "Verification failed", 403


# ------------------------------------------------------------------------
# 7. WEBHOOK -- POST (Incoming Messages)
# ------------------------------------------------------------------------
@app.route("/webhook", methods=["POST"])
def handle_webhook():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "no data"}), 200

    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages")

        if not messages:
            # Ho sakta hai yeh status update ho (delivered/read) - ignore karo
            return jsonify({"status": "ignored"}), 200

        message = messages[0]
        from_number = message.get("from")  # e.g. "923001234567"
        msg_type = message.get("type")

        if msg_type != "text":
            send_whatsapp_message(
                from_number,
                "Filhal main sirf text messages samajh sakta hoon. Barah-e-karam "
                "apna sawal likh kar bhejein. 🙏",
            )
            return jsonify({"status": "ok"}), 200

        user_text = message["text"]["body"].strip()
        log.info(f"📩 From {from_number}: {user_text}")

        # ---- Admin Command Handling: /resume <number> ----
        if from_number == ADMIN_NUMBER and user_text.lower().startswith("/resume"):
            parts = user_text.split()
            if len(parts) == 2:
                target = parts[1].replace("+", "").strip()
                resume_bot(target)
                send_whatsapp_message(ADMIN_NUMBER, f"✅ Bot resumed for +{target}")
            else:
                send_whatsapp_message(ADMIN_NUMBER, "Usage: /resume 923001234567")
            return jsonify({"status": "ok"}), 200

        # ---- Agar yeh number pehle se paused hai, bot chup rahega ----
        if is_paused(from_number):
            log.info(f"🔇 Bot paused for {from_number}, auto-reply skip.")
            return jsonify({"status": "paused"}), 200

        # ---- Fast-path: direct human request keywords ----
        if contains_human_request(user_text):
            send_whatsapp_message(from_number, FALLBACK_MESSAGE)
            pause_bot(from_number, reason="user requested human")
            notify_admin(from_number, user_text)
            return jsonify({"status": "handover"}), 200

        # ---- Gemini se jawab lo ----
        reply = ask_gemini(from_number, user_text)

        if HANDOVER_TAG in reply:
            clean_reply = reply.replace(HANDOVER_TAG, "").strip()
            send_whatsapp_message(from_number, clean_reply)
            pause_bot(from_number, reason="info not in knowledge base")
            notify_admin(from_number, user_text)
        else:
            send_whatsapp_message(from_number, reply)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        log.error(f"Webhook processing error: {e}")
        return jsonify({"status": "error", "detail": str(e)}), 200


# ------------------------------------------------------------------------
# 8. HEALTH CHECK (Render ke liye, aur UptimeRobot jaisi services ke liye)
# ------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "academy": "Concept Science Academy (CWL)",
        "paused_users": len([k for k, v in HANDOVER_STATE.items() if v.get("paused")]),
    }), 200


# ------------------------------------------------------------------------
# 9. LOCAL RUN (Render Gunicorn se run karta hai, yeh sirf local testing ke liye)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
