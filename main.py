import os
import secrets
import html
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from supabase import create_client
from dotenv import load_dotenv
import stripe

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"],
    default_limits_exempt_when=lambda: request.method == "OPTIONS"
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
ADMIN_BYPASS_KEY = "2026JU"

client = Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=0)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
stripe.api_key = STRIPE_SECRET_KEY


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Admin-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    return response

MODEL_MAP = {
    "groq": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
}

MODEL_ALIASES = {v["model"]: k for k, v in MODEL_MAP.items()}


def build_prompt(topic, style):
    if style == "eli5":
        return f"Explain this like I'm 5 in 3 short sentences: {topic}"
    if style == "bullets":
        return f"Answer in 3-5 bullet points: {topic}"
    if style == "detailed":
        return f"Provide a detailed answer in 6-8 sentences: {topic}"
    return f"Explain this clearly in 3 sentences: {topic}"


def build_messages(topic, style, custom_instructions=None, chat_history=None):
    prompt = build_prompt(topic, style)
    system_parts = []
    if custom_instructions:
        system_parts.append(custom_instructions.strip())
    if system_parts:
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    else:
        messages = []

    if chat_history:
        for item in chat_history:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in ["user", "assistant"] and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages, prompt


def generate_answer(
    model_choice,
    topic,
    style,
    image_base64=None,
    chat_history=None,
    custom_instructions=None
):
    resolved_key = MODEL_ALIASES.get(model_choice, model_choice)
    config = MODEL_MAP.get(resolved_key)
    if not config:
        return None, "Unsupported model selection.", 400

    provider = config["provider"]
    model_name = config["model"]

    if provider == "groq":
        if not GROQ_API_KEY:
            return None, "Server misconfigured: missing GROQ_API_KEY.", 500
        messages, prompt = build_messages(
            topic,
            style,
            custom_instructions=custom_instructions,
            chat_history=chat_history
        )
        model_to_use = model_name

        if image_base64:
            image_url = image_base64.strip()
            if not image_url.startswith("data:image/"):
                image_url = f"data:image/jpeg;base64,{image_url}"
            model_to_use = "meta-llama/llama-4-scout-17b-16e-instruct"
            vision_prompt = f"Answer this question about the image: {topic}"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }]

        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages
        )
        return response.choices[0].message.content, None, 200

    return None, "Unsupported model selection.", 400

def get_current_user(request):
    auth_header = request.headers.get("Authorization")
    print(f"Auth header received: {auth_header}")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        response = supabase.auth.get_user(token)
        return response.user
    except Exception as e:
        print(f"Auth error: {e}")
        return None


def is_pro_user(user_id):
    try:
        response = supabase.table("subscriptions").select("status").eq(
            "user_id", user_id
        ).limit(1).execute()
        rows = response.data or []
        if not rows:
            return False
        status = rows[0].get("status")
        return status in ["active", "trialing"]
    except Exception as e:
        print(f"Subscription lookup error: {e}")
        return False


def count_questions_today(user_id):
    start_of_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()
    response = supabase.table("questions").select(
        "id", count="exact"
    ).eq("user_id", user_id).gte("created_at", start_of_day).execute()
    return response.count or 0


@app.route("/ask", methods=["POST"])
@limiter.limit("10 per minute")
def ask():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    data = request.json
    topic = data.get("topic", "")
    model_choice = data.get("model", "groq")
    style = data.get("style", "normal")
    image_base64 = data.get("image_base64")
    chat_history = data.get("chat_history") if isinstance(data.get("chat_history"), list) else None
    custom_instructions = data.get("custom_instructions", "")

    if not topic or len(topic) > 500:
        return jsonify({"error": "Invalid input"}), 400

    is_pro = is_pro_user(user.id)

    if not is_pro:
        if style not in ["normal", "eli5"]:
            return jsonify({"error": "Upgrade to Pro to use this response style."}), 403
        if image_base64:
            return jsonify({"error": "Upgrade to Pro to ask questions about images."}), 403
        try:
            asked_today = count_questions_today(user.id)
        except Exception as e:
            print(f"Usage count error: {e}")
            return jsonify({"error": "Usage check failed."}), 500
        if asked_today >= 5:
            return jsonify({
                "error": "You've used all 5 free questions today. Upgrade to Pro for unlimited!"
            }), 403

    try:
        answer, error_msg, status_code = generate_answer(
            model_choice,
            topic,
            style,
            image_base64=image_base64 if is_pro else None,
            chat_history=chat_history,
            custom_instructions=custom_instructions
        )
        if error_msg:
            return jsonify({"error": error_msg}), status_code
        if not answer:
            return jsonify({"error": "AI service returned no answer."}), 502
    except Exception as e:
        print(f"AI error: {e}")
        return jsonify({"error": "AI service timed out or failed. Please try again."}), 502

    supabase.table("questions").insert({
        "question": topic,
        "answer": answer,
        "user_id": user.id
    }).execute()

    return jsonify({"answer": answer})


@app.route("/ask-admin", methods=["POST"])
@limiter.limit("10 per minute")
def ask_admin():
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != ADMIN_BYPASS_KEY:
        return jsonify({"error": "Unauthorized."}), 401

    data = request.json or {}
    topic = data.get("topic", "")
    model_choice = data.get("model", "groq")
    style = data.get("style", "normal")
    image_base64 = data.get("image_base64")
    chat_history = data.get("chat_history") if isinstance(data.get("chat_history"), list) else None
    custom_instructions = data.get("custom_instructions", "")

    if not topic or len(topic) > 500:
        return jsonify({"error": "Invalid input"}), 400

    try:
        answer, error_msg, status_code = generate_answer(
            model_choice,
            topic,
            style,
            image_base64=image_base64,
            chat_history=chat_history,
            custom_instructions=custom_instructions
        )
        if error_msg:
            return jsonify({"error": error_msg}), status_code
        if not answer:
            return jsonify({"error": "AI service returned no answer."}), 502
    except Exception as e:
        print(f"Admin AI error: {e}")
        return jsonify({"error": "AI service timed out or failed. Please try again."}), 502

    return jsonify({"answer": answer})


@app.route("/subscription-status", methods=["GET"])
def subscription_status():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    return jsonify({"status": "pro" if is_pro_user(user.id) else "free"})


@app.route("/saved-chats", methods=["GET"])
def list_saved_chats():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    try:
        response = supabase.table("saved_chats").select(
            "id,name,messages,created_at"
        ).eq("user_id", user.id).order("created_at", desc=True).limit(50).execute()
        return jsonify({"items": response.data or []})
    except Exception as e:
        print(f"Saved chats list error: {e}")
        return jsonify({"error": "Failed to load saved chats."}), 500


@app.route("/saved-chats", methods=["POST"])
def save_chat():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    data = request.json or {}
    name = (data.get("name") or "").strip()
    messages = data.get("messages")
    if not name:
        return jsonify({"error": "Chat name is required."}), 400
    if not isinstance(messages, list) or len(messages) < 2:
        return jsonify({"error": "Need at least 2 messages to save."}), 400
    try:
        response = supabase.table("saved_chats").insert({
            "user_id": user.id,
            "name": name,
            "messages": messages,
        }).execute()
        rows = response.data or []
        return jsonify({"item": rows[0] if rows else None})
    except Exception as e:
        print(f"Save chat error: {e}")
        return jsonify({"error": "Failed to save chat."}), 500


@app.route("/saved-chats/<chat_id>", methods=["DELETE"])
def delete_saved_chat(chat_id):
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    try:
        supabase.table("saved_chats").delete().eq("id", chat_id).eq("user_id", user.id).execute()
        return jsonify({"ok": True})
    except Exception as e:
        print(f"Delete chat error: {e}")
        return jsonify({"error": "Failed to delete chat."}), 500


@app.route("/share", methods=["POST"])
def share_answer():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    data = request.json or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    if not question or not answer:
        return jsonify({"error": "Question and answer are required."}), 400
    slug = secrets.token_urlsafe(6).replace("_", "").replace("-", "")[:10]
    try:
        supabase.table("shared_answers").insert({
            "slug": slug,
            "question": question,
            "answer": answer,
        }).execute()
        return jsonify({
            "slug": slug,
            "url": f"/share/{slug}",
            "public_url": f"/share.html?slug={slug}"
        })
    except Exception as e:
        print(f"Share save error: {e}")
        return jsonify({"error": "Failed to create share link."}), 500


@app.route("/share/<slug>", methods=["GET"])
def get_shared_answer(slug):
    try:
        response = supabase.table("shared_answers").select(
            "slug,question,answer,created_at"
        ).eq("slug", slug).limit(1).execute()
        rows = response.data or []
        if not rows:
            return jsonify({"error": "Not found."}), 404
        item = rows[0]
        question = html.escape(item.get("question", ""))
        answer = html.escape(item.get("answer", ""))
        page = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Shared Answer</title>
          <style>
            body {{ font-family: Inter, system-ui, sans-serif; background:#0b1220; color:#edf3ff; padding:20px; }}
            .card {{ max-width:760px; margin:0 auto; border:1px solid #253554; border-radius:14px; padding:16px; background:#111a2f; }}
            .q,.a {{ border:1px solid #253554; border-radius:10px; padding:12px; margin-top:10px; white-space:pre-wrap; }}
            .label {{ color:#9db0d3; font-size:12px; margin-bottom:6px; }}
          </style>
        </head>
        <body>
          <div class="card">
            <h1>Shared Answer</h1>
            <div class="q"><div class="label">Question</div>{question}</div>
            <div class="a"><div class="label">Answer</div>{answer}</div>
          </div>
        </body>
        </html>
        """
        return page
    except Exception as e:
        print(f"Share fetch error: {e}")
        return jsonify({"error": "Failed to load shared answer."}), 500


@app.route("/share-data/<slug>", methods=["GET"])
def get_shared_answer_data(slug):
    try:
        response = supabase.table("shared_answers").select(
            "slug,question,answer,created_at"
        ).eq("slug", slug).limit(1).execute()
        rows = response.data or []
        if not rows:
            return jsonify({"error": "Not found."}), 404
        return jsonify({"item": rows[0]})
    except Exception as e:
        print(f"Share data fetch error: {e}")
        return jsonify({"error": "Failed to load shared answer."}), 500


@app.route("/stats", methods=["GET"])
def stats():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    try:
        now = datetime.now(timezone.utc)
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_week = start_today - timedelta(days=6)

        total_resp = supabase.table("questions").select(
            "id", count="exact"
        ).eq("user_id", user.id).execute()
        today_resp = supabase.table("questions").select(
            "id", count="exact"
        ).eq("user_id", user.id).gte("created_at", start_today.isoformat()).execute()
        week_rows_resp = supabase.table("questions").select(
            "created_at"
        ).eq("user_id", user.id).gte("created_at", start_week.isoformat()).execute()

        week_rows = week_rows_resp.data or []
        daily_counts_map = {}
        for i in range(7):
            day = (start_week + timedelta(days=i)).date().isoformat()
            daily_counts_map[day] = 0
        for row in week_rows:
            created_at = row.get("created_at")
            if not created_at:
                continue
            day_key = created_at[:10]
            if day_key in daily_counts_map:
                daily_counts_map[day_key] += 1
        daily_counts = [{"date": day, "count": count} for day, count in daily_counts_map.items()]

        return jsonify({
            "total_questions": total_resp.count or 0,
            "questions_today": today_resp.count or 0,
            "questions_this_week": sum(item["count"] for item in daily_counts),
            "daily_counts": daily_counts,
            "daily_counts_map": daily_counts_map,
        })
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({"error": "Failed to load stats."}), 500


@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        return jsonify({"error": "Stripe not configured."}), 500

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url="https://ai-summary-tool-seven.vercel.app/?success=true",
            cancel_url="https://ai-summary-tool-seven.vercel.app/?canceled=true",
            client_reference_id=user.id,
            metadata={"user_id": user.id},
            subscription_data={"metadata": {"user_id": user.id}},
        )
        return jsonify({"url": session.url})
    except Exception as e:
        print(f"Stripe checkout error: {e}")
        return jsonify({"error": "Failed to create checkout session."}), 500


@app.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except Exception as e:
            print(f"Webhook signature error: {e}")
            return jsonify({"error": "Invalid signature."}), 400
    else:
        try:
            event = stripe.Event.construct_from(request.get_json(), stripe.api_key)
        except Exception as e:
            print(f"Webhook parse error: {e}")
            return jsonify({"error": "Invalid payload."}), 400

    if event["type"] in ["customer.subscription.created", "customer.subscription.updated"]:
        subscription = event["data"]["object"]
        status = subscription.get("status")
        customer_id = subscription.get("customer")
        user_id = None
        metadata = subscription.get("metadata") or {}
        user_id = metadata.get("user_id")

        if user_id and customer_id:
            try:
                supabase.table("subscriptions").upsert({
                    "user_id": user_id,
                    "stripe_customer_id": customer_id,
                    "status": status
                }).execute()
            except Exception as e:
                print(f"Subscription upsert error: {e}")

    return jsonify({"received": True})


@app.route("/history", methods=["GET"])
def history():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    try:
        response = supabase.table("questions").select("question,answer,created_at").eq(
            "user_id", user.id
        ).order("created_at", desc=True).limit(5).execute()
        items = response.data or []
        return jsonify({"items": items})
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"error": "Failed to load history."}), 500


@app.route("/admin/questions", methods=["POST"])
def admin_questions():
    data = request.json or {}
    password = data.get("password", "")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "Unauthorized."}), 401

    try:
        response = supabase.table("questions").select(
            "id,question,answer,user_id,created_at"
        ).order("created_at", desc=True).limit(200).execute()
        rows = response.data or []

        email_cache = {}
        for row in rows:
            uid = row.get("user_id")
            if not uid:
                row["email"] = None
                continue
            if uid in email_cache:
                row["email"] = email_cache[uid]
                continue
            try:
                user = supabase.auth.admin.get_user_by_id(uid)
                email = user.user.email if user and user.user else None
            except Exception:
                email = None
            email_cache[uid] = email
            row["email"] = email

        return jsonify({"items": rows})
    except Exception as e:
        print(f"Admin error: {e}")
        return jsonify({"error": "Failed to load admin data."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
