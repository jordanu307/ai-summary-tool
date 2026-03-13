import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"]
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

client = Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=0)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

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


def generate_answer(model_choice, topic, style):
    resolved_key = MODEL_ALIASES.get(model_choice, model_choice)
    config = MODEL_MAP.get(resolved_key)
    if not config:
        return None, "Unsupported model selection.", 400

    prompt = build_prompt(topic, style)

    provider = config["provider"]
    model_name = config["model"]

    if provider == "groq":
        if not GROQ_API_KEY:
            return None, "Server misconfigured: missing GROQ_API_KEY.", 500
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
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

    if not topic or len(topic) > 500:
        return jsonify({"error": "Invalid input"}), 400

    try:
        answer, error_msg, status_code = generate_answer(model_choice, topic, style)
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
