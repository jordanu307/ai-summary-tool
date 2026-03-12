import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

client = Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=0)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

MODEL_MAP = {
    "groq": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "openai": {"provider": "openai", "model": "gpt-4o-mini"},
    "gemini": {"provider": "gemini", "model": "gemini-1.5-flash"},
    "claude": {"provider": "claude", "model": "claude-haiku-4-5-20251001"},
}

MODEL_ALIASES = {v["model"]: k for k, v in MODEL_MAP.items()}


def generate_answer(model_choice, topic):
    resolved_key = MODEL_ALIASES.get(model_choice, model_choice)
    config = MODEL_MAP.get(resolved_key)
    if not config:
        return None, "Unsupported model selection.", 400

    prompt = f"Explain this clearly in 3 sentences: {topic}"

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

    if provider == "openai":
        if not OPENAI_API_KEY or not openai_client:
            return None, "Server misconfigured: missing OPENAI_API_KEY.", 500
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, None, 200

    if provider == "gemini":
        if not GEMINI_API_KEY:
            return None, "Server misconfigured: missing GEMINI_API_KEY.", 500
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return getattr(response, "text", None), None, 200

    if provider == "claude":
        if not ANTHROPIC_API_KEY or not anthropic_client:
            return None, "Server misconfigured: missing ANTHROPIC_API_KEY.", 500
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text if response.content else None
        return content, None, 200

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

    if not topic or len(topic) > 500:
        return jsonify({"error": "Invalid input"}), 400

    try:
        answer, error_msg, status_code = generate_answer(model_choice, topic)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
