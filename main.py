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

client = Groq(api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_current_user(request):
    """
    Checks the Authorization header for a valid Supabase token.
    Returns the user if valid, or None if not.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ")[1]

    try:
        response = supabase.auth.get_user(token)
        return response.user
    except Exception:
        return None


@app.route("/ask", methods=["POST"])
@limiter.limit("10 per minute")
def ask():
    # Step 1: Verify the user is logged in
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    # Step 2: Validate input
    data = request.json
    topic = data.get("topic", "")

    if not topic or len(topic) > 500:
        return jsonify({"error": "Invalid input"}), 400

    # Step 3: Ask the AI
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": f"Explain this clearly in 3 sentences: {topic}"}
        ]
    )

    answer = response.choices[0].message.content

    # Step 4: Save to Supabase with the user's ID
    supabase.table("questions").insert({
        "question": topic,
        "answer": answer,
        "user_id": user.id
    }).execute()

    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)