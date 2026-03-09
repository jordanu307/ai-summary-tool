import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

client = Groq(api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    topic = data["topic"]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": f"Explain this clearly in 3 sentences: {topic}"}
        ]
    )

    answer = response.choices[0].message.content

    supabase.table("questions").upsert({
        "question": topic,
        "answer": answer
    }).execute()

    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
