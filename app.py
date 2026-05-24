from flask import Flask, render_template, request, jsonify
from agenticrag import AgenticRAG

app = Flask(__name__)

rag = AgenticRAG(
    pdf_files = [
        "./document1.pdf", 
        "./document2.pdf",
    ],
    openai_api_key = "YOUR_OPENAI_API_KEY",
    tavily_api_key = "YOUR_TAVILY_API_KEY",
    verbose        = False,   # agents thinking
)



@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():

    data         = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "No message received."}), 400

    try:
        response = rag.chat(user_message)
        return jsonify({"response": response})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"response": "Something went wrong. Please try again."}), 500


@app.route("/clear", methods=["POST"])
def clear():
    rag.clear_memory()
    return jsonify({"status": "Memory cleared."})


@app.route("/history", methods=["GET"])
def history():
    messages = rag.memory.load_memory_variables({}).get("chat_history", [])

    formatted = [
        {
            "role"    : "user" if m.type == "human" else "assistant",
            "content" : m.content,
        }
        for m in messages
    ]

    return jsonify({"history": formatted})


if __name__ == "__main__":
    app.run(debug=True)