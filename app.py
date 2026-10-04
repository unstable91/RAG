import os

from flask import Flask, render_template, request, jsonify

from rag_backend import ask_question

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON request."}), 400

        message = data.get("message", "")
        if not isinstance(message, str):
            return jsonify({"error": "Message must be a string."}), 400

        message = message.strip()
        if not message:
            return jsonify({"error": "Please enter a question."}), 400
        if len(message) > 1000:
            return jsonify({"error": "Question is too long (max 1000 characters)."}), 400

        result = ask_question(message)
        return jsonify({"answer": result["answer"], "sources": result["sources"]})

    except Exception as e:
        print("ERROR:", repr(e))
        return jsonify({"error": "An error occurred while processing your question."}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Medical RAG"})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    print(f"\nOpen: http://{host}:{port}\n")
    # use_reloader=False: the reloader would load the embedding model twice
    app.run(host=host, port=port, debug=debug, use_reloader=False)
