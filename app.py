from flask import Flask, render_template, request, jsonify
from rag_gemini import GeminiRAG  # Assuming rag_gemini.py defines GeminiRAG class

app = Flask(__name__)

# Initialize GeminiRAG (adjust init as needed for your implementation)
rag = GeminiRAG()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': "No message received."}), 400
    # Get response from GeminiRAG
    response = rag.chat(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
