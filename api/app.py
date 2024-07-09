from flask import Flask, jsonify, request
from qa_chatbot import QAChatbot
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Exemplo de utilização
model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
context_file = "./api/dataset.json"  # Nome do arquivo JSON com o contexto
chatbot = QAChatbot(model_name, context_file)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Nenhuma mensagem fornecida"}), 400

    response_message = chatbot.get_answer(user_message)
    return jsonify({"response": response_message})

if __name__ == '__main__':
    app.run(debug=True)
