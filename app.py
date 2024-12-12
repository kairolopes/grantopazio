import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import os

# Carregar o JSON do arquivo no diretório atual
json_path = "Perguntas_e_respostas.json"

with open(json_path, "r") as file:
    data = json.load(file)

# Configurar modelo e índice FAISS
questions = [item["question"] for item in data["qa_pairs"]]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Criar a API Flask
app = Flask(__name__)

# Adicionando o endpoint para a raiz
@app.route('/')
def home():
    return jsonify({"message": "Bem-vindo à API do Petshop!"}), 200

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "A pergunta está vazia ou não foi enviada"}), 400

    user_embedding = model.encode([user_question])
    _, indices = index.search(user_embedding, k=1)
    matched_question = questions[indices[0][0]]
    response = next(item["resposta"] for item in data["qa_pairs"] if item["question"] == matched_question)

    return jsonify({"response": response})

# Adicionar endpoint de verificação de integridade
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
