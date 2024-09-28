import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

class QAChatbot:
    def __init__(self, model_name: str, context_file: str):
        self.model_name = model_name
        self.context = self._load_context(context_file)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

        # Extraindo perguntas binárias do contexto
        self.binary_questions = self.context.get('perguntas_binarias', {})

    def _load_context(self, context_file: str):
        with open(context_file, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
            return context_data.get('contexto', {})

    def _flatten_context(self) -> str:
        flattened_context = ""
        for key, value in self.context.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for subkey, subvalue in item.items():
                            flattened_context += f"{subkey}: {subvalue}\n"
            else:
                flattened_context += f"{key}: {value}\n"
        return flattened_context.strip()

    def _check_yes_no_question(self, question: str) -> str:
        # Verifica se a pergunta é uma das perguntas binárias definidas
        question_normalized = question.strip().lower()  # Normalize a pergunta para facilitar a comparação
        
        for binary_question, response in self.binary_questions.items():
            if binary_question.lower() in question_normalized:
                return response
        
        return None  # Se não for uma pergunta binária

    def get_answer(self, question: str) -> str:
        yes_no_response = self._check_yes_no_question(question)
        if yes_no_response is not None:
            return yes_no_response

        # Para perguntas normais
        response = self.qa_pipeline(question=question, context=self._flatten_context())
        return response['answer']

if __name__ == '__main__':
    # Código para testar a classe QAChatbot diretamente
    model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
    context_file = "./api/dataset.json"  # Nome do arquivo JSON com o contexto
    chatbot = QAChatbot(model_name, context_file)

    # Testando algumas perguntas
    perguntas = [
        "Você tem filhos?",
        "Você consegue falar em inglês fluentemente?",
        "Qual é o seu nome?"
    ]

    for pergunta in perguntas:
        resposta = chatbot.get_answer(pergunta)
        print(f"Pergunta: {pergunta}\nResposta: {resposta}\n")
