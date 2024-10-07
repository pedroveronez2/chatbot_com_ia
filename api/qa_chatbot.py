import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

class QAChatbot:
    def __init__(self, model_name: str, context_file: str):
        self.model_name = model_name
        self.context = self._load_context(context_file)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)


    def _load_context(self, context_file: str):
        with open(context_file, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
            return context_data.get('contexto', {})



    def get_answer(self, question: str) -> str:
        # Caso não seja uma pergunta binária, utilize o pipeline de QA para responder
        response = self.qa_pipeline(question=question, context=self.context)
        return response['answer']


if __name__ == '__main__':
    # Código para testar a classe QAChatbot diretamente
    model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
    context_file = "./api/dataset.json"  # Nome do arquivo JSON com o contexto
    chatbot = QAChatbot(model_name, context_file)

    # Testando algumas perguntas
    perguntas = [
    "Qual é o seu nome?",
    "Quantos anos você tem?",
    "Onde você mora?",
    "Qual é a sua nacionalidade?",
    "Qual é o seu estado civil?",
    "Qual é o seu nível de educação?",
    "Qual curso você está estudando?",
    "Em qual faculdade você estuda?",
    "Em que semestre você está?",
    "Qual é a sua experiência profissional?",
    "Quais linguagens de programação você conhece?",
    "Quais frameworks você utiliza?",
    "Quais bancos de dados você já trabalhou?",
    "Quais ferramentas você usa no seu dia a dia?",
    "Quais são as suas soft skills?",
    "Quais idiomas você fala?",
    "Quais são os seus interesses?",
    "Quais hobbies você tem?",
    "Você pode falar sobre um projeto pessoal que desenvolveu?",
    "Você participou de algum evento relacionado à sua área?",
    "O que você gosta de fazer no seu tempo livre?"
        
]

    for pergunta in perguntas:
        resposta = chatbot.get_answer(pergunta)
        print(f"Pergunta: {pergunta}\nResposta: {resposta}\n")
