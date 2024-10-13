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
            return context_data['context2']  # Retorna o contexto completo como um dicionário ou texto

    def get_answer(self, question: str) -> str:
        # Para responder perguntas, convertemos o contexto em uma string e utilizamos o pipeline
        context_str = json.dumps(self.context, ensure_ascii=False)  # Garante que caracteres especiais sejam preservados
        response = self.qa_pipeline(question=question, context=context_str)
        return response['answer']

if __name__ == '__main__':
    # Código para testar a classe QAChatbot diretamente
    model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
    context_file = "./data/context.json"  # Nome do arquivo JSON com o contexto
    chatbot = QAChatbot(model_name, context_file)

    # Testando algumas perguntas
    perguntas = [
    
    "Qual é o seu nome?",
    "Quantos anos você tem?",
    "Qual curso você está estudando?",
    "Onde você estuda?",
    "Quais são suas áreas de foco profissional?",
    "Você tem experiência prática em Ciência de Dados e Machine Learning?",
    "Quais competências você desenvolveu?",
    "Como você se descreve profissionalmente?",
    "Quais serviços você presta como freelancer?",
    "Como sua experiência profissional contribui para seu crescimento?",
    "De que forma você continua aprendendo?",
    "Quais são suas principais soft skills?",
    "O que impulsiona sua evolução profissional?",
    "Quais conquistas você já alcançou em seus projetos?",
    "Quais são suas perspectivas futuras na área de tecnologia?",
    
]

    for pergunta in perguntas:
        resposta = chatbot.get_answer(pergunta)
        print(f"Pergunta: {pergunta}\nResposta: {resposta}\n")
