import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import re

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
        if self._is_about_me_request(question):
            return self.context.get('sobre_mim', '')
        elif self._is_list_request(question):
            return self._generate_list_response(question)
        else:
            response = self.qa_pipeline(question=question, context=self._flatten_context())
            return response['answer']

    def _is_about_me_request(self, question: str) -> bool:
        about_me_patterns = [
            r'\bfale um pouco sobre você\b',
            r'\bfale um pouco sobre voce\b',
            r'\bfale sobre você\b',
            r'\bfale sobre voce\b',
        ]
        for pattern in about_me_patterns:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return True
        return False

    def _is_list_request(self, question: str) -> bool:
        list_patterns = [
            r'\bquais são\b',
            r'\bquais sao\b',
            r'\blista d[a-zA-Z]\b',
            r'\benumerar\b',
        ]
        for pattern in list_patterns:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return True
        return False

    def _generate_list_response(self, question: str) -> str:
        if 'soft skills' in question.lower():
            return self.context.get('Soft Skills', '')
        elif 'habilidades' in question.lower():
            return self._format_skills_list(self.context.get('Habilidades', []), include_description=False)
        elif 'idiomas' in question.lower():
            return self._format_languages_list(self.context.get('Idiomas', []))
        elif 'hobbies' in question.lower():
            return self.context.get('Hobbies', '')
        else:
            return "Desculpe, não entendi a pergunta ou não tenho informações sobre isso."

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

    def _format_skills_list(self, skills_list: list, include_description=True) -> str:
        formatted_list = ""
        for skill in skills_list:
            if isinstance(skill, dict):
                habilidade = skill.get('habilidade', '')
                descricao = skill.get('descricao', '')
                if include_description:
                    formatted_list += f"{habilidade}: {descricao}\n"
                else:
                    formatted_list += f"{habilidade}\n"
        return formatted_list.strip()

    def _format_languages_list(self, languages_list: list) -> str:
        formatted_list = ""
        for language in languages_list:
            if isinstance(language, dict):
                idioma = language.get('idioma', '')
                nivel = language.get('nível', '')
                formatted_list += f"{idioma} ({nivel})\n"
        return formatted_list.strip()

# Exemplo de uso:
# chatbot = QAChatbot(model_name='pierreguillou/bert-base-cased-squad-v1.1-portuguese', context_file='./api/dataset.json')
# resposta = chatbot.get_answer("fale um pouco sobre você")
# print(resposta)
