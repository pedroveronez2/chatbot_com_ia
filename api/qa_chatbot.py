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
        if self._is_list_request(question):
            return self._generate_list_response(question)
        else:
            response = self.qa_pipeline(question=question, context=self._flatten_context())
            return response['answer']

    def _is_list_request(self, question: str) -> bool:
        list_patterns = [
            r'\bquais são\b',
            r'\bquais sao\b',
            r'\blista das\b',
            r'\benumerar\b',
        ]
        for pattern in list_patterns:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return True
        return False

    def _generate_list_response(self, question: str) -> str:
        if 'soft skills' in question.lower():
            return self.context.get('soft skills', '')
        elif 'habilidades' in question.lower():
            return self._format_skills_list(self.context.get('Habilidades Técnicas', []), include_description=False)
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
                nome = skill.get('nome', '')
                descricao = skill.get('descricao', '')
                if include_description:
                    formatted_list += f"{nome}: {descricao}\n"
                else:
                    formatted_list += f"{nome}\n"
        return formatted_list.strip()

    def _format_languages_list(self, languages_list: list) -> str:
        formatted_list = ""
        for language in languages_list:
            if isinstance(language, dict):
                nome = language.get('nome', '')
                nivel = language.get('nível', '')
                formatted_list += f"{nome} ({nivel})\n"
        return formatted_list.strip()