# training.py
import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, load_metric
from qa_chatbot import QAChatbot

# Função para preparar os dados de treinamento
def prepare_train_data(context, questions_answers):
    data = []
    for qa in questions_answers:
        question = qa['question']
        answers = qa['answers']
        for answer in answers:
            data.append({
                'context': context,
                'question': question,
                'answers': {
                    'text': [answer['text']],
                    'answer_start': [answer['answer_start']]
                }
            })
    return data

# Exemplo de dados de perguntas e respostas para treinamento
questions_answers = [
    {
        'question': 'Qual é o nome completo?',
        'answers': [{'text': 'Pedro Augusto de Carvalho Veronez', 'answer_start': 6}]
    },
    {
        'question': 'Quantos anos voce tem?',
        'answers': [{'text': '20 anos', 'answer_start': 32}]
    },
    # Adicione mais perguntas e respostas aqui
]

# Carregar o contexto e preparar os dados de treinamento
with open('./api/dataset.json', 'r', encoding='utf-8') as f:
    context_data = json.load(f)
context = context_data['contexto']
train_data = prepare_train_data(context, questions_answers)

# Criar um dataset do Hugging Face
dataset = Dataset.from_list(train_data)

# Inicializar o modelo e tokenizer
model_name = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preparar os argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Função de processamento dos dados
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If the answer is not fully inside the context, label is (0, 0)
        if not (offset[context_start][0] <= start_char and offset[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(offset.index((start_char, start_char), context_start))
            end_positions.append(offset.index((end_char, end_char), context_start))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenizar o dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Inicializar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Treinar o modelo
trainer.train()

# Avaliar o modelo
results = trainer.evaluate()

# Calcular a porcentagem de acertos e a quantidade de erros
total_questions = len(tokenized_dataset)
correct_answers = results['eval_accuracy'] * total_questions
errors = total_questions - correct_answers

print(f"Porcentagem de acertos: {results['eval_accuracy'] * 100:.2f}%")
print(f"Quantidade de erros: {errors}")
