# Chatbot com IA

Este projeto implementa um chatbot com IA utilizando o modelo `pierreguillou/bert-base-cased-squad-v1.1-portuguese` da biblioteca Hugging Face Transformers. O chatbot é capaz de responder perguntas sobre a vida profissional de uma pessoa fictícia, "Pedro Augusto de Carvalho Veronez", com base em um contexto fornecido em formato JSON.

## Sumário

- [Instalação](#instalação)
- [Pré-processamento de Dados](#pré-processamento-de-dados)
- [Treinamento do modelo](#treinamento-do-modelo)
- [Teste do modelo](#teste-do-chatbot)
- [Insights](#insights)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Execução da Aplicação](#execução-da-aplicação)
- [Uso da API](#uso-da-api)

## Instalação

Certifique-se de ter o Python 3.6 ou superior instalado em sua máquina. Você pode instalar as dependências necessárias usando o `pip`:

```bash
pip install flask flask-cors transformers
```
## Pré-processamento de Dados

Os dados utilizados pelo chatbot estão armazenados em um arquivo JSON (`context.json`) que contém informações sobre a pessoa. Existem dois tipos de contexto:

1. **Chave e Valor**
2. **Texto**

## Treinamento do Modelo

O treinamento do modelo será realizado considerando os dois tipos de contexto mencionados:

- **Tipo 1: Chave e Valor**
- **Tipo 2: Texto**

O objetivo do treinamento é avaliar qual dos dois contextos proporciona respostas mais precisas para as perguntas do modelo de Pergunta e Resposta (QA).

## Teste do Chatbot

Você pode testar a classe `QAChatbot` diretamente no arquivo `qa_chatbot.py`. Ao final do arquivo, algumas perguntas de exemplo são fornecidas para verificar a funcionalidade do chatbot.

## Insights

Os insights extraídos através do treinamento e testes mostraram que o modelo se adapta bem aos dois tipos de contexto, cada um com suas vantagens e desvantagens:

- **Tipo Texto**: O chatbot apresenta maior precisão nas respostas, mas o pré-processamento de dados pode ser um pouco mais complexo e demorado.
- **Tipo Chave e Valor**: O processamento é mais simples, mas a precisão das respostas pode ser inferior em comparação ao tipo texto.

## Estrutura do Projeto

A estrutura do projeto é a seguinte:

```bash
chatbot/
│
└── api/
    └── app.py    
    └── qa_chatbot.py
└── data/
    └── context.json
```

### Arquivo app.py
Este arquivo contém a configuração da aplicação Flask e define a rota /chat para interagir com o chatbot.

### Arquivo qa_chatbot.py
Este arquivo contém a implementação da classe QAChatbot, que carrega o modelo, o tokenizador e o contexto, além de fornecer o método para responder perguntas.



## Execução da Aplicação
Para iniciar a aplicação Flask, execute o seguinte comando no terminal:

```
python ./api/app.py
```
A aplicação será iniciada em http://127.0.0.1:5000/.

# Uso da API
Após iniciar a aplicação, você pode enviar uma solicitação POST para a rota /chat para interagir com o chatbot. O corpo da solicitação deve estar no seguinte formato:

```
{
    "message": "sua pergunta aqui"
}
```

## Exemplo de Solicitação cURL

```
curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "Quais são os seus projetos?"}'
```
## Resposta
A resposta será um JSON contendo a resposta do chatbot:

```
{
    "response": "Chatbot com IA, Detector de Imagens com IA, Gerenciador de Tarefas, Análise de Sentimentos em Redes Sociais"
}
```