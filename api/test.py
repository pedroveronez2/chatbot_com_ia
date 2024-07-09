# Carregar o contexto do arquivo JSON
import json


with open('./api/dataset.json', 'r', encoding='utf-8') as f:
    context = json.load(f)
    
print(context)