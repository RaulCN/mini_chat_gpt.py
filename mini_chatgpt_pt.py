import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")

# Carrega os parâmetros do modelo salvos em .pth
model_path = "mini_GPT2_Pt.pth"
model_state_dict = torch.load(model_path) #n é possível upar arquivos muito grandes no github, considere usar o outro arquivo que usa os parâmetros do arquivo pré-treinado

# Cria uma nova instância do modelo e carrega os parâmetros salvos
model = AutoModelForCausalLM.from_pretrained("pierreguillou/gpt2-small-portuguese", state_dict=model_state_dict)

# Define uma função para gerar respostas do chatbot
def generate_response(input_text, model, tokenizer):
    # Processa a entrada do usuário com o tokenizer
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Cria a máscara de atenção
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    
    # Gera uma resposta usando o modelo
    output = model.generate(input_ids, max_length=100, num_beams=3, no_repeat_ngram_size=2, early_stopping=True, attention_mask=attention_mask, pad_token_id=tokenizer.bos_token_id, temperature=0.9)

    # Decodifica a resposta gerada em texto
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Retorna a resposta gerada
    return generated_text

# Loop do chatbot
while True:
    # Solicita a entrada do usuário
    input_text = input("> ")
    
    # Gera uma resposta usando o modelo
    output_text = generate_response(input_text, model, tokenizer)
    
    # Imprime a resposta gerada
    print("Chatbot:", output_text)
