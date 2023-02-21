from transformers import AutoTokenizer, T5ForConditionalGeneration

import torch
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("t5-small")
with open('arxiv_titles.txt') as f:
    titles = f.readlines()

# Training on arXiv Titles:
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# optimizer = torch.optim.Adam(model.parameters())
# for title in tqdm(titles):
#     input_ids = tokenizer("", return_tensors="pt").input_ids
#     labels = tokenizer(title, return_tensors="pt").input_ids
#     outputs = model(input_ids=input_ids, labels=labels)
#     loss = outputs.loss
#     logits = outputs.logits
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
# model.save_pretrained("t5_arxiv_model", from_pt=True)

# Hypothesis Generation from arXiv Trained Model:
model = T5ForConditionalGeneration.from_pretrained("t5_arxiv_model")
optimizer = torch.optim.Adam(model.parameters())
input_ids = tokenizer("", return_tensors="pt").input_ids  # Batch size 1
best_total_grad = 0.0
n_hypotheses = 1000 # number of hypotheses to generate before evaluation 
while True:

    for h in range(n_hypotheses):

        generated_outputs = model.generate(input_ids, do_sample=True, top_k=10000)
        generated_hypothesis = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        not_generated_hypothesis = "Not "+generated_hypothesis
        print("generated hypothesis:", generated_hypothesis)
        generated_outputs = tokenizer.encode(generated_hypothesis, return_tensors="pt")
        not_generated_outputs = tokenizer.encode(not_generated_hypothesis, return_tensors="pt")

        inputs = tokenizer.encode("", return_tensors="pt")

        outputs = model(input_ids=inputs, labels=generated_outputs)
        loss = outputs.loss
        loss.backward()
        total_grad = 0.0
        for param in model.parameters():
            total_grad += torch.sum(torch.abs(param.grad))
        model.zero_grad()

        outputs = model(input_ids=inputs, labels=not_generated_outputs)
        loss = outputs.loss
        loss.backward()
        not_total_grad = 0.0
        for param in model.parameters():
            not_total_grad += torch.sum(torch.abs(param.grad))
        model.zero_grad()

        if total_grad + not_total_grad > best_total_grad:
            best_total_grad = total_grad + not_total_grad
            hypothesis = generated_hypothesis
        # print("hypothesis:", hypothesis) 

    print()
    print("Is it likely that:", hypothesis)
    print()

    response = input("y/n?")

    if response == 'n':
        hypothesis = "Not "+hypothesis
    
    generated_outputs = tokenizer.encode(hypothesis, return_tensors="pt") 
        
    outputs = model(input_ids=inputs, labels=generated_outputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    hypothesis = ""
    best_total_grad = 0.0
    model.save_pretrained("t5_arxiv_model", from_pt=True)

    
