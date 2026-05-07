from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

print("\n QUESTION ANSWERING SYSTEM ")

dataset = load_dataset("squad", split="train[:1000]")
print("\nDataset Loaded Successfully!")
print(dataset)

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def qa_model(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )
    start_score = torch.softmax(outputs.start_logits, dim=1)[0][start].item()
    end_score = torch.softmax(outputs.end_logits, dim=1)[0][end - 1].item()
    score = (start_score + end_score) / 2
    return {"answer": answer, "score": score}

print("\nModel Loaded Successfully!")

 
print("\n ASK YOUR OWN QUESTION ")
print("Type 'exit' to quit.\n")

while True:
    user_context = input("Enter a paragraph (context): ").strip()
    if user_context.lower() == 'exit':
        break

    user_question = input("Enter your question: ").strip()
    if user_question.lower() == 'exit':
        break

    user_result = qa_model(question=user_question, context=user_context)

    print("\n Result ")
    print("Answer:", user_result['answer'])
    print("Confidence Score:", round(user_result['score'], 4))
    print("--------------\n")
