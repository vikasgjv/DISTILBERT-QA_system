from flask import Flask, request, jsonify, render_template
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = Flask(__name__)

# Load dataset and model on startup
print("Loading dataset...")
dataset = load_dataset("squad", split="train[:1000]")
print("Dataset Loaded Successfully!")

print("Loading model...")
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()
print("Model Loaded Successfully!")



def qa_model(question, context):
    """Run QA inference and return answer with confidence score."""
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    input_ids = inputs["input_ids"][0]
    seq_len   = input_ids.shape[0]

    start_idx = int(torch.argmax(outputs.start_logits))
    end_idx   = int(torch.argmax(outputs.end_logits))

    # Ensure valid span: end must be >= start, and both within range
    if end_idx < start_idx:
        end_idx = start_idx
    end_idx = min(end_idx + 1, seq_len)  # +1 for slice, clamped to seq length

    tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])
    answer = tokenizer.convert_tokens_to_string(tokens).strip()

    start_score = torch.softmax(outputs.start_logits, dim=1)[0][start_idx].item()
    end_score   = torch.softmax(outputs.end_logits,   dim=1)[0][end_idx - 1].item()
    score       = round((start_score + end_score) / 2, 4)

    return {"answer": answer, "score": score}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    context  = data.get("context", "").strip()

    if not question or not context:
        return jsonify({"error": "Both question and context are required."}), 400

    try:
        result = qa_model(question=question, context=context)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500


@app.route("/dataset-sample", methods=["GET"])
def dataset_sample():
    try:
        index = int(request.args.get("index", 0))
    except ValueError:
        return jsonify({"error": "Index must be an integer."}), 400

    if index < 0 or index >= len(dataset):
        return jsonify({"error": f"Index out of range. Valid range: 0–{len(dataset)-1}."}), 400

    sample = dataset[index]
    return jsonify({
        "context":  sample["context"],
        "question": sample["question"],
        "answer":   sample["answers"]["text"][0]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5500)
