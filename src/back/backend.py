from pathlib import Path

import datasets
import torch
import yaml
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

data_p = Path(__file__).parent / "model"


def load(p: Path):
    with open(p) as f:
        return yaml.full_load(f.read())


config = load(data_p / "config.yaml")

print("[INFO] Loaded model configuration", config)

model_name = config.get("model_name", "bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(data_p / "checkpoint")

id2label: dict[int, str] = load(data_p / "id2label.yaml")

print("[INFO] Loaded num_classes", len(id2label))


@app.get("/")
def test():
    return "Server is working !"


@app.get("/classes")
def classes():
    return list(id2label.values())


@app.get("/predict")
def predict(title: str, text: str):
    title = title.strip("'")
    text = text.strip("'")

    print(f"[INFO] {title=}")
    print(f"[INFO] {text=}")

    input_ids = tokenizer(
        [title + text], padding="max_length", truncation=True, max_length=512
    )

    dataset = datasets.Dataset.from_dict(input_ids)
    dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    output = model(dataset["input_ids"], dataset["attention_mask"])
    logits: torch.Tensor = output["logits"].detach()

    probs = logits.sigmoid()[0]
    preds = torch.argwhere(probs > config.get("sigmoid_threshold", 0.5)).flatten()
    preds = [id2label[p.item()] for p in preds]
    print(preds)

    return preds
