import json
import random
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import datasets
import torch
from dash import html
from dash.dependencies import Input, Output, State
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ui import get_app

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    "notebooks/artifacts/e772df6f4f224b67b914a2beb3acc792/checkpoint-2439"
)

data_p = Path("data")
id2label: dict[int, str] = json.loads((data_p / "id2label.json").read_text())

app = get_app()


@app.callback(
    Output("container", "children"),
    [Input("predict", "n_clicks")],
    [Input("title", "value"), Input("text", "value")],
    prevent_initial_call=True,
)
def update_title(n_clicks, title, text):
    ctx = dash.ctx
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "predict.n_clicks":
        input_ids = tokenizer(
            [title + text], padding="max_length", truncation=True, max_length=256
        )

        dataset = datasets.Dataset.from_dict(input_ids)
        dataset.set_format("torch", columns=["input_ids"])

        output = model(dataset["input_ids"])
        logits: torch.Tensor = output["logits"].detach()

        preds = logits.softmax(dim=-1)[0]
        preds = preds.argsort(descending=True).numpy()[:5]

        return [
            dbc.Badge(children=id2label[str(tag)], style={"margin": "10px"})
            for tag in preds
        ]


if __name__ == "__main__":
    app.run_server(debug=True)
