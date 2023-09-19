"""
Train a bert model on our dataset
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AdamW,
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    get_scheduler,
)

from src.dataset.dataset import Dataset
from src.models.callback import get_callback
from src.models.trainer import SOTrainer
from src.run.multi.BERT.MultiLabelBert import MultiLabelBert

os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()

    config_p = Path(args.config)

    print("Config : ", config_p)

    config = None
    with open(str(config_p)) as f:
        config = yaml.load(f.read(), yaml.FullLoader)
        config = defaultdict(lambda: False, config)

    model_name = config.get("model_name", "bert-base-uncased")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Import dataset")
    n = config.get("dataset_size", -1)
    topics = Dataset("topics1", n=n)

    all_targets = [f"target{t}" for t in range(1, 6)]

    target_min_freq = config.get("target_min_freq", 0)

    # We need at least more than one of a class
    more_than_one = pd.Series(
        topics.df[all_targets].values.flatten()
    ).value_counts() > int(n * target_min_freq)
    invalid_targets = more_than_one[more_than_one == False].index

    # Line contains targets that appear more than once !
    # Removing targets from the 5 targets array
    multi_targets = topics.df[all_targets].parallel_apply(
        lambda x: list(filter(lambda y: y not in invalid_targets, x)),
        axis=1,
    )

    not_empty = ~multi_targets.parallel_apply(lambda x: len(x)).isin([0])
    df = topics.df.loc[not_empty]
    multi_targets = multi_targets[not_empty]

    # Weight distribution for BCEntropy
    nb_pos = multi_targets.explode().value_counts()
    weights = (nb_pos.sum() - nb_pos) / nb_pos

    # Here we make the binarizer learn our targets in advance to save memory
    mlb = MultiLabelBinarizer()
    ohe_targets = mlb.fit_transform(multi_targets)

    print("classes = ", len(mlb.classes_))

    df["text"] = df["original_title"] + df["original_text"]

    df["labels"] = df.parallel_apply(
        lambda row: np.array(
            mlb.transform([multi_targets[row.name]])[0], dtype=np.float32
        ),
        axis=1,
    )

    dataset: datasets.Dataset = datasets.Dataset.from_pandas(df)

    def preprocess(batch):
        output = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.get("window", 512),
        )

        return output

    import random

    # PREPROCESS
    dataset = dataset.map(preprocess, batched=True)

    print(dataset)

    print("Random processed sample : ")
    sample = dataset[random.randint(0, len(dataset))]
    print("text = ", sample["text"])
    print("text_labels : ", [sample[f"target{i+1}"] for i in range(5)])
    print("labels", sample["labels"])

    dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels", "token_type_ids"]
    )

    dataset = dataset.train_test_split(test_size=0.2, seed=0)

    print("Creating trainer")
    lr = config.get("lr", 4e-3)
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 4)
    gradient_accumulation_steps = config.get("batch_size_acc", 64)

    id2label = {i: cls for i, cls in enumerate(mlb.classes_)}
    label2id = {v: k for k, v in id2label.items()}

    output_p = Path("output").resolve()
    loc = output_p / config_p.stem
    loc.mkdir(exist_ok=True, parents=True)

    # Save run data
    with open(str(loc / "id2label.yaml"), "w") as f:
        yaml.dump(id2label, f)

    with open(str(loc / "label2id.yaml"), "w") as f:
        yaml.dump(label2id, f)

    with open(str(loc / "config.yaml"), "w") as f:
        yaml.dump(config, f)

    w = torch.zeros(len(weights))
    for id, label in id2label.items():
        w[id] = weights[label]
    w = w.cuda()

    loss_name = config.get("loss", "cross_entropy")
    loss = None
    if loss_name == "cross_entropy":
        loss = torch.nn.BCEWithLogitsLoss(weight=w)
    elif loss_name == "mult_label_soft_margin":
        loss = torch.nn.MultiLabelSoftMarginLoss(weight=w)
    else:
        raise NotImplementedError()

    model = MultiLabelBert.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=len(mlb.classes_),
        id2label=id2label,
        label2id=label2id,
        loss_fct=loss,
    )

    # Weight distribution for BCEntropy
    # train_targets = pd.Series(
    #     np.array([dataset["train"][f"target{i+1}"] for i in range(5)]).flatten()
    # )
    # nb_pos = train_targets.value_counts()
    # weights = (nb_pos.sum() - nb_pos) / nb_pos

    # model.bert.requires_grad_(False)

    peft_config = config.get("peft", False)
    if peft_config:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=peft_config.get("r", 8),
                lora_alpha=peft_config.get("alpha", 8),
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
            ),
        )
        model.print_trainable_parameters()

    print(model)
    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Set the number of training steps
    num_train_steps = (
        len(dataset["train"]) / (batch_size * gradient_accumulation_steps)
    ) * epochs

    # Create the scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps * 0.02),
        # num_warmup_steps=0,
        num_training_steps=num_train_steps,
    )

    torch.set_float32_matmul_precision("medium")

    with open(str(loc / "classes.yaml"), "w") as f:
        yaml.dump(mlb.classes_, f)

    checkpoints_p = loc / "checkpoints"
    checkpoints_p.mkdir(exist_ok=True)

    args = TrainingArguments(
        str(checkpoints_p),
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size // 2, 1),
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        eval_steps=config.get("eval_steps", 10),
        save_strategy="epoch",
    )

    iteration = {"i": 0}

    example_p = loc / "examples"
    example_p.mkdir(exist_ok=True)

    def compute_metrics(eval_preds: EvalPrediction):
        threshold = config.get("sigmoid_threshold", 0.5)
        logits = eval_preds.predictions
        ids = eval_preds.label_ids

        y_prob = torch.tensor(logits).sigmoid()
        # y_prob = torch.tensor(logits).softmax(dim=-1)

        y_pred = np.zeros(y_prob.shape)
        y_pred[np.where(y_prob >= threshold)] = 1

        # y_pred = y_prob.argmax(dim=-1)
        # y_pred = torch.nn.functional.one_hot(y_pred, num_classes=y_prob.shape[-1])

        y_labels = mlb.inverse_transform(y_pred)
        labels = mlb.inverse_transform(ids)

        pd.DataFrame({"target": labels, "pred": y_labels}).to_csv(
            example_p / "example{}.csv".format(iteration["i"])
        )

        iteration["i"] += 1

        avg_metrics = {
            "jaccard_score": {"f": jaccard_score},
            "f1_score": {"f": f1_score},
            "roc_auc_score": {"f": roc_auc_score},
            "precision_score": {"f": precision_score, "zero_division": 0.0},
            "recall_score": {"f": recall_score, "zero_division": 0.0},
        }
        avg_types = ["micro", "macro", "weighted", "samples"]

        metrics = {
            **{
                f"{name}_{avg}": score["f"](
                    ids,
                    y_pred,
                    average=avg,
                    # Add custom args
                    **{sk: sv for sk, sv in score.items() if sk != "f"},
                )
                for avg in avg_types
                for name, score in avg_metrics.items()
            },
            "accuracy_score": accuracy_score(ids, y_pred),
        }

        return metrics

    use_mlflow = config.get("mlflow", True)

    # Start run
    callbacks = []
    if use_mlflow:
        end_run = topics.mlflow_run(config_p.stem)
        callbacks.append(get_callback(end_run))

    trainer = SOTrainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=[optimizer, scheduler],
        mlflow=use_mlflow,
    )

    print("Training")
    trainer.train()
