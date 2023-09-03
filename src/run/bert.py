"""
Train a bert model on our dataset
"""

from transformers import AutoTokenizer

from src.dataset.dataset import Dataset

if __name__ == "__main__":
    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Import dataset")
    topics = Dataset("topics1")

    dataset = topics.to_datasets(
        ["target1"], {"target1": "labels"}, tokenizer=tokenizer, sentence_length=380
    )

    print("Creating trainer")
    trainer = topics.trainer(
        model_name,
        dataset,
        epochs=30,
        gradient_accumulation_steps=4,
        batch_size=8,
        lr=1e-4,
        save_dataset=False,
    )
    print("Training")
    trainer.train()
