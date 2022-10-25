"""
    test.py
    Created by pierre
    10/3/22 - 4:07 PM
    Description:
    # Enter file description
 """
import torch
from transformers import AutoModelForCausalLM, Seq2SeqTrainer, TrainingArguments
from evaluate import load

from src.models import PronunciationGPT
from src.poetry_datasets import load_dataset

model_name = "EleutherAI/gpt-neo-125M"
txt = "A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most of the tokenizers are available in two flavors: a full python implementation and a “Fast” implementation based on the Rust library 🤗 Tokenizers. The “Fast” implementations allows:"

accuracy = load("accuracy")
perplexity = load("perplexity")


def compute_metrics(eval_pred):
    # Predictions and labels are grouped in a namedtuple called EvalPrediction
    predictions, labels = eval_pred
    # Get the index with the highest prediction score (i.e. the predicted labels)
    predictions = np.argmax(predictions, axis=1)
    # Compare the predicted labels with the reference labels
    results =  compute(predictions=predictions, references=labels)
    # results: a dictionary with string keys (the name of the metric) and float
    # values (i.e. the metric values)
    return results


def main(argv=None):
    dataset = load_dataset(
        "data/datasets/gutenberg_pronunciation_chunked_128_96_8")
    embeddings_p = torch.load(
        "etc/pronunciation_embeddings/p_32_8_s_32_8_sg_1_negative_5/pronunciation_embeddings.pt")
    embeddings_s = torch.load(
        "etc/pronunciation_embeddings/p_32_8_s_32_8_sg_1_negative_5/stress_embeddings.pt")

    gpt = AutoModelForCausalLM.from_pretrained(model_name)
    model = PronunciationGPT(gpt, embeddings_p, embeddings_s)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model.to(device)

    # model.freeze_gpt()
    args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        half_precision_backend="cuda_amp",
        eval_steps=50,
        logging_steps=10,
        save_steps=200,
        report_to=["tensorboard"],
        output_dir="etc/logging/"
    )
    trainer = Seq2SeqTrainer(
        model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    trainer.train()

    print(torch.cuda.max_memory_allocated() * 1e-9, "Gb")
    exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
