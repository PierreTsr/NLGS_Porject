"""
    fine_tune.py
    Created by Pierre Tessier
    10/25/22 - 7:35 AM
    Description:
    # Enter file description
 """
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from evaluate import load
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from src import load_dataset
from src.models import PronunciationGPT

accuracy = load("accuracy")
perplexity = load("perplexity")
bleu = load("bleu")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    num_training_epochs_pronunciation: float = field(
        default=1.0,
        metadata={"help": "Number of training epochs with GPT frozen to train the PronunciationGPT model."}
    )
    num_training_epochs_full: float = field(
        default=3.0,
        metadata={"help": "Number of training epochs with the full model unfrozen."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: str = field(
        metadata={
            "help": "Path to the dataset dictionary for training, testing and evaluation."
        }
    )
    embeddings_path: str = field(
        metadata={
            "help": "Path to the pretrained pronunciation embeddings."
        }
    )


def get_checkpoint(training_args: Seq2SeqTrainingArguments):
    last_checkpoint = None
    if Path(training_args.output_dir).is_dir() and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(list(Path(training_args.output_dir).glob("*"))) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    return checkpoint


def main(model_args: ModelArguments, training_args: Seq2SeqTrainingArguments, data_args: DataTrainingArguments):
    dataset = load_dataset(data_args.dataset_path)
    embeddings_p = torch.load(Path(data_args.embeddings_path) / "pronunciation_embeddings.pt")
    embeddings_s = torch.load(Path(data_args.embeddings_path) / "stress_embeddings.pt")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    gpt = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model = PronunciationGPT(gpt, embeddings_p, embeddings_s)

    device = torch.device("cpu")
    if torch.cuda.is_available() and not training_args.no_cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        model.to(device)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions_txt = tokenizer.batch_decode(predictions)
        references = tokenizer.batch_decode(labels)
        results_acc = accuracy.compute(predictions=predictions.flatten(), references=labels.flatten())
        results_bleu = bleu.compute(predictions=predictions_txt, references=[[txt] for txt in references])
        res = {**results_acc, **results_bleu}
        print(res)
        return res

    def preprocess_logits_for_metrics(logits, labels):
        logits = torch.argmax(logits, -1)
        return logits

    if training_args.do_train:
        training_args.num_train_epochs = model_args.num_training_epochs_pronunciation
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        checkpoint = get_checkpoint(training_args)
        model.freeze_gpt()
        trainer.train(resume_from_checkpoint=checkpoint)

        training_args.num_train_epochs = model_args.num_training_epochs_full
        training_args.overwrite_output_dir = False
        training_args.resume_from_checkpoint = None
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        checkpoint = get_checkpoint(training_args)
        model.unfreeze_gpt()
        trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
    else:
        args = parser.parse_args_into_dataclasses()

    raise SystemExit(main(*args))
