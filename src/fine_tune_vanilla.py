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

import torch
from evaluate import load
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser, AutoModelForCausalLM, \
    AutoTokenizer, Adafactor
from transformers.optimization import AdafactorSchedule

from src import load_dataset, CMUDictionary, CMULinker, PronunciationTokenizer, MeterMetrics
from metrics import AlliterationMetrics,RhymingMetrics, DistinctMetrics

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
    num_training_epochs: float = field(
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


def main(model_args: ModelArguments, training_args: Seq2SeqTrainingArguments, data_args: DataTrainingArguments):
    dataset = load_dataset(data_args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    cmu = CMUDictionary()
    linker = CMULinker(tokenizer, cmu)
    tokenizer_p = PronunciationTokenizer(linker, tokenizer)

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    alliterations = AlliterationMetrics(linker, tokenizer_p, verbose=False)
    rhymes = RhymingMetrics(linker, tokenizer_p, verbose=False)
    meter = MeterMetrics(linker, tokenizer_p, verbose=False)
    distinct = DistinctMetrics(tokenizer, 4, verbose=False)

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
        results_alli = alliterations.compute(predictions)
        results_rhymes = rhymes.compute(predictions)
        results_meter = meter.compute(predictions)
        results_distinct = distinct.compute(predictions)
        res = {
            **results_acc,
            **results_bleu,
            **results_alli,
            **results_rhymes,
            **results_meter,
            **results_distinct,
        }
        print(res)
        return res

    def preprocess_logits_for_metrics(logits, labels):
        logits = torch.argmax(logits, -1)
        return logits

    if training_args.do_train:
        training_args.num_train_epochs = model_args.num_training_epochs
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            optimizers=(optimizer, lr_scheduler)
        )
        trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=str(Path(sys.argv[1]).absolute()))
    else:
        args = parser.parse_args_into_dataclasses()

    raise SystemExit(main(*args))
