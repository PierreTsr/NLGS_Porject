"""
    plot_results.py
    Created by Pierre Tessier
    12/12/22 - 12:13 AM
    Description:
    # Enter file description
 """
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from wandb import Api
from wandb.apis.public import Run

models_in_order = [
    "gpt-neo-125M",
    "gpt-neo-1.3B",
    "gpt-neo-2.7B",
    "gpt-neo-125M-fine-tuned",
    "gpt-neo-1.3B-fine-tuned",
    "gpt-neo-2.7B-fine-tuned",
    "gpt-neo-2.7B-custom"
]

model_names = {
    "gpt-neo-125M": r"GPT-Neo 125M",
    "gpt-neo-1.3B": r"GPT-Neo 1.3B",
    "gpt-neo-2.7B": r"GPT-Neo 2.7B",
    "gpt-neo-125M-fine-tuned": r"$\mbox{GPT-Neo}_{\mbox{G}}$ 125M",
    "gpt-neo-1.3B-fine-tuned": r"$\mbox{GPT-Neo}_{\mbox{G}}$ 1.3B",
    "gpt-neo-2.7B-fine-tuned": r"$\mbox{GPT-Neo}_{\mbox{G}}$ 2.7B",
    "gpt-neo-2.7B-custom": r"$\mbox{GPT-Neo}_{\mbox{P}}$ 2.7B"
}



def build_run_df(run: Run):
    history = run.scan_history()
    config = run.config
    keys = {"quatrain", "distinct-1", "distinct-2", "distinct-3", "alliteration_2", "alliteration_3", "alliteration_4", "perfect_rhymes", "weak_rhymes", "correct_meter_distance"}
    df = {key: [] for key in keys}
    for entry in history:
        for key in keys.intersection(entry.keys()):
            df[key].append(entry[key])
    df = pd.DataFrame.from_dict(df)
    df["model"] = config["model"].split("/")[-1]
    df["dataset"] = config["dataset"].split("/")[-1]
    return df

def build_sweep_df(api: Api, path: str):
    runs = api.runs(path)
    dfs = [build_run_df(run) for run in tqdm(runs)]
    df = pd.concat(dfs, ignore_index=True, axis=0)
    return df

def plot_distinct(df, path=None, ax=None):
    df = df[["model", "dataset", "quatrain", "distinct-1", "distinct-2", "distinct-3"]]
    dfs = []
    for i, distinct in enumerate(["distinct-1", "distinct-2", "distinct-3"]):
        partial = df.loc[:,["model", "dataset", "quatrain"]]
        partial.insert(3, "distinct", df[distinct])
        partial.insert(4, "n-gram", i+1)
        dfs.append(partial)
    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    full_df = full_df[full_df.quatrain]
    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(full_df, x="model", y="distinct", hue="n-gram", errorbar="ci", order=models_in_order, ax=ax)
    ax.set_ylabel(r"Distinct Score $(\uparrow)$")
    ax.legend(title="n-gram", loc="lower left")
    if not subplot:
        ax.set_xlabel("Model")
        xlabels = [model_names[model] for model in models_in_order]
        ax.set_xticklabels(xlabels, rotation=45)
        plt.tight_layout()
    else:
        ax.set(xlabel=None)

    if path is not None:
        plt.savefig(path / "all_models_distinct.png", dpi=300)
    elif not subplot:
        plt.show()


def plot_alliteration(df, path=None, ax=None):
    df = df[["model", "dataset", "quatrain", "alliteration_2", "alliteration_3", "alliteration_4"]]
    dfs = []
    for i, alliteration in enumerate(["alliteration_2", "alliteration_3", "alliteration_4"]):
        partial = df.loc[:,["model", "dataset", "quatrain"]]
        partial.insert(3, "alliteration", df[alliteration])
        partial.insert(4, "Repetitions", i+2)
        dfs.append(partial)
    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    full_df = full_df[full_df.quatrain]
    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(full_df, x="model", y="alliteration", hue="Repetitions", errorbar="ci", order=models_in_order, ax=ax)
    ax.set_ylabel(r"Alliteration Frequencies $(\uparrow)$")
    ax.legend(loc="center left")
    if not subplot:
        ax.set_xlabel("Model")
        xlabels = [model_names[model] for model in models_in_order]
        ax.set_xticklabels(xlabels, rotation=45)
        plt.tight_layout()
    else:
        ax.set(xlabel=None)

    if path is not None:
        plt.savefig(path / "all_models_alliteration.png", dpi=300)
    elif not subplot:
        plt.show()

def plot_rhymes(df, path=None, ax=None):
    df = df[["model", "dataset", "quatrain", "weak_rhymes", "perfect_rhymes"]]
    dfs = []
    for t, rhymes in zip(["Perfect", "Weak"], ["perfect_rhymes", "weak_rhymes"]):
        partial = df.loc[:,["model", "dataset", "quatrain"]]
        partial.insert(3, "rhymes", df[rhymes])
        partial.insert(4, "type", t)
        dfs.append(partial)
    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    full_df = full_df[full_df.quatrain]
    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(full_df, x="model", y="rhymes", hue="type", errorbar="ci", order=models_in_order, ax=ax)
    ax.set_ylabel(r"Rhyme Frequencies $(\uparrow)$")
    ax.legend(title="Rhyme Type", loc="upper left")
    if not subplot:
        ax.set_xlabel("Model")
        xlabels = [model_names[model] for model in models_in_order]
        ax.set_xticklabels(xlabels, rotation=45)
        plt.tight_layout()
    else:
        ax.set(xlabel=None)

    if path is not None:
        plt.savefig(path / "all_models_rhymes.png", dpi=300)
    elif not subplot:
        plt.show()
def plot_quatrain(df, path=None, ax=None):
    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(df, x="model", y="quatrain", errorbar="ci", order=models_in_order, ax=ax)
    ax.set_ylabel(r"Rate of correct quatrains $(\uparrow)$")
    if not subplot:
        ax.set_xlabel("Model")
        xlabels = [model_names[model] for model in models_in_order]
        ax.set_xticklabels(xlabels, rotation=45)
        plt.tight_layout()
    else:
        ax.set(xlabel=None)

    if path is not None:
        plt.savefig(path / "all_models_quatrain.png", dpi=300)
    elif not subplot:
        plt.show()
def plot_meter(df, path=None, ax=None):
    df = df[["model", "dataset", "quatrain", "correct_meter_distance"]]
    df = df[df.quatrain]
    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    df.loc[:,"dataset"] = df.loc[:,"dataset"].map({'pentameter_prompts': 'Pentameter', 'mixed_meter_prompts': 'Mixed Meter'})
    sns.barplot(df, x="model", y="correct_meter_distance", hue="dataset" ,errorbar="ci", order=models_in_order,
                hue_order=["Pentameter", "Mixed Meter"], ax=ax)
    ax.legend(title="Meter Type", loc="lower left")
    ax.set_ylabel(r"Damerau-Levenshtein Distance $(\downarrow)$")
    if not subplot:
        ax.set_xlabel("Model")
        xlabels = [model_names[model] for model in models_in_order]
        ax.set_xticklabels(xlabels, rotation=45)
        plt.tight_layout()
    else:
        ax.set(xlabel=None)

    if path is not None:
        plt.savefig(path / "all_models_meter.png", dpi=300)
    elif not subplot:
        plt.show()
def main(args):
    results_path = Path("etc/wandb_process/all_models.csv")
    plots_path = Path("doc")
    if results_path.exists():
        df = pd.read_csv(results_path.absolute(), index_col=0)
    else:
        api = Api()
        df = build_sweep_df(api, "pierretsr/NLGS_Project_all_models")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path.absolute())


    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    fig, axs = plt.subplots(nrows=5, ncols=1, sharex="all", figsize=(6, 12))
    sns.set_palette(sns.color_palette("mako", 7))
    plot_quatrain(df, ax=axs[0])
    sns.set_palette(sns.color_palette("mako", 3))
    plot_distinct(df, ax=axs[1])
    plot_alliteration(df, ax=axs[2])
    sns.set_palette(sns.color_palette("mako", 2))
    plot_rhymes(df, ax=axs[3])
    plot_meter(df, ax=axs[4])
    xticklabels = [model_names[model] for model in models_in_order]
    plt.xticks(range(len(models_in_order)),labels=xticklabels, rotation=45)

    plt.tight_layout()
    plt.savefig(plots_path / "all_models_all_metrics.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    raise SystemExit(main(args))
