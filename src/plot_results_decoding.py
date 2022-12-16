"""
    plot_results_decoding.py
    Created by Pierre Tessier
    12/14/22 - 7:08 PM
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
    df["num_beams"] = config["num_beams"]
    df["temperature"] = config["temperature"]
    df["meter_coeff"] = config["meter_mixin_coeff"]
    df["rhyme_coeff"] = config["rhyme_mixin_coeff"]
    df["alliteration_coeff"] = config["alliteration_mixin_coeff"]
    return df

def plot_alliteration(df, path=None, ax=None):
    df = df[["alliteration_coeff", "quatrain", "alliteration_2", "alliteration_3", "alliteration_4"]]
    dfs = []
    for i, alliteration in enumerate(["alliteration_2", "alliteration_3", "alliteration_4"]):
        partial = df.loc[:,["alliteration_coeff", "quatrain"]]
        partial.insert(2, "alliteration", df[alliteration])
        partial.insert(3, "Repetitions", i+2)
        dfs.append(partial)
    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    full_df = full_df[full_df.quatrain]
    full_df = full_df.fillna({"alliteration_coeff":"Baseline"})

    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(full_df, x="alliteration_coeff", y="alliteration", hue="Repetitions", errorbar="ci",
                order=["Baseline", 0.01, 0.1, 1.0], ax=ax)
    ax.set_ylabel(r"Alliteration Frequencies $(\uparrow)$")
    ax.set_xlabel("Alliteration Coeff")
    ax.legend(loc="upper right", title="Repetitions")
    if not subplot:
        plt.tight_layout()

    if path is not None:
        plt.savefig(path / "decoding_alliteration.png", dpi=300)
    elif not subplot:
        plt.show()
def plot_rhymes(df, path=None, ax=None):
    df = df[["rhyme_coeff", "quatrain", "weak_rhymes", "perfect_rhymes"]]
    dfs = []
    for t, rhymes in zip(["Perfect", "Weak"], ["perfect_rhymes", "weak_rhymes"]):
        partial = df.loc[:,["rhyme_coeff", "quatrain"]]
        partial.insert(2, "rhymes", df[rhymes])
        partial.insert(3, "type", t)
        dfs.append(partial)
    full_df = pd.concat(dfs, ignore_index=True, axis=0)
    full_df = full_df[full_df.quatrain]
    full_df = full_df.fillna({"rhyme_coeff":"Baseline"})

    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(full_df, x="rhyme_coeff", y="rhymes", hue="type", errorbar="ci",
                order=["Baseline", 0.01, 0.1, 1.0], ax=ax)
    ax.set_ylabel(r"Rhyme Frequencies $(\uparrow)$")
    ax.set_xlabel("Rhyme Coeff")
    ax.legend(title="Rhyme Type")
    if not subplot:
        plt.tight_layout()

    if path is not None:
        plt.savefig(path / "decoding_rhymes.png", dpi=300)
    elif not subplot:
        plt.show()

def plot_meter(df, path=None, ax=None):
    df = df[["meter_coeff", "quatrain", "correct_meter_distance"]]
    df = df[df.quatrain]
    df = df.fillna({"meter_coeff":"Baseline"})

    subplot = ax is not None
    if not subplot:
        fig, ax = plt.subplots()
    sns.barplot(df, x="meter_coeff", y="correct_meter_distance" ,errorbar="ci",
                order=["Baseline", 0.01, 0.1, 1.0, 10], ax=ax)
    ax.set_ylabel(r"Damerau-Levenshtein Distance $(\downarrow)$")
    ax.set_xlabel("Meter Coeff")
    if not subplot:
        plt.tight_layout()

    if path is not None:
        plt.savefig(path / "decoding_meter.png", dpi=300)
    elif not subplot:
        plt.show()
def build_sweep_df(api: Api, path: str):
    runs = api.runs(path)
    dfs = [build_run_df(run) for run in tqdm(runs)]
    dfs = list(filter(lambda x: bool(x.size), dfs))
    df = pd.concat(dfs, ignore_index=True, axis=0)
    return df
def main(args):
    results_path = Path("etc/wandb_process/constrained_decoding.csv")
    baseline_path = Path("etc/wandb_process/all_models.csv")
    plots_path = Path("doc")

    if results_path.exists():
        df = pd.read_csv(results_path, index_col=0)
    else:
        api = Api()
        df = build_sweep_df(api, "pierretsr/NLGS_Project_decoding")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path)
    baseline = pd.read_csv(baseline_path, index_col=0)
    baseline = baseline.loc[baseline.model == "gpt-neo-1.3B-fine-tuned"]
    full_df = pd.concat((df, baseline), ignore_index=True, axis=0)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6,7))
    sns.set_palette(sns.color_palette("mako", 3))
    plot_alliteration(full_df, ax=axs[0])

    sns.set_palette(sns.color_palette("mako", 2))
    plot_rhymes(full_df, ax=axs[1])

    sns.set_palette(sns.color_palette("mako", 5))
    plot_meter(full_df, ax=axs[2])

    plt.tight_layout()
    plt.savefig(plots_path/"decoding_all_metrics.png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    raise SystemExit(main(args))
