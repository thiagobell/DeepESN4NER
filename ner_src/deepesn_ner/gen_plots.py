import json
import matplotlib.pyplot as plt
import os
from typing import Optional
import re
from deepesn_ner.config import ner_models_folder
import p_tqdm
from collections import defaultdict

def gen_metric_train_report(model_path, trial_number: Optional[str], metric: str, overwrite: bool):
    """

    Args:
        train_report_path:
        metric: one of train_loss, entity_train, entity_dev, token_train, token_dev'

    Returns:

    """
    if trial_number is None:
        train_report_path = os.path.join(model_path, f"train_report.json")
    else:
        train_report_path = os.path.join(model_path, f"train_report.{trial_number}.json")
    with open(train_report_path) as f:
        train_report = json.load(f)

    if metric == "train_loss":
        metric_getter = lambda x: x["train_loss"]
        series_name = metric
    elif metric == "entity_train" or metric == "entity_dev":
        # may return None
        metric_getter = lambda x: {
            "micro f1": x.get(metric, {}).get("micro f1", None),
            "macro f1": x.get(metric, {}).get("macro f1", None)
        }
        series_name = metric
    elif metric == "token_train" or metric == "token_dev":
        metric_getter = lambda x: {
            "micro f1": x.get(metric, {}).get("classification report", {}).get("micro avg", {}).get("f1-score", None),
            "macro f1": x.get(metric, {}).get("classification report", {}).get("macro avg", {}).get("f1-score", None)
        }
        series_name = metric
    else:
        raise ValueError(f"invalid metric {metric}")
    if trial_number is None:
        plot_file = f"{series_name}.png"
    else:
        plot_file = f"{series_name}.{trial_number}.png"
    plot_file = os.path.join(model_path, plot_file)
    if os.path.exists(plot_file) and not overwrite:
        print(f"already found plot for model {model_path}")
        return

    ys = defaultdict(list) # key -> y axis of series
    xs = defaultdict(list) # key -> x axis of series
    for epoch in range(len(train_report)):
        metric_value = metric_getter(train_report[epoch])
        if type(metric_value) == dict:
            for key in metric_value:
                if metric_value[key] is not None:
                    ys[key].append(metric_value[key])
                    xs[key].append(epoch)
        else:
            ys[series_name].append(metric_value)
            xs[series_name].append(epoch)
    plt.figure()
    for series in ys.keys():
        plt.plot(xs[series], ys[series], label=series)
    plt.legend()
    plt.title(series_name)
    plt.savefig(plot_file)
    plt.close()

def process_model(model, metric, overwrite):
    m_path = os.path.join(ner_models_folder, model)
    for file_name in os.listdir(m_path):
        mt = re.match(report_parser, file_name)
        if mt:
            trial_number = mt.groups()[1]
            gen_metric_train_report(m_path, trial_number, metric, overwrite)


if __name__ == "__main__":
    from itertools import repeat, product
    report_parser = r"train_report(.([0-9]*)|).json"
    models = os.listdir(ner_models_folder)
    overwrite = True
    metrics_to_gen = ["token_train", "token_dev", "train_loss", "entity_dev", "entity_train"]
    #"train_loss",
    for metric in metrics_to_gen:
        print(metric)
        #for model in models:
        #process_model(model, metric)
        p_tqdm.p_map(process_model, models, repeat(metric), repeat(overwrite), desc=f"gen plots({metric})", num_cpus=10)
