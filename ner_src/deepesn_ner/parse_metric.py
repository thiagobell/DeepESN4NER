import os
import json
import tabulate
from deepesn_ner.config import ner_models_folder
import pandas as pd
from sys import argv
import re
import numpy as np

def get_metric_from_file(path):
    with open(path) as f:
        data = json.load(f)
        values = []
        for epoch in data:
            if "entity_dev" in epoch:
                values.append(epoch['entity_dev']['micro f1'])
        return values

def get_metric(model_folder):

    # there may be more than one trial
    trial_parser = r"train_report.[0-9]*.json"
    metric_values_max = []
    metric_values_last = []
    for filen in os.listdir(model_folder):
        if re.match(trial_parser, filen) is not None:
            values = get_metric_from_file(os.path.join(model_folder, filen))
            if len(values) > 0:
                metric_values_max.append(max(values))
                metric_values_last.append(values[-1])
    if len(metric_values_max) > 0:
        mean_max = np.mean(metric_values_max)
        stddev_max = np.std(metric_values_max)
        mean_last = np.mean(metric_values_last)
        stddev_last = np.std(metric_values_last)
        return f"{mean_max:.4f} ± {stddev_max:.4f}", f"{mean_last:.4f} ± {stddev_last:.4f}"

    # need to check for legacy experiments
    legacy_report = os.path.join(model_folder, "train_report.json")

    if os.path.exists(legacy_report):
        # there is a single trial
        values = get_metric_from_file(legacy_report)
        if len(values) > 0:
            metric = f"{max(values):.4f}", f" {values[-1]:.4f}"
        else:
            metric = None

        if metric is None:
            return f"EPOCH NOT FOUND", f"EPOCH NOT FOUND"
        return metric
    return "NO REPORT FOUND", "NO REPORT FOUND"



if __name__ == "__main__":
    models = os.listdir(ner_models_folder)
    out_data = []
    out_data.append(["name of model", "max metric", "last metric"])
    for model in sorted(models):
        m_path = os.path.join(ner_models_folder, model)
        out_data.append([model, *get_metric(m_path)])

    if len(argv) > 1:
        print(argv)
        if argv[1] == "csv":

            df = pd.DataFrame(out_data)
            df.to_csv("metric_report.csv")
    else:
        print(tabulate.tabulate(out_data))