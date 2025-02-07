import jsonlines
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def print_result(result_store_path, feature_file, bin_num, audit_model):

    probabilities = []

    with jsonlines.open(os.path.join(result_store_path, feature_file)) as reader:
        for obj in reader:
            probability = obj.get('probability', '')
            probabilities.append(probability)

    hists = []
    bins = np.arange(0, 1.1, 1 / bin_num)
    for prob in probabilities:
        hist, _ = np.histogram(prob, bins)
        hists.append(hist)

    n = len(hists)
    y_true = [0] * (n // 2) + [1] * (n // 2)

    feature_columns = [f'feature{i + 1}' for i in range(bin_num)]

    test_data = pd.DataFrame(
        {**{feature_columns[i]: [hist[i] for hist in hists] for i in range(bin_num)}}
    )

    loaded_predictor = TabularPredictor.load(audit_model)

    y_predic = loaded_predictor.predict(test_data)
    
    y_probabilities = loaded_predictor.predict_proba(test_data)
    y_probabilities = y_probabilities.iloc[:, 1]

    print('evaluation number:', n)

    print(f"AUC: {roc_auc_score(y_true, y_probabilities)}")
    print(f"ACC: {accuracy_score(y_true, y_predic)}")
    print(f"PRE: {precision_score(y_true, y_predic)}")
    print(f"REC: {recall_score(y_true, y_predic)}")
    print(f"f1: {f1_score(y_true, y_predic)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='HealthCareMagic')
    parser.add_argument('--method', choices=['Audit'], help='methods')
    parser.add_argument('--llm', type=str, default='llama3')
    parser.add_argument('--data_store_path', type=str, default='Data')
    parser.add_argument('--result_store_path', type=str, default='Result')
    parser.add_argument('--defence', choices=['wo', 'prompt_modify', 'paraphrasing'])
    parser.add_argument('--bin_num', type=int, default=10)
    parser.add_argument('--audit_model', type=str)
    args = parser.parse_args()
    method = args.method
    data_store_path = args.data_store_path
    result_store_path = args.result_store_path
    dataset_name = args.dataset_name
    model = args.llm
    bin_num = args.bin_num

    if model.find('llama') != -1:
        model_id = "./Model/llama-3-8b-Instruct"
    elif model.find('gpt') != -1:
        model_id = "gpt-4o-mini"

    prefix = '-wo'

    if args.defence == 'wo':
        prefix = '-wo'
    elif args.defence == 'prompt_modify':
        prefix = '-prompt_modify'
    elif args.defence == 'paraphrasing':
        prefix = '-paraphrasing'

    feature_file = dataset_name + model_id.replace('/', '-').replace(".", "-") + prefix + '-' + method + '_feature.jsonl'

    print_result(result_store_path, feature_file, bin_num, audit_model)
        