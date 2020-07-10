import csv
import glob
from sklearn.metrics import classification_report, f1_score

import numpy as np

from biomed.properties_manager import PropertiesManager


class Evaluator:
    def __init__(self):
        self.Y_test_75_binary, self.Y_test_75_multi = self.get_y_data()

    def get_y_data(self):
        Y_test_75_binary = np.load('../training_data/Y_test_75_binary.npy')
        Y_test_75_multi = np.load('../training_data/Y_test_75_multi.npy')
        return Y_test_75_binary, Y_test_75_multi

    def get_preds_from_csv_file(self, file):
        input_data = self.get_list_from_csv(file)
        preds = list()
        for row in input_data:
            preds.append(row[1])
        preds_target = self.Y_test_75_multi if preds[0] == "doid" else self.Y_test_75_binary
        preds = preds[1:] if preds[0] == 'doid' else self.convert_buggy_doid_to_real_binary(preds[1:])
        # print(f"{file} len(y) {len(self.Y_test_75_binary)} len(preds) {len(preds)}")
        return preds, preds_target

    def get_list_from_csv(self, file, delimiter=','):
        input_data = list()
        read_csv = csv.reader(file, delimiter=delimiter)
        for row in read_csv:
            input_data.append(row)
        return input_data

    def convert_buggy_doid_to_real_binary(self, preds):
        for i, entry in enumerate(preds):
            preds[i] = '0' if entry == '-1' else '1'
        return preds

    def generate_stats_and_write_to_file(self, preds):
        pm = PropertiesManager()
        preds_target = self.Y_test_75_multi if pm.classifier == "doid" else self.Y_test_75_binary
        preds = preds if pm.classifier == 'doid' else self.convert_buggy_doid_to_real_binary(preds)
        if len(preds) != len(preds_target):
            print('Prediction lengths don\'t match')
            return

        tp = 0
        for i in range(len(preds)):
            if preds[i] == preds_target[i]:
                if preds[i] not in ('0', '-1'):
                    tp += 1
        score = classification_report(preds_target, preds)
        f1_score_macro_accurate = f1_score(preds_target, preds, average='macro')
        print('classification report:\n', score,
              '\nmore accurate macro f1 score', f1_score_macro_accurate,
              '\n correctly predicted:', tp)

        with open('../results/all_results.txt', 'a') as file:
            file.write(f"classifier={pm.classifier}, model={pm.model}, preprocessing={pm.preprocessing['variant']},"
                       f"ngrams={pm.tfidf_transformation_properties['ngram_range'][1]}\n"
                       f"{score}\nf1_score_macro_accurate={f1_score_macro_accurate}\ntrue positives={tp}\n\n")


if __name__ == '__main__':
    f1 = Evaluator()
    for file in glob.glob("*.csv*"):
        with open(file, 'r') as file:
            preds, preds_target = f1.get_preds_from_csv_file(file)
            if len(preds) == len(preds_target):
                matches, matches_pos, matches_neg = 0, 0, 0
                for i in range(len(preds)):
                    if preds[i] == preds_target[i]:
                        if preds[i] not in ('0', '-1'):
                            # print(preds[i], preds_target[i])
                            matches += 1
                    # else:
                    #     print(preds[i], preds_target[i])
                score = classification_report(preds_target, preds)
                print(file.name, '\nclassification report:\n', score, 'correctly predicted:', matches)
