import csv
import torch

def certainty_score(prob):
    if prob > 0.90:
        return 'E'
    elif prob > 0.75:
        return 'D'
    elif prob > 0.50:
        return 'C'
    elif prob > 0.35:
        return 'B'
    else:
        return 'A'

def output_results(filename, preds, targets, dima, dimb):
    preds_softmax_v = torch.softmax(preds, dim=-1)
    preds_class_v = preds_softmax_v.argmax(dim=-1)
    is_correct_v = preds_class_v == targets

    preds2 = preds.view(preds.size()[0],dima,dimb,preds.size()[2])
    targets2 = targets.view(targets.size()[0],dima,dimb)
    preds_class = preds_class_v.view(preds.size()[0],dima, dimb)
    is_correct  = is_correct_v.view(preds.size()[0],dima, dimb)
    with open(filename, 'w') as f:
        for i in range(preds2.shape[0]):
            pred_class_matrix = preds_class[i].tolist()
            target_class_matrix = targets2[i].tolist()
            is_correct_matrix = is_correct[i].tolist()

            f.write('Predicted Class Matrix:\n')
            for row in pred_class_matrix:
                f.write(' '.join(map(str, row)) + '\n')

            f.write('\nActual Class Matrix:\n')
            for row in target_class_matrix:
                f.write(' '.join(map(str, row)) + '\n')

            f.write('\nIs Correct Matrix:\n')
            for row in is_correct_matrix:
                f.write(' '.join(map(str, row)) + '\n')

            f.write('\n' + '-' * 50 + '\n')  # Add a line to separate samples
