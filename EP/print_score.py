import pickle as pkl
import json
import os

fold_name='ad739d780d8f2e6f0b825cefee967044'
def print_score(root, fold_name):
    file_name='{}/{}/score_list.pkl'.format(root,fold_name)

    json_name = '{}/{}/exp_dict.json'.format(root,fold_name)

    # print(fold_name)

    with open(json_name) as f:
      js = json.load(f)
    # if 'tiered' not in js['dataset_train_root']:
    #     return 0
    print(fold_name)
    print(js)
    print(' lr {} data_root {} model {}'.format(js['lr'],js['dataset_train_root'],js['model']))

    with open(file_name, 'rb') as infile:
        data = pkl.load(infile)
    acc_max=0
    for d in data:
        e = d['epoch']
        acc = d['test_accuracy']
        # dataset_root = d['dataset_test_root']
        # con = d['test_confidence']
        con = 0

        # print('epcoh: {}, test_accuracy: {}, test_confidence: {}'.format(e,acc, con))
        if acc>acc_max:
            ss = 'epcoh: {}, test_accuracy: {}, test_confidence: {}'.format(e,acc, con)
            acc_max = acc
    print('max: {}'.format(ss))

d = './logs/finetune-omni'
ll = os.listdir(d)
for l in ll:
    print_score(d,l)

