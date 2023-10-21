
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt


class AUMRecorder():
    def __init__(self, all_sample_ids=None, labels=None, n_classes=8):
        """
        Initiliaze an AUMRecords object to record AUM related information for all samples

        :param sample_ids: Shape[num_samples]
        :param labels: Shape[num_labels]
        """
        if all_sample_ids is None:
            self.records = None
        elif labels is None:
            self.records = {sample_id: {'logits':[], 'margin':[], 'aum':[], 'pred':[], 'gt':None} for sample_id in all_sample_ids} 
        else:
            self.records = {sample_id: {'logits':[], 'margin':[], 'aum':[], 'pred':[], 'gt':label} for (sample_id, label) in zip(all_sample_ids, labels)}

        self.n_classes = n_classes


    def update(self, logits, sample_ids):
        """
        Updates AUMRecords for given samples

        :param logits: Shape[num_samples x num_classes], each row contains logits for a sample
        :param sample_ids: Shape[num_samples]
        :param labels: Shape[num_labels]
        """

        top2_val, top2_idx = torch.topk(logits,2)
        margin = top2_val[:,0] - top2_val[:,1]
        max_idx = top2_idx[:,0]

        # convert everything into numpy array
        logits = logits.tolist()
        margin = margin.tolist()
        max_idx = max_idx.tolist()


        for i, sample_id in enumerate(sample_ids):
            record = self.records[sample_id]
            record['logits'].append(logits[i])
            record['margin'].append(margin[i])
            aum = sum(record['margin']) / len((record['margin']))
            record['aum'].append(aum)
            record['pred'].append(max_idx[i])
            self.records[sample_id] = record

    def save(self, dir_path):
        """
        Save AUM records as a json file
        """
        filename = os.path.join(dir_path, 'aum.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=4)
        print('Successfully saved AUM records in: ', filename)
        
    def load(self, path):
        """
        Load saved AUM json file to AUM records dictionary
        """
        with open(path) as f:
            self.records = json.load(f)
        print('Successfully load AUM records from: ', path)


    def visual(self, sample_id, metrics, n_class = 8, plt_line='-', plt_marker=','):
        """
        Visualize AUM records

        :param metrics: visualization metrics, options: 'logits', 'margin', 'aum', 'pred'
        """

        y = self.records[sample_id][metrics]
        gt = self.records[sample_id]['gt']
        if metrics == 'logits':
            logits = np.array(y)
            [plt.plot(logits[:,c], plt_line, label=c, marker=plt_marker) for c in range(n_class)]
        else:
            plt.plot(y, plt_line, marker=plt_marker)

        if metrics == 'logits' and gt is not None:
            title = 'Id: ' + sample_id + ', Metrics: ' + metrics + ', GT: ' + str(gt) 
        else:
            title = 'Id: ' + sample_id + ', Metrics: ' + metrics
        plt.title(title)
        plt.xlim(0)
        plt.legend()
        plt.show()

    def visual_multisample(self, sample_ids, metrics, curve_labels=None, curve_colors=None, title_info=None, plt_line='-', plt_marker=',', alpha=0.5):
        """
        Visualize AUM records for multiple samples together in one plot

        :param metrics: visualization metrics, options: 'margin', 'aum', 'pred'
        """

        if curve_labels is None:
            curve_labels = sample_ids

        if curve_colors is None:
            for (id, curve_label) in zip(sample_ids, curve_labels):
                plt.plot(self.records[id][metrics], plt_line, label=curve_label, marker=plt_marker, alpha=alpha)
        else:
            for (id, curve_label, curve_color) in zip(sample_ids, curve_labels, curve_colors):
                plt.plot(self.records[id][metrics], plt_line, label=curve_label, color=curve_color, marker=plt_marker, alpha=alpha)            

        title = 'Mutisample' + ', Metrics: ' + metrics + title_info
        plt.title(title)
        plt.ylabel(metrics)
        plt.xlabel('epochs')
        plt.xlim(0)
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()



# ########### Unit Test ###########
# # - input example
# all_sample_ids = [100, 200, 300] # list
# # labels = None
# labels = [2, 1, 0] # list
# logits = torch.rand((2,3))
# sample_ids = [100, 200]
# saving_path = '/home/pzou3/1_ResearchProjects/1_imbalancedSF/log/A_Output'

# # - initialization
# AUMRecords = AUMRecorder(all_sample_ids, labels)
# # AUMRecords = AUMRecorder(all_sample_ids)

# # - update
# print('---First Update---')
# AUMRecords.update(logits, sample_ids)
# print(AUMRecords.records[100])
# print(AUMRecords.records)


# # - update 2,3
# print('---More Updates---')
# logits = torch.rand((2,3))
# AUMRecords.update(logits, sample_ids)
# AUMRecords.update(logits, sample_ids)
# print(AUMRecords.records[100])
# print(AUMRecords.records)

# # - save
# print('---Save---')
# AUMRecords.save(saving_path)

# # - load
# print('---Load---')
# AUMRecords = AUMRecorder()
# print(AUMRecords.records)
# AUMRecords.load(saving_path+'/aum.json')
# print(AUMRecords.records)

# # - visual
# sample_id = str(100)
# visual_metrics = 'logits'
# # visual_metrics = 'margin'
# # visual_metrics = 'aum'
# # visual_metrics = 'pred'
# AUMRecords.visual(sample_id, visual_metrics, n_class=logits.shape[1])

# # - visualize multiple isamples
# sample_ids = [str(100), str(200), str(300)]
# visual_metrics = 'aum'  # 'margin', 'aum', 'pred'
# AUMRecords.visual_multisample(sample_ids, visual_metrics)
# curve_labels = ['correct: 100','incorrect: 200','incorrect: 300']
# AUMRecords.visual_multisample(sample_ids, visual_metrics, curve_labels=curve_labels) # specify curve_labels