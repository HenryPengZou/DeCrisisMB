import torch
import numpy as np


class MemoryBank:
    def __init__(self, n_classes=6, memory_per_class = 200, selection_strategy=None, selection_top_threshold=0.8, selection_sharpening_T=0.5):

        self.bank_input = [None] * n_classes
        self.bank_mask = [None] * n_classes
        self.bank_label = [None] * n_classes
        self.bank_selection_score = [None] * n_classes
        self.bank_statistics = [None] * n_classes
        self.n_classes = n_classes
        self.memory_per_class = memory_per_class

        self.selection_strategy = selection_strategy
        self.selection_top_threshold = selection_top_threshold
        self.selection_sharpening_T = selection_sharpening_T


    def update_memorybank(self, input_ids, attention_mask, labels, selection_score=None):
        # update by high-confidence pseudo-labeled data
        _, labels_hard = torch.max(labels, dim=-1)
        if input_ids.shape[0] > 0:
            for c in range(self.n_classes):
                # update each class queue
                mask_c = labels_hard == c
                if input_ids[mask_c].shape[0] > 0:
                    new_input_ids = input_ids[mask_c].cpu().numpy()
                    new_attention_mask = attention_mask[mask_c].cpu().numpy()
                    new_labels = labels[mask_c].cpu().numpy()
                    if selection_score is not None:
                        new_selection_score = selection_score[mask_c].cpu().numpy()


                    if self.bank_input[c] is None: # if empty, first updates
                        self.bank_input[c] = new_input_ids
                        self.bank_mask[c] = new_attention_mask
                        self.bank_label[c] = new_labels
                        if selection_score is not None:
                            self.bank_selection_score[c] = new_selection_score
                                                     
                    else: # not empty, update to existing list (FIFO queue)
                        self.bank_input[c] = np.concatenate((new_input_ids, self.bank_input[c]), axis=0)[:self.memory_per_class, :]
                        self.bank_mask[c] = np.concatenate((new_attention_mask, self.bank_mask[c]), axis=0)[:self.memory_per_class, :]
                        self.bank_label[c] = np.concatenate((new_labels, self.bank_label[c]), axis=0)[:self.memory_per_class, :]
                        if selection_score is not None:
                            # print('Tree-new_selection_score: ', new_selection_score)
                            # print('Tree-self.bank_selection_score[c]: ', self.bank_selection_score[c])
                            self.bank_selection_score[c] = np.concatenate((new_selection_score, self.bank_selection_score[c]), axis=0)[:self.memory_per_class]
                
                    # compute # psl stored in each queue; length of queues
                    self.bank_statistics[c] = self.bank_label[c].shape[0]
            print('Tree -bank statistics: ', self.bank_statistics)

    def sample_memorybank(self, samples_per_class, replace=False, shuffle=False, cw_mean_prob=None, t=None):
        sampled_input, sampled_mask, sampled_label = None, None, None
    

        # sample each class queue
        for c in range(self.n_classes):
            # if # psl in any queue < # samples_per_class, skip to avoid introduce bias; sampling only when all queue has psl > # sample_per_class
            if self.bank_statistics[c] is None or self.bank_statistics[c] < samples_per_class:
                return torch.tensor([]).long(),torch.tensor([]).long(),torch.tensor([]).long()
                # break
                # TODO: memorybank start to sample when first class has more than 5 samples, even though others not, check and revise.

            # selection strategy: 'top', 'proportional', 'top_proportional'
            # 'proportional': samples with larger margin/AUM have more possibility to be selected
            # generate selection probs
            select_probs = None
            if self.bank_selection_score[c] is not None:
                if self.selection_strategy == 'top':
                    idx = int(self.bank_statistics[c] * self.selection_top_threshold)
                    # print('Tree-idx: ', idx)
                    # print('Tree-self.bank_statistics[c]: ', self.bank_statistics[c])
                    threshold_v = sorted(self.bank_selection_score[c], reverse=True)[idx] # get the value at top threshold
                    selection_mask = (self.bank_selection_score[c] >= threshold_v).flatten()
                    select_probs = np.zeros(self.bank_statistics[c])
                    select_probs[selection_mask] = 1 / sum(selection_mask)
                    # print('Tree-select_probs: ', select_probs)

                elif self.selection_strategy == 'proportional':
                    st = self.bank_selection_score[c] **(1/self.selection_sharpening_T)
                    select_probs = st / np.sum(st) # can use sharpening temperature to adjust entropy/degree
                    # select_probs = self.bank_selection_score[c] / np.sum(self.bank_selection_score[c]) # can use sharpening temperature to adjust entropy/degree
                # print('Tree - select_probs: ', select_probs)


            if self.bank_input[c] is not None:  # if empty, skip
                if cw_mean_prob is not None:    # sampling strategy: int[(((1/n_classes)/p)^t)*N] or N
                    n_samples = int(torch.round( (((1/self.n_classes)/cw_mean_prob[c])**t) *samples_per_class))
                    replace = True
                    sample_indices = np.random.choice(self.bank_statistics[c], n_samples, replace=replace, p=select_probs) 
                else:   # sampling strategy: random sampling equal #
                    sample_indices = np.random.choice(self.bank_statistics[c], samples_per_class, replace=replace, p=select_probs) 


                sampled_input_c = self.bank_input[c][sample_indices]
                sampled_mask_c = self.bank_mask[c][sample_indices]
                sampled_label_c = self.bank_label[c][sample_indices]

                # concatenate all sampled data in each class 
                if sampled_input is None: # if empty, first update
                    sampled_input, sampled_mask, sampled_label = sampled_input_c, sampled_mask_c, sampled_label_c
                else: # not empty, update to existing list
                    sampled_input = np.concatenate((sampled_input,sampled_input_c), axis=0)
                    sampled_mask = np.concatenate((sampled_mask,sampled_mask_c), axis=0)
                    sampled_label = np.concatenate((sampled_label,sampled_label_c), axis=0)
                sampled_input, sampled_mask, sampled_label = torch.tensor(sampled_input), torch.tensor(sampled_mask), torch.tensor(sampled_label)

        
        if sampled_input is not None:
            if shuffle:
                np.random.shuffle(sampled_input)
                np.random.shuffle(sampled_mask)
                np.random.shuffle(sampled_label)
        else:
            sampled_input, sampled_mask,sampled_label = torch.tensor([]).long(),torch.tensor([]).long(),torch.tensor([]).long()

        return sampled_input, sampled_mask, sampled_label
  


# # Unit Test
# input_ids = torch.rand((3,3))
# attention_mask = torch.ones((3,3))
# labels = torch.tensor([1,2,2])

# n_classes = 6
# memory_per_class = 200
# samples_per_class = 5
# # initial_data = torch.ones((samples_per_class, n_classes))

# mb = MemoryBank(n_classes=6, memory_per_class = 200)
# mb.update_memorybank(input_ids, attention_mask, labels)
# print(mb.bank_input)
# sampled_input, sampled_mask, sampled_label = mb.sample_memorybank(samples_per_class, shuffle=True)
# print(sampled_input)
# print(sampled_label)