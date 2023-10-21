
import os
import random
import math
import sys
import time

import numpy as np
import pandas as pd
import preprocessor as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from copy import deepcopy

from utils.memorybank import *
from utils.aum import *
from utils.aug import aug
from utils.ema import EMA

def oneRun(log_dir_path_multiRun=None, **params):
    import torch
    ##### Default Setting
    ## Set path
    # add sys path
    root = './'
    sys.path.append(root)    

    # output directory: used to store saved model and aum.json
    output_dir_path = './experiment/threecrises'

    # set data path
    base_path = './data'
    target_list = ['threecrises']
    print('\nData base directory: ', base_path)
    print('Target list: ', target_list)   

    # set log_dir_path: used to store log, plots, saved model for current training
    import time
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    if log_dir_path_multiRun is None:
        log_root = os.getcwd() + '/log/'
        log_dir_path = log_root + cur_time + '/'
    else:
        log_root = log_dir_path_multiRun
        log_dir_path = log_dir_path_multiRun + cur_time + '/'
    
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)



    ## Set default hyperparameters
    n_labeled_per_class = 10        if 'n_labeled_per_class' not in params else params['n_labeled_per_class']
    n_unlabeled_per_class = None    if 'n_unlabeled_per_class' not in params else params['n_unlabeled_per_class']

    seed = 42                       if 'seed' not in params else params['seed']
    device_idx = 1                  if 'device_idx' not in params else params['device_idx']
    val_interval = 25               if 'val_interval' not in params else params['val_interval'] # 20, 25
    steps = 5000                    if 'steps' not in params else params['steps']
    early_stop_tolerance = 10       if 'early_stop_tolerance' not in params else params['early_stop_tolerance'] # 5, 6, 10
    lr = 2e-5                       if 'lr' not in params else params['lr']
    bs = 32                         if 'bs' not in params else params['bs']      # original: 32     
    bs_u = 32                       if 'bs_u' not in params else params['bs_u']      # original: 32   
    # TODO: think about every sizes: bs, memory_per_class, samples_per_class
    max_length = 64

    # - data augmentation
    aug_mode = True                 if 'aug_mode' not in params else params['aug_mode']
    num_aug = 2                     if 'num_aug' not in params else params['num_aug']
    alpha_sr = 0.1                  if 'alpha_sr' not in params else params['alpha_sr']
    alpha_rs = 0.1                  if 'alpha_rs' not in params else params['alpha_rs']
    aug_dict = {'num_aug': num_aug, 'alpha_sr': alpha_sr, 'alpha_rs': alpha_rs}
    strong_aug_mode = False         if 'strong_aug_mode' not in params else params['strong_aug_mode']

    # - mixup
    mixup_mode = False              if 'mixup_mode' not in params else params['mixup_mode']            
    mixup_method = 3                if 'mixup_method' not in params else params['mixup_method'] #[1,2,3]
    mixup_alpha = 0.75              if 'mixup_alpha' not in params else params['mixup_alpha'] # alpha: 2, 0.75 
    mixup_layers = [i for i in range(13)]   if 'mixup_layers' not in params else params['mixup_layers'] # mixup_layers: [[-1], [i for i in range(13)], [0], [12], [7,9,12], [6,8,11]]

    # - semi-supervised 
    weight_u_loss = 10              if 'weight_u_loss' not in params else params['weight_u_loss']
    u_loss_rampup_length = 1000     if 'u_loss_rampup_length' not in params else params['u_loss_rampup_length']
    threshold_mode = 'hard_t'       if 'threshold_mode' not in params else params['threshold_mode'] # hard_t, sat, sat_global
    labeling_mode = 'hard'          if 'labeling_mode' not in params else params['labeling_mode'] # hard, soft, sharpening
    u_loss = 'L2'                   if 'u_loss' not in params else params['u_loss'] # cross_entropy, L2

    # - consistency regularization
    consis_reg = 'avg_pred'         if 'avg_pred' not in params else params['avg_pred'] # avg_pred, weak_sup_strong
    
    # - pseudo-labeling
    psl_mode = False                if 'psl_mode' not in params else params['psl_mode']
    psl_threshold_h = 0.75          if 'psl_threshold_h' not in params else params['psl_threshold_h']
    threshold_rampup_length = 100   if 'threshold_rampup_length' not in params else params['threshold_rampup_length']
    sharpening_T = 0.5              if 'sharpening_T' not in params else params['sharpening_T']

    # - ema
    ema_mode = False                if 'ema_mode' not in params else params['ema_mode']
    ema_momentum = 0.999            if 'ema_momentum' not in params else params['ema_momentum']


    # - dibias
    # threshold_mode = 'debias'
    marginal_loss = False           if 'marginal_loss' not in params else params['marginal_loss']
    tau = 0.4                       if 'tau' not in params else params['tau']
    qhat_momentum = 0.99            if 'qhat_momentum' not in params else params['qhat_momentum'] # 0.99, 0.999 

    # - memorybank
    use_memorybank = False          if 'use_memorybank' not in params else params['use_memorybank']     # original: 200
    memory_per_class = 200          if 'memory_per_class' not in params else params['memory_per_class']     # original: 5
    samples_per_class = 5           if 'samples_per_class' not in params else params['samples_per_class']
    sampling_strategy = None        if 'sampling_strategy' not in params else params['sampling_strategy'] # 'avg_probs', None
    sampling_temperature = 1        if 'sampling_temperature' not in params else params['sampling_temperature'] # Range:[0~2]
    selection_strategy = None       if 'selection_strategy' not in params else params['selection_strategy'] # selection_strategy = 'top', 'proportional', 'top_proportional'
    selection_metrics = None        if 'selection_metrics' not in params else params['selection_metrics'] # selection_metrics = 'margin', 'aum'    
    selection_top_threshold = 0.8   if 'selection_top_threshold' not in params else params['selection_top_threshold']
    selection_sharpening_T = 0.5    if 'selection_sharpening_T' not in params else params['selection_sharpening_T']

    # - investigate
    investigate = None              if 'investigate' not in params else params['investigate'] # ['psl_acc', 'psl_num', 'psl_acc_num']
    if investigate == 'psl_num' or investigate == 'psl_acc_num':
    # (b) use same # psl per classs for training per iteration, here psl are not guaranteed to be correct  (invest. num) -> (memory bank)
        use_memorybank = True

    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    # Check & set device
    if torch.cuda.is_available():     
        device = torch.device("cuda", device_idx)
        print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
    else:
        print('\nNo GPU available, using the CPU instead.')
        device = torch.device("cpu")



    ##### Data Processing 
    # read data
    def read_data_split(event_list, split):
        total_df = pd.DataFrame()
        for event in event_list:
            df = pd.read_csv(base_path+'/%s/%s_%s.tsv' % (event, event, split), sep='\t')
            total_df = pd.concat([total_df, df])
        return total_df

    target_train_df = read_data_split(target_list, 'train')
    target_dev_df = read_data_split(target_list, 'dev')
    target_test_df = read_data_split(target_list, 'test')

    # labels mapping
    labels_set = sorted(set(target_train_df['class_label'].to_list()))  # use sorted() here because set() output order is unstable

    n_labels = len(labels_set)
    labels_mapping = {label: idx for idx, label in enumerate(labels_set)}
    print('\nAvailable labels and mapping: \n', labels_mapping)
    print('n_labels: ', n_labels)
    print('')

    data = [target_train_df, target_dev_df, target_test_df]
    data_name = ['target_train_df', 'target_dev_df', 'target_test_df']
    for idx, df in enumerate(data):
        df['class_label'] = df['class_label'].map(labels_mapping)
        print('Original %s samples: %d' % (data_name[idx], df.shape[0]))


    ## Data Preprocessing
    # TODO: preprocess data and save before training
    import preprocessor as p
    def clean_tweet(df):   
        """
        Clean Tweet:
        1.remove URL, Mention/Username, Hashtag sign, Emoji, Smiley, Number
        2.lowercasing, remove punctuation, remove retweet 'RT'
        """ 
        # remove URL, Mention/Username, Emoji, Smiley, Number
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)  # p.OPT.NUMBER
        df['cleaned_tweet'] = df['tweet_text'].apply(lambda x: p.clean(x))

        # lowercasing, remove punctuation, Hashtag sign, 'RT'
        df['cleaned_tweet'] = df['cleaned_tweet'].str.replace('RT ','')
        df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()
        df['cleaned_tweet'] = df['cleaned_tweet'].str.replace('[^\w\s]',' ')

        return df


    for idx, df in enumerate(data):
        df = clean_tweet(df)
        print('Cleaned %s samples: %d' % (data_name[idx], df.shape[0]))


    ## Data Spliting
    def train_split(labels, n_labeled_per_class, unlabeled_per_class=None):
        """Split the original training set into labeled training set, unlabeled training set, development set
        Arguments:
            labels {list} -- List of labeles for original training set
            n_labeled_per_class {int} -- Number of labeled data per class   

        Keyword Arguments:
            unlabeled_per_class {int or None} -- Number of unlabeled data per class (default: {None})

        Returns:
            [list] -- idx for labeled training set, unlabeled training set, development set
        """
        labels = np.array(labels)
        n_labels = len(set(labels))

        train_labeled_idxs = []
        train_unlabeled_idxs = []

        for i in range(n_labels):
            idxs = np.where(labels == i)[0]
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            if unlabeled_per_class:
                train_unlabeled_idxs.extend(idxs[n_labeled_per_class:n_labeled_per_class+unlabeled_per_class])
            else: 
                train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)

        return train_labeled_idxs, train_unlabeled_idxs


    n_labeled_per_class = n_labeled_per_class
    n_unlabeled_per_class = n_unlabeled_per_class
    labels = list(target_train_df["class_label"])
    train_labeled_idxs, train_unlabeled_idxs = train_split(labels, n_labeled_per_class, unlabeled_per_class=n_unlabeled_per_class)


    target_train_labeled_df = target_train_df.iloc[train_labeled_idxs]
    target_train_unlabeled_df = target_train_df.iloc[train_unlabeled_idxs]

    print('\nn_labeled_per_class: ', n_labeled_per_class) # labeled target samples per class
    print('n_unlabeled_per_class: ', n_unlabeled_per_class) # labeled target samples per class
    print('target_train_labeled_df samples: %d' % (target_train_labeled_df.shape[0]))
    print('target_train_unlabeled_df samples: %d' % (target_train_unlabeled_df.shape[0]))

    # check n_smaples_per_class
    print('Check n_smaples_per_class in the original training set: ', target_train_df['class_label'].value_counts().to_dict())
    print('Check n_smaples_per_class in the clearned labeled training set: ', target_train_labeled_df['class_label'].value_counts().to_dict())
    print('Check n_smaples_per_class in the clearned unlabeled training set: ', target_train_unlabeled_df['class_label'].value_counts().to_dict())


    # return 1
    # sys.exit()





    ##### Create Torch Dataset and DataLoader
    from transformers import BertTokenizer

    ## Load the BERT tokenizer.
    print('\nLoading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ## Data Augmentation
    from utils.aug import aug

    ## Create Torch Dataset and DataLoader
    from torch.utils.data import DataLoader

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, df, test=False, return_list=False):
            self.sentences = list(df['cleaned_tweet'])
            self.labels = list(df['class_label'])
            self.sample_ids = list(df['tweet_id'])
            self.test = test
            self.return_list = return_list

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence, label = self.sentences[idx], self.labels[idx]
            encodings = tokenizer.encode_plus(sentence, max_length=max_length, truncation=True, padding='max_length', return_attention_mask=True)

            item = {key: torch.tensor(val) for key, val in encodings.items()}
            if not self.test:
                item["labels"] = torch.tensor(label)
            sample_id = self.sample_ids[idx]
            item['sample_ids'] = sample_id

            if self.return_list:   
                return [item] 
            else:
                return item

    class AugDataset_L(torch.utils.data.Dataset):
        def __init__(self, df, test=False, aug_dict=None):
            self.sentences = list(df['cleaned_tweet'])
            self.labels = list(df['class_label'])
            self.sample_ids = list(df['tweet_id'])
            self.test = test
            self.aug_dict = aug_dict

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence, label = self.sentences[idx], self.labels[idx]
            
            # select only one aug sentence for labeled data
            num_aug, alpha_sr, alpha_rs = aug_dict["num_aug"], aug_dict["alpha_sr"], aug_dict["alpha_rs"]
            aug_sentence = aug(sentence, num_aug=num_aug, alpha_sr=alpha_sr, alpha_rs=alpha_rs, return_ori=False)[0]

            encodings = tokenizer.encode_plus(aug_sentence, max_length=max_length, truncation=True, padding='max_length', return_attention_mask=True)
            item = {key: torch.tensor(val) for key, val in encodings.items()}
            if not self.test:
                item["labels"] = torch.tensor(label)
            sample_id = self.sample_ids[idx]
            item['sample_ids'] = sample_id
            return item

    class AugDataset_U(torch.utils.data.Dataset):
        def __init__(self, df, test=False, aug_dict=None):
            self.sentences = list(df['cleaned_tweet'])
            self.labels = list(df['class_label'])
            self.sample_ids = list(df['tweet_id'])
            self.test = test
            self.aug_dict = aug_dict

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence, label = self.sentences[idx], self.labels[idx]
            
            num_aug, alpha_sr, alpha_rs = aug_dict["num_aug"], aug_dict["alpha_sr"], aug_dict["alpha_rs"]
            aug_sentences = aug(sentence, num_aug=num_aug, alpha_sr=alpha_sr, alpha_rs=alpha_rs, return_ori=True)
            
            items = []
            for s in aug_sentences:
                encodings = tokenizer.encode_plus(s, max_length=max_length, truncation=True, padding='max_length', return_attention_mask=True)
                item = {key: torch.tensor(val) for key, val in encodings.items()}
                if not self.test:
                    item["labels"] = torch.tensor(label)
                sample_id = self.sample_ids[idx]
                item['sample_ids'] = sample_id
                items.append(item)
            return items


    class AugDataset_U_ws(torch.utils.data.Dataset):
        def __init__(self, df, test=False, aug_dict=None, strong_aug=None, train_unlabeled_idxs=None):
            self.sentences = list(df['cleaned_tweet'])
            self.labels = list(df['class_label'])
            self.sample_ids = list(df['tweet_id'])
            self.test = test
            self.aug_dict = aug_dict
            self.strong_aug = strong_aug
            self.train_unlabeled_idxs = train_unlabeled_idxs

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sentence, label = self.sentences[idx], self.labels[idx]
            
            # - weak aug 
            num_aug, alpha_sr, alpha_rs = aug_dict["num_aug"], aug_dict["alpha_sr"], aug_dict["alpha_rs"]
            w_aug = aug(sentence, num_aug=num_aug, alpha_sr=alpha_sr, alpha_rs=alpha_rs, return_ori=True)[0]
            
            # - strong aug
            s_aug = self.strong_aug.iloc[self.train_unlabeled_idxs[idx]][random.randint(0,1)]
            # - strong_weak aug: avoid overfit on limited strong aug sentences
            # s_aug = aug(s_aug, num_aug=1, alpha_sr=alpha_sr, alpha_rs=alpha_rs, return_ori=True)
            # print('Tree s_aug: ', s_aug)
            # print('Tree ori: ', sentence)

            aug_sentences = [w_aug, s_aug]

            items = []
            for s in aug_sentences:
                encodings = tokenizer.encode_plus(s, max_length=max_length, truncation=True, padding='max_length', return_attention_mask=True)
                item = {key: torch.tensor(val) for key, val in encodings.items()}
                if not self.test:
                    item["labels"] = torch.tensor(label)
                sample_id = self.sample_ids[idx]
                item['sample_ids'] = sample_id
                items.append(item)
            return items


    # - Create Datasets
    target_dev_dataset, target_test_dataset = TextDataset(target_dev_df), TextDataset(target_test_df)
    if aug_mode:
        if strong_aug_mode:
            bt_data_path = '/data/hurricane/bt_data.csv'   # back-translated data path
            df_bt_dta = pd.read_csv(bt_data_path)
            target_train_labeled_dataset, target_train_unlabeled_dataset = AugDataset_L(target_train_labeled_df, aug_dict=aug_dict), AugDataset_U_ws(target_train_unlabeled_df, aug_dict=aug_dict, strong_aug=df_bt_dta, train_unlabeled_idxs=train_unlabeled_idxs)
        else:
            target_train_labeled_dataset, target_train_unlabeled_dataset = AugDataset_L(target_train_labeled_df, aug_dict=aug_dict), AugDataset_U(target_train_unlabeled_df, aug_dict=aug_dict)
    elif aug_mode==False and psl_mode==True:
        target_train_labeled_dataset, target_train_unlabeled_dataset = TextDataset(target_train_labeled_df), TextDataset(target_train_unlabeled_df, return_list=True)
    else:
        target_train_labeled_dataset, target_train_unlabeled_dataset = TextDataset(target_train_labeled_df), TextDataset(target_train_unlabeled_df)


    # - Create DataLoaders
    # We'll take training samples in random order. 
    target_train_labeled_loader= DataLoader(target_train_labeled_dataset, batch_size=bs, shuffle=True)
    target_train_unlabeled_loader= DataLoader(target_train_unlabeled_dataset, batch_size=bs_u, shuffle=True)
    target_dev_loader = DataLoader(target_dev_dataset, batch_size=bs, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=bs, shuffle=True)





    ##### Model & Optimizer & Learning Rate Scheduler

    ## Model
    from models.Bert4Mixup import Bert4MixupForSequenceClassification
    net = Bert4MixupForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = n_labels, # The number of output labels  
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    net.to(device)

    ## Optimizer & Learning Rate Scheduler
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch), 'W' stands for 'Weight Decay fix"
    optimizer_net = AdamW(net.parameters(),
                    lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    ## EMA Initialization
    net.train()
    if ema_mode:
        ema = EMA(net, ema_momentum)
        ema.register()




    ##### Training Loop

    ## Helper function for training
    import time
    import datetime

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def linear_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def save_model(save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        net.eval()
        # use ema model for evaluation
        if ema_mode:
            ema.apply_shadow() 
        ema_model = net.state_dict()
        # restore training mode 
        if ema_mode:
            ema.restore()
        net.train()

        torch.save({'model': net.state_dict(),
                    'optimizer': optimizer_net.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    # 'it': it,
                    'ema_model': ema_model},
                   save_filename)

        print(f"model saved: {save_filename}")    



    ## Evaluation
    # define evaluation metrics
    import torch
    from torchmetrics import F1Score
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassConfusionMatrix
    f1 = F1Score(num_classes=n_labels, average='macro')
    accuracy = Accuracy(num_classes=n_labels, average='weighted')
    accuracy_classwise = Accuracy(num_classes=n_labels, average='none')
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_labels)


    @torch.no_grad()
    def evaluation(loader, final_eval=False):
        """Evaluation"""
        # print("\nRunning Evaluation...")

        t1 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        net.eval()
        if ema_mode:
            # use ema model for evaluation
            ema.apply_shadow() # does ema model need .eval() and .train()?
            
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_f1 = 0

        # - For calculating classwise accuracy, note: need to avoid nan value when there is a class that does not have any data
        preds_all = []
        target_all = []

        # Evaluate data for one epoch
        for batch in loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                result = net(b_input_ids, token_type_ids=None,attention_mask=b_input_mask,
                            labels=b_labels,return_dict=True)

            loss = result.loss
            logits = result.logits

            # Move logits and labels to CPU
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu()
            target = b_labels.cpu()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate over all batches.
            total_eval_accuracy += accuracy(preds, target).item()
            total_eval_f1 += f1(preds, target).item()
            total_eval_loss += loss.item()

            # For calculating classwise acc
            preds_all.append(preds)
            target_all.append(target)


        # Report the final accuracy, f1, loss for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(loader)
        avg_val_f1 = total_eval_f1 / len(loader)
        avg_val_loss = total_eval_loss / len(loader)
        # Calculate classwise acc
        accuracy_classwise_ = accuracy_classwise(torch.cat(preds_all), torch.cat(target_all)).numpy().round(3)
        # print('===accuracy_classwise_', accuracy_classwise_)

        if final_eval:
            confmat_result = confusion_matrix(torch.cat(preds_all), torch.cat(target_all))
            return avg_val_accuracy, avg_val_f1, avg_val_loss, list(accuracy_classwise_), confmat_result
        else:
            return avg_val_accuracy, avg_val_f1, avg_val_loss, list(accuracy_classwise_)





    ## Kick off the training!
    import torch.nn.functional as F

    t0 = time.time() # Measure how long the training epoch takes.
    start = 0 
    best_acc = 0
    best_model_step = 0
    pslt_global, pslt_confidence = 0, 0
    cw_u_avg_prob, cw_u_avg_conf = torch.zeros(n_labels), torch.zeros(n_labels)
    psl_total_eval = 0
    psl_correct_eval = 0
    cw_psl_total, cw_psl_correct = torch.zeros(n_labels, dtype=int), torch.zeros(n_labels, dtype=int)
    cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_labels, dtype=int), torch.zeros(n_labels, dtype=int)
    cw_psl_total_accum, cw_psl_correct_accum = torch.zeros(n_labels, dtype=int), torch.zeros(n_labels, dtype=int)
    training_stats = []

    print('\n\nn_labeled_per_class: ', n_labeled_per_class) # labeled target samples per class
    print("Aug_mode: ", aug_mode)
    print("Aug_dict: ", aug_dict)
    print('Total steps: ', steps)

    criterion = nn.CrossEntropyLoss()
    data_iter_t = iter(target_train_labeled_loader)
    data_iter_t_unl = iter(target_train_unlabeled_loader)
    len_train_target = len(target_train_labeled_loader)
    len_train_target_unl = len(target_train_unlabeled_loader)

    # self-adaptive thresholding setting
    p_model = (torch.ones(n_labels) / n_labels).to(device)
    time_p = p_model.mean()

    # initial qhat, ema_mean_prob, ema_mean_conf
    qhat = (torch.ones([1, n_labels], dtype=torch.float)/n_labels).to(device)
    ema_mean_prob = (torch.ones([1, n_labels], dtype=torch.float)/n_labels).to(device)
    ema_mean_conf = (torch.ones([1, n_labels], dtype=torch.float)/n_labels).to(device)
    
    # initialize memorybank
    if use_memorybank:
        memorybank = MemoryBank(n_classes=n_labels, memory_per_class = memory_per_class, 
                                selection_strategy=selection_strategy, selection_top_threshold=selection_top_threshold, selection_sharpening_T=selection_sharpening_T)

    # initiliaze AUMRecorder
    all_sample_ids = target_train_df['tweet_id']
    labels = target_train_df["class_label"]
    AUMRecords = AUMRecorder(all_sample_ids, labels)


    net.train()
    for step in range(start, steps+1):

        # --- Check Performance on Validation set every val_interval batches. ---#
        # if step % val_interval == 0 and not step == 0:
        if step % val_interval == 0:   
            acc_test, f1_test, loss_test, acc_test_cw = evaluation(target_test_loader)
            acc_val, f1_val, loss_val, acc_val_cw = evaluation(target_dev_loader)
            acc_train, f1_train, loss_train, acc_train_cw = evaluation(target_train_labeled_loader)

            # check accuracy of pseudo-labels
            # avg_psl_acc_eval = total_psl_acc_eval/val_interval
            # total_psl_acc_eval = 0
            
            # restore training mode 
            if ema_mode:
                ema.restore()
            net.train()


            print('Step %d acc %f f1 %f loss %f acc_train %f f1_train %f loss_train %f acc_val %f f1_val %f loss_val %f psl_cor %d psl_totl %d pslt_global %f pslt_confidence %f' % 
                    (step, acc_test, f1_test, loss_test, acc_train, f1_train, loss_train, acc_val, f1_val, loss_val, psl_correct_eval, psl_total_eval, pslt_global, pslt_confidence),
                    'Tim {:}'.format(format_time(time.time() - t0)))


            # Record all statistics from this evaluation.
            training_stats.append(
                {   'step': step,
                    'acc_test': acc_test,
                    'f1_test': f1_test,
                    'loss_test': loss_test,
                    'acc_train': acc_train,
                    'f1_train': f1_train,
                    'loss_train': loss_train,
                    'acc_val': acc_val,
                    'f1_val': f1_val,
                    'loss_val': loss_val,
                    'psl_correct': psl_correct_eval,   
                    'psl_total': psl_total_eval,
                    'pslt_global': pslt_global,  
                    'pslt_confidence': pslt_confidence,
                    'cw_acc_train': acc_train_cw,
                    'cw_acc_val': acc_val_cw,
                    'cw_acc_test': acc_test_cw,
                    'cw_u_avg_prob': cw_u_avg_prob.tolist(),
                    'cw_u_avg_conf': cw_u_avg_conf.tolist(), 
                    # 'cw_psl_total': cw_psl_total.tolist(),
                    # 'cw_psl_correct': cw_psl_correct.tolist(),  
                    'cw_psl_total_eval': cw_psl_total_eval.tolist(),
                    'cw_psl_correct_eval': cw_psl_correct_eval.tolist(),
                    'cw_psl_acc_eval': (cw_psl_correct_eval/cw_psl_total_eval).tolist(),
                    'cw_psl_total_accum': cw_psl_total_accum.tolist(),
                    'cw_psl_correct_accum': cw_psl_correct_accum.tolist(),
                    'cw_psl_acc_accum': (cw_psl_correct_accum/cw_psl_total_accum).tolist(),
                })

            if psl_mode:
                # check classwise psl accuracy for the current eval
                print('Tree test: ', cw_psl_total_eval.tolist(), cw_psl_correct_eval.tolist())
                print('Tree test: ', (cw_psl_correct_eval/cw_psl_total_eval).tolist())


            # Early stopping & Save best model
            # - best criterion: acc_val (TODO: consider changes to avg of acc and F1 ?)
            if acc_val >= best_acc:
                best_acc = acc_val
                best_model_step = step
                early_stop_count = 0
                save_model('model_best.pth', output_dir_path)
            else:
                early_stop_count+=1
                if early_stop_count >= early_stop_tolerance:
                    print('Early stopping trigger at step: ', step)
                    break

            # initialize pseudo labels evaluation
            psl_total_eval, psl_correct_eval = 0, 0
            cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_labels, dtype=int), torch.zeros(n_labels, dtype=int)


        # --- Done ---#


        if step % len_train_target == 0:
            data_iter_t = iter(target_train_labeled_loader)
        if step % len_train_target_unl == 0:
            data_iter_t_unl = iter(target_train_unlabeled_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        bs_l_actual = data_t['input_ids'].size()[0]



        # prepare model input data and target: labeled and unlabeled
        if psl_mode==True:
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                if consis_reg == 'weak_sup_strong':
                # if strong_aug_mode:
                    # use w_aug to generate psl for supervising s_aug -> consistency regularizaion
                    w_aug_data = data_t_unl[0]
                    out = net(w_aug_data['input_ids'].to(device), 
                                attention_mask=w_aug_data['attention_mask'].to(device), 
                                token_type_ids=None,     
                                labels=None,
                                return_dict=True
                                )
                    logits = out.logits
                    p_avg = torch.softmax(logits, dim=1)
                    l_avg = logits

                else:
                    # average predictions -> consistency regularization
                    p_augs = [] 
                    l_augs = []
                    for aug_data in data_t_unl:
                        # aug_data['input_ids'], aug_data['attention_mask'], aug_data['labels']
                        out = net(aug_data['input_ids'].to(device), 
                                    attention_mask=aug_data['attention_mask'].to(device), 
                                    token_type_ids=None,     
                                    labels=None,
                                    return_dict=True
                                    )
                        logits = out.logits
                        p = torch.softmax(logits, dim=1)
                        p_augs.append(p)
                        l_augs.append(logits)
                    probs_u = torch.cat(p_augs)
                    p_avg = torch.mean(torch.stack(p_augs), dim=0)
                    l_avg = torch.mean(torch.stack(l_augs), dim=0)



            ## Label guessing: select high confident predictions as pseudo-labels: None, hard_t, sat
            max_probs, max_idx = torch.max(p_avg, dim=-1)
            pslt_confidence = max_probs.mean().item()
            # print('===max_probs: ', max_probs)
            # print('===p_avg: ', p_avg)


            ## -Info
            # Count cw_u_avg_prob(mean_prob), cw_u_avg_conf(mean_conf) for all unlabeled data 
            current_logit = l_avg
            mean_prob = torch.softmax(current_logit, dim=-1).mean(dim=0)
            mean_conf = (torch.zeros(n_labels) / n_labels).to(device)   # repeat
            for i in range(n_labels):
                # if there exists samples in class i, update mean confidence
                if sum(max_idx==i) > 0:
                    mean_conf[i] = max_probs[max_idx==i].mean()
                # if there does not exist samples in class i, mean confidence stays the same
                else:
                    mean_conf[i] = ema_mean_conf[0, i]
            ema_mean_prob = ema_momentum * ema_mean_prob + (1 - ema_momentum) * mean_prob
            ema_mean_conf = ema_momentum * ema_mean_conf + (1 - ema_momentum) * mean_conf
            cw_u_avg_prob, cw_u_avg_conf = ema_mean_prob[0], ema_mean_conf[0]

            # print('Tree testt: ', ema_mean_prob[0])



            # Update/Compute AUM related info to AUMRecords
            for aug_data in data_t_unl:
                sample_ids = aug_data['sample_ids'].tolist()
                logits = l_avg
                AUMRecords.update(logits, sample_ids)

            # Retrieve AUM-related selection information
            if selection_strategy is not None:
                if selection_metrics == 'margin':
                    margin_metrics = l_avg  # margin between the largest two logits/probs
                    top2 = torch.topk(margin_metrics,2).values
                    selection_scores = top2[:,0] - top2[:,1]

                elif selection_metrics == 'aum':
                    # Retrieve aum from AUMRecords
                    aug_data = data_t_unl[0] # aum info same for all weak augmentations of the sample data
                    sample_ids = aug_data['sample_ids'].tolist()
                    selection_scores = torch.tensor([AUMRecords.records[sample_id]['aum'][-1] for sample_id in sample_ids])




            if threshold_mode=='hard_t':      
                # hard threshold
                pslt_global = psl_threshold_h
                u_psl_mask = max_probs >= pslt_global

            elif threshold_mode == 'debias':
                # debias by adjusting logits (or probabilities) via mean_probs
                current_logit = l_avg
                debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
                # - update qhat
                mean_prob = torch.softmax(current_logit, dim=-1).mean(dim=0)  #repeat
                qhat = ema_momentum * qhat + (1 - ema_momentum) * mean_prob
                # - get psl_mask
                debiased_max_probs, debiased_max_idx = torch.max(debiased_prob, dim=-1)
                pslt_global = psl_threshold_h
                u_psl_mask = debiased_max_probs.ge(pslt_global)

            elif threshold_mode == 'debias2':
                # debias by adjusting logits via mean_conf
                current_logit = l_avg
                debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
                # - update qhat by confidence of each class
                mean_conf = (torch.zeros(n_labels) / n_labels).to(device)   # repeat
                for i in range(n_labels):
                    # if there exists samples in class i, update mean confidence
                    if sum(max_idx==i) > 0:
                        mean_conf[i] = max_probs[max_idx==i].mean()
                    # if there does not exist samples in class i, mean confidence stays the same
                    else:
                        mean_conf[i] = qhat[0, i]
                qhat = ema_momentum * qhat + (1 - ema_momentum) * mean_conf
                # - get psl_mask
                debiased_max_probs, debiased_max_idx = torch.max(debiased_prob, dim=-1)
                pslt_global = psl_threshold_h
                u_psl_mask = debiased_max_probs.ge(pslt_global)

                # TODO: record class-wise mean confidence
                print(qhat)
                

            elif threshold_mode=='sat':           
                # self-adaptive threshold
                time_p = time_p * ema_momentum +  max_probs.mean() * (1-ema_momentum)
                # time_p =  max_probs.mean() 
                pslt_global = time_p
                p_model = p_model * ema_momentum + torch.mean(probs_u, dim=0) * (1-ema_momentum)
                p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
                u_psl_mask = max_probs.ge(pslt_global * p_model_cutoff[max_idx])

            elif threshold_mode=='sat_global': 
                # self-adaptive threshold
                time_p = time_p * ema_momentum +  max_probs.mean() * (1-ema_momentum)
                # time_p =  max_probs.mean() 
                pslt_global = time_p
                u_psl_mask = max_probs.ge(pslt_global)

            elif threshold_mode=='linear_rampup':      
                # hard threshold
                pslt_global = psl_threshold_h * linear_rampup(step, threshold_rampup_length)
                u_psl_mask = max_probs >= pslt_global


            ## pseudo-labeling mode
            if labeling_mode=='soft':
                # soft labels
                u_label_psl = p_avg[u_psl_mask]
                u_label_psl = u_label_psl.detach()
            elif labeling_mode=='sharpening':
                # soft labels + sharpening 
                p_avg = p_avg[u_psl_mask]
                pt = p_avg**(1/sharpening_T)
                u_label_psl = pt / pt.sum(dim=1, keepdim=True)
                u_label_psl = u_label_psl.detach()              
            else:
                # hard labels
                u_label_psl = max_idx[u_psl_mask]
                u_label_psl =  F.one_hot(u_label_psl, num_classes=n_labels).to(device)


            if consis_reg == 'weak_sup_strong':
                s_aug_data = data_t_unl[1]
                u_labels_psl = u_label_psl
                u_inputs_psl = s_aug_data['input_ids'][u_psl_mask]
                u_masks_psl = s_aug_data['attention_mask'][u_psl_mask]
            else:
                u_labels_psl = torch.cat([u_label_psl for i in range(len(data_t_unl))])
                u_inputs_psl = torch.cat([_['input_ids'][u_psl_mask] for _ in data_t_unl])
                u_masks_psl = torch.cat([_['attention_mask'][u_psl_mask] for _ in data_t_unl])
                if selection_strategy is not None:
                    u_selection_scores_psl = torch.cat([selection_scores[u_psl_mask] for i in range(len(data_t_unl))])  



            ## investigate influence of incorrect psl, # psl, both
            if investigate == 'psl_acc':
                # (a) delete incorrect psl (invest. acc)
                gt_labels_u = data_t_unl[0]['labels'][u_psl_mask].to(device)
                _, u_label_psl_hard = torch.max(u_label_psl, dim=-1)
                mask_cor_psl = u_label_psl_hard == gt_labels_u

                u_labels_psl = torch.cat([u_label_psl[mask_cor_psl] for i in range(len(data_t_unl))])
                u_inputs_psl = torch.cat([_['input_ids'][u_psl_mask][mask_cor_psl] for _ in data_t_unl])
                u_masks_psl = torch.cat([_['attention_mask'][u_psl_mask][mask_cor_psl] for _ in data_t_unl])

            elif investigate == 'psl_num':
                # (b) use same # psl per classs for training per iteration, here psl are not guaranteed to be correct  (invest. num) -> (memory bank)
                use_memorybank = True

            elif investigate == 'psl_acc_num':
                # (c) replace incorrect psl with correct psl (invest. both)  -> results is quite interesting and counter-intuitive, need further pondering!!!
                u_labels_psl = torch.cat([F.one_hot(_['labels'], num_classes=n_labels)[u_psl_mask] for _ in data_t_unl]).to(device)
                # (d) use same # psl per classs for training per iteration, replace incorrect psl with correct psl (invest. both)
                use_memorybank = True




            if not use_memorybank:
                labels_l_onehot = F.one_hot(data_t['labels'], num_classes=n_labels).to(device)
                all_labels = torch.cat([labels_l_onehot, u_labels_psl])
                all_inputs = torch.cat([data_t['input_ids'], u_inputs_psl])
                all_masks = torch.cat([data_t['attention_mask'], u_masks_psl])
            else:
                # update memorybank
                if selection_strategy is not None:
                    memorybank.update_memorybank(u_inputs_psl, u_masks_psl, u_labels_psl, selection_score=u_selection_scores_psl)
                else:
                    memorybank.update_memorybank(u_inputs_psl, u_masks_psl, u_labels_psl)

                # sample memorybank
                if sampling_strategy == 'avg_probs':
                    sampled_input, sampled_mask, sampled_label = memorybank.sample_memorybank(samples_per_class, cw_mean_prob=cw_u_avg_prob, t=sampling_temperature)
                else:
                    sampled_input, sampled_mask, sampled_label = memorybank.sample_memorybank(samples_per_class)
                labels_l_onehot = F.one_hot(data_t['labels'], num_classes=n_labels)
                
                all_labels = torch.cat([labels_l_onehot, sampled_label]).to(device)
                all_inputs = torch.cat([data_t['input_ids'], sampled_input])
                all_masks = torch.cat([data_t['attention_mask'], sampled_mask])   
        

            data = all_inputs
            mask = all_masks
            target = all_labels


            ## check the total and correct number of pseudo-labels
            # psl_total = u_labels_psl.shape[0]
            # _, u_label_psl_hard = torch.max(u_labels_psl, dim=-1)
            # gt_labels_u = torch.cat([_['labels'][u_psl_mask] for _ in data_t_unl])
            # psl_correct = torch.sum(u_label_psl_hard == gt_labels_u).item()

            gt_labels_u = data_t_unl[0]['labels'][u_psl_mask].to(device)
            psl_total = torch.sum(u_psl_mask).item()
            _, u_label_psl_hard = torch.max(u_label_psl, dim=-1)
            psl_correct = torch.sum(u_label_psl_hard == gt_labels_u).item()

            psl_total_eval +=  psl_total
            psl_correct_eval += psl_correct

            # check class-wise total and correct number of pseudo-labels 
            cw_psl_total = torch.bincount(u_label_psl_hard, minlength=n_labels).to('cpu')
            cw_psl_correct = torch.bincount(u_label_psl_hard[u_label_psl_hard == gt_labels_u], minlength=n_labels).to('cpu')

            cw_psl_total_eval += cw_psl_total
            cw_psl_correct_eval += cw_psl_correct

            cw_psl_total_accum += cw_psl_total
            cw_psl_correct_accum += cw_psl_correct            


        else:
            data = data_t['input_ids']
            mask = data_t['attention_mask']
            target = data_t['labels']         


        mixup_dict = None
        if mixup_mode:
            # mixup preparation
            mixup_dict = {} 
            mixup_layers = mixup_layers
            mixup_dict["layer_num"] = np.random.choice(mixup_layers)

            lam = np.random.beta(mixup_alpha, mixup_alpha)
            mixup_dict["lam"] = max(lam, 1-lam)

            num_data_inbatch = data.size()[0]

            if mixup_method == 1:
                # mixup_method 1: mixup all
                shuffled_indices = np.random.permutation(num_data_inbatch)

            elif mixup_method == 2:
                # mixup_method 2: mixup labeled with labeled, mixup unlabeled with labeled
                shuffled_indices_l = np.random.permutation(bs_l_actual)
                shuffled_indices_u = np.random.choice(bs_l_actual, num_data_inbatch-bs_l_actual)
                shuffled_indices = np.concatenate([shuffled_indices_l,shuffled_indices_u])

            elif mixup_method == 3:
                # mixup_method 3: mixup labeled with labeled, mixup unlabeled with both labeled and unlabeled
                shuffled_indices_l = np.random.permutation(bs_l_actual)
                shuffled_indices_u = np.random.choice(num_data_inbatch, num_data_inbatch-bs_l_actual)
                shuffled_indices = np.concatenate([shuffled_indices_l,shuffled_indices_u])

            elif mixup_method == 4:
                # mixup_method 4: mixup labeled with labeled only
                shuffled_indices_l = np.random.permutation(bs_l_actual)
                shuffled_indices_u = np.arange(num_data_inbatch-bs_l_actual) + bs_l_actual
                shuffled_indices = np.concatenate([shuffled_indices_l,shuffled_indices_u])

            elif mixup_method == 5:
                # mixup_method 5: mixup unlabeled with unlabeled only
                shuffled_indices_l = np.arange(bs_l_actual)
                shuffled_indices_u = np.random.choice(num_data_inbatch-bs_l_actual, num_data_inbatch-bs_l_actual) + bs_l_actual
                shuffled_indices = np.concatenate([shuffled_indices_l,shuffled_indices_u])       

            elif mixup_method == 6:
                # mixup_method 6: mixup on seperate labled and unlabeled
                shuffled_indices_l = np.random.permutation(bs_l_actual)
                shuffled_indices_u = np.random.choice(num_data_inbatch-bs_l_actual, num_data_inbatch-bs_l_actual) + bs_l_actual
                shuffled_indices = np.concatenate([shuffled_indices_l,shuffled_indices_u])

            # undo_shuffled_indices = np.argsort(shuffled_indices) # returns indices to undo the shuffle
            mixup_dict["shuffled_indices"] = shuffled_indices

            # mixup labels
            ori_labels = target
            shuffled_labels = ori_labels[shuffled_indices]
            if psl_mode==True:
                mixed_labels = lam*(ori_labels) + (1-lam)*(shuffled_labels)
            else:
                mixed_labels = lam*F.one_hot(ori_labels, num_classes=n_labels) + (1-lam)*F.one_hot(shuffled_labels, num_classes=n_labels)
            target = mixed_labels

        # ToDo: add compatability for the attention_mask after mixup 


        # forward pass
        out_1 = net(data.to(device), 
                    token_type_ids=None, 
                    attention_mask=mask.to(device), 
                    labels=None,
                    return_dict=True,
                    mixup_dict=mixup_dict
                    )


        # compute loss
        loss_1 = 0
        if psl_mode:
            if mixup_mode:
                mixed_logits = out_1.logits
                logits_x = mixed_logits[:bs_l_actual]
                labels_x = mixed_labels[:bs_l_actual]
                logits_u = mixed_logits[bs_l_actual:]
                labels_u = mixed_labels[bs_l_actual:]
            else:
                all_logits = out_1.logits
                logits_x = all_logits[:bs_l_actual]
                labels_x = all_labels[:bs_l_actual]
                logits_u = all_logits[bs_l_actual:]
                labels_u = all_labels[bs_l_actual:] 

            if marginal_loss == True:
                # adaptive marginal loss
                logits_u = logits_u + tau*torch.log(qhat)
                # need to use cross entropy loss?

            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * labels_x, dim=1))

            Lu = 0
            if logits_u.shape[0] > 0:
                if u_loss == 'L2':
                    probs_u = torch.softmax(logits_u, dim=1)
                    Lu = torch.mean((probs_u - labels_u)**2)
                elif u_loss == 'cross_entropy':
                    Lu = -torch.mean(torch.sum(F.log_softmax(logits_u, dim=1) * labels_u, dim=1))

            # invest why Lu mean loss is so small, 1e-9
            # print('===probs_x: ', torch.softmax(logits_x, dim=1))

            
            weight_u_loss_step = weight_u_loss * linear_rampup(step, u_loss_rampup_length)
            loss_1 = Lx + weight_u_loss_step * Lu
            # print('Tree loss: ', Lx, Lu, loss_1)

        else:
            logits_1 = out_1.logits
            loss_1 = criterion(logits_1, target.to(device))


        # check avg confidence for labeled samples
        if not psl_mode:
            logits_1 = out_1.logits
            probs_x = logits_1.softmax(dim=1)
            max_probs, max_idx = torch.max(probs_x, dim=-1)
            pslt_confidence = max_probs.mean().item()
            # print('==max_probs', max_probs[:10])


        # backward pass
        # Always clear any previously calculated gradients before performing backward pass
        net.zero_grad()
        total_loss = loss_1
        total_loss.backward(retain_graph=True)
        optimizer_net.step()
        if ema_mode:
            ema.update()

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t0)))

    # # save AUMRecords for analysis
    # AUMRecords.save(output_dir_path)


    # retrieve the saved best model and evaluate on test set
    def load_model(load_path):
        checkpoint = torch.load(load_path)

        net.load_state_dict(checkpoint['model'])
        optimizer_net.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # it = checkpoint['it']
        if ema_mode:
            ema_model = deepcopy(net)
            ema_model.load_state_dict(checkpoint['ema_model'])
            ema.load(ema_model)
        print('model loaded from path: ', load_path)

    load_path = os.path.join(output_dir_path, 'model_best.pth')
    load_model(load_path)
    acc_test, f1_test, loss_test, acc_test_cw, confmat_test = evaluation(target_test_loader, final_eval=True)

    # TODO: save confmat and visualize confmat



    ##### Results Summary and Visualization
    ## Quantatitive Results
    pd.set_option('precision', 4)
    df_stats= pd.DataFrame(training_stats)
    df_stats = df_stats.set_index('step')

    # save training statistics
    training_stats_path = log_dir_path + 'training_statistics.csv'   
    df_stats.to_csv(training_stats_path)     
    print('Save training statistics in: ', training_stats_path)


    ## Previous module for retireve best model result, can remove if identical with the updated code result
    # check the test performance at the step that has the lowest loss/highest acc on validation set
    # best_row_idx = df_stats['loss_val'].argmin()
    best_row_idx = df_stats['acc_val'].argmax()
    best_step = df_stats.index[best_row_idx]
    best_test_acc = df_stats['acc_test'].iloc[best_row_idx]
    best_test_f1 = df_stats['f1_test'].iloc[best_row_idx]
    print('\nBest_step2: ', best_step, '\nbest_test_acc: ', best_test_acc, '\nbest_test_f1: ', best_test_f1)
    best_data2 = {'record_time': cur_time,
                'best_step': best_step, 'test_acc':best_test_acc, 'test_f1': best_test_f1,     
                }


    # save best record in both the training statisitcs and also a summary file
    best_data = {'record_time': cur_time,
                'best_step': best_model_step, 'test_acc':acc_test, 'test_f1': f1_test,     
                }
    print('\nBest_step: ', best_model_step, '\nbest_test_acc: ', acc_test, '\nbest_test_f1: ', f1_test)
    

    best_data.update(params) # record tuned hyper-params
    best_data2.update(params) # record tuned hyper-params

    best_df = pd.DataFrame([best_data])   
    best_df2 = pd.DataFrame([best_data2])       

    best_csv_path = log_root + 'summary.csv'
    if not os.path.exists(best_csv_path):
        best_df.to_csv(best_csv_path, mode='a', index=False, header=True)
    else:
        best_df.to_csv(best_csv_path, mode='a', index=False, header=False)

    best_df.to_csv(training_stats_path, mode='a', index=False, header=True)
    best_df2.to_csv(training_stats_path, mode='a', index=False, header=True)
    print('Save best record in: ', best_csv_path)
        


    ## Visualization - Plot Training Curves
    import matplotlib.pyplot as plt
    import seaborn as sns

    # select data range and types to plot
    df_stats_1 = df_stats
    plot_types = ['f1', 'acc', 'loss', 'psl', 'pslt']

    for plot_type in plot_types:
        plt.figure()
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        # plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        for idx, key in enumerate(df_stats.keys().tolist()):
            if key.split('_')[0] == plot_type:
                plt.plot(df_stats_1[key], '--', label=key)
        # Label the plot.
        plt.xlabel("iteration")
        plt.ylabel("peformance")
        plt.legend()
        # plt.show()
        plt.savefig(log_dir_path+plot_type+'.png', bbox_inches='tight')

    return best_data









###### multiRun 

def multiRun(log_home=None, **params):
    import statistics
    import pandas as pd
    import os
    import random
    import time

    num_runs = 3 # 3, 1
    seeds_list = [42, 0, 1]
    unit_test_mode = False

    # create a folder for this multiRun
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    if log_home is None:
        log_root = os.getcwd() + '/log/'
    else:
        log_root = log_home + '/log/'
    log_dir_path_muliRun = log_root + cur_time + '/'
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(log_dir_path_muliRun):
        os.makedirs(log_dir_path_muliRun)


    results = []
    test_accs = []
    test_f1s = []
    for i in range(num_runs):
        if not unit_test_mode:
            result = oneRun(log_dir_path_multiRun=log_dir_path_muliRun, **params, seed=seeds_list[i])
        else:
            result = {'test_acc': random.random(), 'test_f1': random.random()}
        results.append(result)
        
        test_accs.append(result['test_acc'])
        test_f1s.append(result['test_f1'])


    st_dev_acc = round(statistics.pstdev(test_accs), 3)
    mean_acc = round(statistics.mean(test_accs), 3)
    st_dev_f1 = round(statistics.pstdev(test_f1s), 3)
    mean_f1 = round(statistics.mean(test_f1s), 3)

    final = {'record_time': cur_time,
            'Mean_std_acc': '%.1f ± %.1f' % (100*mean_acc, 100*st_dev_acc),
            'Mean_std_f1': '%.1f ± %.1f' % (100*mean_f1, 100*st_dev_f1)}

    final.update(params)

    df = pd.DataFrame([final])

    csv_path = log_root + 'summary2.csv'
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', index=False, header=True)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=True)

    print('\nSave best record in: ', csv_path)




# Unit Test

# multiRun()
# oneRun(weight_u_loss=1, psl_mode=True)
# multiRun(device_idx=0)

# multiRun(psl_mode=True)

# lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill {}
# user: ps aux | grep python





