# import own scripts
import preprocess_data as prepData
import losses

# data handling
from datasets import Dataset
import pandas as pd

# modeling
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding, get_scheduler


class TransformerSentimentClassifier(torch.nn.Module):

    def __init__(self, config):
        super(TransformerSentimentClassifier, self).__init__()

        # initialise language model
        self.plm_name     = config["plm_name"]
        self.lm_config    = AutoConfig.from_pretrained(self.plm_name)
        self.lm_tokenizer = AutoTokenizer.from_pretrained(self.plm_name)
        self.lm = AutoModel.from_pretrained(self.plm_name)

        # initialise key params
        self.depth = config["cls_depth"]
        self.width = config["cls_width"]
        
        # get activation function
        if config["cls_activation"] == "ReLU":
            activation = nn.ReLU()
        elif config["cls_activation"] == "Sigmoid":
            activation = nn.Sigmoid()
        elif config["cls_activation"] == "Tanh":
            activation = nn.Tanh()

        # initialise classifier
        self.classifier = nn.Sequential()
        self.classifier.append(nn.Dropout(config["cls_dropout_st"]))

        ## basic settings
        self.emb_dim = self.lm_config.hidden_size
        cur_width = self.emb_dim
        self.output_width = 3
        
        ## append layers
        for i in range(self.depth):

            # last layer
            if i == self.depth - 1:
                self.classifier.append(nn.Linear(cur_width, self.output_width))
                break

            # for all hidden layers (append activation and dropout only here!)
            self.classifier.append(nn.Linear(cur_width, self.width))
            self.classifier.append(activation)
            self.classifier.append(nn.Dropout(config["cls_dropout_hidden"]))
            cur_width = self.width


    def forward(self, ids, mask, token_type_ids):
        if self.plm_name == "bert-base-cased":
            _, output_1 = self.lm(input_ids = ids, attention_mask = mask, token_type_ids = token_type_ids)
            output_2 = self.classifier(output_1)

        if self.plm_name == "roberta-base" or self.plm_name == "roberta-large":
            output_1 = self.lm(input_ids = ids, attention_mask = mask, token_type_ids = token_type_ids)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            output_2 = self.classifier(pooler)

        return output_2
    
def get_dataset(config, filename):

    # load data in pandas dataframe
    pandas_df = pd.read_csv(filename, sep = "\t", header = None).rename(columns = {0: "y", 1: "aspect", 2: "target_term", 3: "target_location", 4: "sentence"})
    
    # preprocess data to get model inputs and labels
    preprocess_df = prepData.preprocess(pandas_df, config)

    # transform into huggingface dataset
    hf_df = Dataset.from_pandas(preprocess_df)

    return hf_df

def get_dataloader(hf_dataset_tok, batch_size, shuffle, data_collator):
    return DataLoader(hf_dataset_tok, batch_size = batch_size, shuffle = shuffle, collate_fn = data_collator)

def init_training(config, train_filename, dev_filename, device):
    """
    Helper function that initialises all objects needed for training based on a config file.
    """
    # how many epochs we want to train for (at maximum)
    max_epochs = int(config["max_epochs"])

    # model initialisation
    model = TransformerSentimentClassifier(config)

    # freeze language model weights if required
    if config["plm_freeze"]:
        for param in model.lm.parameters():
            param.requires_grad = False

    # move to device
    model.to(device)

    # get data, preprocess and tokenize it
    hf_train = get_dataset(config, train_filename)
    hf_dev = get_dataset(config, dev_filename)

    def tokenize_func(hf_dataset):
        return model.lm_tokenizer(hf_dataset["inputs"], truncation = True)
    
    hf_train_tok = hf_train.map(tokenize_func, batched = True)
    hf_dev_tok   = hf_dev.map(tokenize_func,   batched = True)

    hf_train_tok = hf_train_tok.remove_columns(["inputs"])
    hf_dev_tok   = hf_dev_tok.remove_columns(["inputs"])

    # create dataloaders
    data_collator = DataCollatorWithPadding(tokenizer = model.lm_tokenizer, padding = True, return_tensors = "pt") # dynamic batch-wise padding
    trainloader = get_dataloader(hf_train_tok, config["batch_size"], shuffle = True, data_collator = data_collator)
    devloader   = get_dataloader(hf_dev_tok,   config["batch_size"], shuffle = True, data_collator = data_collator)

    # get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["wd"])

    # get learning rate scheduler
    num_warmup_steps   = config["warmup"] * len(trainloader)
    num_training_steps = (max_epochs - config["warmup"]) * len(trainloader)
    lr_scheduler = get_scheduler(name = config["lr_s"], optimizer = optimizer,
                                 num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps)
    
    # get loss weights that we will apply
    num_positives = torch.tensor([390, 58, 1055], dtype = torch.float)
    num_negatives = torch.tensor([1113, 1445, 448], dtype = torch.float)

    if config["crit_w"] == "invClassFreq": # in our problem roughly (3, 24, 0.5) for (neg, neutral, pos)
        weights = (num_negatives / num_positives).to(device)
    elif config["crit_w"] == "invSqrtClassFreq": # roughly (1.69, 5, 0.65) for (neg, neutral, pos) --> smoother than previous option, neutral class not as crazy important
        weights = (torch.sqrt(num_negatives) / torch.sqrt(num_positives)).to(device)
    else:
        weights = None
    
    # get criterion based on which we will compute the loss
    if config["crit"] == "BCE":
        criterion = nn.BCEWithLogitsLoss(pos_weight = weights)
    elif config["crit"] == "Focal":
        criterion = losses.FocalLoss()
    elif config["crit"] == "TopK":
        criterion = losses.TopKLoss(k = 50, pos_weight = weights)
    
    return (max_epochs, trainloader, devloader, model, optimizer, lr_scheduler, criterion)