# import own scripts
import src.preprocess_data as prepData

# basic stuff
import os
import numpy as np

# data handling
from datasets import Dataset
import pandas as pd

# modeling
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler

# evaluation
from sklearn.metrics import accuracy_score

# hyperparam optimization
from ray import air, tune
from ray.air import session
from ray.tune import JupyterNotebookReporter

# suppress hugginface messages
from transformers import logging
logging.set_verbosity_error()

# suppress progress bar
from datasets.utils import disable_progress_bar
disable_progress_bar()


def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # same for pytorch
    random_seed = 1 # or any of your favorite number 
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


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

        # initliase classifier
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
    

class FocalLoss(nn.Module):

    def __init__(self, alpha = 1, gamma = 2, reduction = 'mean', eps = 1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none")

        prob = torch.sigmoid(inputs)
        prob = torch.clamp(prob, min = self.eps, max = 1.0) # avoid vanishing gradients

        pt = torch.where(targets == 1, prob, 1 - prob)

        F_loss = self.alpha * ((1 - pt) ** self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        

class TopKLoss(nn.Module):

    def __init__(self, k, reduction='mean', pos_weight = None):
        super(TopKLoss, self).__init__()
        self.k = k # 0 < k <= 100 (if 100, this is simply BCE_Loss)
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none", pos_weight = self.pos_weight)
        num_losses = np.prod(BCE_loss.shape, dtype=np.int64) # with batch size 10, we would have shape (10, 3) and num_losses = 30
        TopK_loss, _ = torch.topk(BCE_loss.view((-1, )), int(num_losses * self.k / 100))
        
        if self.reduction == 'mean':
            return torch.mean(TopK_loss)
        elif self.reduction == 'sum':
            return torch.sum(TopK_loss)
        else:
            return TopK_loss


def get_datasets(config):
    # paths to data
    datadir = config["data_path"] + "\\data\\"
    trainfile =  datadir + "traindata.csv"
    devfile   =  datadir + "devdata.csv"

    # load data in pandas dataframe
    train = pd.read_csv(trainfile, sep = "\t", header = None).rename(columns = {0: "y", 1: "aspect", 2: "target_term", 3: "target_location", 4: "sentence"})
    dev   = pd.read_csv(devfile  , sep = "\t", header = None).rename(columns = {0: "y", 1: "aspect", 2: "target_term", 3: "target_location", 4: "sentence"})
    
    # preprocess data to get model inputs and labels
    train_prep = prepData.preprocess(train, config)
    dev_prep   = prepData.preprocess(dev,   config)

    # transform into huggingface dataset
    hf_train = Dataset.from_pandas(train_prep)
    hf_dev   = Dataset.from_pandas(dev_prep)

    return hf_train, hf_dev


def get_dataloader(hf_dataset_tok, batch_size, shuffle, data_collator):
    return DataLoader(hf_dataset_tok, batch_size = batch_size, shuffle = shuffle, collate_fn = data_collator)


def get_device(model):
    # where we want to run the model (so this code can run on cpu, gpu, multiple gpus depending on system)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    return device, model.to(device)


def init_training(config):
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

    # initialise device
    device, model = get_device(model)

    # get data, preprocess and tokenize it
    hf_train, hf_dev = get_datasets(config)

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
        criterion = FocalLoss()
    elif config["crit"] == "TopK":
        criterion = TopKLoss(k = 50, pos_weight = weights)
    
    return (max_epochs, trainloader, devloader, model, optimizer, lr_scheduler, criterion, device)


def train_epoch(dataloader, model, optimizer, lr_scheduler, criterion, device):
    
    model.train()
    losses = []
    lbls = torch.Tensor([])
    preds = torch.Tensor([])
    
    for batch in dataloader:
        # get ground truth labels of this batch
        lbls = torch.cat((lbls, batch["labels"].argmax(dim = 1)))

        # move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # cardiffnlp model does not have token type ids
        try:
            token_type_ids = batch["token_type_ids"]
        except KeyError:
            token_type_ids = None
    
        # zero optimizer gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(batch["input_ids"], batch["attention_mask"], token_type_ids)
        loss = criterion(outputs, batch['labels'].float())
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # logging
        losses.append(loss.item())
        preds = torch.cat((preds, outputs.argmax(dim = 1).cpu()))
        
    # compute stats
    acc = accuracy_score(lbls, preds)
    loss_mean = np.mean(losses)
    
    return acc, loss_mean


def val_epoch(dataloader, model, criterion, device):
    
    model.eval()
    losses = []
    lbls = torch.Tensor([])
    preds = torch.Tensor([])
    
    with torch.no_grad():
        for batch in dataloader:
            # get ground truth labels of this batch
            lbls = torch.cat((lbls, batch["labels"].argmax(dim = 1)))

            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # cardiffnlp model does not have token type ids
            try:
                token_type_ids = batch["token_type_ids"]
            except KeyError:
                token_type_ids = None

            # forward
            outputs = model(batch["input_ids"], batch["attention_mask"], token_type_ids)
            loss = criterion(outputs, batch['labels'].float())
            
            # logging
            losses.append(loss.item())
            preds = torch.cat((preds, outputs.argmax(dim = 1).cpu()))

    # compute stats
    acc = accuracy_score(lbls, preds)
    loss_mean = np.mean(losses)
    
    return acc, loss_mean


def train_evaluate_model(max_epochs, trainloader, devloader, model,
                         optimizer, lr_scheduler, criterion, device,
                         verbose = True, ray = False, return_obj = True, save_best_model = False):
    """
    Function that aggregates everything in one place to start model training.
    """
    
    # train and evaluate the model
    trn_losses = []
    dev_losses = []
    trn_accs = []
    dev_accs = []
    max_dev_acc = 0

    for epoch in range(1, max_epochs + 1):
        
        ##TRAINING##
        model.train()
        trn_acc, trn_loss = train_epoch(trainloader, model, optimizer,
                                        lr_scheduler, criterion, device)
        
        ##VALIDATION##
        model.eval()
        dev_acc, dev_loss = val_epoch(devloader, model, criterion, device)
        
        ##REPORT##
        if verbose:
            print(f"Epoch [{epoch}/{max_epochs}] -> Trn Loss: {round(trn_loss, 4)}, Dev Loss: {round(dev_loss, 4)}, \
Trn Acc: {round(trn_acc, 4)}, Dev Acc: {round(dev_acc, 4)}")
        
        if ray:
            cur_lr = optimizer.param_groups[0]["lr"]
            session.report({"trn_loss": trn_loss, "dev_loss": dev_loss,
                           "trn_acc": trn_acc, "dev_acc": dev_acc, "cur_lr": cur_lr})
        
        ##LOGGING##
        if return_obj:
            trn_losses.append(trn_loss)
            dev_losses.append(dev_loss)
            trn_accs.append(trn_acc)
            dev_accs.append(dev_acc)

        if save_best_model:
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                model_name = model.lm._get_name()
                path = os.path.abspath("") + "\\models\\" + model_name + "_finetuned.pt"
                torch.save(model.state_dict(), path)

    if return_obj:
        return model, trn_losses, dev_losses, trn_accs, dev_accs
    

def evaluate(dataloader, model, device):
    
    model.to(device)
    model.eval()
    lbls = torch.Tensor([])
    preds = torch.Tensor([])
    
    with torch.no_grad():
        for batch in dataloader:
            # get ground truth labels of this batch
            lbls = torch.cat((lbls, batch["labels"].argmax(dim = 1)))

            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # cardiffnlp model does not have token type ids
            try:
                token_type_ids = batch["token_type_ids"]
            except KeyError:
                token_type_ids = None

            # forward
            outputs = model(batch["input_ids"], batch["attention_mask"], token_type_ids)
            
            # logging
            preds = torch.cat((preds, outputs.argmax(dim = 1).cpu()))

    # compute stats
    acc = accuracy_score(lbls, preds)

    return acc, lbls, outputs.detach().cpu(), preds


def ray_trainable(config):
    """
    Function that wraps everything into one function to allow for raytune hyperparameter training.
    """

    # ensure reproducibility
    set_reproducible()

    # initialise objects for training
    (max_epochs, trainloader, devloader,
     model, optimizer, lr_scheduler,
     criterion, device) = init_training(config)
    
    # perform training (no return!)
    train_evaluate_model(max_epochs, trainloader, devloader, model,
                         optimizer, lr_scheduler, criterion, device,
                         verbose = False, ray = True, return_obj = False, save_best_model = False)
    

def trial_str_creator(trial):
    """
    Trial name creator for ray tune logging.
    """
    return f"{trial.trial_id}"


def run_ray_experiment(train_func, config, ray_path, num_samples, metric_columns, parameter_columns):

    reporter = JupyterNotebookReporter(
        metric_columns = metric_columns,
        parameter_columns= parameter_columns,
        max_column_length = 15,
        max_progress_rows = 20,
        max_report_frequency = 10, # refresh output table every second
        print_intermediate_tables = True
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"CPU": 16, "GPU": 1}
        ),
        tune_config = tune.TuneConfig(
            metric = "trn_loss",
            mode = "min",
            num_samples = num_samples,
            trial_name_creator = trial_str_creator,
            trial_dirname_creator = trial_str_creator,
            ),
        run_config = air.RunConfig(
            local_dir = ray_path,
            progress_reporter = reporter,
            verbose = 1),
        param_space = config
    )

    result_grid = tuner.fit()
    
    return result_grid


def open_validate_ray_experiment(experiment_path, trainable):
    # open & read experiment folder
    print(f"Loading results from {experiment_path}...")
    restored_tuner = tune.Tuner.restore(experiment_path, trainable = trainable, resume_unfinished = False)
    result_grid = restored_tuner.get_results()
    print("Done!\n")

    # Check if there have been errors
    if result_grid.errors:
        print(f"At least one of the {len(result_grid)} trials failed!")
    else:
        print(f"No errors! Number of terminated trials: {len(result_grid)}")
        
    return restored_tuner, result_grid