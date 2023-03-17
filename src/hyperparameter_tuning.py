# import own scripts
import src.preprocess_data as prepData

# basic stuff
import numpy as np

# data handling
from datasets import Dataset
import pandas as pd

# modeling
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding, get_scheduler

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


class TransformerSentimentClassifier(torch.nn.Module):

    def __init__(self, config):
        super(TransformerSentimentClassifier, self).__init__()

        # initialise language model
        self.lm_config    = AutoConfig.from_pretrained(config["plm_name"])
        self.lm_tokenizer = AutoTokenizer.from_pretrained(config["plm_name"])
        self.lm           = AutoModel.from_pretrained(config["plm_name"], output_attentions=False)

        # get input and output dimensions of classifier
        self.emb_dim     = self.lm_config.hidden_size
        self.output_size = 3

        # initialise classifier
        self.classifier = nn.Sequential()

        ## get activation function
        if config["cls_activation"] == "ReLU":
            activation = nn.ReLU()
        elif config["cls_activation"] == "Sigmoid":
            activation = nn.Sigmoid()
        elif config["cls_activation"] == "Tanh":
            activation = nn.Tanh()
        
        ## get dropout layer (if no dropout, just set parameter to 0)
        dropout = nn.Dropout(config["cls_dropout"])

        ## initialise classifier head
        input_size = self.emb_dim
        for idx, channel in enumerate(config["cls_channels"]):
            self.classifier.append(nn.Linear(input_size, channel))
            input_size = channel # update input size for next layer
            if idx < len(config["cls_channels"]) - 1: # append activation + dropout only in hidden layers
                self.classifier.append(activation)
                self.classifier.append(dropout)
            else: # if last layer, apply Sigmoid to make sum of predictions = 1
                self.classifier.append(nn.Sigmoid())

    def forward(self, x):
        x : torch.Tensor = self.lm(x['input_ids'], x['attention_mask']).last_hidden_state
        global_vects     = x.mean(dim=1) # THIS SHOULD BE OPTIMIZED (how to handle output vectors from language model)
        x                = self.classifier(global_vects)
        return x.squeeze(-1)


def get_datasets(config):
    # paths to data
    datadir = config["data_path"] + "\\data\\"
    trainfile =  datadir + "traindata.csv"
    devfile   =  datadir + "devdata.csv"

    # load data in pandas dataframe
    train = pd.read_csv(trainfile, sep = "\t", header = None).rename(columns = {0: "y", 1: "aspect", 2: "target_term", 3: "target_location", 4: "sentence"})
    dev   = pd.read_csv(devfile  , sep = "\t", header = None).rename(columns = {0: "y", 1: "aspect", 2: "target_term", 3: "target_location", 4: "sentence"})
    
    # preprocess data to get model inputs and labels
    train_prep = prepData.preprocess(train, enrich_inputs = config["input_enrichment"])
    dev_prep   = prepData.preprocess(dev,   enrich_inputs = config["input_enrichment"])

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

    # get criterion based on which we will compute the loss
    if config["criterion"] == "BCE":
        criterion = torch.nn.BCELoss(reduction = "mean")
    
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
    
        # zero optimizer gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(batch)
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
    
    for batch in dataloader:
        # get ground truth labels of this batch
        lbls = torch.cat((lbls, batch["labels"].argmax(dim = 1)))

        # move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # forward
        outputs = model(batch)
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
                         verbose = True, ray = False, return_obj = True):
    """
    Function that aggregates everything in one place to start model training.
    """
    
    # train and evaluate the model
    trn_losses = []
    dev_losses = []
    trn_accs = []
    dev_accs = []

    for epoch in range(1, max_epochs + 1):
        
        ##TRAINING##
        trn_acc, trn_loss = train_epoch(trainloader, model, optimizer,
                                        lr_scheduler, criterion, device)
        
        ##TESTING##
        dev_acc, dev_loss = val_epoch(devloader, model, criterion, device)
        
        ##REPORT##
        if verbose:
            print(f"Epoch [{epoch}/{max_epochs}] -> Trn Loss: {round(trn_loss, 2)}, Dev Loss: {round(dev_loss, 4)}, \
Trn Acc: {round(trn_acc, 2)}, Dev Acc: {round(dev_acc, 4)}")
        
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

    if return_obj:
        return model, trn_losses, dev_losses, trn_accs, dev_accs
    

def ray_trainable(config):
    """
    Function that wraps everything into one function to allow for raytune hyperparameter training.
    """
    # initialise objects for training
    (max_epochs, trainloader, devloader,
     model, optimizer, lr_scheduler,
     criterion, device) = init_training(config)
    
    # perform training (no return!)
    train_evaluate_model(max_epochs, trainloader, devloader, model,
                         optimizer, lr_scheduler, criterion, device,
                         verbose = False, ray = True, return_obj = False)
    

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