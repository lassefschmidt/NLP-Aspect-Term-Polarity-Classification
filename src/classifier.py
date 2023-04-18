from typing import List

import torch

import numpy as np

import models 

import pandas as pd

from sklearn.metrics import accuracy_score


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self):
        self.config = {
          # basic infos
          "max_epochs": 1,
          "batch_size": 32,
          
          # data preprocessing
          "input_enrichment": "question_sentence_target",
          
          # pre-trained language model (transformer)
          # "bert-base-cased", "cardiffnlp/twitter-roberta-base-sentiment-latest", "roberta-base"
          "plm_name": "roberta-base",
          "plm_freeze": False,

          # classifier (linear layers)
          "cls_dropout_st":     0,
          "cls_channels":       [3],
          "cls_activation":     "ReLU",
          "cls_dropout_hidden": 0.2,
          "cls_depth":          2,
          "cls_width":          384,
          
          # optimizer
          "lr": 5e-5,
          "wd": 1e-2,

          # scheduler
          "lr_s": "linear",
          "warmup": 0,
          
          # loss function
          "criterion": "BCE"
          }
        
        self.final_model_path = "best_model.pt"


    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        # Preprocess and Initialize objects for training 
        (max_epochs, trainloader, devloader,
        model, optimizer, lr_scheduler,
        criterion) = models.init_training(self.config, train_filename, dev_filename, device)

        # Helper function for training
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
        
        # Helper function for validation
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

        # train and evaluate the model
        trn_losses = []
        dev_losses = []
        trn_accs = []
        dev_accs = []
        max_dev_acc = 0

        for epoch in range(1, max_epochs + 1):
            
            ##TRAINING##
            trn_acc, trn_loss = train_epoch(trainloader, model, optimizer,
                                            lr_scheduler, criterion, device)
            
            ##TESTING##
            dev_acc, dev_loss = val_epoch(devloader, model, criterion, device)
            
            ##REPORT##
            print(f"Epoch [{epoch}/{max_epochs}] -> Trn Loss: {round(trn_loss, 2)}, Dev Loss: {round(dev_loss, 4)}, \
                Trn Acc: {round(trn_acc, 2)}, Dev Acc: {round(dev_acc, 4)}")
            
            ##LOGGING##
            trn_losses.append(trn_loss)
            dev_losses.append(dev_loss)
            trn_accs.append(trn_acc)
            dev_accs.append(dev_acc)

            ##SAVE BEST MODEL##
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                torch.save(model.state_dict(), self.final_model_path)
        
        return trn_losses, dev_losses, trn_accs, dev_accs


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Load the final_model
        model = models.TransformerSentimentClassifier(self.config)
        model.load_state_dict(torch.load(self.final_model_path))
        model.eval()
        lbls = torch.Tensor([])
        preds = torch.Tensor([])
        
        # Process datafile (we have to do this again because we need to be able intake datafile of dev and test)
        def tokenize_func(hf_dataset):
          return model.lm_tokenizer(hf_dataset["inputs"], truncation = True)
        hf_dev = models.get_datasets(self.onfig, data_filename)
        hf_dev_tok   = hf_dev.map(tokenize_func,   batched = True)
        hf_dev_tok   = hf_dev_tok.remove_columns(["inputs"])
        data_collator = models.DataCollatorWithPadding(tokenizer = model.lm_tokenizer, padding = True, return_tensors = "pt")
        devloader   = models.get_dataloader(hf_dev_tok,   self.config["batch_size"], shuffle = True, data_collator = data_collator)

        with torch.no_grad():
            for batch in devloader:
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

        pred_labels = ['negative' if l == 0 else 'neutral' if l == 1 else 'positive' for l in preds]

        return pred_labels





