import torch
import torch.nn.functional as F
import copy
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'

# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef}

# resnet50
resnet50_list = []

# for FedProx
FedProx_mu = 0.01

# for MOON
MOON_temperature = 0.5
MOON_mu = 1.0

class CNN_femnist(torch.nn.Module):
    def __init__(self, args, image_size = 28, num_class = 62):
        super(CNN_femnist, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 64 * int(image_size / 4) * int(image_size / 4), out_features = 2048),
            torch.nn.ReLU(),
        )

        self.logits = torch.nn.Linear(in_features = 2048, out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x):
        h = self.encoder(x)
        x = self.logits(h)
        return x, h
    
class CNN_celeba(torch.nn.Module):
    def __init__(self, args, image_size = 84, num_class = 2):
        super(CNN_celeba, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same'),
            torch.nn.BatchNorm2d(num_features = 32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
        )

        self.logits = torch.nn.Linear(in_features = 32 * int(image_size / 16) * int(image_size / 16), out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = True

    def forward(self, x):
        h = self.encoder(x)
        x = self.logits(h)
        return x, h
    
class LSTM_shakespeare(torch.nn.Module):
    def __init__(self, args, embedding_dim = 8, hidden_size  = 256, num_class = 80):
        super(LSTM_shakespeare, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings = num_class, embedding_dim = embedding_dim)
        self.encoder   = torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, batch_first = True)
        self.logits    = torch.nn.Linear(in_features = hidden_size, out_features = num_class)

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = False

    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.encoder(x)
        h = x[:, -1, :]
        x = self.logits(h)
        return x, h
    
class Resnet50_covid19(torch.nn.Module):
    def __init__(self, args, num_class = 2, freeze = True):
        super(Resnet50_covid19, self).__init__()

        if len(resnet50_list) == 0:
            resnet50 = torch.hub.load('pytorch/vision:v0.15.2', 'resnet50', weights = 'ResNet50_Weights.DEFAULT')
            resnet50.fc = torch.nn.Identity() # remove last FC layer
            for p in resnet50.parameters(): # freeze resnet50
                p.requires_grad = False
            resnet50.to(device)
            resnet50_list.append(resnet50)
        assert(len(resnet50_list) == 1)
        
        self.logits = torch.nn.Linear(in_features = 2048, out_features = num_class)

        self.t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.optim = args.client_optim
        self.lr    = args.client_lr
        self.reuse_optim = args.reuse_optim
        self.optim_state = None

        self.binary = True

    def forward(self, x):
        resnet50 = resnet50_list[0]
        
        # x is of shape (batch_size, 1, 224, 224)
        x = x.expand(-1, 3, -1, -1)
        x = self.t(x)
        h = resnet50(x)
        x = self.logits(h)
        return x, h

def model_train(model, data_loader, num_client_epoch):
    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            p, _ = model(x)
            loss = F.cross_entropy(p, y)
            loss.backward()
        
            optim.step()
            optim.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())

# for FedProx
def model_train_FedProx(model, data_loader, num_client_epoch, global_model):
    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    for current_client_epoch in range(num_client_epoch):
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            p, _ = model(x)
            loss = F.cross_entropy(p, y)
            
            # FedProx
            for p1, p2 in zip(model.parameters(), global_model.parameters()):
                ploss = (p1 - p2.detach()) ** 2
                loss += FedProx_mu * ploss.sum()

            loss.backward()
            optim.step()
            optim.zero_grad()

            # stability
            for p in model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())

# for MOON
def model_train_MOON(model, global_model, data_loader, previous_features, current_global_epoch, num_global_epoch):
    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].train()

    model.train()
    optim = model.optim(model.parameters(), lr = model.lr)

    # load previous optimizer state
    if model.reuse_optim and model.optim_state is not None:
        optim.load_state_dict(model.optim_state)
    
    cos = torch.nn.CosineSimilarity(dim=-1)

    for batch_id, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        
        # feed into model
        p, features = model(x)
        if batch_id == 0:
            total_features = torch.empty((0, features.size()[1]), dtype=torch.float32).to(device)
        total_features = torch.cat([total_features, features], dim=0)
        loss = F.cross_entropy(p, y)

        # for MOON
        features_tsne = np.squeeze(features)
        _, global_feat = global_model(x)
        global_feat_copy = copy.copy(global_feat)
        posi = cos(features_tsne, global_feat_copy.to(device))
        logits = posi.reshape(-1,1)
        if previous_features == None or torch.count_nonzero(previous_features) == 0:
            previous_features = torch.zeros_like(features_tsne)
            nega = cos(features_tsne, previous_features)
            logits = torch.cat((posi.reshape(-1,1), nega.reshape(-1,1)), dim=1)
        if previous_features.dim() == 3:
            for prev_feat in previous_features[:, batch_id*y.size()[0]:(batch_id+1)*y.size()[0], :]:
                prev_nega = cos(features_tsne,prev_feat)
                logits = torch.cat((logits, prev_nega.reshape(-1,1)), dim=1)
        
        logits /= MOON_temperature # 0.5
        cos_labels = torch.zeros(logits.size(0)).long().to(device)
        loss_contrastive = F.cross_entropy(logits, cos_labels)
        if torch.count_nonzero(previous_features) != 0:
            loss += MOON_mu * loss_contrastive

        loss.backward()
        optim.step()
        optim.zero_grad()
        
        # stability
        for p in model.parameters():
            torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

    # save optimizer state
    if model.reuse_optim:
        model.optim_state = copy.deepcopy(optim.state_dict())
    
    return total_features

def model_eval(model, data_loader, wandb_log, metric_prefix = 'prefix/', returns = False):
    # for covid19 with resnet50
    if isinstance(model, Resnet50_covid19):
        resnet50_list[0].eval()

    model.eval()
    epoch_labels   = []
    epoch_predicts = []
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                
                p, _ = model(x)
                
                epoch_labels  .append(y)
                epoch_predicts.append(p)
   
    epoch_labels   = torch.cat(epoch_labels  ).detach().to('cpu')
    epoch_predicts = torch.cat(epoch_predicts).detach().to('cpu')
    
    if returns:
        return epoch_labels, epoch_predicts
    else:
        cal_metrics(epoch_labels, epoch_predicts, wandb_log, metric_prefix, model.binary)

def cal_metrics(labels, preds, wandb_log, metric_prefix, binary):
    # loss
    loss = F.cross_entropy(preds, labels)
    wandb_log[metric_prefix + 'loss'] = loss
        
    if not binary: # multi-class    
        # get probability
        preds = torch.softmax(preds, axis = 1)

        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds, multi_class = 'ovr')
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1

        # get class prediction
        preds = preds.argmax(axis = 1)
        
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric
    
    else: # binary
        # get probability
        preds = torch.softmax(preds, axis = 1)[:, 1]
        
        # ROC AUC
        try:
            wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds)
        except Exception:
            wandb_log[metric_prefix + 'auc'] = -1
        
        # get class prediction
        preds = preds.round()
        
        # accuracy and mcc
        for metric_name, metric_func in metrics_no_avg.items():
            metric = metric_func(labels, preds)
            wandb_log[metric_prefix + metric_name] = metric

        # precision, recall, f1 score
        for metric_name, metric_func in metrics_with_avg.items():
            metric = metric_func(labels, preds, average = avg, zero_division = 0)
            wandb_log[metric_prefix + metric_name] = metric