import torch
import copy
from data_preprocessing import Dataset
from models import model_train, model_train_FedProx, model_train_MOON, model_eval

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# client is an object, which contains its own dataset, dataloader, local model copy (for simplification, only local parameters) and training loop fucntion
class Client(object):
    def __init__(self, args, client_name, client_data_dict):
        super(Client, self).__init__()
        self.client_name   = client_name
        self.num_sample    = len(client_data_dict['y'])
        self.client_epoch  = args.client_epoch
        self.client_bs     = args.client_bs
        
        # for FedProx
        self.FedProx = args.FedProx

        # for MOON
        self.MOON = args.MOON
        
        # datasets and data loaders
        self.dataset = Dataset(client_data_dict['x'], client_data_dict['y'])
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.client_bs, shuffle = not self.MOON)
            
    def local_train(self, client_model, global_model, previous_feature, current_global_epoch, num_global_epoch):
        client_model.to(device)

        client_features = []
        if self.MOON:
            for current_client_epoch in range(self.client_epoch):
                # client model train
                if (previous_feature != None) and (client_features == []):
                    client_features_tensor = previous_feature
                elif (previous_feature == None) and (client_features == []):
                    client_features_tensor = None
                elif client_features != []:
                    client_features_tensor = torch.zeros((len(client_features), client_features[0].shape[0], client_features[0].shape[1]))
                    for idx, prev in enumerate(client_features):
                        client_features_tensor[idx] = copy.deepcopy(prev.detach())
                    client_features_tensor = client_features_tensor.cuda()
                    
                client_feat = model_train_MOON(client_model, global_model, self.data_loader, client_features_tensor, current_global_epoch, num_global_epoch)
                client_features.append(client_feat)
        elif self.FedProx:
            model_train_FedProx(client_model, self.data_loader, self.client_epoch, global_model)
        else:
            model_train(client_model, self.data_loader, self.client_epoch)

        client_model.to('cpu')
        last_client_features = []
        if self.MOON:
            last_client_features = client_features[-1]
        
        return last_client_features
        
    def local_eval(self, client_model):
        client_model.to(device)
        labels, preds = model_eval(client_model, self.data_loader, {}, '', True)
        client_model.to('cpu')
        return labels, preds

def get_clients(args, data_dict):
    clients = []
    for client_name, client_data_dict in data_dict.items():
        client = Client(args, client_name, client_data_dict)
        clients.append(client)
    return clients
