import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms

# get data dictionary for femnist dataset
def get_data_dict_femnist(json_path, min_sample = 64, image_size = 28):
    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # return value
    data_dict = {}

    for user, num_sample in zip(data['users'], data['num_samples']):
        # discard a user if it has too few samples
        if num_sample < min_sample:
            continue

        xs = []
        for x in data['user_data'][user]['x']:
            x = torch.as_tensor(x).reshape(1, image_size, image_size)
            xs.append(x)
        xs = torch.stack(xs)
        ys = torch.as_tensor(data['user_data'][user]['y']).long()
        
        data_dict[user] = {'x' : xs, 'y' : ys}
     
    return data_dict

# get data dictionary for celeba dataset
def get_data_dict_celeba(json_path, image_path, min_sample = 8, image_size = 84):
    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    if not os.path.exists(image_path):
        raise Exception("folder doesnt exist:", image_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # transformer
    t = transforms.ToTensor()

    # return value
    data_dict = {}

    for user, num_sample in zip(data['users'], data['num_samples']):
        # discard a user if it has too few samples
        if num_sample < min_sample:
            continue

        xs = []
        for x in data['user_data'][user]['x']:
            x = Image.open(image_path + x)
            x = x.resize((image_size, image_size)).convert('RGB')
            x = t(x)
            xs.append(x)
        xs = torch.stack(xs)
        ys = torch.as_tensor(data['user_data'][user]['y']).long()
        
        data_dict[user] = {'x' : xs, 'y' : ys}
     
    return data_dict

# get data dictionary for shakespeare dataset
def get_data_dict_shakespeare(json_path, min_sample = 64, seq_length = 80, num_class = 80):
    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # return value
    data_dict = {}

    # all 80 chars
    all_chars_sorted = ''' !"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}'''
    # all 75 chars (not 80 chars, because some chars are missing in train/test/both datasets)
    # all_chars_sorted   = ''' !"&'(),-.12345678:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz'''
    assert(len(all_chars_sorted) == num_class)
    
    for user, num_sample in zip(data['users'], data['num_samples']):
        # discard a user if it has too few samples
        if num_sample < min_sample:
            continue

        xs, ys = [], []
        for x, y in zip(data['user_data'][user]['x'], data['user_data'][user]['y']):
            assert(len(x) == seq_length)
            
            y = all_chars_sorted.find(y)
            if y == -1: # cannot find character
                raise Exception('wrong character:', y)
            ys.append(y)

            seq = torch.as_tensor([all_chars_sorted.find(c) for c in x])
            xs.append(seq)

        xs = torch.stack(xs)
        ys = torch.as_tensor(ys).long()
        
        data_dict[user] = {'x' : xs, 'y' : ys}
     
    return data_dict

# get data dictionary for covid19 dataset
def get_data_dict_covid19(dir_path, min_sample = 64, image_size = 224):
    if not os.path.exists(dir_path):
        raise Exception("folder doesnt exist:", dir_path)
    
    # return value
    data_dict = {}

    # transformer
    # t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t = transforms.ToTensor()
    
    # positive
    pos_path = dir_path + 'positive/'
    for image_path in os.listdir(pos_path):
        user = image_path[:6]
        if user not in data_dict:
            data_dict[user] = {'x' : [], 'y' : []}
        
        x = Image.open(pos_path + image_path)
        x = x.resize((image_size, image_size)) # .convert('RGB')
        x = t(x).reshape(1, image_size, image_size)
        data_dict[user]['x'].append(x)
        data_dict[user]['y'].append(1)

    # negative
    neg_path = dir_path + 'negative/'
    for image_path in os.listdir(neg_path):
        user = image_path[:6]
        if user not in data_dict:
            data_dict[user] = {'x' : [], 'y' : []}
        
        x = Image.open(neg_path + image_path)
        x = x.resize((image_size, image_size))
        x = t(x)
        data_dict[user]['x'].append(x)
        data_dict[user]['y'].append(0)

    to_remove = []
    for user in data_dict.keys():
        if len(data_dict[user]['y']) < min_sample:
            to_remove.append(user)
        data_dict[user]['x'] = torch.stack(data_dict[user]['x'])
        data_dict[user]['y'] = torch.as_tensor(data_dict[user]['y']).long()

    # remove users who have too few samples
    for user in to_remove:
        data_dict.pop(user)

    return data_dict

# dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        
    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]

        return x, y