# extract data of each client from global json file and store it in a standalone new file (i.e. one json file per cilent)
import json 
import os

def data_split(json_path, save_path):

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
        
    with open(json_path, 'r') as f:
        all_data = json.load(f)
     
    with open(save_path + 'user_overview.json', 'w') as f:
        json.dump({'user_ids' : all_data['users'], 'num_samples' : all_data['num_samples']}, f)

    for user, num_sample in zip(all_data['users'], all_data['num_samples']):
        os.mkdir(save_path + user)
        for i in range(num_sample):
            with open(save_path + user + '/' + str(i) + '.json', 'w') as f:
                json.dump({'x' : all_data['user_data'][user]['x'][i], 'y' : all_data['user_data'][user]['y'][i]}, f)

if __name__ == '__main__':
    data_split('../femnist/train/all_data_0_niid_0_keep_0_train_9.json', '../femnist/train/')
    data_split('../femnist/test/all_data_0_niid_0_keep_0_test_9.json', '../femnist/test/')