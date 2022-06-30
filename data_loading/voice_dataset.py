import time

import numpy as np
import pandas as pd
import torch.utils.data

import options

opt = options.voice_option
WIN_LEN = opt['seq_len']


class KSHOTTensorDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes, features, classes, domains):
        assert (features.shape[0] == classes.shape[0] == domains.shape[0])

        self.num_classes = num_classes
        self.features_per_class = []
        self.classes_per_class = []
        self.domains_per_class = []

        for class_idx in range(self.num_classes):
            indices = np.where(classes == class_idx)
            self.features_per_class.append(np.random.permutation(features[indices]))
            self.classes_per_class.append(np.random.permutation(classes[indices]))
            self.domains_per_class.append(np.random.permutation(domains[indices]))

        # Get the data amount for each label, it is the minimum number among all labels for each domain
        self.data_num = min(
            [len(feature_per_class) for feature_per_class in self.features_per_class])

        for class_idx in range(self.num_classes):
            self.features_per_class[class_idx] = torch.from_numpy(self.features_per_class[class_idx]
                                                                  [:self.data_num]).float()
            self.classes_per_class[class_idx] = torch.from_numpy(self.classes_per_class[class_idx][:self.data_num])
            self.domains_per_class[class_idx] = torch.from_numpy(self.domains_per_class[class_idx][:self.data_num])

    def __getitem__(self, index):
        features = torch.FloatTensor(self.num_classes, *(
            self.features_per_class[0][0].shape))  # make FloatTensor with shape num_classes x F-dim1 x F-dim2...
        classes = torch.LongTensor(self.num_classes)
        domains = torch.LongTensor(self.num_classes)

        rand_indices = [i for i in range(self.num_classes)]
        np.random.shuffle(rand_indices)

        for i in range(self.num_classes):
            features[i] = self.features_per_class[rand_indices[i]][index]
            classes[i] = self.classes_per_class[rand_indices[i]][index]
            domains[i] = self.domains_per_class[rand_indices[i]][index]

        return features, classes, domains

    def __len__(self):
        return self.data_num


class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, file='', transform=None, domain=None, complementary=True, max_source=np.inf, num_bin=np.inf):

        st = time.time()

        self.df = pd.read_csv(file)
        self.domain = domain
        self.complementary = complementary
        self.transform = transform
        self.max_source = max_source
        self.num_bin = num_bin

        self.num_domains = []
        self.features_data = []
        self.labels_data = []
        self.domains_data = []
        self.datasets = []
        self.dataset = []
        self.kshot_datasets = []

        if self.complementary:
            self.df = self.df[self.df['domain'] != self.domain]  # Data instances that are not target domain
            print('Number of data loaded: {:d}'.format(len(self.df)))
        else:
            self.df = self.df[self.df['domain'] == self.domain]  # Data instances that are target domain
            print('Number of data loaded: {:d}'.format(len(self.df)))

        spt = time.time()

        self.preprocessing()
        print('Loading data done with rows: {:d}\tPreprocessing Time: {:f}\tTotal Time: {:f}'.format(len(self.df.index),
                                                                                                     time.time() - spt,
                                                                                                     time.time() - st))

    def preprocessing(self):
        original_domains = opt['domains']
        if self.complementary:
            domains = set(opt['domains']) - {self.domain}  # All domains except args.tgt, for source dataset
        else:
            domains = {self.domain}  # One domain is args.tgt, for target dataset
        domains = list(domains)
        domains.sort()

        valid_domains = []
        for idx in range(len(self.df) // WIN_LEN):
            domain_data = self.df.iloc[idx*WIN_LEN, 0]
            label_data = self.df.iloc[idx*WIN_LEN, 1]
            feature_data = self.df.iloc[idx*WIN_LEN:(idx+1)*WIN_LEN, 3:].values
            feature_data = feature_data.T

            # Verify if data is valid by checking if the domain exist in proposed domain set
            for i in range(len(domains)):
                if domains[i] == domain_data and domains[i] not in valid_domains:
                    valid_domains.append(domains[i])
                    break
                else:
                    continue

            # Convert domain string to domain number
            if domain_data in valid_domains:
                domain_number = original_domains.index(domain_data)
            else:
                print("Invalid domain, stopped")
                continue

            # Add data in window size to list
            self.domains_data.append(domain_number)
            self.labels_data.append(self.class_to_number(label_data))
            self.features_data.append(feature_data)

        self.features_data = np.array(self.features_data, dtype=np.float)
        self.labels_data = np.array(self.labels_data)
        self.domains_data = np.array(self.domains_data)

        valid_domains.sort()
        print("Valid domains: " + str(valid_domains))

        # Randomize domain order
        self.num_domains = len(valid_domains) if len(valid_domains) < self.max_source else self.max_source
        domains_data_list = list(set(self.domains_data))
        np.random.shuffle(domains_data_list)
        domains_data_list = domains_data_list[:self.num_domains]

        # Create Tensor for each domain
        for domain_idx in domains_data_list:
            # Indices for each domain
            indices = np.where(self.domains_data == domain_idx)[0]

            dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.features_data[indices]).float(),
                                                     torch.from_numpy(self.labels_data[indices]),
                                                     torch.from_numpy(self.domains_data[indices]))
            self.datasets.append(dataset)

            # Group by class for each domain
            kshot_dataset = KSHOTTensorDataset(len(np.unique(self.labels_data)),
                                               self.features_data[indices],
                                               self.labels_data[indices],
                                               self.domains_data[indices])
            self.kshot_datasets.append(kshot_dataset)

        # Concatenate datasets which are seperated by domain
        self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    @staticmethod
    def class_to_number(class_label):
        dic = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
        }
        return dic[class_label]

    def get_num_domains(self):
        return self.num_domains

    def get_datasets_per_domain(self):
        # The dataset is seperated by domains and each domain is seperated by labels.
        return self.kshot_datasets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.dataset[idx]
