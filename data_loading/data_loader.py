import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

import options
from data_loading.voice_dataset import VoiceDataset


def number_to_domain(domain_number):
    dic = options.voice_option['number_to_domain']
    return dic[domain_number]


def split_data(data, valid_split, test_split, train_max_rows, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(data) * valid_split)
    test_size = math.floor(len(data) * test_split)
    train_size = len(data) - valid_size - test_size

    train_data, valid_data, test_data = torch.utils.data.random_split(data, [train_size, valid_size, test_size])

    # Limit data size according to user given parameter
    if len(train_data) > train_max_rows:
        train_data = torch.utils.data.Subset(train_data, range(train_max_rows))
    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return train_data, valid_data, test_data


# Used only when calibration enabled, augmented data is used for adaptation,
# real data is used for validating and testing
def split_val_test_data(data, valid_split, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(data) * valid_split)
    test_size = len(data) - valid_size

    valid_data, test_data = torch.utils.data.random_split(data, [valid_size, test_size])

    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return valid_data, test_data


def datasets_to_dataloader(datasets, batch_size, concat=False, num_workers=0, shuffle=True, drop_last=False):
    if concat:
        data_loader = None
        if len(datasets):
            if type(datasets) is torch.utils.data.dataset.Subset:
                datasets = [datasets]
            if sum([len(dataset) for dataset in datasets]) > 0:  # at least one dataset has data
                data_loader = DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers,
                                         drop_last=drop_last, pin_memory=False)
        else:
            print("No data in {}".format(datasets))
        return data_loader
    else:
        data_loaders = []
        for dataset in datasets:
            if len(dataset) == 0:
                continue
            else:
                data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, drop_last=drop_last, pin_memory=False))
        return data_loaders


def domain_data_loader(args, domains, file_path, batch_size, train_max_rows=np.inf, valid_max_rows=np.inf,
                       test_max_rows=np.inf, valid_split=0.1, test_split=0.1, separate_domains=False, num_workers=0):
    entire_datasets = []
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    st = time.time()

    # Domains is 'rest' or 'all' for source dataset, 'args.tgt' is for target dataset
    if domains == 'all':
        domains = 'rest'  # MAML does not allow 'all' for source dataset
    elif domains == 'rest':
        domains = 'rest'
    else:
        domains = [domains]

    # Processed domains are the rest for source dataset, Processed domain is 'args.tgt' for target dataset
    if domains == 'rest':
        target_domains = [args.tgt]
        processed_domains = set(options.voice_option['domains']) - set(target_domains)
        processed_domains_list = list(processed_domains)
        processed_domains_list.sort()
        print('Processed domains: ' + str(processed_domains_list))
        print('Processed domains length: ' + str(len(processed_domains_list)))
    else:
        target_domains = domains
        print('Processed domain: ' + str(target_domains))

    for target_domain in target_domains:  # Only support one domain currently
        if args.dataset in ['voice']:
            '''Complementary signify if the dataset is target domain only (False), or domains are the rest
             of target domain (True)'''
            if separate_domains:
                if domains == 'rest':  # For source dataset
                    voice_dataset = VoiceDataset(file=file_path, domain=target_domain,
                                                 complementary=True,
                                                 max_source=args.num_source, num_bin=args.num_bin)
                else:  # For target dataset
                    voice_dataset = VoiceDataset(file=file_path, domain=target_domain,
                                                 complementary=False,
                                                 max_source=args.num_source, num_bin=args.num_bin)

                voice_dataset_per_domain = voice_dataset.get_datasets_per_domain()
                entire_datasets = voice_dataset_per_domain
            else:
                if domains == 'rest':  # For source dataset
                    voice_dataset = VoiceDataset(file=file_path, domain=target_domain,
                                                 complementary=True,
                                                 max_source=args.num_source, num_bin=args.num_bin)
                else:  # For target dataset
                    voice_dataset = VoiceDataset(file=file_path, domain=target_domain,
                                                 complementary=False,
                                                 max_source=args.num_source, num_bin=args.num_bin)

                entire_datasets.append(voice_dataset)
        else:
            print("Invalid dataset, please choose one of the dataset in ['voice'].")

    # Split each domain dataset into train, valid, and test dataset
    for data in entire_datasets:
        feature, label, domain = data[0]
        domain_number = domain[0].tolist()
        domain_str = number_to_domain(domain_number)

        train_data, valid_data, test_data = split_data(data, valid_split, test_split, train_max_rows,
                                                       valid_max_rows, test_max_rows)
        train_datasets.append(train_data)
        valid_datasets.append(valid_data)
        test_datasets.append(test_data)

        print('Domain: {:s},\tEntire: {:d} instances per class,\tTrain: {:d},\tValid: {:d},\tTest: {:d}'.format(
            domain_str, len(data), len(train_data), len(valid_data), len(test_data)))

    # Limit the number of domains
    train_datasets = train_datasets[:args.num_source]
    valid_datasets = valid_datasets[:args.num_source]
    test_datasets = test_datasets[:args.num_source]
    print('#Time cost: {:f} seconds'.format(time.time() - st))

    if separate_domains:
        # Actual batch size is multiplied by num_class
        train_data_loaders = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=False,
                                                    drop_last=True, num_workers=num_workers)
        valid_data_loaders = datasets_to_dataloader(valid_datasets, batch_size=batch_size, concat=False,
                                                    drop_last=True, num_workers=num_workers)
        test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False,
                                                   drop_last=True, num_workers=num_workers)

        data_loaders = []
        for i in range(len(train_data_loaders)):
            # Train is for support set, Test is for query set.
            data_loader = {
                'train': train_data_loaders[i],  # for support set
                'valid': valid_data_loaders[i]
                if len(valid_data_loaders) == len(train_data_loaders) else None,  # for validation in target dataset
                'test': test_data_loaders[i],  # for query set
                'num_domains': len(train_data_loaders)
            }
            data_loaders.append(data_loader)
        return data_loaders
    else:
        train_data_loader = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=True,
                                                   drop_last=True, num_workers=num_workers)
        valid_data_loader = datasets_to_dataloader(valid_datasets, batch_size=batch_size, concat=True,
                                                   drop_last=True, num_workers=num_workers)
        test_data_loader = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=True,
                                                  drop_last=True, num_workers=num_workers)

        data_loader = {
            'train': train_data_loader,  # for support set
            'valid': valid_data_loader,  # for validation in target dataset
            'test': test_data_loader,  # for query set
            'num_domains': sum([dataset.dataset.get_num_domains() for dataset in train_datasets]),  # Todo
        }
        return data_loader


def calibrated_domain_data_loader(args, domains, file_path, augment_file_path, batch_size,
                                  valid_max_rows=np.inf, test_max_rows=np.inf, valid_split=0.2, num_workers=0):
    st = time.time()

    target_domains = [domains]
    print('Processed domain: ' + str(target_domains))

    train_datasets = None
    valid_test_datasets = None
    for target_domain in target_domains:  # Only support one domain currently
        if args.dataset in ['voice']:
            '''Complementary signify if the dataset is target domain only (False), or domains are the rest
             of target domain (True)'''
            voice_training_dataset = VoiceDataset(file=augment_file_path, domain=target_domain, complementary=False,
                                                  max_source=args.num_source, num_bin=args.num_bin)
            voice_val_testing_dataset = VoiceDataset(file=file_path, domain=target_domain, complementary=False,
                                                     max_source=args.num_source, num_bin=args.num_bin)

            voice_training_dataset_per_domain = voice_training_dataset.get_datasets_per_domain()
            voice_val_testing_dataset_per_domain = voice_val_testing_dataset.get_datasets_per_domain()

            for train_dataset in voice_training_dataset_per_domain:
                train_datasets = train_dataset
            for valid_test_dataset in voice_val_testing_dataset_per_domain:
                valid_test_datasets = valid_test_dataset
        else:
            print("Invalid dataset, please choose one dataset in ['voice']")

    # Split each dataset into valid and test dataset
    valid_datasets, test_datasets = split_val_test_data(valid_test_datasets, valid_split, valid_max_rows, test_max_rows)

    print('Domain: {:s},\tTrain: {:d},\tValid: {:d},\tTest: {:d}'.format(
        str(target_domains), len(train_datasets), len(valid_datasets), len(test_datasets)))
    print('#Time cost: {:f} seconds'.format(time.time() - st))

    # Actual batch size is multiplied by num_class
    train_data_loaders = DataLoader(train_datasets, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    valid_data_loaders = DataLoader(valid_datasets, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    test_data_loaders = DataLoader(test_datasets, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)

    # Train is for support set, Test is for query set.
    data_loaders = []
    data_loader = {
        'train': train_data_loaders,  # for support set
        'valid': valid_data_loaders,  # for validation in target dataset
        'test': test_data_loaders,  # for query set
        'num_domains': len(train_data_loaders)
    }
    data_loaders.append(data_loader)

    return data_loaders


def calibrated_domain_data_loader2(args, domains, file_path, augment_file_path, batch_size, train_max_rows=np.inf,
                               valid_max_rows=np.inf,
                               test_max_rows=np.inf,
                               valid_split=0.2, test_split=0.2):
    st = time.time()

    target_domain = [domains]
    print('Processed domain: ' + str(target_domain))

    for domain in target_domain:  # Only support one domain currently
        if args.dataset in ['voice']:
            '''Complementary signify if the dataset include target domain only (false), or domains exclude target 
            domain (true)'''
            voice_training_dataset = VoiceDataset(file=augment_file_path, domain=domain, complementary=False,
                                                  max_source=args.num_source, num_bin=args.num_bin)

            voice_training_dataset_per_domain = voice_training_dataset.get_datasets_per_domain()

            for valid_test_dataset in voice_training_dataset_per_domain:
                valid_test_datasets = valid_test_dataset

        else:
            print("Invalid dataset, please choose one dataset in ['voice']")

    # Split each dataset into valid and test dataset
    train_datasets, valid_datasets, test_datasets = split_data(valid_test_datasets, valid_split, test_split,
                                                               train_max_rows, valid_max_rows, test_max_rows)

    print('#Multi domain?:{:d}\tTrain: {:d} instances per class\tValid: {:d}\tTest: {:d}'.format(
        False, len(train_datasets), len(valid_datasets), len(test_datasets)))
    print('#Time cost: {:f} seconds'.format(time.time() - st))

    # Actual batch size is multiplied by num_class
    train_data_loaders = DataLoader(train_datasets, batch_size=batch_size,
                                    shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    # Set validation batch_size = 10 to boost validation speed
    valid_data_loaders = DataLoader(valid_datasets, batch_size=batch_size,
                                    shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    test_data_loaders = DataLoader(test_datasets, batch_size=batch_size,
                                   shuffle=True, num_workers=0, drop_last=True, pin_memory=False)

    # Train is for support set, Test is for query set.
    data_loaders = []
    data_loader = {
        'train': train_data_loaders,  # for support set
        # for validation in target dataset
        'valid': valid_data_loaders,
        'test': test_data_loaders,  # for query set
        'num_domains': len(train_data_loaders)
    }
    data_loaders.append(data_loader)

    return data_loaders
