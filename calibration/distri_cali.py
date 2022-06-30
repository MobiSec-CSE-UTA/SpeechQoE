import numpy as np
import pandas as pd
from numpy.random import multivariate_normal

import options


def domain_to_number(domain_name):
    dic = options.voice_option['domain_to_number']
    return dic[domain_name]


def number_to_domain(domain_number):
    dic = options.voice_option['number_to_domain']
    return dic[domain_number]


def mean_cov_calculate(file_path, seq_len, target_domain):
    data = pd.read_csv(file_path)

    user_list = []
    for user in data["domain"]:
        user_list.append(user)
    user_list = list(set(user_list))  # remove duplicate value
    # Exclude target domain
    user_list.remove(target_domain)
    data = data.loc[data['domain'].isin(user_list)]

    # Convert domain from string to number based on a given dictionary
    domain_to_number_df = options.voice_option['domain_to_number']
    data["domain"].replace(domain_to_number_df, inplace=True)
    user_numbers_list = []
    for user in user_list:
        user_number = domain_to_number(user)
        user_numbers_list.append(user_number)

    label_list = []
    for label in data["label"]:
        label_list.append(label)
    label_list = list(set(label_list))  # remove duplicate value

    base_means = [[] for _ in range(len(user_numbers_list))]
    base_cov = [[] for _ in range(len(user_numbers_list))]

    feature_len = data.shape[1] - 3

    file_data_arrays = np.zeros((seq_len, 1))
    mu_arrays = np.zeros((seq_len, 1))
    sig_arrays = np.zeros((seq_len, seq_len, 1))
    label_mu_arrays = np.zeros((seq_len, feature_len, 1))
    label_sig_arrays = np.zeros((seq_len, seq_len, feature_len, 1))

    for index, user in enumerate(user_numbers_list):
        user_data = data[data['domain'] == user]
        user_data = user_data.iloc[:, 1:]
        for label in label_list:
            label_data = user_data[user_data['label'] == label]
            label_data = label_data.iloc[:, 1:]
            # file_list = [file for file in label_data["file"]]
            # file_list = list(set(file_list))

            for feature_index in range(0, feature_len):
                for idx in range(len(label_data) // seq_len):
                    # Append all data with row as sequence
                    seq_data = label_data.iloc[idx*seq_len:(idx + 1)*seq_len, feature_index + 1]

                    file_data_array = seq_data.values.reshape((-1, 1))
                    file_data_arrays = np.concatenate([file_data_arrays, file_data_array], axis=1)

                file_data_arrays = file_data_arrays[:, 1:]
                mu_vector = np.mean(file_data_arrays, axis=1)
                # file_data_arrays = file_data_arrays[:,0]
                sig_matrix = np.cov(file_data_arrays, rowvar=True)

                mu_vector = mu_vector.reshape((-1, 1))
                sig_matrix = sig_matrix.reshape((seq_len, -1, 1))
                mu_arrays = np.concatenate([mu_arrays, mu_vector], axis=1)
                sig_arrays = np.concatenate([sig_arrays, sig_matrix], axis=2)

                file_data_arrays = np.zeros((seq_len, 1))

            mu_arrays = mu_arrays[:, 1:]
            sig_arrays = sig_arrays[:, :, 1:]
            mu_arrays = mu_arrays.reshape((seq_len, -1, 1))
            sig_arrays = sig_arrays.reshape((seq_len, -1, feature_len, 1))
            label_mu_arrays = np.concatenate([label_mu_arrays, mu_arrays], axis=2)
            label_sig_arrays = np.concatenate([label_sig_arrays, sig_arrays], axis=3)

            mu_arrays = np.zeros((seq_len, 1))
            sig_arrays = np.zeros((seq_len, seq_len, 1))

        label_mu_arrays = label_mu_arrays[:, :, 1:]
        label_sig_arrays = label_sig_arrays[:, :, :, 1:]
        base_means[index].append(label_mu_arrays)
        base_cov[index].append(label_sig_arrays)

        label_mu_arrays = np.zeros((seq_len, feature_len, 1))
        label_sig_arrays = np.zeros((seq_len, seq_len, feature_len, 1))

    print("Means and covariances for source domains have been calculated.")
    return user_numbers_list, base_means, base_cov


def resampling(file_path, target_domain, output, base_means, base_covs, num_aug_shot, num_class, k, alpha, seq_len,
               source_domain):
    # Extract query from target domain
    data = pd.read_csv(file_path)
    data = data.loc[data['domain'] == target_domain]

    feature_len = data.shape[1] - 3

    label_list = []
    for label in data["label"]:
        label_list.append(label)
    label_list = list(set(label_list))  # remove duplicate value

    file_data_arrays = np.zeros((seq_len, 1))
    mu_arrays = np.zeros((seq_len, 1))
    label_mu_arrays = np.zeros((seq_len, feature_len, 1))

    for label in label_list:
        label_data = data[data['label'] == label]
        label_data = label_data.iloc[:, 1:]
        for feature_index in range(0, feature_len):
            for idx in range(len(label_data) // seq_len):
                # Append all data with row as sequence
                seq_data = label_data.iloc[idx * seq_len:(idx + 1) * seq_len, feature_index + 2]
                file_data_array = seq_data.values.reshape((-1, 1))
                file_data_arrays = np.concatenate([file_data_arrays, file_data_array], axis=1)

            file_data_arrays = file_data_arrays[:, 1:]
            mu_vector = np.mean(file_data_arrays, axis=1)

            mu_vector = mu_vector.reshape((-1, 1))
            mu_arrays = np.concatenate([mu_arrays, mu_vector], axis=1)
            file_data_arrays = np.zeros((seq_len, 1))

        mu_arrays = mu_arrays[:, 1:]
        mu_arrays = mu_arrays.reshape((seq_len, -1, 1))
        label_mu_arrays = np.concatenate([label_mu_arrays, mu_arrays], axis=2)

        mu_arrays = np.zeros((seq_len, 1))

    label_mu_arrays = label_mu_arrays[:, :, 1:]
    query = label_mu_arrays

    # Calibrate distribution
    dist = []
    for i in range(len(base_means)):
        base_means_array = np.array(base_means[i])
        dist.append(np.linalg.norm(query - base_means_array))

    # Find the closest domain(s) and their corresponding distance
    index = np.argpartition(dist, k)[:k]
    index_list = index.tolist()
    closest_domains = [source_domain[i] for i in index_list]
    for closest_domain in closest_domains:
        domain_str = number_to_domain(closest_domain)
        print("The most similar doamin(s) to target domain," + target_domain + ",is: " + str(domain_str))

    closest_distances = [dist[i].tolist() for i in index_list]
    for closest_distance in closest_distances:
        print("The corresponding distance(s) is: {:.2f}".format(closest_distance))

    # Unpack means and covs from list to pure narray
    closest_base_means = []
    for i in index_list:
        base_mean = base_means[i]
        for j in base_mean:
            closest_base_means.append(j)
    closest_base_covs = []
    for i in index_list:
        base_cov = base_covs[i]
        for j in base_cov:
            closest_base_covs.append(j)

    # Add one more dimension for domain
    closest_base_means_reshape = []
    for closest_base_mean in closest_base_means:
        closest_base_mean_reshape = closest_base_mean[:, :, :,  np.newaxis]
        closest_base_means_reshape.append(closest_base_mean_reshape)
    closest_base_covs_reshape = []
    for closest_base_cov in closest_base_covs:
        closest_base_cov_reshape = closest_base_cov[:, :, :, :, np.newaxis]
        closest_base_covs_reshape.append(closest_base_cov_reshape)

    query_reshape = query[:, :, :, np.newaxis]

    # mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, np.newaxis, :]], axis=1)  # Todo: support k>1
    # calibrated_mean = np.mean(mean, axis=1)
    # calibrated_cov = np.mean(np.array(base_cov)[index], axis=1) + alpha

    mean = np.concatenate(closest_base_means_reshape, axis=3)
    mean = np.concatenate([mean, query_reshape], axis=3)
    calibrated_mean = np.mean(mean, axis=3)

    cov = np.concatenate(closest_base_covs_reshape, axis=4)
    calibrated_cov = np.mean(cov, axis=4) + alpha

    # calibrated_mean = calibrated_mean.reshape(seq_len, feature_len, num_class)
    # calibrated_cov = calibrated_cov.reshape(seq_len, seq_len, feature_len, num_class)

    # Resampling from calibrated distribution
    random_feature_arrays = np.zeros([seq_len * num_aug_shot, 1])
    random_label_arrays = np.zeros([1, feature_len])

    # calibrated_cov_feature_1 = np.ones([seq_len, seq_len])*100  # Fixed covariance

    for label in range(num_class):
        calibrated_mean_label = calibrated_mean[:, :, label]
        calibrated_cov_label = calibrated_cov[:, :, :, label]

        for feature in range(feature_len):
            calibrated_mean_feature = calibrated_mean_label[:, feature]
            calibrated_cov_feature = calibrated_cov_label[:, :, feature]
            random_sample = multivariate_normal(calibrated_mean_feature, calibrated_cov_feature, size=num_aug_shot)
            # random_sample = multivariate_normal(calibrated_mean_feature, calibrated_cov_feature_1, size=num_aug_shot)
            random_sample = random_sample.reshape(-1, 1)
            random_feature_arrays = np.concatenate([random_feature_arrays, random_sample], axis=1)

        random_feature_arrays = random_feature_arrays[:, 1:]
        random_label_arrays = np.concatenate([random_label_arrays, random_feature_arrays], axis=0)

        random_feature_arrays = np.zeros([seq_len * num_aug_shot, 1])

    random_label_arrays = random_label_arrays[1:, :]

    # Add columns on generated samples to make it consistent with original format
    # Generate label column
    label_arr = np.zeros(shape=(len(random_label_arrays), 1))
    label_seq = seq_len * num_aug_shot
    for idx in range(num_class):
        label_arr[idx * label_seq:(idx + 1) * label_seq, 0] = idx + 1

    # Generate domain column
    domain_arr = np.zeros(shape=(len(random_label_arrays), 1))

    # Generate file column to keep the output format consistent with source dataset
    file_arr = np.zeros(shape=(len(random_label_arrays), 1))

    random_sample = np.concatenate((domain_arr, label_arr, file_arr, random_label_arrays), axis=1)

    df = pd.DataFrame(random_sample, columns=['domain', 'label', 'file', '1', '2', '3', '4', '5', '6', '7', '8',
                                              '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
    df['domain'] = target_domain
    df['file'] = '_'

    df.to_csv(output, index=False)

    print("Data augmentation file created in {}. Target domain: {}, shot number:{}, sequence length:{}".format(
        str(output), str(target_domain), str(num_aug_shot), str(seq_len)))


if __name__ == "__main__":
    pass
