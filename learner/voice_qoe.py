import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from emodel import infer_by_emodel
from torch.utils.data import DataLoader

from copy import deepcopy
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VoiceQOE:
    def __init__(self, args, opt, model, tensorboard, source_dataloaders, target_dataloader, lr):
        self.device = device

        self.args = args
        self.opt = opt
        self.tensorboard = tensorboard

        self.net = model.MAML_BN(opt).to(device)
        self.task_lr = lr  # learning rate for getting task specific parameter
        self.meta_lr = opt['learning_rate']  # learning rate for getting generic parameter
        self.update_step_train = 30
        self.update_step_test = 45
        self.class_criterion = nn.L1Loss()  # MAE Loss
        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.NLLLoss()
        self.optimizer = optim.Adam([{'params': self.net.parameters()}], lr=self.meta_lr,
                                    weight_decay=opt['weight_decay'])

        self.source_dataloaders = source_dataloaders
        self.target_dataloader = target_dataloader

        self.target_support_set = next(iter(target_dataloader['train']))

        # Iterators for support and query sets in source dataset
        self.iters_spt = [iter(self.source_dataloaders[i]['train']) for i in range(len(self.source_dataloaders))]
        self.iters_qry = [iter(self.source_dataloaders[i]['test']) for i in range(len(self.source_dataloaders))]

    def save_checkpoint(self, epoch, lowest_loss, checkpoint_path):
        state = {'epoch': epoch, 'lowest_loss': lowest_loss,
                 'class_classifier': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, checkpoint_path)
        return

    def load_checkpoint(self, checkpoint_path):
        path = checkpoint_path
        checkpoint = torch.load(path)

        self.net.load_state_dict(checkpoint['class_classifier'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint

    def reset_layer(self):
        """Remember that Pytorch accumulates gradients.x We need to clear them before each instance,
        that means we have to call net.zero_grad() before each backward."""
        self.net.zero_grad()

    def get_label_and_data(self, data):
        input_of_data, class_label_of_data, domain_label_of_data = data
        input_of_data = input_of_data.to(device)
        class_label_of_data = class_label_of_data.to(device)
        domain_label_of_data = domain_label_of_data.to(device)

        return input_of_data, class_label_of_data, domain_label_of_data

    def get_mae(self, classifier, params, criterion, data, label):
        preds_of_data = classifier(data, params)
        preds_of_data = torch.flatten(preds_of_data)
        loss_of_mse = criterion(preds_of_data, label)
        # pred_label = torch.round(preds_of_data)

        # labels = [i for i in range(len(self.opt['classes']))]
        # cm = confusion_matrix(label.cpu().numpy(), pred_label.detach().numpy(), labels=labels)

        return loss_of_mse

    def fuse_acc(self, classifier, params, criterion, data, label):
        preds_voice = classifier(data, params)
        preds_voice = torch.flatten(preds_voice)

        # E-model prediction
        path_qos = "./emodel/dataset/qos_feature_emodel.csv"
        path_model = "./emodel/emodel.pt"

        dataset = infer_by_emodel.EmodelDataset(path_qos)
        dl = DataLoader(dataset, batch_size=64, shuffle=True)

        model = infer_by_emodel.MLP(2)
        model.load_state_dict(torch.load(path_model))
        model.eval()

        index = preds_voice.size()[0]
        preds_emodel = infer_by_emodel.preds(dl, model)
        preds_emodel = preds_emodel[0:index, :]
        preds_emodel = torch.from_numpy(preds_emodel)
        preds_emodel = torch.flatten(preds_emodel)

        preds = 0.99*preds_voice+0.01*preds_emodel

        loss_of_mse = criterion(preds, label)

        return loss_of_mse

    def log_loss_results(self, condition, epoch, loss_avg, nshot=5):
        self.tensorboard.log_scalar(condition + '/loss_sum_' + str(nshot) + 'shot', loss_avg, epoch)
        return loss_avg

    def generate_random_tasks(self, num_repeat=1):
        # Generate random tasks and its corresponding support and query sets
        with torch.no_grad():
            support_set_from_domains = []
            query_set_from_domains = []
            set_index = []

            for domain_i in range(len(self.source_dataloaders)):  # for each task
                for i in range(len(self.iters_spt[domain_i])):
                    try:
                        train_batch_i = next(self.iters_spt[domain_i])
                    except StopIteration:
                        self.iters_spt[domain_i] = iter(self.source_dataloaders[domain_i]['train'])
                        train_batch_i = next(self.iters_spt[domain_i])

                    try:
                        test_batch_i = next(self.iters_qry[domain_i])
                    except StopIteration:
                        self.iters_qry[domain_i] = iter(self.source_dataloaders[domain_i]['test'])
                        test_batch_i = next(self.iters_qry[domain_i])

                    support_set_from_domains.append(train_batch_i)
                    query_set_from_domains.append(test_batch_i)

            # Index generated from set_from_domain
            for i in range(len(support_set_from_domains)):
                set_index.append(i)

            # Generate support and query set for each tasks
            synthetic_supports = []
            synthetic_queries = []

            for i in range(num_repeat):
                # Indices for each class and its corresponding domain
                selected_domain_indices = np.random.choice(set_index, self.opt[
                    'num_class'] * self.args.nshot)  # Generate random indices to choose domain for each class
                class_indices = np.array(list(range(self.opt['num_class'])) * self.args.nshot)
                class_domain_indices = np.array(list(zip(class_indices, selected_domain_indices)))

                # Created empty tensors for support and query sets
                tmp_spt_feat = torch.FloatTensor(support_set_from_domains[0][0].shape[0], 0,
                                                 *(support_set_from_domains[0][0].shape[2:]))
                tmp_spt_cl = torch.LongTensor(support_set_from_domains[0][1].shape[0], 0)
                tmp_spt_dl = torch.LongTensor(support_set_from_domains[0][2].shape[0], 0)

                tmp_qry_feat = torch.FloatTensor(query_set_from_domains[0][0].shape[0], 0,
                                                 *(support_set_from_domains[0][0].shape[2:]))
                tmp_qry_cl = torch.LongTensor(query_set_from_domains[0][1].shape[0], 0)
                tmp_qry_dl = torch.LongTensor(query_set_from_domains[0][2].shape[0], 0)

                # Extract data based on randomly generated index
                for class_id, domain_id in class_domain_indices:
                    class_id = torch.tensor([class_id])

                    # Support set
                    feature, class_label, domain_label = support_set_from_domains[domain_id]
                    # select class dimension
                    indices = class_label == class_id
                    feature = feature[indices].view(feature.shape[0], 1, *feature.shape[2:])
                    class_label = class_label[indices].view(-1, 1)
                    domain_label = domain_label[indices].view(-1, 1)

                    tmp_spt_feat = torch.cat((tmp_spt_feat, feature), dim=1)
                    tmp_spt_cl = torch.cat((tmp_spt_cl, class_label), dim=1)
                    tmp_spt_dl = torch.cat((tmp_spt_dl, domain_label), dim=1)

                    # Query set
                    feature, class_label, domain_label = query_set_from_domains[domain_id]
                    # select class dimension
                    indices = class_label == class_id
                    feature = feature[indices].view(feature.shape[0], 1, *feature.shape[2:])
                    class_label = class_label[indices].view(-1, 1)
                    domain_label = domain_label[indices].view(-1, 1)

                    tmp_qry_feat = torch.cat((tmp_qry_feat, feature), dim=1)
                    tmp_qry_cl = torch.cat((tmp_qry_cl, class_label), dim=1)
                    tmp_qry_dl = torch.cat((tmp_qry_dl, domain_label), dim=1)

                tmp_spt_feat = tmp_spt_feat[0, :]
                tmp_spt_cl = tmp_spt_cl[0, :]
                tmp_spt_dl = tmp_spt_dl[0, :]

                tmp_qry_feat = tmp_qry_feat[0, :]
                tmp_qry_cl = tmp_qry_cl[0, :]
                tmp_qry_dl = tmp_qry_dl[0, :]

                synthetic_supports.append([tmp_spt_feat, tmp_spt_cl, tmp_spt_dl])  # Shape: (3 * (# of task))
                synthetic_queries.append([tmp_qry_feat, tmp_qry_cl, tmp_qry_dl])

        return synthetic_supports, synthetic_queries

    def train(self, epoch):
        num_synthetic_domains = len(self.source_dataloaders)
        num_task = num_synthetic_domains * self.args.ntask

        synthetic_supports, synthetic_queries = self.generate_random_tasks(num_repeat=num_task)

        self.net.train()
        self.reset_layer()

        losses_q = [0 for _ in range(self.update_step_train + 1)]  # losses_q[i] is the loss on step i

        for task_i in range(num_task):
            support_set = synthetic_supports[task_i]
            query_set = synthetic_queries[task_i]

            labeled_feature_spt, labeled_class_spt, _ = self.get_label_and_data(support_set)
            labeled_feature_qry, labeled_class_qry, _ = self.get_label_and_data(query_set)

            fast_weights = self.net.parameters()  # weights trained on support set

            for k in range(0, self.update_step_train + 1):
                # Get the query set MSE with parameters updated by support set
                loss_query = self.get_mae(self.net, fast_weights, self.class_criterion, labeled_feature_qry,
                                          labeled_class_qry)
                if k == self.update_step_train:
                    losses_q[k] += loss_query
                else:
                    losses_q[k] += float(loss_query)

                if k != self.update_step_train:  # ignore the last unnecessary update
                    # 1. run the i-th task and compute loss
                    loss_support = self.get_mae(self.net, fast_weights, self.class_criterion,
                                                labeled_feature_spt, labeled_class_spt)
                    # 2. compute grad on theta_pi, line 7
                    grad = torch.autograd.grad(loss_support, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad, line 8
                    fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))

        loss_query = losses_q[-1] / num_task

        self.optimizer.zero_grad()
        loss_query.backward()
        self.optimizer.step()

        losses_q = np.array([loss if isinstance(loss, float) else loss.item() for loss in losses_q]) / num_task

        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print('###############################################################')
        print('[Train] epoch:' + str(epoch) + '\tloss:' + str(losses_q))

        self.log_loss_results('train', epoch=epoch, loss_avg=loss_query)
        return

    def evaluation(self, epoch, condition, nshot=5):
        num_task = 1

        losses_q = [0.0 for _ in range(self.update_step_test + 1)]  # losses_q[i] is the loss on step i

        self.reset_layer()

        """ In order to avoid ruin the state of running_mean/variance and bn_weight/bias,
        we fine tune on the copied model instead of self.net"""
        net = deepcopy(self.net)
        net.train()

        """Be aware we use support set to fine tune the trained model, query set is for testing.
        Parameter nshot is the number of shot for fine tune, all data in  validation dataset or test
        dataset is used for accuracy testing"""
        support_set = [x[:nshot].data.clone() for x in self.target_support_set]  # Evaluation with only n-shots
        support_set[0] = support_set[0].view(-1, *(support_set[0].shape[2:]))
        support_set[1] = support_set[1].view(-1)
        support_set[2] = support_set[2].view(-1)
        print("Total number of data used for adaption: " + str(len(support_set[1])))

        query_set_concat = [[], [], []]
        for batch_idx, query_set in enumerate(self.target_dataloader[condition]):
            query_set[0] = query_set[0].view(-1, *(query_set[0].shape[2:]))
            query_set[1] = query_set[1].view(-1)
            query_set[2] = query_set[2].view(-1)

            query_set_concat[0].append(query_set[0])
            query_set_concat[1].append(query_set[1])
            query_set_concat[2].append(query_set[2])

        query_set_concat[0] = torch.cat(query_set_concat[0])
        query_set_concat[1] = torch.cat(query_set_concat[1])
        query_set_concat[2] = torch.cat(query_set_concat[2])

        labeled_feature_spt, labeled_class_spt, _ = self.get_label_and_data(support_set)
        labeled_feature_qry, labeled_class_qry, _ = self.get_label_and_data(query_set_concat)

        fast_weights = net.parameters()
        for k in tqdm(range(0, self.update_step_test + 1), total=self.update_step_test + 1):
            # class_loss_query = self.get_mae(net, fast_weights, self.class_criterion, labeled_feature_qry,
            #                                 labeled_class_qry)
            class_loss_query = self.fuse_acc(net, fast_weights, self.class_criterion, labeled_feature_qry,
                                             labeled_class_qry)
            with torch.no_grad():
                losses_q[k] += float(class_loss_query)

            if k != self.update_step_test:
                # 1. run the i-th task and compute loss for k=1~K-1
                loss_support = self.get_mae(net, fast_weights, self.class_criterion, labeled_feature_spt,
                                            labeled_class_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss_support, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))

        loss_query = losses_q[-1] / num_task

        losses_q = np.array([loss if isinstance(loss, float) else loss.item() for loss in losses_q]) / num_task
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

        print('---------------------------------------------------')
        print('[{:s}] epoch:{:d} nshot:{:d}'.format(condition, epoch, nshot) + '\tloss:' + str(losses_q))

        self.log_loss_results(condition, epoch=1, loss_avg=loss_query)

        return losses_q[-1]

    def validation(self, epoch):
        # Fixed shot size 5 for validation
        loss = self.evaluation(epoch, 'valid', nshot=5)

        return loss

    def test(self, epoch):
        # The accuracy for 1 to 10 shots
        accuracy_of_test_data = []

        # Test by 1 to 10 shots
        for i in range(10):
            loss_for_shot = self.evaluation(epoch, 'test', nshot=i + 1)
            accuracy_of_test_data.append(loss_for_shot)

        return accuracy_of_test_data
