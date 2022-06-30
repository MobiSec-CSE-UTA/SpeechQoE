import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_flatten_dim = 128 * 7  # Calculate by (the shape feeding into FC layer)/(batch size), check line 180.
input_channel_dim = 1


class MAML_BN(nn.Module):
    def __init__(self, opt):
        super(MAML_BN, self).__init__()
        self.opt = opt

        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        # Stride size is defined in forward function
        conv1 = nn.Conv1d(input_channel_dim, 2, kernel_size=6)
        bn1 = nn.BatchNorm1d(2)
        conv2 = nn.Conv1d(2, 4, kernel_size=6)
        bn2 = nn.BatchNorm1d(4)
        conv3 = nn.Conv1d(4, 8, kernel_size=3)
        bn3 = nn.BatchNorm1d(8)
        conv4 = nn.Conv1d(8, 16, kernel_size=3)
        bn4 = nn.BatchNorm1d(16)
        conv5 = nn.Conv1d(16, 32, kernel_size=2)
        bn5 = nn.BatchNorm1d(32)
        conv6 = nn.Conv1d(32, 64, kernel_size=2)
        bn6 = nn.BatchNorm1d(64)
        conv7 = nn.Conv1d(64, 128, kernel_size=1)
        bn7 = nn.BatchNorm1d(128)
        fc1 = nn.Linear(feature_flatten_dim, 512)
        bn8 = nn.BatchNorm1d(512)
        fc2 = nn.Linear(512, 256)
        bn9 = nn.BatchNorm1d(256)
        fc3 = nn.Linear(256, 128)
        bn10 = nn.BatchNorm1d(128)
        fc4 = nn.Linear(128, self.opt['num_class'])

        layers = [('conv', conv1), ('bn', bn1), ('conv', conv2), ('bn', bn2), ('conv', conv3), ('bn', bn3),
                  ('conv', conv4), ('bn', bn4), ('conv', conv5), ('bn', bn5), ('conv', conv6), ('bn', bn6),
                  ('conv', conv7), ('bn', bn7), ('fc', fc1), ('bn', bn8), ('fc', fc2), ('bn', bn9), ('fc', fc3),
                  ('bn', bn10), ('fc', fc4)]

        for (name, layer) in layers:
            if name == 'bn':
                w = nn.Parameter(torch.ones_like(layer.weight))
                self.vars.append(w)
                b = nn.Parameter(torch.zeros_like(layer.bias))
                self.vars.append(b)
                running_mean = nn.Parameter(torch.zeros_like(layer.weight), requires_grad=False)
                running_var = nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            else:
                w = nn.Parameter(torch.ones_like(layer.weight))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                b = nn.Parameter(torch.zeros_like(layer.bias))
                self.vars.append(b)

    def load_checkpoint(self, checkpoint):
        state_keys_fe = list(checkpoint['feature_extractor'].keys())
        state_keys_cc = list(checkpoint['class_classifier'].keys())

        keys_without_bn_fe = []
        keys_without_bn_cc = []

        for key in state_keys_fe:
            if 'num_batches_tracked' in key or 'running' in key:
                continue
            else:
                keys_without_bn_fe.append(key)

        for key in state_keys_cc:
            if 'num_batches_tracked' in key or 'running' in key:
                continue
            else:
                keys_without_bn_cc.append(key)

        assert (len(keys_without_bn_fe) + len(keys_without_bn_cc) == len(self.vars))

        for i in range(len(self.vars)):
            if i < len(keys_without_bn_fe):
                self.vars[i] = nn.Parameter(checkpoint['feature_extractor'][keys_without_bn_fe[i]])
            else:
                self.vars[i] = nn.Parameter(
                    checkpoint['class_classifier'][keys_without_bn_cc[i - len(keys_without_bn_fe)]])

    def forward(self, my_input, my_vars=None, training=True):
        #print('The size of input0: ' + str(my_input.size()))
        if my_vars is None:
            my_vars = self.vars
        idx = 0
        bn_idx = 0

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.relu(my_input, True)
        #print('The size of input1: ' + str(my_input.size()))
        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.relu(my_input, True)
        #print('The size of input2: ' + str(my_input.size()))
        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.max_pool1d(my_input, 2)
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.max_pool1d(my_input, 2)
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b, stride=1)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        # print('The shape before feeding into fc layer: ' + str(my_input.size()))
        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.linear(my_input.view(-1, feature_flatten_dim), w, b)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.linear(my_input, w, b)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.linear(my_input, w, b)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.linear(my_input, w, b)
        idx += 2

        assert idx == len(my_vars)
        assert bn_idx == len(self.vars_bn)

        return F.log_softmax(my_input, 1)

    def zero_grad(self, my_vars=None):
        """
        :param my_vars:
        :return:
        """
        with torch.no_grad():
            if my_vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in my_vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


if __name__ == '__main__':
    pass
