import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Calculate by (the shape feeding into FC layer)/(batch size), check line 150, (x,y,z) => (y*z).
feature_flatten_dim = 86528  # Given (x,y,z) => (x*y*z)/(5*5)
input_channel_dim = 18


class MAML_BN(nn.Module):
    def __init__(self, opt):

        super(MAML_BN, self).__init__()
        self.opt = opt

        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        conv1 = nn.Conv1d(input_channel_dim, 32, kernel_size=3)
        bn1 = nn.BatchNorm1d(32)
        conv2 = nn.Conv1d(32, 64, kernel_size=3)
        bn2 = nn.BatchNorm1d(64)
        conv3 = nn.Conv1d(64, 128, kernel_size=3)
        bn3 = nn.BatchNorm1d(128)
        conv4 = nn.Conv1d(128, 256, kernel_size=3)
        bn4 = nn.BatchNorm1d(256)
        conv5 = nn.Conv1d(256, 512, kernel_size=3)
        bn5 = nn.BatchNorm1d(512)
        fc1 = nn.Linear(feature_flatten_dim, 1024)
        bn6 = nn.BatchNorm1d(1024)
        fc2 = nn.Linear(1024, 256)
        bn7 = nn.BatchNorm1d(256)
        fc3 = nn.Linear(256, self.opt['num_class'])

        layers = [('conv', conv1), ('bn', bn1), ('conv', conv2), ('bn', bn2), ('conv', conv3), ('bn', bn3),
                  ('conv', conv4), ('bn', bn4), ('conv', conv5), ('bn', bn5), ('fc', fc1), ('bn', bn6), ('fc', fc2),
                  ('bn', bn7), ('fc', fc3)]

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
        if my_vars is None:
            my_vars = self.vars
        idx = 0
        bn_idx = 0

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b)
        idx += 2
        my_input = F.max_pool1d(my_input, 2)
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b)
        idx += 2
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = my_vars[idx], my_vars[idx + 1]
        my_input = F.conv1d(my_input, w, b)
        idx += 2
        my_input = F.max_pool1d(my_input, 2)
        my_input = F.relu(my_input, True)

        w, b = my_vars[idx], my_vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        my_input = F.batch_norm(my_input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        #print('The shape before feeding into fc layer: ' + str(my_input.size()))
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
