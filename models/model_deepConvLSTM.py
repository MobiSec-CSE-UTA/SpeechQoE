import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch._VF as VF

sys.path.append('..')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_channel_dim = 1


class MAML_BN(nn.Module):
    def __init__(self, opt):

        super(MAML_BN, self).__init__()
        self.opt = opt

        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        conv1 = nn.Conv1d(input_channel_dim, 32, kernel_size=2)
        bn1 = nn.BatchNorm1d(32)
        conv2 = nn.Conv1d(32, 64, kernel_size=1)
        bn2 = nn.BatchNorm1d(64)
        conv3 = nn.Conv1d(64, 128, kernel_size=1)
        bn3 = nn.BatchNorm1d(128)
        lstm1 = nn.LSTM(128, self.opt['num_class'])

        layers = [('conv', conv1), ('bn', bn1), ('conv', conv2), ('bn', bn2), ('conv', conv3), ('bn', bn3),
                  ('lstm', lstm1)]

        for (name, layer) in layers:
            if name == 'bn':
                w = nn.Parameter(torch.ones_like(layer.weight))
                self.vars.append(w)
                b = nn.Parameter(torch.zeros_like(layer.bias))
                self.vars.append(b)
                running_mean = nn.Parameter(torch.zeros_like(layer.weight), requires_grad=False)
                running_var = nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name == 'conv':
                w = nn.Parameter(torch.ones_like(layer.weight))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                b = nn.Parameter(torch.zeros_like(layer.bias))
                self.vars.append(b)

            elif name == 'lstm':
                for weight in layer._flat_weights:
                    w = torch.zeros_like(weight)
                    w.data = weight.clone()
                    w = nn.Parameter(w)
                    self.vars.append(w)

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

        # LSTM layer
        batch_size = my_input.shape[0]
        hx = (torch.zeros(1, batch_size, self.opt['num_class']).to(device),
              torch.zeros(1, batch_size, self.opt['num_class']).to(device))
        flat_weights = [my_vars[idx], my_vars[idx + 1], my_vars[idx + 2], my_vars[idx + 3]]

        lstm_out = VF.lstm(my_input.permute(2, 0, 1), hx, flat_weights, has_biases=True, num_layers=1, dropout=0.0,
                           train=True,
                           bidirectional=False,
                           batch_first=False)

        idx += 4
        my_input, hidden_out = lstm_out[0], lstm_out[1:]
        my_input = my_input[-1]
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
