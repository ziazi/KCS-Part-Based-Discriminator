import torch
import torch.nn as nn
import bonePos as Pos


class KCSi_discriminator(nn.Module):
    def __init__(self, activation, in_features, channel, mid_channel, predict_with_sigmoid=True):
        super(KCSi_discriminator, self).__init__()
        self.residual_layer_1 = nn.Linear(in_features, channel, bias=True)
        self.residual_layer_2 = nn.Linear(channel, channel, bias=True)
        self.residual_layer_3 = nn.Linear(channel, channel, bias=True)

        self.mlp_layer_1 = nn.Linear(channel, mid_channel, bias=True)
        self.mlp_layer_pred = nn.Linear(mid_channel, 1, bias=False)
        self.activation = activation
        self.predict_with_sigmoid = predict_with_sigmoid

    def forward(self, x):
        res1 = self.activation(self.residual_layer_1(x))
        res2 = self.activation(self.residual_layer_2(res1))
        res3 = self.activation(self.residual_layer_3(res2) + res1)
        mlp_1 = self.activation(self.mlp_layer_1(res3))
        if self.predict_with_sigmoid:
            mlp_pred = nn.Sigmoid()(self.mlp_layer_pred(mlp_1))
        else:
            mlp_pred = self.activation(self.mlp_layer_pred(mlp_1))

        return mlp_pred


class Pos3dDiscriminator(nn.Module):
    def __init__(self, num_joints=17, channel=1000, mid_channel=100, activation="LeakyRelu"):
        super(Pos3dDiscriminator, self).__init__()
        functions = {"LeakyRelu": nn.LeakyReLU(), "Relu": nn.ReLU(), "Sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh()}
        activation = functions[activation]
        self.kcs_util = Pos.KCS_util(num_joints)
        features = self.kcs_util.compute_features()
        self.kcsi_ll = KCSi_discriminator(activation, features, channel, mid_channel)
        self.kcsi_rl = KCSi_discriminator(activation, features, channel, mid_channel)
        self.kcsi_torso = KCSi_discriminator(activation, features, channel, mid_channel)
        self.kcsi_lh = KCSi_discriminator(activation, features, channel, mid_channel)
        self.kcsi_rh = KCSi_discriminator(activation, features, channel, mid_channel)
        self.optimizer = None

    def forward(self, inputs_3d):
        ext_inputs_3d = self.kcs_util.extend(inputs_3d)
        ext_inputs_3d = self.kcs_util.center(ext_inputs_3d)
        bv = self.kcs_util.bone_vectors(ext_inputs_3d)
        kcs_ll = self.kcs_util.kcs_layer(bv, "ll")
        kcs_rl = self.kcs_util.kcs_layer(bv, "rl")
        kcs_torso = self.kcs_util.kcs_layer(bv, "torso")
        kcs_lh = self.kcs_util.kcs_layer(bv, "lh")
        kcs_rh = self.kcs_util.kcs_layer(bv, "rh")

        ll_pred = self.kcsi_ll(kcs_ll.view((inputs_3d.size(0), -1)))
        rl_pred = self.kcsi_rl(kcs_rl.view((inputs_3d.size(0), -1)))
        torso_pred = self.kcsi_torso(kcs_torso.view((inputs_3d.size(0), -1)))
        lh_pred = self.kcsi_lh(kcs_lh.view((inputs_3d.size(0), -1)))
        rh_pred = self.kcsi_rh(kcs_rh.view((inputs_3d.size(0), -1)))

        return torch.stack([ll_pred, rl_pred, torso_pred, lh_pred, rh_pred])

    def predict(self, input_3d, threshold):
        network_pred = self.forward(input_3d.repeat([1, 1, 1]))
        network_pred[network_pred < threshold] = 0
        network_pred[network_pred >= threshold] = 1
        network_pred = torch.prod(network_pred)
        return network_pred

    def set_optimizer(self, optimizer="SGD", lr=0.01, momentum=0.9, eps=1e-10, weight_decay=0.0):
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return self

    def step(self, inputs_3d, labels, loss_fn):
        assert (self.optimizers is not None)
        self.optimizer.zero_grad()
        batch_loss = loss_fn(self.forward(inputs_3d), labels)
        batch_loss.backward()
        self.optimizer.step()
        return self
