import torch
import torch.nn as nn
from core.util import idx2onehot


class VAE(nn.Module):

    def __init__(self, encoder_sizes, latent_size, decoder_sizes,
                 conditional=False, num_labels=0, device='cpu'):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_sizes) == list

        self.encoder = Encoder(
            encoder_sizes, latent_size, conditional, num_labels, device)
        self.decoder = Decoder(
            decoder_sizes, latent_size, conditional, num_labels, device)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, z, c=None):

        if c is not None:
            assert z.size(0) == c.size(0)

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size,
                 conditional=False, num_labels=0, device='cpu'):

        super().__init__()

        self.device = device
        self.num_labels = num_labels
        self.conditional = conditional
        layer_sizes.insert(0, 784)
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="full_connection_{:d}".format(i),
                module=nn.Linear(in_size, out_size)
            )
            self.MLP.add_module(name="activate_{:d}".format(i), module=nn.ReLU())

        self.linear_mean = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels, device=self.device)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_mean(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size,
                 conditional=False, num_labels=0, device='cpu'):

        super().__init__()

        self.device = device
        self.num_labels = num_labels
        layer_sizes.append(784)
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1],
                                                    layer_sizes)):
            self.MLP.add_module(
                name="full_connection_{:d}".format(i),
                module=nn.Linear(in_size, out_size)
            )
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="activate_{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="Sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels, device=self.device)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
