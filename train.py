import argparse
import os
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from core.util import set_random_seed
from core.network import VAE


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld


def train_epoch(model, optimizer, train_loader, device, args, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        if args.conditional:
            recon_batch, mu, logvar, _ = model(data, label)
        else:
            recon_batch, mu, logvar, _ = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data))
            )

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test_epoch(model, test_loader, device, args, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            if args.conditional:
                recon_batch, mu, logvar, _ = model(data, label)
            else:
                recon_batch, mu, logvar, _ = model(data)

            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(
                        args.batch_size, 1, 28, 28)[:n]]
                )
                if not (os.path.exists(args.fig_dir)):
                    os.mkdir(args.fig_dir)
                save_image(
                    comparison.cpu(),
                    os.path.join(args.fig_dir,
                                 'reconstruction_' + str(epoch) + '.png'
                                 ), nrow=n
                )

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def train(model, optimizer, train_loader, test_loader, device, args):
    for epoch in range(1, args.num_epochs + 1):
        train_epoch(model, optimizer, train_loader, device, args, epoch)
        test_epoch(model, test_loader, device, args, epoch)
        with torch.no_grad():
            z = torch.randn(64, args.latent_size, device=device)

            if args.conditional:
                c = torch.randint(0, 10, [z.size(0), 1], device=device)
                sample = model.inference(z, c).cpu()
            else:
                sample = model.inference(z).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                os.path.join(args.fig_dir, 'sample_' + str(epoch) + '.png')
            )


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--data-dir', type=str, default='./data', metavar='D',
                        help='directory of training and testing data (default: ./data)')
    parser.add_argument('--fig-dir', type=str, default='fig',
                        help='directory to store output figures (default: ./fig)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='set learning rate (default: 0.001)')
    parser.add_argument('--encoder-sizes', nargs='+', type=int, default=[256],
                        help='hidden sizes of encoder layers (default: [256])')
    parser.add_argument('--decoder-sizes', nargs='+', type=int, default=[256],
                        help='hidden sizes of decoder layers (default: [256])')
    parser.add_argument('--latent-size', type=int, default=64,
                        help='the dimension of latent variable')
    parser.add_argument('--conditional', action='store_true', default=False,
                        help='activate conditional VAE (CVAE)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='L',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    set_random_seed(args)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_dir, train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VAE(
        encoder_sizes=args.encoder_sizes,
        latent_size=args.latent_size,
        decoder_sizes=args.decoder_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0,
        device=device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, train_loader, test_loader,
          device, args)


if __name__ == '__main__':
    main()
