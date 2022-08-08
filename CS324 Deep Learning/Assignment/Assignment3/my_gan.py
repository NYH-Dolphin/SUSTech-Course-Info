import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image


class Generator(nn.Module):

    def __init__(self, latent_dim):
        """
        :param latent_dim: 输入图片维度，这里输入的是一个MINST 手写数据集，code z 可以是一个 10x1 维的向量，表示数字 0-9
        """
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 256)

        self.BN2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.BN3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 784)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

        # 3x32x32   -> 1x28x28
        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

    def forward(self, z):
        # Generate images from z
        z = self.leaky_relu1(self.fc1(z))
        z = self.leaky_relu2(self.BN2(self.fc2(z)))
        z = self.leaky_relu3(self.BN3(self.fc3(z)))
        z = self.fc4(z)
        return torch.tanh(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

    def forward(self, img):
        # return discriminator score for img
        img = self.leaky_relu1(self.fc1(img))
        img = self.leaky_relu2(self.fc2(img))
        img = self.fc3(img)
        return torch.sigmoid(img)


def modify_scale(imgs):
    out = 0.5 * (imgs + 1)  # 将x的范围由(-1,1)伸缩到(0,1)
    out = out.view(-1, 1, 28, 28)
    return out


def tensor2img(imgs_tensor):
    """
        :param imgs_tensor: (batch_size, 784)
        :return: 转换成 (batch_size, 1, 28, 28)的大小
    """
    img = 0.5 * (imgs_tensor + 1)  # 将x的范围由(-1,1)伸缩到(0,1)
    img = img.view(args.batch_size, 1, 28, 28)
    return img


def train(dataloader, D, G, optimizer_G, optimizer_D):
    criterion = nn.BCELoss()
    for epoch in range(args.n_epochs):
        print('epoch{0}'.format(epoch))
        G.train()
        D.train()
        sum_loss_G = 0
        sum_loss_D = 0
        for i, (imgs, _) in enumerate(dataloader):
            # imgs.cuda()

            # Train Discriminator
            # -------------------
            for _ in range(5):
                # 真实的部分
                imgs_tensor = torch.flatten(imgs, start_dim=1)
                result_truth = D(imgs_tensor)  # 真实的计算结果
                real_labels = torch.ones_like(result_truth, dtype=torch.float)  # 真实的标签，全部为1
                real_loss = criterion(result_truth, real_labels)

                # 虚假的部分
                z = torch.randn((args.batch_size, args.latent_dim))
                imgs_generate = G(z)
                result_generate2 = D(imgs_generate.detach())  # 生成器生成图片的计算结果
                fake_labels = torch.zeros_like(result_generate2, dtype=torch.float)  # 生成器的标签 fake，全部为0
                fake_loss = criterion(result_generate2, fake_labels)

                loss_D = real_loss + fake_loss
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                sum_loss_D += loss_D.item()

            # Train Generator
            # -------------------
            gen_imgs = None
            for _ in range(2):
                z = torch.randn((args.batch_size, args.latent_dim), requires_grad=True)
                imgs_generate = G(z)
                result_generate = D(imgs_generate)
                real_labels = torch.ones_like(result_generate, dtype=torch.float)  # 生成器的标签 fake，全部为0
                loss_G = criterion(result_generate, real_labels)
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                sum_loss_G += loss_G.item()
                gen_imgs = tensor2img(imgs_generate)

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_imgs[:25], 'images/{}.png'.format(batches_done), nrow=5, normalize=True)
                print("save_image")
        print('G loss: {0}, D loss: {1}'.format(sum_loss_G, sum_loss_D))


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
