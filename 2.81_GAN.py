"""
Пример реализации генеративно-состязательной сети (GAN) с использованием
библиотеки PyTorch для обучения на наборе данных MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import МNIST

# Определение генератора
class Generator(nn.Module):

    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNormld(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNormld(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNormld(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_shape),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.image_shape)
        return img

# Определение дискриминатора
class Discriminator(nn.Module):

    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape

        self.model = nn.Sequential(
            nn.Linear(image_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Параметры
latent_dim = 100
image_shape = (28, 28)
lr = 0.0002
batch_size = 64
epochs = 10

# Инициализация генератора и дискриминатора
generator = Generator(latent_dim, image_shape)
discriminator = Discriminator(image_shape)

# Оптимизаторы
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
adversarial_loss = nn.BCELoss()

# Обучение GAN
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(data_loader):

        # Генерация шума
        z = torch.randn(batch_size, latent_dim)

        # Генерация изображений
        gen_imgs = generator(z)

        # Обучение дискриминатора
        real_loss = adversarial_loss(discriminator(imgs), torch.ones(batch_size, 1))
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), torch.zeros(batch_size, 1))
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Обучение генератора
        gen_loss = adversarial_loss(discriminator(gen_imgs), torch.ones(batch_size, 1))

        optimizer_G.zero_grad()
        gen_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {gen_loss.item()}]")
