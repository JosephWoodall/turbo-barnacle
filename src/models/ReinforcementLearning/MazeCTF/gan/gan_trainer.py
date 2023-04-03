import torch


class GanTrainer:
    """ """
    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.criterion = criterion

    def train_step(self, real_images, noise):
        """

        :param real_images: 
        :param noise: 

        """
        # Train the discriminator
        self.discriminator_optimizer.zero_grad()
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones_like(real_preds)
        real_loss = self.criterion(real_preds, real_targets)
        fake_images = self.generator(noise).detach()
        fake_preds = self.discriminator(fake_images)
        fake_targets = torch.zeros_like(fake_preds)
        fake_loss = self.criterion(fake_preds, fake_targets)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Train the generator
        self.generator_optimizer.zero_grad()
        fake_images = self.generator(noise)
        fake_preds = self.discriminator(fake_images)
        generator_loss = self.criterion(fake_preds, real_targets)
        generator_loss.backward()
        self.generator_optimizer.step()

        return discriminator_loss.item(), generator_loss.item()
