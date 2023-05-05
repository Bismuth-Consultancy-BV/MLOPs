#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2022 Erik Linder-Norén and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions.

import argparse
import os
import time
import datetime
import sys
import numpy
import torch

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from modeling_pix2pix import GeneratorUNet, Discriminator
from datasets import load_dataset
from accelerate import Accelerator
from torch import nn



input_dataset = "huggan/facades" #"Dataset to use"
starting_epoch = 0 #"epoch to start training from")
total_epochs = 500 #"number of epochs of training"
batch_size = 1 #"size of the batches"
lr = 0.0002 #"adam: learning rate"
b1 = 0.5 #"adam: decay of first order momentum of gradient"
b2 = 0.999 #"adam: decay of first order momentum of gradient"
decay_epoch = 100 #"epoch from which to start lr decay"
image_size = 256 #"size of images for training"
sample_interval = 500 #"interval between sampling of images from generators"
checkpoint_interval = -1 #"interval between model checkpoints"
mixed_precision = "no" #["no", "fp16", "bf16"]
use_cpu = False #"If passed, will train on the CPU."
lambda_L1 = 100 # Loss weight of L1 pixel-wise loss between translated image and real image

# Custom weights initialization called on Generator and Discriminator
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def initialize_networks(netG, netD):
    if starting_epoch != 0:
        # Load pretrained models
        netG.load_state_dict(torch.load(f"saved_models/{input_dataset}/generator_{starting_epoch}.pth"))
        netD.load_state_dict(torch.load(f"saved_models/{input_dataset}/discriminator_{starting_epoch}.pth"))
    else:
        # Initialize weights
        netG.apply(weights_init_normal)
        netD.apply(weights_init_normal)
    return netG, netD

def transforms(examples):
    # Configure dataloaders
    transform = Compose(
            [
                Resize((image_size, image_size), Image.BICUBIC),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    # random vertical flip
    imagesA = []
    imagesB = []
    for imageA, imageB in zip(examples['imageA'], examples['imageB']):
        if numpy.random.random() < 0.5:
            imageA = Image.fromarray(numpy.array(imageA)[:, ::-1, :], "RGB")
            imageB = Image.fromarray(numpy.array(imageB)[:, ::-1, :], "RGB")
        imagesA.append(imageA)
        imagesB.append(imageB)

    # transforms
    examples["A"] = [transform(image.convert("RGB")) for image in imagesA]
    examples["B"] = [transform(image.convert("RGB")) for image in imagesB]

    del examples["imageA"]
    del examples["imageB"]

    return examples

def sample_images(batches_done, accelerator, val_dataloader, netG):
    """Saves a generated sample from the validation set"""
    batch = next(iter(val_dataloader))
    real_A = batch["A"]
    real_B = batch["B"]
    fake_B = netG(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    if accelerator.is_main_process:
        save_image(img_sample, f"images/{batches_done}.png", nrow=5, normalize=True)


def training_function():
    accelerator = Accelerator(cpu=use_cpu, mixed_precision=mixed_precision)

    os.makedirs("images/", exist_ok=True)
    os.makedirs("saved_models/", exist_ok=True)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, image_size // 2 ** 4, image_size // 2 ** 4)

    # Define Networks
    netG = GeneratorUNet()
    netD = Discriminator()

    # Define Loss functions
    criterionGAN = torch.nn.MSELoss() #TODO: CHECK
    criterionL1 = torch.nn.L1Loss()

    # Define Optimizers
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(b2, b2))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(b2, b2))

    netG, netD = initialize_networks(netG, netD)

    dataset = load_dataset(input_dataset)
    transformed_dataset = dataset.with_transform(transforms)

    splits = transformed_dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    dataloader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0)
    val_dataloader = DataLoader(val_ds, batch_size=10, shuffle=True, num_workers=0)

    netG, netD, optimizer_G, optimizer_D, dataloader, val_dataloader = accelerator.prepare(netG, netD, optimizer_G, optimizer_D, dataloader, val_dataloader)

    # ----------
    #  Training
    # ----------

            # def forward():
            #     fake_B = netG(real_A)

            # def backward_D():
            #     pred_fake = netD(fake_B.detach(), real_A)
            #     loss_D_fake = criterionGAN(pred_fake, fake)

            #     pred_real = netD(real_B, real_A)
            #     loss_D_real = criterionGAN(pred_real, valid)

            #     loss_D = (loss_D_fake + loss_D_real) * 0.5
            #     accelerator.backward(loss_D)

            # def backward_G():
            #     pred_fake = netD(fake_B, real_A)
            #     loss_G_GAN = criterionGAN(pred_fake, valid)

            #     loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            #     loss_G = loss_G_GAN + loss_G_L1
            #     accelerator.backward(loss_G)

            # def optimize_parameters():
            #     forward()
            #     # Update D
            #     optimizer_D.zero_grad()
            #     backward_D()
            #     optimizer_D.step()
            #     # Update G
            #     optimizer_G.zero_grad()
            #     backward_G()
            #     optimizer_G.step()


    prev_time = time.time()

    for epoch in range(starting_epoch, total_epochs):
        print("")
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = batch["A"]
            real_B = batch["B"]
            fake = torch.zeros((real_A.size(0), *patch), device=accelerator.device)
            valid = torch.ones((real_A.size(0), *patch), device=accelerator.device)

            # def forward():
            fake_B = netG(real_A)

            # Update D
            optimizer_D.zero_grad()

            # def backward_D():
            pred_fake = netD(fake_B.detach(), real_A)
            loss_D_fake = criterionGAN(pred_fake, fake)
            pred_real = netD(real_B, real_A)
            loss_D_real = criterionGAN(pred_real, valid)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            accelerator.backward(loss_D)

            # Update D
            optimizer_D.step()
            # Update G
            optimizer_G.zero_grad()

            # def backward_G():
            pred_fake = netD(fake_B, real_A)
            loss_G_GAN = criterionGAN(pred_fake, valid)
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            accelerator.backward(loss_G)

            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = total_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch+1,
                    total_epochs,
                    i+1,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_G_L1.item(),
                    loss_G_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done, accelerator, val_dataloader, netG)

        # if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        #     if accelerator.is_main_process:
        #         unwrapped_generator = accelerator.unwrap_model(netG)
        #         unwrapped_discriminator = accelerator.unwrap_model(netD)
        #         # Save model checkpoints
        #         torch.save(unwrapped_generator.state_dict(), "saved_models/%s/generator_%d.pth" % (input_dataset, epoch))
        #         torch.save(unwrapped_discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (input_dataset, epoch))

    if accelerator.is_main_process:
        unwrapped_generator = accelerator.unwrap_model(netG)
        unwrapped_discriminator = accelerator.unwrap_model(netD)

        torch.save(unwrapped_generator.state_dict(), f"saved_models/generator.pth")
        torch.save(unwrapped_discriminator.state_dict(), f"saved_models/discriminator.pth")


training_function()
