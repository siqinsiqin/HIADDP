from __future__ import absolute_import, division, print_function

from collections import OrderedDict

# import pytorch_lightning as pl
import torch
from monai.metrics import DiceMetric
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv
from monai.transforms import AsDiscrete, Compose, EnsureType
from ptflops import get_model_complexity_info
from torch import nn

# Pytorch Lightning module
from models.models_3d.mipt.ucaps.layers import ConvSlimCapsule3D


class UCaps3D(nn.Module):
    def __init__(
            self,
            in_channels=2,
            out_channels=4,
            lr_rate=2e-4,
            rec_loss_weight=0.1,
            margin_loss_weight=1.0,
            class_weight=None,
            share_weight=False,
            sw_batch_size=128,
            cls_loss="CE",
            val_patch_size=(32, 32, 32),
            overlap=0.75,
            connection="skip",
            val_frequency=100,
            weight_decay=2e-6,
            **kwargs,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_weight = share_weight
        self.connection = connection
        #
        # self.lr_rate = self.hparams.lr_rate
        # self.weight_decay = self.hparams.weight_decay
        #
        # self.cls_loss = self.hparams.cls_loss
        # self.margin_loss_weight = self.hparams.margin_loss_weight
        # self.rec_loss_weight = self.hparams.rec_loss_weight
        # self.class_weight = self.hparams.class_weight

        # # Defining losses
        # self.classification_loss1 = MarginLoss(class_weight=self.class_weight, margin=0.2)
        #
        # if self.cls_loss == "DiceCE":
        #     self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, ce_weight=self.class_weight)
        # elif self.cls_loss == "CE":
        #     self.classification_loss2 = DiceCELoss(
        #         softmax=True, to_onehot_y=True, ce_weight=self.class_weight, lambda_dice=0.0
        #     )
        # elif self.cls_loss == "Dice":
        #     self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=0.0)
        # self.reconstruction_loss = nn.MSELoss(reduction="none")
        #
        # self.val_frequency = self.hparams.val_frequency
        # self.val_patch_size = self.hparams.val_patch_size
        # self.sw_batch_size = self.hparams.sw_batch_size
        # self.overlap = self.hparams.overlap

        # Building model
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            dimensions=3,
                            in_channels=self.in_channels,
                            out_channels=16,
                            kernel_size=5,
                            strides=1,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (
                        "conv2",
                        Convolution(
                            dimensions=3,
                            in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            strides=1,
                            dilation=2,
                            padding=4,
                            bias=False,
                        ),
                    ),
                    (
                        "conv3",
                        Convolution(
                            dimensions=3,
                            in_channels=32,
                            out_channels=64,
                            kernel_size=5,
                            strides=1,
                            padding=4,
                            dilation=2,
                            bias=False,
                            act="tanh",
                        ),
                    ),
                ]
            )
        )

        self.primary_caps = ConvSlimCapsule3D(
            kernel_size=3,
            input_dim=1,
            output_dim=16,
            input_atoms=64,
            output_atoms=4,
            stride=1,
            padding=1,
            num_routing=1,
            share_weight=self.share_weight,
        )
        self._build_encoder()
        self._build_decoder()
        self._build_reconstruct_branch()

        # For validation
        self.post_pred = Compose(
            [EnsureType(), AsDiscrete(argmax=True, to_onehot=self.out_channels, n_classes=self.out_channels)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=self.out_channels, n_classes=self.out_channels)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        self.example_input_array = torch.rand(1, self.in_channels, 32, 32, 32)

    def forward(self, x):
        # Contracting
        x = self.feature_extractor(x)
        x = x.unsqueeze(dim=1)
        conv_cap_1_1 = self.primary_caps(x)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_3_1.size()
        conv_cap_3_1 = conv_cap_3_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_2_1.size()
        conv_cap_2_1 = conv_cap_2_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_1_1.size()
        conv_cap_1_1 = conv_cap_1_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])

        # Expanding
        if self.connection == "skip":
            x = self.decoder_conv[0](conv_cap_4_1)
            x = torch.cat((x, conv_cap_3_1), dim=1)
            x = self.decoder_conv[1](x)
            x = self.decoder_conv[2](x)
            x = torch.cat((x, conv_cap_2_1), dim=1)
            x = self.decoder_conv[3](x)
            x = self.decoder_conv[4](x)
            x = torch.cat((x, conv_cap_1_1), dim=1)

        logits = self.decoder_conv[5](x)

        return logits

    def _build_encoder(self):
        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 3
        self.encoder_output_dim = [16, 16, 8, 8, 8, self.out_channels]
        self.encoder_output_atoms = [8, 8, 16, 16, 32, 64]

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = self.primary_caps.output_dim
                input_atoms = self.primary_caps.output_atoms
            else:
                input_dim = self.encoder_output_dim[i - 1]
                input_atoms = self.encoder_output_atoms[i - 1]

            stride = 2 if i % 2 == 0 else 1

            self.encoder_conv_caps.append(
                ConvSlimCapsule3D(
                    kernel_size=self.encoder_kernel_size,
                    input_dim=input_dim,
                    output_dim=self.encoder_output_dim[i],
                    input_atoms=input_atoms,
                    output_atoms=self.encoder_output_atoms[i],
                    stride=stride,
                    padding=1,
                    dilation=1,
                    num_routing=3,
                    share_weight=self.share_weight,
                )
            )

    def _build_decoder(self):
        self.decoder_conv = nn.ModuleList()
        if self.connection == "skip":
            self.decoder_in_channels = [self.out_channels * self.encoder_output_atoms[-1], 384, 128, 256, 64, 128]
            self.decoder_out_channels = [256, 128, 128, 64, 64, self.out_channels]

        for i in range(6):
            if i == 5:
                self.decoder_conv.append(
                    Conv["conv", 3](self.decoder_in_channels[i], self.decoder_out_channels[i], kernel_size=1)
                )
            elif i % 2 == 0:
                self.decoder_conv.append(
                    UpSample(
                        dimensions=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        scale_factor=2,
                    )
                )
            else:
                self.decoder_conv.append(
                    Convolution(
                        dimensions=3,
                        kernel_size=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        strides=1,
                        padding=1,
                        bias=False,
                    )
                )

    def _build_reconstruct_branch(self):
        self.reconstruct_branch = nn.Sequential(
            nn.Conv3d(self.decoder_in_channels[-1], 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, self.in_channels, 1),
            # nn.Sigmoid(),
        )


if __name__ == '__main__':
    # todo ThreeD-UCaps: ThreeD Capsules Unet for Volumetric Image Segmentation
    x = torch.randn((1, 2, 64, 64, 64))
    model = UCaps3D(in_channels=2, out_channels=1)
    SIZE = 64
    macs, params = get_model_complexity_info(model, (2, SIZE, SIZE, SIZE), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print(y)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
