
class CycleGANOptions:
    def __init__(self, model="cycle_gan", input_nc=3, output_nc=3, ngf=64, ndf=64, netD="basic", netG="resnet_6blocks",
                 n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02, no_dropout=False):
        self.model = model
        self.input_nc = input_nc  # of input image channels: 3 for RGB and 1 for grayscale
        self.output_nc = output_nc  # of output image channels: 3 for RGB and 1 for grayscale
        self.ngf = ngf  # of gen filters in the last conv layer
        self.ndf = ndf  # of discrim filters in the first conv layer
        self.netD = netD  # specify discriminator architecture [basic | n_layers | pixel]. The basic model is a
        # 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
        self.netG = netG  # specify generator architecture
        # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.n_layers_D = n_layers_D  # only used if netD==n_layers
        self.norm = norm  # instance normalization or batch normalization [instance | batch | none]
        self.init_type = init_type  # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_gain = init_gain  # scaling factor for normal, xavier and orthogonal.
        self.no_dropout = no_dropout  # no dropout for the generator
