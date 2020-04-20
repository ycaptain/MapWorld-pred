
class CycleGANOptions:
    def __init__(self):
        self.dataroot = None
        self.name = "cyclegan-exp"
        self.gpu_ids = 0
        self.model = "cycle_gan"
        self.input_nc = 3  # of input image channels: 3 for RGB and 1 for grayscale
        self.output_nc = 3  # of output image channels: 3 for RGB and 1 for grayscale
        self.ngf = 64  # of gen filters in the last conv layer
        self.ndf = 64  # of discrim filters in the first conv layer
        self.netD = "basic"  # specify discriminator architecture [basic | n_layers | pixel]. The basic model is a
        # 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
        self.netG = "resnet_6blocks"  # specify generator architecture
        # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.n_layers_D = 3  # only used if netD==n_layers
        self.norm = "batch"  # instance normalization or batch normalization [instance | batch | none]
        self.init_type = "normal"  # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.02  # scaling factor for normal, xavier and orthogonal.
        self.no_dropout = False  # no dropout for the generator
