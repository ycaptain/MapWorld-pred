import torch


class CycleGANOptions:
    def __init__(self, model="cycle_gan", input_nc=3, output_nc=3, ngf=64, ndf=64, netD="basic", netG="resnet_6blocks",
                 n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02, no_dropout=False, gpu_ids="0",
                 lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5, pool_size=50, gan_mode="lsgan", lr=0.0002,
                 beta1=0.5, lr_policy='linear', n_epochs=100, n_epochs_decay=100, lr_decay_iters=50, epoch_count=1):
        str_ids = gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id > 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

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

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity
        self.pool_size = pool_size  # the size of image buffer that stores previously generated images
        self.gan_mode = gan_mode
        self.lr = lr  # initial learning rate for adam
        self.beta1 = beta1  # momentum term of adam

        self.lr_policy = lr_policy
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.lr_decay_iters = lr_decay_iters
        self.epoch_count = epoch_count
