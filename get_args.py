import argparse
def get_args(know_args=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--note', dest='note', type=str, default="", help='note what you want')
    """ Data Setting """
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    
    """ Mask Setting """
    parser.add_argument('--mask_type', dest='mask_type', type=str, default="grid", help='grid, tri')
    parser.add_argument('--varmap_type', dest='varmap_type', type=str, default="notuse", help='notuse, var(warp), warp(var), small_grid')
    parser.add_argument('--varmap_threshold', dest='varmap_threshold', type=float, default=0.7, help='0 to 1 , if -1: not use')
    # if ksize = 0 will calculate by sigma , see more detail on cv2. # kszie1 = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;  # 但有點不太懂為什麼是這個公式
    parser.add_argument('--guassian_ksize', dest='guassian_ksize',type=int, default=5, help='guassian_blur kernel size') 
    parser.add_argument('--guassian_sigma', dest='guassian_sigma',type=float, default=0.0, help='guassian_blur sigma')
    parser.add_argument('--guassian_blur', dest='guassian_blur',default=False, action="store_true", help='mask output with guassian_blur or not')

    """ Train Setting """
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0,1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--load_name', type = str, default = '', help = 'load model name')
    parser.add_argument('--epochs', type = int, default = 40, help = 'number of epochs of training')
    # parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_epoch', type = int, default = 0)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    
    # optimizer setting :
    parser.add_argument('--lr_g', type = float, default = 2e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 2e-4, help = 'Adam: learning rate')
    # parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for optimizer')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    
    # loss weight setting
    parser.add_argument('--lambda_l1', type = float, default = 100, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 10, help = 'the parameter of FML1Loss (perceptual loss)')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    
    # Model Setting
    
    parser.add_argument('--D_iter', dest='D_iter', type=int, default=5, help='d iter per batch')
    parser.add_argument('--G_iter', dest='G_iter', type=int, default=1, help='g iter per batch')
    parser.add_argument('--type', dest='type', type=str, default="wgangp", help='GAN LOSS TYPE')
    parser.add_argument('--gp_lambda', dest='gp_lambda', type=int, default=10, help='Gradient penalty lambda hyperparameter')
    
    parser.add_argument('--mask_weight', dest='mask_weight', type=float, default=1.0, help='weight of mask_loss')
    parser.add_argument('--matt_weight', dest='matt_weight', type=float, default=100.0, help='weight of matt_loss')

    parser.add_argument('--in_out_area_split', dest='in_out_area_split',default=False, action="store_true", help='split to in_area_mask_loss and out_area_mask_loss')
    parser.add_argument('--in_area_weight', dest='in_area_weight', type=float, default=0.5, help='in_area_mask weight')
    parser.add_argument('--out_area_weight', dest='out_area_weight', type=float, default=0.5, help='out_area_mask weight')
    
    # Network parameters
    # MaskEstimator
    parser.add_argument('--backbone', dest='backbone', type=str, default="convnext_base_in22k", help='models in timm')
    # InpaintGenerator from Deepfillv2
    parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 48, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    
    # Log Setting
    parser.add_argument('--data_dir',dest='data_dir',type=str, help="warppeData dir")
    # parser.add_argument('--train_dir',dest='train_dir',type=str,default="./data/celebAHQ/train_data",help="train dataset dir")
    # parser.add_argument('--val_dir',dest='val_dir',type=str,default="./data/celebAHQ/val_data",help="val dataset dir")
    parser.add_argument('--log_dir', dest='log_dir', default='./log/', help='log dir')
    parser.add_argument('--wandb',default=False, action="store_true")

    args = parser.parse_args(know_args) if know_args != None else parser.parse_args()
    return args


 
 
   
    
   