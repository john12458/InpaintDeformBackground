import argparse
def get_args(know_args=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--note', dest='note', type=str, default="", help='note what you want')
    # Mask Setting
    parser.add_argument('--mask_type', dest='mask_type', type=str, default="grid", help='grid, tri')
    parser.add_argument('--varmap_type', dest='varmap_type', type=str, default="notuse", help='notuse, var(warp), warp(var), small_grid')
    parser.add_argument('--varmap_threshold', dest='varmap_threshold', type=float, default=0.7, help='0 to 1 , if -1: not use')

    parser.add_argument('--guassian_blur', dest='guassian_blur',default=False, action="store_true", help='mask output with guassian_blur or not')

    
    # if ksize = 0 will calculate by sigma , see more detail on cv2.
    # kszie1 = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;  # 但有點不太懂為什麼是這個公式
    parser.add_argument('--guassian_ksize', dest='guassian_ksize',type=int, default=5, help='guassian_blur kernel size')
    parser.add_argument('--guassian_sigma', dest='guassian_sigma',type=float, default=0.0, help='guassian_blur sigma')
    
    
    
    
    # train Setting
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for optimizer')
    
    # Model Setting
    parser.add_argument('--backbone', dest='backbone', type=str, default="convnext_base_in22k", help='models in timm')
    parser.add_argument('--use_attention', dest='use_attention',default=False, action="store_true", help='add attention layers')

    
    parser.add_argument('--D_iter', dest='D_iter', type=int, default=5, help='d iter per batch')
    parser.add_argument('--G_iter', dest='G_iter', type=int, default=1, help='g iter per batch')
    parser.add_argument('--type', dest='type', type=str, default="wgangp", help='GAN LOSS TYPE')
    parser.add_argument('--gp_lambda', dest='gp_lambda', type=int, default=10, help='Gradient penalty lambda hyperparameter')
    
    parser.add_argument('--mask_weight', dest='mask_weight', type=float, default=1.0, help='weight of mask_loss')
    parser.add_argument('--matt_weight', dest='matt_weight', type=float, default=100.0, help='weight of matt_loss')
    parser.add_argument('--regularzation_weight', dest='regularzation_weight', type=float, default=0.05, help='weight of regularzation')

    parser.add_argument('--in_out_area_split', dest='in_out_area_split',default=False, action="store_true", help='split to in_area_mask_loss and out_area_mask_loss')
    parser.add_argument('--in_area_weight', dest='in_area_weight', type=float, default=0.5, help='in_area_mask weight')
    parser.add_argument('--out_area_weight', dest='out_area_weight', type=float, default=0.5, help='out_area_mask weight')



    # Dir
    parser.add_argument('--data_dir',dest='data_dir',type=str, help="warppeData dir")
    # parser.add_argument('--train_dir',dest='train_dir',type=str,default="./data/celebAHQ/train_data",help="train dataset dir")
    # parser.add_argument('--val_dir',dest='val_dir',type=str,default="./data/celebAHQ/val_data",help="val dataset dir")
    parser.add_argument('--log_dir', dest='log_dir', default='./log/', help='log dir')
   
    # Other
    parser.add_argument('--wandb',default=False, action="store_true")

    args = parser.parse_args(know_args) if know_args != None else parser.parse_args()
    return args
