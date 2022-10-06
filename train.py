import configargparse
import os
import torch.multiprocessing as mp
import torch
from main_semi_v1 import main


def get_argparser():
    # parser = argparse.ArgumentParser()
    config_path = "/mnt/zhangyunyang/deeplabv3/config_semi.yml"
    parser = configargparse.ArgParser(default_config_files=[str(config_path)], description="Hyper-parameters.")

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/mnt/zhangyunyang/TorchSemiSeg/DATA',
                        help="path to Dataset")
    parser.add_argument("--labeled_ratio", type=int, default=8,
                        help="ratio of labeled set")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--name", type=str, default=None,
                        help='Name of saved log')

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--num_members", type=int, default=2, help="num_members > 1")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--dist", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--epoch", type=int, default=52,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--unl_batch_size", type=int, default=16,
                        help='batch size for unlabeled data(default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--base_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="epoch interval for eval (default: 100)")
    return parser


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


if __name__ == '__main__':
    opts = get_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    if opts.dist:
        opts.nprocs = torch.cuda.device_count()
        port = find_free_port()
        opts.dist_url = f"tcp://127.0.0.1:{port}"
        opts.nodes = 0
        opts.world_size = opts.nprocs
        mp.spawn(main, nprocs=opts.nprocs, args=(opts.nprocs, opts))
    else:
        opts.nodes = 0
        main(0, 0, opts)
