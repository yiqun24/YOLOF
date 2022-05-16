import os
import argparse
import time
import torch
import distributed_utils
import torch.backends.cudnn as cudnn
from criterion import build_criterion
from data.coco import COCODataset
from data.transforms import ValTransforms, BaseTransforms, TrainTransforms
from evaluator.coco_evaluator import COCOAPIEvaluator
from misc import get_total_grad_norm
from optimizer import build_optimizer
from warmup_schedule import build_warmup
from model.yolof import build_model
from misc import CollateFunc


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOF Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--schedule', type=str, default='1x', choices=['1x', '2x', '3x', '9x'],
                        help='training schedule. Attention, 9x is designed for YOLOF53-DC5.')
    parser.add_argument('-lr', '--base_lr', type=float, default=0.03,
                        help='base learning rate')
    parser.add_argument('-lr_bk', '--backbone_lr', type=float, default=0.01,
                        help='backbone learning rate')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in data loading')
    parser.add_argument('--num_gpu', default=1, type=int,
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                        default=2, help='interval between evaluations')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='path to save weight')

    # input image size
    parser.add_argument('--train_min_size', type=int, default=800,
                        help='The shorter train size of the input image')
    parser.add_argument('--train_max_size', type=int, default=1333,
                        help='The longer train size of the input image')
    parser.add_argument('--val_min_size', type=int, default=800,
                        help='The shorter val size of the input image')
    parser.add_argument('--val_max_size', type=int, default=1333,
                        help='The longer val size of the input image')

    # model
    parser.add_argument('-v', '--version', default='yolof18', choices=['yolof18', 'yolof50'],
                        help='build yolof')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/root/autodl-tmp',
                        help='data root')

    # Loss
    parser.add_argument('--alpha', default=0.25, type=float,
                        help='focal loss alpha')
    parser.add_argument('--gamma', default=2.0, type=float,
                        help='focal loss gamma')
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, device)

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=CollateFunc(),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # criterion
    criterion = build_criterion(args=args, device=device, num_classes=num_classes)

    # build model
    model = build_model(args=args,
                        device=device,
                        num_classes=num_classes,
                        pretrained=True)

    model = model.to(device).train()

    # optimizer
    optimizer = build_optimizer(model=model,
                                base_lr=args.base_lr,
                                backbone_lr=args.backbone_lr,
                                name='sgd',
                                momentum=0.9,
                                weight_decay=1e-4)

    # lr scheduler
    epoch = {
        '1x': {'max_epoch': 12,
               'lr_epoch': [8, 11],
               'multi_scale': None},
        '2x': {'max_epoch': 24,
               'lr_epoch': [16, 22],
               'multi_scale': [400, 500, 600, 700, 800]},
        '3x': {'max_epoch': 36,
               'lr_epoch': [24, 33],
               'multi_scale': [400, 500, 600, 700, 800]},
    }
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=epoch[args.schedule]['lr_epoch'])
    # warmup scheduler
    warmup = 'linear'
    wp_iter = 1500
    warmup_factor = 0.00066667
    warmup_scheduler = build_warmup(name=warmup,
                                    base_lr=args.base_lr,
                                    wp_iter=wp_iter,
                                    warmup_factor=warmup_factor)

    # training configuration
    max_epoch = epoch[args.schedule]['max_epoch']
    epoch_size = len(dataset) // (args.batch_size * args.num_gpu)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(max_epoch):
        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < wp_iter and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == wp_iter and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, args.base_lr, args.base_lr)

            # to device
            images = images.to(device)
            masks = masks.to(device)

            # inference
            outputs = model(images, mask=masks)

            # compute loss
            cls_loss, reg_loss, total_loss = criterion(outputs=outputs,
                                                       targets=targets,
                                                       anchor_boxes=model.anchor_boxes)

            loss_dict = dict(
                cls_loss=cls_loss,
                reg_loss=reg_loss,
                total_loss=total_loss
            )
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(total_loss):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            total_loss.backward()
            if args.grad_clip_norm > 0.:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad()

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                print(
                    '[Epoch %d/%d][Iter %d/%d][lr: %.6f][lr_bk: %.6f][Loss: cls %.2f || reg %.2f || gnorm: %.2f || '
                    'size [%d, %d] || time: %.2f] '
                    % (epoch + 1,
                       max_epoch,
                       iter_i,
                       epoch_size,
                       cur_lr_dict['lr'],
                       cur_lr_dict['lr_bk'],
                       loss_dict_reduced['cls_loss'].item(),
                       loss_dict_reduced['reg_loss'].item(),
                       total_norm,
                       args.train_min_size, args.train_max_size,
                       t1 - t0),
                    flush=True)

                t0 = time.time()

        lr_scheduler.step()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch + 1))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                    torch.save(model.state_dict(), os.path.join(path_to_save, weight_name))
                else:
                    print('eval ...')
                    model_eval = model

                    # set eval mode
                    model_eval.trainable = False
                    model_eval.eval()

                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map * 100)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, weight_name))

                        # set train mode.
                    model_eval.trainable = True
                    model_eval.train()


def build_dataset(args, device):
    # transform
    epoch = {
        '1x': {'max_epoch': 12,
               'lr_epoch': [8, 11],
               'multi_scale': None},
        '2x': {'max_epoch': 24,
               'lr_epoch': [16, 22],
               'multi_scale': [400, 500, 600, 700, 800]},
        '3x': {'max_epoch': 36,
               'lr_epoch': [24, 33],
               'multi_scale': [400, 500, 600, 700, 800]},
    }
    transforms = {
        '1x': [{'name': 'RandomHorizontalFlip'},
               {'name': 'RandomShift', 'max_shift': 32},
               {'name': 'ToTensor'},
               {'name': 'Resize'},
               {'name': 'Normalize'},
               {'name': 'PadImage'}],

        '2x': [{'name': 'RandomHorizontalFlip'},
               {'name': 'RandomShift', 'max_shift': 32},
               {'name': 'ToTensor'},
               {'name': 'Resize'},
               {'name': 'Normalize'},
               {'name': 'PadImage'}],

        '3x': [{'name': 'DistortTransform',
                'hue': 0.1,
                'saturation': 1.5,
                'exposure': 1.5},
               {'name': 'RandomHorizontalFlip'},
               {'name': 'RandomShift', 'max_shift': 32},
               {'name': 'JitterCrop', 'jitter_ratio': 0.3},
               {'name': 'ToTensor'},
               {'name': 'Resize'},
               {'name': 'Normalize'},
               {'name': 'PadImage'}]}
    trans_config = transforms[args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = TrainTransforms(trans_config=trans_config,
                                      min_size=args.train_min_size,
                                      max_size=args.train_max_size,
                                      random_size=epoch[args.schedule]['multi_scale'],
                                      pixel_mean=[0.485, 0.456, 0.406],
                                      pixel_std=[0.229, 0.224, 0.225],
                                      fmt='RGB')
    val_transform = ValTransforms(min_size=args.val_min_size,
                                  max_size=args.val_max_size,
                                  pixel_mean=[0.485, 0.456, 0.406],
                                  pixel_std=[0.229, 0.224, 0.225],
                                  fmt='RGB')
    color_augment = BaseTransforms(min_size=args.train_min_size,
                                   max_size=args.train_max_size,
                                   random_size=epoch[args.schedule]['multi_scale'],
                                   pixel_mean=[0.485, 0.456, 0.406],
                                   pixel_std=[0.229, 0.224, 0.225],
                                   fmt='RGB')
    # dataset
    data_dir = args.root
    num_classes = 80
    # dataset
    dataset = COCODataset(img_size=args.train_max_size,
                          data_dir=data_dir,
                          image_set='train2017',
                          transform=train_transform,
                          color_augment=color_augment)
    # evaluator
    evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                 device=device,
                                 transform=val_transform)

    print('==============================')
    print('Training model on COCO')
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


if __name__ == '__main__':
    train()
