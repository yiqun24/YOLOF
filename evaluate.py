import argparse
import torch

from data.transforms import ValTransforms
from evaluator.coco_evaluator import COCOAPIEvaluator
from misc import TestTimeAugmentation
from model.yolof import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOF Detection')
    # basic
    parser.add_argument('--min_size', default=800, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1333, type=int,
                        help='the min size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    # model
    parser.add_argument('-v', '--version', default='yolof50', choices=['yolof18', 'yolof50'],
                        help='build YOLOF')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()


def coco_test(model, data_dir, device, transform, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            test_set=True,
            transform=transform)

    else:
        # eval
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            test_set=False,
            transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    num_classes = 80
    if args.dataset == 'coco-val':
        print('eval on coco-val ...')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
    else:
        print('unknown dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)
    data_dir = args.root
    # YOLOF config
    print('Model: ', args.version)

    # build model
    model = build_model(args=args,
                        device=device,
                        num_classes=num_classes,
                        pretrained=False)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None

    # transform
    transform = ValTransforms(min_size=args.min_size,
                              max_size=args.max_size,
                              pixel_mean=[0.485, 0.456, 0.406],
                              pixel_std=[0.229, 0.224, 0.225],
                              fmt='RGB')
    # evaluation
    with torch.no_grad():
        if args.dataset == 'coco-val':
            coco_test(model, data_dir, device, transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, transform, test=True)
