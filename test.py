import argparse
import cv2
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data.coco import coco_class_index, coco_class_labels, COCODataset
from data.transforms import ValTransforms
from misc import TestTimeAugmentation
from model.yolof import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOF Detection')

    # basic
    parser.add_argument('--min_size', default=800, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1333, type=int,
                        help='the min size of input image')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visualize results.')
    parser.add_argument('-vs', '--visual_threshold', default=0.35, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')

    # model
    parser.add_argument('-v', '--version', default='yolof50', choices=['yolof18', 'yolof50'],
                        help='build yolof')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='NMS threshold')
    parser.add_argument('-bg', '--background', action='store_true', default=False,
                        help='add background class')

    # dataset
    parser.add_argument('--root', default='./dataset',
                        help='data root')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)

    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1 - t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img,
              bboxes,
              scores,
              cls_inds,
              vis_thresh,
              class_colors,
              class_names,
              class_indexs=None):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            cls_color = class_colors[cls_id]
            cls_id = class_indexs[cls_id]

            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img


def test(args,
         net,
         device,
         dataset,
         transforms=None,
         vis_thresh=0.4,
         class_colors=None,
         class_names=None,
         class_indexs=None,
         show=False,
         test_aug=None):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        image, _ = dataset.pull_image(index)

        h, w, _ = image.shape
        orig_size = np.array([[w, h, w, h]])

        # prepare
        x = transforms(image)[0]
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # inference
        if test_aug is not None:
            # test augmentation:
            bboxes, scores, cls_inds = test_aug(x, net)
        else:
            bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")

        # rescale
        bboxes *= orig_size

        # vis detection
        img_processed = visualize(
            img=image,
            bboxes=bboxes,
            scores=scores,
            cls_inds=cls_inds,
            vis_thresh=vis_thresh,
            class_colors=class_colors,
            class_names=class_names,
            class_indexs=class_indexs)
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) + '.jpg'), img_processed)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_dir = args.root
    class_names = coco_class_labels
    class_indexs = coco_class_index
    num_classes = 80
    dataset = COCODataset(
        data_dir=data_dir,
        image_set='val2017',
        transform=None)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]


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

    # run
    test(args=args,
         net=model,
         device=device,
         dataset=dataset,
         transforms=transform,
         vis_thresh=args.visual_threshold,
         class_colors=class_colors,
         class_names=class_names,
         class_indexs=class_indexs,
         show=args.show,
         test_aug=test_aug, )
