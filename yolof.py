import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOF(nn.Module):
    def __init__(self,
                 backbone,
                 encoder,
                 decoder,
                 device,
                 num_classes=20,
                 pos_ignore_thresh=0.15,
                 neg_ignore_thresh=0.7,
                 topk=1000):
        super(YOLOF, self).__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.num_classes = num_classes
        self.num_anchors = len([[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]])
        # Ignore thresholds:
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
        self.fmp_size = None
        self.stride = 32
        self.anchor_sizes = torch.as_tensor([[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]])  # aspect ratio?
        self.anchor_boxes = None
        self.scale_clamp = math.log(1000.0 / 16)
        self.topk = topk

    def generate_anchor(self, fmp_size):
        if self.fmp_size is not None and self.fmp_size == fmp_size:
            return self.anchor_boxes
        # feature map size
        fmp_h, fmp_w = fmp_size
        anchor_x, anchor_y = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)], indexing='xy')
        anchor_xy = torch.dstack([anchor_x, anchor_y]).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2]
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1).to(self.device)
        anchor_xy *= self.stride  # what's stride?
        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_sizes[None, :, :].repeat(fmp_h * fmp_w, 1, 1).to(self.device)
        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4)

        self.anchor_boxes = anchor_boxes
        self.fmp_size = fmp_size

        return anchor_boxes

    def box_transform(self, anchor_boxes, bbox_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4]
            bbox_reg: (List[tensor]) [B, M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = bbox_reg[..., :2] * anchor_boxes[..., 2:]
        # restriction
        pred_ctr_offset = torch.clamp(pred_ctr_offset, max=32, min=-32)
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset
        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dw_dh = bbox_reg[..., 2:]
        pred_dw_dh = torch.clamp(pred_dw_dh, max=self.scale_clamp)
        pred_wh = anchor_boxes[..., 2:] * torch.exp(pred_dw_dh)

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_boxes = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_boxes

    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  # x_min
        y1 = dets[:, 1]  # y_min
        x2 = dets[:, 2]  # x_max
        y2 = dets[:, 3]  # y_max

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            # reserve all the bounding box whose ovr less than thresh
            index = np.where(ovr <= self.nms_thresh)[0]
            order = order[index + 1]

        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """

        cls_indices = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_indices)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_indices = cls_indices[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            indices = np.where(cls_indices == i)[0]
            if len(indices) == 0:
                continue
            c_bboxes = bboxes[indices]
            c_scores = scores[indices]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[indices[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_indices = cls_indices[keep]

        return bboxes, scores, cls_indices

    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]
        x = self.backbone(x)
        x = self.encoder(x)
        H, W = x.shape[2:]
        cls_pred, reg_pred = self.decoder(x)

        anchor_boxes = self.generate_anchors(fmp_size=[H, W])  # [M, 4]
        # scores
        scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

        # topk
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]

        # decode box
        bboxes = self.box_transform(anchor_boxes[None], reg_pred[None])[0]  # [N, 4]

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            indices = np.where(labels == i)[0]
            if len(indices) == 0:
                continue
            c_bboxes = bboxes[indices]
            c_scores = scores[indices]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[indices[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # normalize bbox
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels

    def forward(self, x, mask=None):
        if not self.training:
            return self.inference_single_image(x)
        else:
            x = self.backbone(x)
            x = self.encoder(x)
            H, W = x.shape[2:]
            cls_pred, reg_pred = self.decoder(x)
            anchor_boxes = self.generate_anchor(fmp_size=[H, W])  # [M, 4]
            box_pred = self.box_transform(anchor_boxes[None], reg_pred)  # [B, M, 4]

            if mask is not None:
                # [B, H, W]
                mask = F.interpolate(mask[None], size=[H, W]).bool()[0]
                # [B, H, W, KA] -> [B, HW]
                mask = mask.flatten(1)
                # [B, HW] -> [B, HW, KA] -> [BM,], M= HW x KA
                mask = mask[..., None].repeat(1, 1, self.num_anchors).flatten()

            outputs = {"pred_cls": cls_pred,
                       "pred_box": box_pred,
                       "mask": mask}

            return outputs
