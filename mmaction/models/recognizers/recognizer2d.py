from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss = self.cls_head.loss(cls_score, gt_labels)

        return loss

    def forward_test(self, imgs):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        cls_score = self.cls_head(x, num_segs)

        return cls_score.cpu().numpy()