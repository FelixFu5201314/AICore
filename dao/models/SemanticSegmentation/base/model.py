import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    """
    SegmentationModel是分割模型的基类，
        共包含4组件，即：encoder（backbone）不在此处定义、decoder、segmentation_Head、classificationHead；
        共包含3个函数，即：initialize初始化权重，forward，predict
    """
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = [x, ] + self.encoder(x)  # backbone提取的不同层次的特征
        decoder_output = self.decoder(*features)  # 解码后的特征，一般是[batch_size, c, height, width]，c是输出维度

        masks = self.segmentation_head(decoder_output)   # 对解码后的特征做个卷积得到最后结果[batch_size, num_class, height, width]

        if self.classification_head is not None and self.training:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
