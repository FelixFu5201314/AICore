# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Ref:https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
import numpy as np
import torch
import torch.nn as nn

from dao.register import Registers


@Registers.losses.register
def BCELoss(weight=None, reduction='mean'):
    return nn.BCELoss(weight=weight,  reduction=reduction)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    """
    注意内容:
        1.要求输入的predict和target必须是同样shape的。
        2.要求输入的predict的数值范围应该为0~1
    """
    N, C, H, W = 2, 3, 4, 4
    loss = BCELoss()
    loss_w = BCELoss(weight=torch.ones(C))
    m = nn.Sigmoid()

    # 0. 手动验证二分类 ， CELoss和BCELoss的区别：CELoss = Softmax + BCELoss
    predict = torch.rand([2, 2])  # 预测值
    ce_target = torch.tensor([1, 0])  # 真实值
    # （1）CELoss
    ce_loss = torch.nn.CrossEntropyLoss()
    print(ce_loss(predict, ce_target))

    # （2）Softmax + BCELoss
    soft_max = torch.nn.Softmax(dim=-1)
    soft_out = soft_max(predict)

    bce_target = torch.Tensor([[0, 1],
                               [1, 0]])     # one-hot形式
    bce_loss = torch.nn.BCELoss()
    print(bce_loss(soft_out, bce_target))

    # （3）手动实现个BCELoss
    bce_result = - bce_target * torch.log(soft_out) - (1.0 - bce_target) * torch.log(1.0 - soft_out)
    print(bce_result.mean())

    # （4）Softmax + log + NLLLoss
    log_soft_out = torch.log(soft_out)
    nll_loss = torch.nn.NLLLoss()
    print(nll_loss(log_soft_out, ce_target))

    # 1.  手动验证多分类
    predict = torch.Tensor([[0.5796, 0.4403, 0.9087],
                            [-1.5673, -0.3150, 1.6660]])  # 预测值
    ce_target = torch.tensor([2, 0])  # 真实值
    # （1）. CELoss
    ce_loss = torch.nn.CrossEntropyLoss()
    print('ce_loss:', ce_loss(predict, ce_target))  # ce_loss: tensor(2.1246)

    # （2）.Softmax + BCELoss
    soft_input = torch.nn.Softmax(dim=-1)

    soft_out = soft_input(predict)

    bec_target = torch.Tensor([[0, 0, 1],
                               [1, 0, 0]])
    bce_loss = torch.nn.BCELoss()
    print('bce_loss:', bce_loss(soft_out, bec_target))  # bce_loss: tensor(1.1572)

    # （3）.Softmax + log + NLLLoss
    log_soft_out = torch.log(soft_out)
    nll_loss = torch.nn.NLLLoss()
    print('nll_loss:', nll_loss(log_soft_out, ce_target))  # nll_loss: tensor(2.1246)

    # 可以看出，解决多分类问题时，CELoss和BCELoss的结果不一样了
    # 通过以下步骤解释下

    # 二分类预测值
    predict_2 = torch.rand([3, 2])
    # tensor([[0.6718, 0.8155],
    #         [0.6771, 0.1240],
    #         [0.7621, 0.3166]])
    soft_input = torch.nn.Softmax(dim=-1)
    # 二分类Softmax结果
    soft_out_2 = soft_input(predict_2)
    # tensor([[0.4641, 0.5359],
    #         [0.6349, 0.3651],
    #         [0.6096, 0.3904]])

    # 三分类预测值
    predict_3 = torch.rand([2, 3])
    # tensor([[0.0098, 0.5813, 0.9645],
    #         [0.4855, 0.5245, 0.4162]])
    # 三分类Softmax结果
    soft_out_3 = soft_input(predict_3)
    # tensor([[0.1863, 0.3299, 0.4839],
    #         [0.3364, 0.3498, 0.3139]])

    """
    可以看出，在解决二分类问题时，soft_out_2的结果，每行只有两个元素，且两个元素和为。也就是说，soft_out_2[:][0] + soft_out_2[:][1] = 1

    假设target的第一个元素是0， 那么应对在BCELoss的公式  BCELoss(x,y)=−(y∗log(x)+(1−y)∗log(1−x))  中，
    BCELoss(soft_out_2[0][0],0)=−log(1−soft_out_2[0][0])=−log(soft_out_2[0][1])
    BCELoss(soft_out_2[0][1],1)=−log(soft_out_2[0][1])
    二者是一样的，也就是说，面对二分类问题，BCELoss每一行的结果中每个元素都是一样的，所以做平均值的时候，每行的结果也就是每行每个元素的结果。
    
    但是解决三分类问题时，soft_out_3的结果每行有三个元素，三个元素的和为1。
    还是假设target的第一个元素是0，BCELoss每行的每个元素不一样了。那结果也就不一样了。
    如此BCELoss相比CELoss在解决多分类问题的优势就表现了出来。CELoss只是根据每行的分类结果去取值，而BCELoss考虑了每行全部结果。
    """

    # 2. 测试分割(target是整数形式)， 分类同理只是无H,W，特征为一维的
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, C, H, W), dtype=torch.float32).random_(C)  # 应该会将C转成one-hot格式
    # print("input:{}".format(input))
    # print("target:{}".format(target))
    output = loss(m(input), target)
    print("output:{}".format(output))
    # output_w = loss_w(m(input), target)
    # print("output_w:{}".format(output))
    output.backward()

    # 4. 测试分割(Weighted Binary Cross-Entropy加权交叉熵损失函数)
    loss = BCELoss(weight=torch.ones(C, H, W))
    input = torch.randn((N, C, H, W), requires_grad=True)
    target = torch.empty((N, C, H, W)).random_(2)
    output = loss(m(input), target)
    print("output:{}".format(output))
    output.backward()