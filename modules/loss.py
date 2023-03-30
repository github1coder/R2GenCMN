import torch
import torch.nn as nn


# 语言模型损失函数的计算方法是：将模型的预测结果和参考文本的id序列进行比较，计算交叉熵损失。由于模型的预测结果是一个概率分布，因此需要使用gather函数将模型的预测结果按照参考文本的id序列进行提取。同时，还需要使用mask序列将无用的token的损失剔除。最终，将损失值除以mask序列中1的个数得到平均损失。

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):  # 掩码的作用是在计算损失时忽略填充部分的贡献。
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask  # ?
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss
