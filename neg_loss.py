import torch
import torch.nn as nn
import torch.nn.functional as F


class NEGLoss(nn.Module):
    def __init__(self, ix_to_word, word_freqs, num_negative_samples=5):
        super(NEGLoss, self).__init__()
        self.num_negative_samples = num_negative_samples
        self.num_words = len(ix_to_word)
        self.distr = F.normalize(torch.Tensor(
            [word_freqs[ix_to_word[i]] for i in range(len(word_freqs))]).pow(0.75), dim=0)

    def sample(self, num_samples, positives=[]):
        weights = torch.zeros((self.num_words, 1))
        for w in positives:
            weights[w] += 1.0
        for positive_label in positives:
            for _ in range(num_samples):
                w = torch.multinomial(self.distr, 1)[0]
                while w.item() == positive_label:
                    w = torch.multinomial(self.distr, 1)[0]
                weights[w] += 1.0
        return weights

    def forward(self, input, target):
        weight = self.sample(self.num_negative_samples, positives=target.squeeze().data.numpy())
        return F.nll_loss(input, target, weight=weight)
