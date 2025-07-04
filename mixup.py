import torch
import numpy as np

def mixup(inputs, alpha=0.1, prob=0.7):
    if np.random.rand() < prob:
        batch_size = inputs[0].size(0)
        rand = torch.randperm(batch_size)
        lam = int(np.random.beta(alpha, alpha) * inputs[0].size(2))
        new_inputs = []
        for input in inputs:
            rand_input = input[rand]
            if np.random.rand() < 0.5:
                new_input = torch.cat([input[:, :, :, 0:lam, :],
                                       rand_input[:, :, :, lam:input.size(3), :]], dim=3)
            else:
                new_input = torch.cat([input[:, :, 0:lam, :, :],
                                       rand_input[:, :, lam:input.size(2), :, :]], dim=2)
            new_inputs.append(new_input)
        return new_inputs
    else:
        return inputs