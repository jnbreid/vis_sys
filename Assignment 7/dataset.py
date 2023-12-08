
# function to remove normalization to see images
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean    # initialize mean and std in each dimesion (same input as for normalization)
        self.std = std

    def __call__(self, tensor):
        # for each tensor, mean, std, multiply tensor in each channel by std and translate by mean
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor