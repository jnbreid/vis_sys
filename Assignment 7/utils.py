import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

# adding normal distributed noise to image
def add_noise(img, mean=0, sigma=0.3):
    noisy_img = img + torch.normal(mean * torch.ones(img.shape), sigma)
    return noisy_img.clamp(0,1)

# function to count model parameters that are adjusted during training
def model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# function to make loss smooth
def smooth(f, K=5):
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]
    return smooth_f


# functions to save model/stats and load them
import os

def save_model(model, optimizer, epoch, stats, margin):
    if(not os.path.exists("checkpoints")):
        os.makedirs("checkpoints")
    savepath = f"checkpoints/checkpoint_epoch_{epoch}_margin_{margin}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return

# function to load a model and its stats
def load_model(model, optimizer, savepath):
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    return model, optimizer, epoch, stats


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

# class for tripletloss (from last session)
class TripletLoss(nn.Module):
    """ Implementation of the triplet loss function """
    def __init__(self, margin=0.2, reduce="mean"):
        """ Module initializer """
        assert reduce in ["mean", "sum"]
        super().__init__()
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, anchor, positive, negative):
        """ Computing pairwise distances and loss functions """
        # L2 distances
        d_ap = (anchor - positive).pow(2).sum(dim=-1)
        d_an = (anchor - negative).pow(2).sum(dim=-1)

        # triplet loss function
        loss = (d_ap - d_an + self.margin)
        loss = torch.maximum(loss, torch.zeros_like(loss))

        # averaging or summing
        loss = torch.mean(loss) if(self.reduce == "mean") else torch.sum(loss)

        return loss

# function to visualize training progress (mostly from last session)
def visualize_progress(train_loss, val_loss, start=0):
    """ Visualizing loss and accuracy """
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_train = smooth(train_loss, 19)
    ax[0].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_yscale("linear")
    ax[0].set_title("Training Progress (linear)")

    ax[1].plot(train_loss, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_train, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (log)")

    smooth_val = smooth(val_loss, 31)
    N_ITERS = len(val_loss)
    ax[2].plot(np.arange(start, N_ITERS)+start, val_loss[start:], c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[2].plot(np.arange(start, N_ITERS)+start, smooth_val[start:], c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("CE Loss")
    ax[2].set_yscale("log")
    ax[2].set_title(f"Valid Progress")

    return

def display_projections(points, labels, ax=None, legend=None, labels_to_keep = None):
    """ Displaying low-dimensional data projections """

    # more different colors to see more different classes in diagrams
    COLORS = ['black', 'grey', 'rosybrown', 'brown', 'red', 'salmon', 'green', 'yellow', 'blue', 'navy', 'darkviolet', 'deeppink',
              'fuchsia', 'gold', 'purple', 'teal', 'lightgreen', 'darkgoldenrod', 'beige','skyblue','pink','crimson','orange',
              'cyan', 'peru']

    if(ax is None):
        _, ax = plt.subplots(1,1,figsize=(12,6))

    for i in range(len(labels_to_keep)):
        idx = np.where(labels_to_keep[i] == labels)

        ax.scatter(points[idx, 0], points[idx, 1], label=i, c=COLORS[i])
    ax.legend(loc="best")


from tqdm import tqdm
# slightly changed training class for more evaluation
class Trainer:

    def __init__(self, model, criterion, train_loader, valid_loader, n_iters=1e4):
        """ Trainer initializer """
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.n_iters = int(n_iters)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_loss = []
        self.valid_loss = []
        return

    @torch.no_grad()
    def valid_step(self, val_iters=100, get_res = False):
        """ Some validation iterations """
        self.model.eval()
        cur_losses = []

        # lists to save calculated distances and wanted 'class' (same person/not same person)
        pred = []
        true_val = []

        for i, ((anchors, positives, negatives),_) in enumerate(self.valid_loader):
            # setting inputs to GPU
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)

            # forward pass and triplet loss
            anchor_emb, positive_emb, negative_emb = self.model(anchors, positives, negatives)

            # also save if instances are same and predicted embedding distance for AUC
            if get_res == True:
              # calculate L2 distances between anchor and other two images
              d_p = torch.sqrt((anchor_emb - positive_emb).pow(2).sum(dim=-1)).detach().cpu().numpy()
              d_n = torch.sqrt((anchor_emb - negative_emb).pow(2).sum(dim=-1)).detach().cpu().numpy()

              # append the predicted distances and the 'fitting value' for 'same person'
              pred = pred + list(d_p)
              true_val = true_val + list(np.zeros(anchors.shape[0]))
              # append the predicted distances and the 'fitting value' for 'not same person'
              pred = pred + list(d_n)
              true_val = true_val + list(np.ones(anchors.shape[0]))


            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            cur_losses.append(loss.item())

            if(i >= val_iters):
                break

        self.valid_loss += cur_losses
        self.model.train()

        if get_res == True:
          return cur_losses, pred, true_val

        return cur_losses

    def fit(self):
        """ Train/Validation loop """

        self.iter_ = 0

        progress_bar = tqdm(range(self.n_iters), total=self.n_iters, initial=0)

        for i in range(self.n_iters):
            for (anchors, positives, negatives), _ in self.train_loader:
                # setting inputs to GPU
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)

                # forward pass and triplet loss
                anchor_emb, positive_emb, negative_emb = self.model(anchors, positives, negatives)
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                self.train_loss.append(loss.item())

                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if(self.iter_ % 250 == 0):
                    cur_losses = self.valid_step()

                # nicer? progressbar
                progress_bar.set_description(f"Train Iter {self.iter_} | Epoch {i}: Loss={round(loss.item(),5)} | Valid loss={np.mean(cur_losses)} (per 250 it)")
                progress_bar.update(1)

                self.iter_ = self.iter_+1
                if(self.iter_ >= self.n_iters):
                    break
            if(self.iter_ >= self.n_iters):
                break
        return
        

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor