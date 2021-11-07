import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


if __name__ == '__main__':
    epoch = 1000
    model_name = 'VAE'
    dataset = 'celeba'

    path_for_loading = f"./checkpoints/testing_0.0/{epoch}_{dataset}_net_{model_name}.pth"
    model = torch.load(path_for_loading).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    if model_name == 'VAE':
        model = model.decoder

    model.eval()
    print(model)

    n_img = 8
    nz = 8
    fixed_noise1 = torch.randn(n_img, nz, 1, 1,
                                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    fixed_noise2 = torch.randn(n_img, nz, 1, 1,
                               device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    direction = (fixed_noise2 - fixed_noise1) / 10.0

    fake_arr = torch.FloatTensor(model(fixed_noise1).detach().cpu())
    for l in range(1,11):
        fixed_noise_new = fixed_noise1 + direction * l

        with torch.no_grad():
            fake = model(fixed_noise_new).detach().cpu()
        fake_arr = torch.dstack((fake_arr, fake))
    plt.imsave(f"image_LS_{epoch}_{model_name}_{dataset}.png", np.transpose(vutils.make_grid(fake_arr, padding=2, normalize=True), (1, 2, 0)).cpu().numpy())