"""
Github: https://github.com/juliusbierk/torchlda
"""

import torch

def lda(X, y):
    """
    This is not the real Latent Dirichlet Allocation
    This is the Linear Discriminant Analysis !
    """

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = torch.tensor(X, device=device, dtype=torch.float)
        y = torch.tensor(y, device=device, dtype=torch.float)

        m1 = torch.mean(X, dim=0)
        m2 = torch.mean(y, dim=0)
        m = (len(X) * m1 + len(y) * m2) / (len(X) + len(y))

        d1 = X - m1[None, :]
        scatter1 = d1.t() @ d1
        d2 = y - m2[None, :]
        scatter2 = d2.t() @ d2
        within_class_scatter = scatter1 + scatter2

        d1 = m1 - m[None, :]
        scatter1 = len(X) * (d1.t() @ d1)
        d2 = m2 - m[None, :]
        scatter2 = len(y) * (d2.t() @ d2)
        between_class_scatter = scatter1 + scatter2

        p = torch.pinverse(within_class_scatter) @ between_class_scatter
        eigenvalues, eigenvectors = torch.eig(p, eigenvectors=True)
        idx = torch.argsort(eigenvalues[:, 0], descending=True)
        eigenvalues = eigenvalues[idx, 0]
        eigenvectors = eigenvectors[idx, :]

        return eigenvectors[0, :].cpu().numpy()