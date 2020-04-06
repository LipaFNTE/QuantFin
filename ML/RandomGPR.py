import torch
import gpytorch as gp
from sklearn.ensemble import RandomForestRegressor


class rGPR:
    def __init__(self,
                 X, y,
                 likelihood=gp.likelihoods.GaussianLikelihood(),
                 mean=gp.means.ConstantMean(),
                 covar=gp.kernels.ScaleKernel(gp.kernels.RBFKernel()),
                 estimators: int = 10,
                 seqential_bootstrap: bool = False,
                 optimizer=torch.optim,
                 learning_rate: float = 0.1):
        self.X = X
        self.y = y
        self.model = gp.models.ExactGP()
        self.likelihood = likelihood
        self.mean = mean
        self.covar = covar

    class ExactGPModel(gp.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gp.means.ConstantMean()
            self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gp.distributions.MultivariateNormal(mean_x, covar_x)
