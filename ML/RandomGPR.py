import torch
import gpytorch as gp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

class rGPR:
    def __init__(self,
                 X, y,
                 likelihood=gp.likelihoods.GaussianLikelihood(),
                 mean=gp.means.ConstantMean(),
                 covar=gp.kernels.ScaleKernel(gp.kernels.RBFKernel()),
                 estimators: int = 10,
                 seqential_bootstrap: bool = False,
                 optimizer=torch.optim,
                 training_iter: int = 50,
                 learning_rate: float = 0.1):
        self.X = X
        self.y = y
        self.likelihood = likelihood
        self.mean = mean
        self.covar = covar
        self.estimators = estimators
        self.sequential_bootstrap = seqential_bootstrap
        self.optimizer = self._optimizer()
        self.training_iter = training_iter
        self.learning_rate = learning_rate
        self.model = self._get_model()

    def generate_samples(self) -> list[pd.DataFrame]:
        if self.sequential_bootstrap:
            samples_index = self._get_samples_index()
            return self._get_data_from_index(samples_index)
        else:
            pass

    def fit(self):
        pass

    def get_seq_sample(self, unique_obs):
        samp = []
        unique = 0
        size = self.X.shape[0]
        while unique < unique_obs:
            v = np.random.randint(0, size)
            if v not in samp:
                unique = unique + 1
            samp.append(v)

    def _get_samples_index(self):
        unique_obs = np.ceil(self.X.shape[0] * (1 - pow(np.e, -1)))
        return [self.get_seq_sample(unique_obs) for _ in range(self.estimators)]

    def _get_data_from_index(self, samples_index):
        return [self.X.iloc[i, :] for i in samples_index]

    def _optimizer(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

    def _get_model(self):
        return ExactGPModel(self.X, self.y, self.likelihood)



