'''
exactgpmodels - Cleaned version containing only actively used classes
'''

import torch
import torch.nn as nn 
import gpytorch
import math 

from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

class SingleTaskDeepKernelNonLinear(gpytorch.models.ExactGP): 
    def __init__(self, input_dim, train_x, train_y, likelihood, depth, dropout, activation, pretrained,latent_dim, feature_extractor, gphyper, kernel_choice='RBF', mean='CONSTANT'):
        super(SingleTaskDeepKernelNonLinear, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.LinearMean(input_size=latent_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

        self.pretrained = pretrained

        if not pretrained: 
            self.feature_extractor = LargeFeatureExtractorNonLin(datadim=input_dim, depth=depth, dr=dropout, activ=activation)
        else: 
            self.feature_extractor = feature_extractor
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if gphyper is not None: 
            self.initialize(**gphyper)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        if self.pretrained:
            projected_x = projected_x.detach()

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SingleTaskDeepKernel(gpytorch.models.ExactGP): 
    def __init__(self, input_dim, train_x, train_y, likelihood, depth, dropout, activation, pretrained, latent_dim, feature_extractor, gphyper, kernel_choice='RBF', mean='CONSTANT'):
        super(SingleTaskDeepKernel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.LinearMean(input_size=latent_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

        self.pretrained = pretrained

        # Set up feature extractor, with parameters frozen if pretrained
        if not pretrained: 
            self.feature_extractor = LargeFeatureExtractor(datadim=input_dim, depth=depth, dr=dropout, activ=activation)
        else: 
            self.feature_extractor = feature_extractor
            # Freeze parameters if feature extractor is pretrained
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if gphyper is not None: 
            self.initialize(**gphyper)

    def forward(self, x):
        # Pass data through the feature extractor without detaching
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Normalize NN outputs

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LargeFeatureExtractorNonLin(torch.nn.Sequential):
    def __init__(self, datadim, depth, dr, activ):
        super(LargeFeatureExtractorNonLin, self).__init__()

        # Deep Kernel Architecture
        self.datadim = datadim
        self.depth = depth # a list that defines the number of the depth of the deep kernel, the number of linear leayers 
        self.activation_function = activ 
        self.droupout_rate = dr
        print('depth', depth)
        final_layer = depth[-1]
        print('final layer', final_layer)

        ### Add the hyperparameters of the
        for i, d in enumerate(self.depth[:-1]): 
            # print(i, d)
            dim1, dim2 = d 
            self.add_module('linear' + str(i+1), torch.nn.Linear(dim1, dim2))
            if self.activation_function == 'relu': 
                self.add_module('activ' + str(i+1), torch.nn.ReLU())
            elif self.activation_function == 'leakyr':
                self.add_module('activ' + str(i+1) , torch.nn.LeakyReLU())
            elif self.activation_function == 'prelu':
                self.add_module('activ' + str(i+1) , torch.nn.PReLU())
            elif self.activation_function == 'selu':
                self.add_module('activ' + str(i+1) , torch.nn.SELU())
            
        print('Final Layer', final_layer[0], final_layer[1])
        self.add_module('final_linear', torch.nn.Linear(int(final_layer[0]), int(final_layer[1])))
        self.add_module('nonlinearity', torch.nn.LeakyReLU())
        self.add_module('dr1', torch.nn.Dropout(self.droupout_rate))

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, datadim, depth, dr, activ):
        super(LargeFeatureExtractor, self).__init__()

        # Deep Kernel Architecture
        self.datadim = datadim
        self.depth = depth # a list that defines the number of the depth of the deep kernel, the number of linear leayers 
        self.activation_function = activ 
        self.droupout_rate = dr
        print('depth', depth)
        final_layer = depth[-1]
        print('final layer', final_layer)

        ### Add the hyperparameters of the
        for i, d in enumerate(self.depth[:-1]): 
            # print(i, d)
            dim1, dim2 = d 
            self.add_module('linear' + str(i+1), torch.nn.Linear(dim1, dim2))
            if self.activation_function == 'relu': 
                self.add_module('activ' + str(i+1), torch.nn.ReLU())
            elif self.activation_function == 'leakyr':
                self.add_module('activ' + str(i+1) , torch.nn.LeakyReLU())
            elif self.activation_function == 'prelu':
                self.add_module('activ' + str(i+1) , torch.nn.PReLU())
            elif self.activation_function == 'selu':
                self.add_module('activ' + str(i+1) , torch.nn.SELU())
            
        print('Final Layer', final_layer[0], final_layer[1])
        self.add_module('final_linear', torch.nn.Linear(int(final_layer[0]), int(final_layer[1])))        
        self.add_module('dr1', torch.nn.Dropout(self.droupout_rate))
