# -*- coding: utf-8 -*-


import torch
import numpy as np


class OlshausenField1996Model:
    def __init__(self, num_inputs, num_units, batch_size,
                 lr_r=1e-2, lr_Phi=1e-2, lmda=5e-3, device=None):
        self.lr_r = lr_r # learning rate of r
        self.lr_Phi = lr_Phi # learning rate of Phi
        self.lmda = lmda # regularization parameter

        self.num_inputs = num_inputs
        self.num_units = num_units
        self.batch_size = batch_size

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Weights
        Phi = torch.randn(self.num_inputs, self.num_units, device=self.device, dtype=torch.float32)
        self.Phi = Phi * torch.sqrt(torch.tensor(1.0/self.num_units, device=self.device, dtype=torch.float32))

        # activity of neurons
        self.r = torch.zeros(self.batch_size, self.num_units, device=self.device, dtype=torch.float32)

    def initialize_states(self):
        self.r = torch.zeros(self.batch_size, self.num_units, device=self.device, dtype=torch.float32)

    def normalize_rows(self):
        norm = torch.norm(self.Phi, p=2, dim=0, keepdim=True)
        self.Phi = self.Phi / torch.clamp(norm, min=1e-8)

    # thresholding function of S(x)=|x|
    def soft_thresholding_func(self, x, lmda):
        return torch.maximum(x - lmda, torch.zeros_like(x)) - torch.maximum(-x - lmda, torch.zeros_like(x))

    # thresholding function of S(x)=ln(1+x^2)
    def ln_thresholding_func(self, x, lmda):
        f = 9*lmda*x - 2*torch.pow(x, 3) - 18*x
        g = 3*lmda - torch.pow(x, 2) + 3
        h = torch.cbrt(torch.sqrt(torch.pow(f, 2) + 4*torch.pow(g, 3)) + f)
        two_croot = torch.cbrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        return (1/3)*(x - h / two_croot + two_croot*g / (1e-8+h))

    # thresholding function https://arxiv.org/abs/2003.12507
    def cauchy_thresholding_func(self, x, lmda):
        sqrt_term = torch.sqrt(torch.clamp(x**2 - lmda, min=0))
        f = 0.5*(x + sqrt_term)
        g = 0.5*(x - sqrt_term)
        return f*(x>=lmda) + g*(x<=-lmda)

    def calculate_total_error(self, error):
        recon_error = torch.mean(error**2)
        sparsity_r = self.lmda*torch.mean(torch.abs(self.r))
        return recon_error + sparsity_r

    def __call__(self, inputs, training=True):
        # Ensure inputs are on the correct device
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        else:
            inputs = inputs.to(self.device)

        # Updates
        error = inputs - self.r @ self.Phi.T

        r = self.r + self.lr_r * (error @ self.Phi)
        self.r = self.soft_thresholding_func(r, self.lmda)
        #self.r = self.cauchy_thresholding_func(r, self.lmda)

        if training:
            error = inputs - self.r @ self.Phi.T
            dPhi = error.T @ self.r
            self.Phi += self.lr_Phi * dPhi

        return error, self.r


