# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import math

from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind
from scipy.stats import f_oneway, kruskal

from torch.autograd import grad

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6
        best_phi = None  # Safety fallback

        print(f"Total environments: {len(environments)}")
        if len(environments) < 2:
            raise ValueError("IRM requires at least two environments (train + validation).")

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        print(f"x_val shape: {x_val.shape}")
        print(f"y_val shape: {y_val.shape}")

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            print(f"Training with reg={reg}")
            self.train(environments[:-1], args, reg=reg)

            sol = self.solution()
            err = (x_val @ sol - y_val).pow(2).mean().item()
            # err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        
        # self.phi = best_phi
        if best_phi is None:
            raise RuntimeError("IRM failed: No reg value improved validation error.")

        print(f"Best reg: {best_reg}, Best error: {best_err}")
        self.phi = best_phi

        self.best_val_error = best_err

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        # print(f"Training on {len(environments)} environments")
        # print(f"Each input has dim {dim_x}")

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                      reg,
                                                                      error,
                                                                      penalty,
                                                                      w_str))

    def solution(self):
        return (self.phi @ self.w).view(-1, 1)


class InvariantCausalPrediction(object):
    def __init__(self, environments, args):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients.view(-1, 1)


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args):
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()

        w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.w = torch.Tensor(w).view(-1, 1)

    def solution(self):
        return self.w


class alphaInvariantCausalPrediction:
    def __init__(self, environments, args):
        self.alpha = args.get("alpha", 0.05)

        X_envs = [e[0].numpy() for e in environments[:-1]]
        y_envs = [e[1].numpy().flatten() for e in environments[:-1]]

        X_val = environments[-1][0].numpy()
        y_val = environments[-1][1].numpy().flatten()

        d = X_envs[0].shape[1]
        self.candidate_subsets = list(self._powerset(range(d)))
        self.accepted_subsets = []

        print(f"[ICP] Number of environments: {len(X_envs)}")
        print(f"[ICP] Input dimension: {d}")
        print(f"[ICP] Testing {len(self.candidate_subsets)} subsets")

        for subset in self.candidate_subsets:
            print(f"[ICP] Testing subset: {subset}")
            if self._is_invariant(X_envs, y_envs, subset):
                print(f"Subset {subset} is invariant (p > {self.alpha})")
                self.accepted_subsets.append(set(subset))
            else:
                print(f"Subset {subset} rejected")

        if self.accepted_subsets:
            self.invariant_set = set.intersection(*self.accepted_subsets)
            print(f"[ICP] Invariant set selected: {self.invariant_set}")
        else:
            self.invariant_set = set()
            print("[ICP] No invariant subsets found!")

        if len(self.invariant_set) == 0:
            self.coef_ = torch.zeros(d, 1)
        else:
            model = LinearRegression()
            X_all = np.vstack(X_envs)
            y_all = np.hstack(y_envs)
            model.fit(X_all[:, list(self.invariant_set)], y_all)

            full_coef = np.zeros((d,))
            full_coef[list(self.invariant_set)] = model.coef_
            self.coef_ = torch.tensor(full_coef, dtype=torch.float32).view(-1, 1)

        self.best_val_error = float(((X_val @ self.coef_.numpy().reshape(-1, 1) - y_val.reshape(-1, 1)) ** 2).mean())

    def _powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    def _is_invariant(self, X_envs, y_envs, subset):
        if len(subset) == 0:
            return False  # No predictors

        residuals = []
        for X, y in zip(X_envs, y_envs):
            X_sub = X[:, list(subset)]
            model = LinearRegression().fit(X_sub, y)
            y_pred = model.predict(X_sub)
            residuals.append(y - y_pred)

        try:
            stat, p = kruskal(*residuals)
            print(f"Kruskal-Wallis p-value: {p:.4f}")
            return p > self.alpha
        except Exception as e:
            print(f"Kruskal-Wallis test failed for subset {subset}: {e}")
            return False

    def solution(self):
        return self.coef_