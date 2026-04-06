import numpy as np
from scipy.optimize import minimize as opt_min


# %%
class PGP_skew():

    def __init__(self, lam_1, lam_2, lam_3, n_asset=3):
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lam_3 = lam_3
        self.n_asset = n_asset

    def optimize_1order(self):
        x_mean = np.mean(self.x, axis=0)
        w = np.zeros(self.n_asset)
        w[x_mean.argmax()] = 1
        return w, x_mean.max() + 10e-5

    def optimize_2order(self):
        self.cov = np.cov(self.x, rowvar=False)

        # cov_inv = np.linalg.inv(self.cov)
        # one_vec = np.ones((self.n_asset, 1))
        # w = (cov_inv @ one_vec) / (one_vec.T @ cov_inv @ one_vec)
        # opt_ = 1 / (one_vec.T @ cov_inv @ one_vec)
        # return w.ravel(), opt_.item()

        def obj_func(w):
            return w.dot(self.cov).dot(w)

        # opt process
        w_init = np.ones(self.n_asset) / self.n_asset
        w_constrain = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.})
        w_bound = [(0, 1) for i in range(self.n_asset)]
        result = opt_min(obj_func, w_init,
                         constraints=w_constrain, bounds=w_bound)
        return result.x, result.fun

    def optimize_3order(self):
        self.x_mean = np.mean(self.x, axis=0)

        def obj_func(w):
            w_std = np.sqrt(w.dot(self.cov).dot(w))
            w_order = (((self.x - self.x_mean).dot(w)) ** 3).mean()
            return - w_order / (w_std ** 3)

        # opt process
        w_init = np.ones(self.n_asset) / self.n_asset
        w_constrain = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.})
        w_bound = [(0, 1) for i in range(self.n_asset)]
        result = opt_min(obj_func, w_init,
                         constraints=w_constrain, bounds=w_bound)
        return result.x, -result.fun

    def optimize_pgp(self):
        self.w1, R_star = self.optimize_1order()
        self.w2, V_star = self.optimize_2order()
        self.w3, S_star = self.optimize_3order()

        def obj_func(w):
            d1 = R_star - (w * self.x_mean).sum()
            d2 = w.dot(self.cov).dot(w) - V_star
            w_std = np.sqrt(w.dot(self.cov).dot(w))
            w_order = (((self.x - self.x_mean).dot(w)) ** 3).mean()
            d3 = S_star - w_order / w_std ** 3
            # obj_1 = self.lam_1 * np.log(np.abs(d1 / R_star))
            # obj_2 = self.lam_2 * np.log(np.abs(d2 / V_star))
            # obj_3 = self.lam_3 * np.log(np.abs(d3 / S_star))
            obj_1 = np.abs(d1 / R_star) ** self.lam_1
            obj_2 = np.abs(d2 / V_star) ** self.lam_2
            obj_3 = np.abs(d3 / S_star) ** self.lam_3
            return obj_1 + obj_2 + obj_3

        w_init = np.ones(self.n_asset) / self.n_asset
        w_constrain = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.})
        w_bound = [(0, 1) for i in range(self.n_asset)]
        result = opt_min(obj_func, w_init,
                         constraints=w_constrain, bounds=w_bound)

        self.result = result
        return result.x, result.fun

    def fit(self, X):
        self.x = X
        w, opt = self.optimize_pgp()
        return w, opt


def get_shape_ratio(ret_ser):
    return_mean = ret_ser.mean()
    return_std = ret_ser.std()
    return return_mean / return_std * np.sqrt(12)


# %%
if __name__ == '__main__':
    model = PGP_skew(1, 2, 3, n_asset=5)
    x = np.random.uniform(-0.1, 0.1, size=(200, 5))
    w2, R2 = model.fit(x)
    print(model.fit(x))
