import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# %%
def generate_group_data(m=1000, n=10, n_group=100, sigma=2, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_gs, X_gs = [], []
    y_temp = np.zeros(shape=(m,))
    idxs = np.random.choice(range(n_group), int(density * n_group), replace=False)
    for i in range(n_group):
        beta_star = np.random.randn(n)
        X_g = np.random.randn(m, n)
        if i in idxs:
            y_g = X_g.dot(beta_star) + np.random.normal(0, sigma, size=m)
            beta_gs.append(beta_star)
        else:
            y_g = np.zeros(shape=(m,))
            beta_gs.append(np.zeros_like(beta_star))
        y_temp += y_g
        X_gs.append(X_g)

    return X_gs, beta_gs, y_temp


# %%

class group_lasso_model:

    def __init__(self, intercept=False, penalty=0.3, max_iters=1000, solver='default', n_params=20):
        self.intercept = intercept
        self.penalty = penalty
        self.max_iters = max_iters
        self.solver = solver
        self.n_params = n_params

    def get_group_index(self, X_g):  # 获取数据的index，组特征的index相同，[1,1,1,1,2,2,2,2,3,3,3...]
        group_idx = []
        for ind, data in enumerate(X_g):
            group_idx.extend([ind + 1] * data.shape[1])
        group_idx = np.asarray(group_idx).astype(int)
        return group_idx, np.hstack(X_g)

    def cvxpy_solver_options(self, solver):  # 确定优化器的使用，一般使用默认
        if solver == 'ECOS':
            solver_dict = dict(solver=solver,
                               max_iters=self.max_iters)
        elif solver == 'OSQP':
            solver_dict = dict(solver=solver,
                               max_iter=self.max_iters)
        else:
            solver_dict = dict(solver=solver)
        return solver_dict

    def get_beta_from_group_index(self, group_idx):  # 通过组特征的index，来生成需要优化的组参数beta
        group_size = []
        beta_var = []
        unique_group = np.unique(group_idx)
        for idx in unique_group:
            group_size.append(len(np.where(group_idx == idx)[0]))
            beta_var.append(cp.Variable(len(np.where(group_idx == idx)[0])))
        return group_size, beta_var

    def fit(self, X_g, y, weights=[]):
        group_idx, X = self.get_group_index(X_g)
        group_sizes, beta_var = self.get_beta_from_group_index(group_idx)

        unique_group_index = np.unique(group_idx)
        num_groups = len(group_sizes)
        n = X.shape[0]

        start_group = 0
        model_prediction = 0
        group_lasso_penalization = 0

        if self.intercept:  # 是否需要加入截距项
            group_idx = np.append(0, group_idx)
            unique_group_index = np.unique(group_idx)
            X = np.column_stack[np.ones(X.shape[0]), X]
            group_sizes = [1] + group_sizes
            beta_var = [cp.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # computer the intercept prediction
            model_prediction = X[:, np.where(group_idx == unique_group_index[0])[0]] @ beta_var[0]
            start_group = 1

        if len(weights) == 0:  # 与adaptive机制对应，给每个项不同的承罚权重
            weights = np.ones_like(group_sizes)

        for i in range(start_group, num_groups):  # 计算预测值与惩罚项的值
            model_prediction += X[:, np.where(group_idx == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cp.sqrt(group_sizes[i]) * cp.norm(beta_var[i], 2) * weights[i]

        # define objective function
        obj_func = 1 / n * cp.sum_squares(y - model_prediction)
        lambd_params = cp.Parameter(nonneg=True)
        object = cp.Minimize(obj_func + lambd_params * group_lasso_penalization)
        problem = cp.Problem(object)

        lambd_values = np.logspace(-self.penalty, 0, self.n_params)

        # solver problem
        beta_with_penalty = {}
        flag = 1
        for lam in lambd_values:  # 遍历惩罚系数，属于超参数的选择
            lambd_params.value = lam

            # 整个try except 的过程是用来保证优化过程能够继续的逻辑。如果某一个优化算法优化失败，会切换其他优化方法
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver = self.cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver)

            except (ValueError, cp.error.SolverError):
                solver = ['ECOS', 'OSQP', 'SCS']

                for elt in solver:
                    solver_dict = self.cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cp.error.SolverError):
                        continue

            self.solver_stats = problem.solver_stats
            sig_beta = [b.value for b in beta_var]
            beta_with_penalty[flag] = sig_beta
            flag += 1
            self.beta_ = beta_with_penalty
        return beta_with_penalty

    def predict_with_multi_penalty(self, X_g):  # 得到具有多个罚项的预测结果，是得到超参数结果的一部分
        pred_with_penlty = {}
        penty_keys = self.beta_.keys()
        for k in penty_keys:
            pred = 0
            g_beta = self.beta_[k]
            for b, x in zip(g_beta, X_g):
                pred += x @ b
            pred_with_penlty[k] = pred
        return pred_with_penlty

    def bic_cti(self, y, beta_dict, pred_y_dict):
        penty_keys = self.beta_.keys()
        lambd_values = np.logspace(-self.penalty, 0, self.n_params)
        n = y.shape[0]
        bics = []
        for k in penty_keys:
            pred_y = pred_y_dict[k]
            se = np.square(pred_y - y).sum() / (2 * (pred_y - y).var())
            n_betas = (np.abs(np.hstack(beta_dict[k])) > 1e-4).astype(int).sum()
            bic = np.log(n) * n_betas + se
            bics.append(bic)
        best_ind = np.argmin(bics)
        return lambd_values[best_ind], pred_y_dict[best_ind + 1], beta_dict[best_ind + 1]

    # use the penalty we define
    # 注：这一部分是我们已经确定好了惩罚系数，在确定的惩罚系数下进行优化。上一部分的逻辑是说在我们没有确定好惩罚系数的
    # 前提下，对惩罚系数进行一系列的遍历，从而可以选择一个最优的惩罚系数。相应的，遍历惩罚系数时间也是成倍数增加的。
    # 我们在实际使用时，是确定好惩罚系数再进行优化的。即，主体的优化使用是这一部分。
    # 这部分主要内容与fit()函数中的类似。
    def fit_with_panelty(self, X_g, y, weights=[]):
        group_idx, X = self.get_group_index(X_g)
        group_sizes, beta_var = self.get_beta_from_group_index(group_idx)

        unique_group_index = np.unique(group_idx)
        num_groups = len(group_sizes)
        n = X.shape[0]

        start_group = 0
        model_prediction = 0
        group_lasso_penalization = 0

        if self.intercept:
            group_idx = np.append(0, group_idx)
            unique_group_index = np.unique(group_idx)
            X = np.column_stack[np.ones(X.shape[0]), X]
            group_sizes = [1] + group_sizes
            beta_var = [cp.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # computer the intercept prediction
            model_prediction = X[:, np.where(group_idx == unique_group_index[0])[0]] @ beta_var[0]
            start_group = 1

        if len(weights) == 0:
            weights = np.ones_like(group_sizes)

        for i in range(start_group, num_groups):
            model_prediction += X[:, np.where(group_idx == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cp.sqrt(group_sizes[i]) * cp.norm(beta_var[i], 2) * weights[i]

        # define objective function
        obj_func = 1 / n * cp.sum_squares(y - model_prediction)
        lambd_params = cp.Parameter(nonneg=True)
        object = cp.Minimize(obj_func + lambd_params * group_lasso_penalization)
        problem = cp.Problem(object)

        # solver problem
        lambd_params.value = self.penalty

        try:
            if self.solver == 'default':
                problem.solve(warm_start=True)
            else:
                solver = self.cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver)

        except (ValueError, cp.error.SolverError):
            solver = ['ECOS', 'OSQP', 'SCS']

            for elt in solver:
                solver_dict = self.cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if 'optimal' in problem.status:
                        break
                except (ValueError, cp.error.SolverError):
                    continue

        self.solver_stats = problem.solver_stats

        sig_beta = [b.value for b in beta_var]
        self.sig_beta = sig_beta
        return sig_beta

    def predict_penalty(self, X_g):  # 在得到beta系数后，对结果进行预测
        pred = 0
        g_beta = self.sig_beta
        for b, x in zip(g_beta, X_g):
            pred += x @ b
        return pred


# %%
if __name__ == '__main__':
    import time

    print(time.asctime())
    X_g, betas, y = generate_group_data()

    # 此处的5对应1e-5到1的区间，对应log取值，
    # n_params 表示取多少个数据，例如，此时一共取10个数据点。
    model = group_lasso_model(intercept=False, penalty=5, n_params=10)

    # 使用类中的fit函数进行训练，返回的是得到的beta的系数，以字典的形式保存。
    b = model.fit(X_g, y)

    # 此时预测的是10个参数分别得到的预测值，预测保存在一个字典中。
    preds = model.predict_with_multi_penalty(X_g)
    # %%
    print(preds)

    plt.plot(y, y)
    for i in preds.keys():
        plt.plot(y, preds[i], '.', alpha=0.3)
    plt.show()
    print('multi_penalty has done')

    # %%
    penty, pred, beta = model.bic_cti(y, beta_dict=b, pred_y_dict=preds)
    print(penty, pred, beta)

    # g_beta = model.fit_with_panelty(X_g, y)
    # pred = model.predict_penalty(X_g)
    # plt.plot(y, y)
    # plt.plot(y, pred, '.')
    # plt.show()
    # print(time.asctime())
