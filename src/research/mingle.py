# -*- coding: utf-8 -*-
"""联合因子-图估计 (MINGLE) 与基于图的配置算法。

方法来源是一项把"图上的邻接关系定义在因子暴露空间、并与因子联合估计"的工作。
本模块按其公开描述实现, 未见其代码, 因此凡描述不明确处均在注释中标出选择。

目标函数
--------
    min_{B,F,W}  ||(R - B F) Ω^{1/2}||_F^2                  收益重构(时间指数加权)
               + α Σ_ij W_ij ||b_i - b_j||^2                暴露平滑(= 2α tr(B'LB))
               + β ||W||_1                                  稀疏
               - γ Σ_i log(d_i)                             对数度(抑制孤立点)
               + δ ||offdiag(Cov_ω(F))||_F^2                因子去相关
    s.t.  B'B = I,  W ≥ 0,  W = W',  diag(W) = 0,  d = W1

关键点: 平滑项约束的是**暴露**在图上的变化, 不是原始收益逐期接近。
个股收益的短期扰动要先经过低秩分解才能影响连边, 这是它与"直接用收益相关建图"
的根本差别。

ADMM 拆分
---------
    B  Sylvester 方程 (P B + B M = E), 特征分解求解
    Z_B  正交约束副本, 极分解(SVD)投影到 Stiefel 流形
    F  梯度步(闭式初值 F = B'R, 再对去相关项走几步)
    W  投影梯度 + 度副本 d 的解析解
双线性项使问题非凸, ADMM 不保证全局最优; 初始化与迭代次数会影响结果。
"""
import numpy as np


# ============================ 联合估计 ============================
def _sylvester(P_eigval, P_eigvec, M, E):
    """解 P B + B M = E, 其中 P 已给出特征分解, M 对称半正定。"""
    th, V = np.linalg.eigh(M)
    G = P_eigvec.T @ E @ V
    Y = G / (P_eigval[:, None] + th[None, :])
    return P_eigvec @ Y @ V.T


def _degree_op(W):
    return W.sum(1)


def fit_mingle(R, K=6, halflife=252, alpha=1.0, beta=None, gamma=1.0, delta=1.0,
               target_degree=10.0, n_outer=150, rho_B=1.0, rho_W=1.0,
               seed=0, tol=1e-5, verbose=False):
    """R: (N, T) 收益矩阵(已按股票去均值)。返回 B, F, W, Psi, info。

    beta 缺省时由 target_degree 自动标定 —— 原工作未给正则强度, 若按样本外表现去调
    就是泄漏。这里改为按**结构目标**(平均度数)标定, 与最终评估无关。
    """
    rng = np.random.default_rng(seed)
    N, T = R.shape
    w_t = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)
    w_t = w_t / w_t.mean()
    s_w = w_t.sum()
    Rw = R * w_t                                   # R Ω

    # 初始化: 加权 PCA
    C0 = (Rw @ R.T) / s_w
    ev, evec = np.linalg.eigh(C0)
    B = evec[:, -K:][:, ::-1].copy()
    F = B.T @ R
    Z_B, U_B = B.copy(), np.zeros_like(B)

    iu = np.triu_indices(N, 1)
    n_edge = len(iu[0])
    w_vec = np.full(n_edge, target_degree / max(N - 1, 1))
    d_cp = np.full(N, target_degree)
    u_W = np.zeros(N)

    def W_from_vec(v):
        Wm = np.zeros((N, N))
        Wm[iu] = v
        return Wm + Wm.T

    beta_auto = beta is None
    hist = []
    for it in range(n_outer):
        W = W_from_vec(w_vec)
        deg = _degree_op(W)
        L = np.diag(deg) - W

        # ---- B: Sylvester ----
        P = 4.0 * alpha * L + rho_B * np.eye(N)
        pv, pvec = np.linalg.eigh(P)
        pv = np.maximum(pv, 1e-9)
        M = 2.0 * (F * w_t) @ F.T
        E = 2.0 * Rw @ F.T + rho_B * (Z_B - U_B)
        B = _sylvester(pv, pvec, M, E)

        # ---- Z_B: 投影到 Stiefel(极分解) ----
        A = B + U_B
        Uu, _, Vt = np.linalg.svd(A, full_matrices=False)
        Z_B = Uu @ Vt
        U_B = U_B + B - Z_B

        # ---- F: 闭式初值 + 去相关梯度步 ----
        F = Z_B.T @ R
        if delta > 0:
            for _ in range(5):
                Cf = (F * w_t) @ F.T / s_w
                Gd = 2.0 * (Cf - np.diag(np.diag(Cf)))
                gradF = -2.0 * Z_B.T @ (Rw - Z_B @ (F * w_t)) + delta * (2.0 / s_w) * Gd @ (F * w_t)
                step = 1.0 / (2.0 * w_t.max() + 1e-9)
                F = F - 0.05 * step * gradF

        # ---- W: 暴露距离 + 投影梯度, 度副本解析解 ----
        Bz = Z_B
        sq = (Bz ** 2).sum(1)
        Zd = sq[:, None] + sq[None, :] - 2.0 * (Bz @ Bz.T)
        np.maximum(Zd, 0, out=Zd)
        z_vec = Zd[iu]
        if beta_auto:
            # 标定 beta: 使"边成本"的中位数与 z 的尺度可比, 再由 gamma 拉起度数
            beta_use = np.median(z_vec) * 0.5
        else:
            beta_use = beta
        c = 2.0 * alpha * z_vec + 2.0 * beta_use

        for _ in range(30):
            deg_w = np.zeros(N)
            np.add.at(deg_w, iu[0], w_vec)
            np.add.at(deg_w, iu[1], w_vec)
            resid = deg_w - d_cp + u_W
            grad = c + rho_W * (resid[iu[0]] + resid[iu[1]])
            lip = rho_W * 2.0 * N
            w_vec = np.maximum(w_vec - grad / lip, 0.0)

        deg_w = np.zeros(N)
        np.add.at(deg_w, iu[0], w_vec)
        np.add.at(deg_w, iu[1], w_vec)
        m = deg_w + u_W
        d_cp = (m + np.sqrt(m ** 2 + 4.0 * gamma / rho_W)) / 2.0
        u_W = u_W + deg_w - d_cp

        obj = float(((R - Z_B @ F) ** 2 * w_t).sum())
        hist.append(obj)
        if verbose and it % 25 == 0:
            print(f'  it{it:3d} recon={obj:.3e} avg_deg={deg_w.mean():.2f} '
                  f'nnz={(w_vec > 1e-8).mean():.3f}')
        if it > 20 and abs(hist[-1] - hist[-2]) < tol * max(abs(hist[-2]), 1e-12):
            break

    W = W_from_vec(w_vec)
    Resid = R - Z_B @ F
    Psi = (Resid ** 2 * w_t).sum(1) / s_w
    info = {'n_iter': it + 1, 'avg_degree': float(W.sum(1).mean()),
            'edge_density': float((w_vec > 1e-8).mean()), 'recon': hist[-1]}
    return Z_B, F, W, Psi, info


def factor_cov(B, F, Psi, halflife=252):
    """Σ = B Cov_ω(F) B' + diag(Psi)。"""
    T = F.shape[1]
    w_t = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)
    w_t = w_t / w_t.mean()
    Cf = (F * w_t) @ F.T / w_t.sum()
    return B @ Cf @ B.T + np.diag(Psi)


def pca_factor_cov(R, K=6, halflife=252):
    """普通因子协方差(不带图约束): 加权 PCA + 对角残差。"""
    N, T = R.shape
    w_t = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)
    w_t = w_t / w_t.mean()
    C0 = ((R * w_t) @ R.T) / w_t.sum()
    ev, evec = np.linalg.eigh(C0)
    B = evec[:, -K:]
    F = B.T @ R
    Resid = R - B @ F
    Psi = (Resid ** 2 * w_t).sum(1) / w_t.sum()
    Cf = (F * w_t) @ F.T / w_t.sum()
    return B @ Cf @ B.T + np.diag(Psi)


def sample_cov(R, halflife=252):
    N, T = R.shape
    w_t = 0.5 ** (np.arange(T - 1, -1, -1) / halflife)
    w_t = w_t / w_t.mean()
    return ((R * w_t) @ R.T) / w_t.sum()


# ============================ 配置算法 ============================
def _fiedler_split(Sigma, idx, adaptive=True):
    """对 idx 子集做谱二分。adaptive: 在前几个非平凡特征向量中选切分质量最好的。

    原描述提到"自适应谱切分"会在窗口间选择是否使用菲德勒向量, 但未说明备选与判据。
    这里的实现: 候选 = 第2~4小的归一化拉普拉斯特征向量, 判据 = 归一化割最小。
    """
    S = Sigma[np.ix_(idx, idx)]
    dsd = np.sqrt(np.maximum(np.diag(S), 1e-16))
    Cm = S / np.outer(dsd, dsd)
    A = np.abs(Cm)          # 亲和 = |相关|: 高相关的应留在同一簇, 谱切分切开弱连接处
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    if (d <= 1e-12).any() or len(idx) < 4:
        h = len(idx) // 2
        return idx[:h], idx[h:], 0
    Dm = 1.0 / np.sqrt(d)
    Ln = np.eye(len(idx)) - (A * Dm[:, None]) * Dm[None, :]
    ev, evec = np.linalg.eigh(Ln)
    cands = range(1, min(4, len(idx)))
    best, best_cut, best_k = None, np.inf, 1
    for k in cands:
        v = evec[:, k]
        m = v > np.median(v)
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        cut = A[np.ix_(m, ~m)].sum()
        ncut = cut / max(A[m].sum(), 1e-12) + cut / max(A[~m].sum(), 1e-12)
        if ncut < best_cut:
            best_cut, best, best_k = ncut, m, k
        if not adaptive:
            break
    if best is None:
        h = len(idx) // 2
        return idx[:h], idx[h:], 0
    return idx[best], idx[~best], best_k


def spectral_leaves(Sigma, n_leaves=24, adaptive=True):
    """递归谱二分到 n_leaves 个叶子簇。返回 (叶子索引列表, 用菲德勒的比例)。"""
    N = Sigma.shape[0]
    leaves = [np.arange(N)]
    used_fiedler = []
    while len(leaves) < n_leaves:
        sizes = [len(x) for x in leaves]
        j = int(np.argmax(sizes))
        if sizes[j] < 4:
            break
        a, b, k = _fiedler_split(Sigma, leaves[j], adaptive)
        if len(a) == 0 or len(b) == 0:
            break
        used_fiedler.append(k == 1)
        leaves = leaves[:j] + [a, b] + leaves[j + 1:]
    frac = float(np.mean(used_fiedler)) if used_fiedler else np.nan
    return leaves, frac


def _cluster_var(Sigma, idx, w):
    s = Sigma[np.ix_(idx, idx)]
    return float(w @ s @ w)


def allocate(Sigma, leaves, intra_w=None):
    """簇间用递归二分的逆方差配置; 簇内权重由 intra_w 给出(缺省等权)。"""
    N = Sigma.shape[0]
    inner = []
    for lf in leaves:
        if intra_w is None:
            wv = np.full(len(lf), 1.0 / len(lf))
        else:
            v = np.maximum(intra_w[lf], 0.0)
            wv = v / v.sum() if v.sum() > 1e-12 else np.full(len(lf), 1.0 / len(lf))
        inner.append(wv)

    order = list(range(len(leaves)))
    alloc = np.ones(len(leaves))

    def rec(items):
        if len(items) <= 1:
            return
        h = len(items) // 2
        left, right = items[:h], items[h:]
        vl = sum(_cluster_var(Sigma, leaves[i], inner[i]) for i in left)
        vr = sum(_cluster_var(Sigma, leaves[i], inner[i]) for i in right)
        tot = vl + vr
        if tot <= 0:
            fl = 0.5
        else:
            fl = 1.0 - vl / tot
        for i in left:
            alloc[i] *= fl
        for i in right:
            alloc[i] *= (1.0 - fl)
        rec(left)
        rec(right)

    rec(order)
    w = np.zeros(N)
    for i, lf in enumerate(leaves):
        w[lf] = alloc[i] * inner[i]
    return w / w.sum()


def contagion_intra(W, leaves):
    """簇内加权度数的倒数 —— 资金向网络边缘节点倾斜。

    需要处理簇内度数接近零的情形, 否则倒数加权会把权重集中到一两只上;
    全图的度数约束不能自动排除分簇后的这一问题。
    """
    N = W.shape[0]
    out = np.zeros(N)
    for lf in leaves:
        sub = W[np.ix_(lf, lf)]
        deg = sub.sum(1)
        pos = deg[deg > 1e-12]
        floor = np.median(pos) * 0.1 if len(pos) else 1.0
        deg = np.maximum(deg, floor)
        out[lf] = 1.0 / deg
    return out
