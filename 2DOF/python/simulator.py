import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as lin


def Make_Kmat(k):
    N = len(k)
    K_mat = np.diag(k) - np.diag(k[1:], k=1) - np.diag(k[1:], k=-1)
    K_mat[0:N - 1, 0:N - 1] = K_mat[0:N - 1, 0:N - 1] + np.diag(k[1:])
    return K_mat


def noise(theta, epsilon=0.1):
    return theta + np.random.normal(loc=0, scale=epsilon, size=theta.shape)


def noise_nonzero(theta, epsilon=0.1):
    theta_noise = theta + np.random.normal(loc=0, scale=epsilon, size=theta.shape)
    if np.all(theta_noise > 0):
        return theta_noise
    else:
        return np.array([10 ** (-10), 10 ** (-10)])


def eigenvalue_problem(theta_noise):
    mass = np.array([16.531e3, 16.131e3])
    M = np.diag(mass)

    k1 = 29.7e6
    k2 = 29.7e6
    k = theta_noise * np.array([k1, k2])

    K = Make_Kmat(k)

    eigval, eigvec = lin.eigh(M, K)
    eigval = eigval[::-1]
    U = eigvec[:, ::-1]

    omega = np.sqrt(1.0 / eigval)  # ω_j = √(1/λ_j)
    normal_mass = U.T @ M @ U
    N = len(eigval)
    bet = lin.solve(normal_mass, U.T @ M @ np.ones(N))
    PF = U * bet  # shape: (N, N)
    return omega, PF


def eigenvalue_problem_analytic(theta_noise):
    m1, m2 = 16.531e3, 16.131e3
    k1_0, k2_0 = 29.7e6, 29.7e6

    k1 = theta_noise * k1_0
    k2 = theta_noise * k2_0
    M_mat = np.diag([m1, m2])

    A = k1 * k2
    B = -(m2 * (k1 + k2) + m1 * k2)
    C = m1 * m2

    w_roots = np.roots([A, B, C])  # 2개의 해
    w_sorted = np.sort(w_roots)  # 오름차순: [w_small, w_large]

    # omega는 sqrt(1/w). w가 작을수록 omega가 큼 => 첫 번째가 더 큰 ω
    omega1 = np.sqrt(1.0 / w_sorted[0])  # 큰 진동수
    omega2 = np.sqrt(1.0 / w_sorted[1])  # 작은 진동수
    omega = np.array([omega1, omega2], dtype=np.float64)

    def get_mode_vector(wi):
        a = (m1 - wi * (k1 + k2))
        b = wi * k2
        # a*v1 + b*v2 = 0 => v2 = -a/b, v1=1
        return np.array([1.0, -a / b], dtype=np.float64)

    v1 = get_mode_vector(w_sorted[0])  # 첫 해 = 큰 ω
    v2 = get_mode_vector(w_sorted[1])  # 두 번째 해

    U_unnorm = np.column_stack([v1, v2])  # shape: (2,2)

    def m_norm(vec):
        return np.sqrt(vec @ (M_mat @ vec))

    U = np.empty_like(U_unnorm)
    for i in range(2):
        vec = U_unnorm[:, i]
        U[:, i] = vec / m_norm(vec)

    normal_mass = U.T @ M_mat @ U
    bet = lin.inv(normal_mass) @ (U.T @ M_mat @ np.ones(2))
    PF = U * bet  # shape: (2,2)
    return omega, PF


def theta2freq(theta_noise):
    mass = np.array([16.531 * 10 ** 3, 16.131 * 10 ** 3])
    M = np.diag(mass)
    k1 = 29.7 * 10 ** 6
    k2 = 29.7 * 10 ** 6
    k = theta_noise * np.array([k1, k2])
    K = Make_Kmat(k)
    eigval, eigvec = lin.eigh(M, K)
    omega = np.sqrt(1. / eigval[::-1])
    freq = omega / (2 * np.pi)
    # f = omega / (2. * np.pi)
    U = eigvec[:, ::-1]
    N = len(eigval)
    normal_mass = U.T @ M @ U
    bet = lin.inv(normal_mass) @ U.T @ M @ np.ones((N,))
    PF = np.zeros((N, N))
    for i in range(N):
        PF[:, i] = bet[i] * U[:, i]
    return np.hstack((freq, np.ravel(PF)))


def transfer_function(omega, PF, h=0.05):
    f = np.arange(0, 512 * 0.04, 0.04)
    w = 2.0 * np.pi * f

    w2 = w ** 2
    numerator = w2[None, :]  # shape: (1, M)
    denominator = (omega[:, None] ** 2
                   - w2[None, :]
                   + 2j * h * omega[:, None] * w[None, :])  # shape: (N, M)
    T = numerator / denominator + 1  # shape: (N, M)

    H = PF @ T
    return H


def simulator(theta, sigma):
    yy = []
    theta_noise = noise(theta, sigma)
    omega, PF = eigenvalue_problem(theta_noise)
    H = transfer_function(omega, PF)
    yy.append(np.abs(H[-1, :]).tolist())
    yy.append(np.abs(H[-2, :]).tolist())
    return yy


def simulator_only(theta_noise):
    yy = []
    omega, PF = eigenvalue_problem(theta_noise[:2])
    H = transfer_function(omega, PF)
    yy.append(np.abs(H[-1, :]).tolist())
    yy.append(np.abs(H[-2, :]).tolist())
    return yy
