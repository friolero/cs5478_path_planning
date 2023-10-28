import numpy as np
import scipy

from base_planner import BasePlanner
from primitives import Node
from utils import distance, knn, sort_with_distance
from utils import vis_path


class CHOMP:
    def __init__(
        self,
        dt=0.04,
        n_samples=1,
        n_waypoints=64,
        max_iterations=1000,
        lr=0.1,
        grad_clip=30,
        eps=3,
        collision_weight=1,
        smooth_weight=1.0,
        dist_threshold=20,
        sigma_start_init=0.001,
        sigma_end_init=0.001,
        sigma_gp_init=0.3,
    ):

        self._dt = dt
        self._n_waypoints = n_waypoints
        self._n_samples = n_samples
        self._max_iterations = max_iterations
        self._lr = lr
        self._grad_clip = grad_clip
        self._eps = eps
        self._collision_weight = collision_weight
        self._smooth_weight = smooth_weight
        self._dist_threshold = dist_threshold

        self._sigma_start_init = sigma_start_init
        self._k_s_inv = np.eye(2 * 2) / (self._sigma_start_init ** 2)

        self._sigma_end_init = sigma_end_init
        self._k_g_inv = np.eye(2 * 2) / (self._sigma_end_init ** 2)

        self._sigma_gp_init = sigma_gp_init
        self._q_c_inv = np.eye(2) / (self._sigma_gp_init ** 2)
        self._q_c_inv = np.zeros((self._n_waypoints - 1, 2, 2)) + self._q_c_inv
        m1 = 12.0 * (self._dt ** -3.0) * self._q_c_inv
        m2 = -6.0 * (self._dt ** -2.0) * self._q_c_inv
        m3 = 4.0 * (self._dt ** -1.0) * self._q_c_inv
        q_inv_u = np.concatenate([m1, m2], -1)
        q_inv_l = np.concatenate([m2, m3], -1)
        self._k_gp_inv = np.concatenate([q_inv_u, q_inv_l], -2)

        self.sigma_inv = self.get_precision_mat()
        self.sigma = np.linalg.inv(self.sigma_inv).astype(np.float64)

    def get_const_vel_mean(self, start_node, end_node):
        const_vel_mean = np.zeros((self._n_waypoints, 4))
        for i in range(self._n_waypoints):
            const_vel_mean[i][:2] = start_node.array() * (
                self._n_waypoints - 1 - i
            ) * 1.0 / (self._n_waypoints - 1) + end_node.array() * i * 1.0 / (
                self._n_waypoints - 1
            )
        const_vel_mean[1:-1, 2:] = (end_node - start_node).array() / (
            (self._n_waypoints - 1) * self._dt
        )
        const_vel_mean = const_vel_mean.reshape(-1)
        return const_vel_mean

    def get_const_vel_cov(self, K_s_inv, K_gp_inv, K_g_inv, precision_mat=True):
        phi = np.eye(2 * 2, dtype=np.float64)
        phi[:2, 2:] = np.eye(2, dtype=np.float64) * self._dt
        diag_phis = phi
        for _ in range(self._n_waypoints - 1 - 1):
            diag_phis = scipy.linalg.block_diag(diag_phis, phi)
        A = np.eye(4 * self._n_waypoints, dtype=np.float64)
        A[4:, :-4] += -1.0 * diag_phis
        B = np.zeros((4, 4 * self._n_waypoints), dtype=np.float64)
        B[:, -4:] = np.eye(4, dtype=np.float64)
        A = np.concatenate([A, B])

        Q_inv = K_s_inv
        for _ in range(self._n_waypoints - 1):
            Q_inv = scipy.linalg.block_diag(Q_inv, K_gp_inv).astype(np.float64)
        Q_inv = scipy.linalg.block_diag(Q_inv, K_g_inv).astype(np.float64)

        K_inv = A.T @ Q_inv @ A
        # precision matrix
        if precision_mat:
            return K_inv.astype(np.float64)
        # covariance matrix
        else:
            return np.linalg.inv(K_inv).astype(np.float64)

    def sample_trajectories(self, start_node, end_node):
        # N * 4
        mean = self.get_const_vel_mean(start_node, end_node)
        # 4N * 4N
        sigma = self.get_const_vel_cov(
            self._k_s_inv, self._k_gp_inv[0], self._k_g_inv, precision_mat=False
        )
        traj_samples = np.random.multivariate_normal(
            mean, sigma, size=self._n_samples
        )
        return traj_samples.reshape((self._n_samples, self._n_waypoints, -1))

    def get_precision_mat(self):
        upper_diag = np.diag(np.ones(self._n_waypoints - 1), k=1).astype(
            np.float64
        )
        lower_diag = np.diag(np.ones(self._n_waypoints - 1), k=-1).astype(
            np.float64
        )
        diag = -2 * np.eye(self._n_waypoints, dtype=np.float64)
        A = upper_diag + diag + lower_diag
        A = np.concatenate(
            (
                np.zeros((1, self._n_waypoints)),
                A,
                np.zeros((1, self._n_waypoints)),
            ),
            axis=0,
        )
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        A = A * 1.0 / self._dt ** 2
        R = A.T @ A
        return R.astype(np.float64)

    def get_ab_mat(self, start_node, end_node):

        AA_ = np.zeros((self._n_waypoints * 2, self._n_waypoints * 2))
        for i in range(self._n_waypoints):
            AA_[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2] = np.eye(2) * 2.0
            if i > 0:
                AA_[(i - 1) * 2 : i * 2, i * 2 : (i + 1) * 2] = np.eye(2) * -1.0
                AA_[i * 2 : (i + 1) * 2, (i - 1) * 2 : i * 2] = np.eye(2) * -1.0
        AA_ /= (self._dt ** 2) * (self._n_waypoints + 1)

        bb_ = np.zeros((self._n_waypoints * 2))
        bb_[:2] = start_node.array()
        bb_[-2:] = end_node.array()
        bb_ /= -(self._dt ** 2) * (self._n_waypoints + 1)
        return AA_, bb_

    def clip(self, traj, map):
        for i in range(self._n_waypoints):
            traj[i, 0] = min(max(0, traj[i][0]), map.row)
            traj[i, 1] = min(max(0, traj[i][1]), map.col)
        return traj

    def postprocess(self, map, traj):
        updated_traj = []
        for i, wp in enumerate(traj):
            if i == 0:
                parent = None
            else:
                parent = updated_traj[-1]
            updated_traj.append(
                Node(
                    min(max(0, int(wp[0])), map.row),
                    min(max(0, int(wp[1])), map.col),
                    parent,
                )
            )
        return updated_traj

    def d2c(self, dist, tol_radius=3):
        return (
            self._collision_weight / (2 * tol_radius) * (dist - tol_radius) ** 2
        )

    def potential_grad(self, dist, x, y, map):
        if x < map.row - 1:
            grad_x = self.d2c(map.nearest_obstacle(x + 1, y)[0]) - self.d2c(
                dist
            )
        else:
            grad_x = self.d2c(dist) - self.d2c(
                map.nearest_obstacle(x - 1, y)[0]
            )
        if y < map.col - 1:
            grad_y = self.d2c(map.nearest_obstacle(x, y + 1)[0]) - self.d2c(
                dist
            )
        else:
            grad_y = self.d2c(dist) - self.d2c(
                map.nearest_obstacle(x, y - 1)[0]
            )
        return np.array([grad_x, grad_y], dtype=np.float32)

    def plan(self, map, start_node, end_node, return_history=False):

        AA_, bb_ = self.get_ab_mat(start_node, end_node)
        Ainv_ = np.linalg.inv(AA_)

        init_traj = self.sample_trajectories(start_node, end_node)[0][..., :2]

        last_err = 1e6
        n_err_increase = 0
        patience = 5
        decay_rate = 0.1
        lr = self._lr

        success = False
        traj_history = [init_traj]
        for n_iter in range(self._max_iterations):
            xi_ = traj_history[-1]
            nabla_smooth = AA_ @ xi_.reshape(-1) + bb_
            xidd = nabla_smooth.reshape((self._n_waypoints, 2))

            nabla_obs = np.zeros((self._n_waypoints, 2))
            for i in range(self._n_waypoints):
                # backward finite difference
                if i == 0:
                    qd = (xi_[i + 1] - xi_[i]) / self._dt
                # forward finite difference
                elif i == (self._n_waypoints - 1):
                    qd = (xi_[i] - xi_[i - 1]) / self._dt
                # center finite difference
                else:
                    qd = (xi_[i + 1] - xi_[i - 1]) / (2 * self._dt)

                J = np.eye(2)
                vel = np.linalg.norm(qd)
                if vel < 1e-3:
                    continue
                xdn = qd / vel
                xdd = J @ xidd[i]
                prj = np.eye(2) - xdn @ xdn.T
                kappa = prj @ xdd / vel ** 2

                dist, delta_cost = map.nearest_obstacle(
                    int(xi_[i][0]), int(xi_[i][1])
                )

                if dist < self._eps:
                    cost = self.d2c(dist, tol_radius=self._eps)
                    delta_cost = self.potential_grad(
                        dist, int(xi_[i][0]), int(xi_[i][1]), map
                    )
                    nabla_obs[i] = J.T * vel @ (prj @ delta_cost - cost * kappa)

            dxi = Ainv_ @ (
                nabla_obs.reshape(-1) + self._smooth_weight * nabla_smooth
            )
            dxi = np.clip(dxi, a_min=-self._grad_clip, a_max=self._grad_clip)
            dxi = dxi.reshape((self._n_waypoints, 2))
            dxi[0, :] = 0.0
            dxi[-1, :] = 0.0
            xi_ = xi_ - dxi * lr
            xi_ = self.clip(xi_, map)
            traj_history.append(xi_)
            if (n_iter % 100) == 0:
                vis_path(map, self.postprocess(map, traj_history[-1]))

            err = np.linalg.norm(dxi)
            if err >= last_err:
                n_err_increase += 1
                if n_err_increase >= patience:
                    lr *= decay_rate
            else:
                n_err_increase = 0
            last_err = err

            if err < self._dist_threshold:
                break
            else:
                print(n_iter, lr, n_err_increase, err)
        if (n_iter % 100) == 0:
            vis_path(map, self.postprocess(map, traj_history[-1]))
        return self.postprocess(map, traj_history[-1]), success


if __name__ == "__main__":
    from map import ImageMap2D
    from utils import vis_path

    map = ImageMap2D("data/2d_map_4.png")  # , distance_field=True)
    start_node, end_node = np.random.choice(
        map.free_conf, size=2, replace=False
    )

    chomp = CHOMP()
    path, success = chomp.plan(map, start_node, end_node, return_history=True)
    import ipdb

    ipdb.set_trace()
