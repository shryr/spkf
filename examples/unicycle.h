/*
 * Copyright(C) 2017. Shehryar Khurshid <shehryar87@hotmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3, as published
 * by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranties of
 * MERCHANTABILITY, SATISFACTORY QUALITY, or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#pragma once

#include <Eigen/Eigen>
#include <random>

/* state vector type: x_k = [x_k, y_k, theta_k, v_k, w_k]^T */
template <typename Scalar>
using StateT = typename Eigen::Matrix<Scalar, 5, 1>;

template <typename Scalar>
using CovarT = typename Eigen::Matrix<Scalar, 5, 5>;

/* control vector type: u_k = [a_k, aw_k]^T */
template <typename Scalar>
using ControlT = typename Eigen::Matrix<Scalar, 2, 1>;

/* measurement type: z_k = [x_k, y_k, theta_k, v_k, w_k] */
template <typename Scalar>
using ObsT = typename Eigen::Matrix<Scalar, 5, 1>;

template <typename Scalar>
using ObsCovarT = typename Eigen::Matrix<Scalar, 5, 5>;

/* process model: x_{k+1} = f(x_k, u_k) + q */
template <typename Scalar>
struct process_t {

    using scalar_t = Scalar;
    using state_t = StateT<scalar_t>;
    using control_t = ControlT<scalar_t>;

    inline bool operator()(Eigen::Ref<state_t> state_k,
                           const Eigen::Ref<const control_t> &control_k,
                           const Eigen::Ref<const state_t> &proc_noise_k,
                           const Scalar del_k) const {

        state_k[0] += state_k[3] * del_k * cos(state_k[2]); /* x_{k+1} = x_k + v_k.dk.cos(theta_k) */
        state_k[1] += state_k[3] * del_k * sin(state_k[2]); /* y_{k+1} = y_k + v_k.dk.sin(theta_k) */
        state_k[2] += state_k[4] * del_k;                   /* theta_{k+1} = theta_k + dk.w_k */
        state_k[3] += control_k[0] * del_k;                 /* v_{k+1} =  v_k + a_k.dk */
        state_k[4] += control_k[1] * del_k;                 /* w_{k+1} = w_k +  aw_k.dk */

        state_k += proc_noise_k;                            /* x_{k+1} = x_{k+1} + q (additive process noise)*/

        return true;
    }
};

/* observation model: z_k = h(x_k) + r_k */
template <typename Scalar>
struct observe_t {

    using scalar_t = Scalar;
    using state_t = StateT<scalar_t>;
    using obs_t = ObsT<scalar_t>;

    inline bool operator()(const Eigen::Ref<const state_t> &state_k,
                           Eigen::Ref<obs_t> obs_k,
                           const Eigen::Ref<const obs_t> &obs_noise_k) const {

        obs_k[0] = state_k[0];
        obs_k[1] = state_k[1];
        obs_k[2] = state_k[2];
        obs_k[3] = state_k[3];
        obs_k[4] = state_k[4];

        obs_k += obs_noise_k;   /* z_k = z_k + r (additive observation noise) */

        return true;
    }
};