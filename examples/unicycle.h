/* 
 * Copyright (c) 2017-2018 Shehryar Khurshid <shryr.kd@gmail.com>
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
