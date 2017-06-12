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

#include "kf_base.h"

namespace spkf {

    /* forward declaration */
    template <class Process_t, class Observe_t>
    class EKF;

    /* EKF traits */
    template <class Process_t, class Observe_t>
    struct traits<EKF<Process_t, Observe_t>> {

        using scalar_t = typename Process_t::scalar_t; /* floating-point type */
        constexpr static const unsigned nx = Process_t::state_t::RowsAtCompileTime;   /* state vector dims */
        constexpr static const unsigned nu = Process_t::control_t::RowsAtCompileTime; /* control vector dims */
        constexpr static const unsigned nz = Observe_t::obs_t::RowsAtCompileTime;     /* observation vector dims */
    };

    template <class Process_t, class Observe_t>
    class EKF : public KFBase<EKF<Process_t, Observe_t>> {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /* base can touch my private bits */
        friend class KFBase<EKF<Process_t, Observe_t>>;

        using Base = KFBase<EKF<Process_t, Observe_t>>;
        using Base::nx;
        using Base::nz;

        using scalar_t = typename Base::scalar_t;                   /* floating-point scalar type */
        using state_t = typename Base::state_t;                     /* state vector type */
        using control_t = typename Base::control_t;                 /* control vector type */
        using obs_t = typename Base::obs_t;                         /* observation vector type */
        using covar_t = typename Base::covar_t;                     /* state covariance matrix type */
        using obs_covar_t = typename Base::obs_covar_t;             /* observation covariance matrix type */
        using proc_jacobian_t = Matrix<scalar_t, nx, nx>;           /* process model Jacobian matrix type */
        using obs_jacobian_t = Matrix<scalar_t, nz, nx>;            /* observation model Jacobian matrix type */
        using cross_covar_t = typename Base::cross_covar_t;

        /* default constructor */
        EKF() = default;

        /* constructor */
        EKF(const Ref<const state_t> &state,
            const Ref<const covar_t> &covar,
            const Ref<const covar_t> &proc_covar,
            const Ref<const obs_covar_t> & obs_covar)
                : Base(state, covar, proc_covar, obs_covar) {

        }

    private:

        inline bool _process(Ref<state_t> state_k,
                             const Ref<const control_t> &control_k,
                             const scalar_t del_k) {

            /* compute process Jacobian at current state */
            _update_proc_jacobian(state_k, control_k, del_k);

            /* update state using the process model */
            auto proc_noise_k = this->proc_noise();
            _f(state_k, control_k, proc_noise_k, del_k);

            return true;
        }

        inline bool _process_covar(Ref<covar_t> covar_k) const {

            auto proc_covar_k = this->proc_covar();
            covar_k = _F * covar_k * _F.transpose() + proc_covar_k;
            return true;
        }

        inline bool _observe(Ref<obs_t> observation_k) {

            /* compute observation Jacobian at current state */
            auto state_k = this->state();
            _update_obs_jacobian(state_k);

            /* predict measurement using the observation model */
            auto obs_noise_k = this->obs_noise();
            _h(state_k, observation_k, obs_noise_k);

            return true;
        }

        inline bool _innovation_covar(Ref<obs_covar_t> inov_covar_k) const {

            const auto& covar_k = this->covar();
            const auto& obs_covar_k = this->obs_covar();

            inov_covar_k = _H * covar_k * _H.transpose() + obs_covar_k;
            return true;
        }

        inline bool _kalman_gain(Ref<cross_covar_t> kalman_gain_k) const {

            const auto& inov_covar_k = this->inov_covar();
            const auto& covar_k = this->covar();

            kalman_gain_k = inov_covar_k.transpose().fullPivHouseholderQr().solve(
                    ( covar_k * _H.transpose() ).transpose()
            ).transpose();

            return true;
        }

        inline bool _update_covar(Ref<covar_t> covar_k) const {

            const auto& kalman_gain_k = this->kalman_gain();
            covar_k = (covar_t::Identity() - kalman_gain_k * _H) * covar_k;
            return true;
        }

        inline bool _update_proc_jacobian(const Ref<const state_t> &state_k,
                                          const Ref<const control_t> &control_k,
                                          const scalar_t del_k) {

            /* forward and backward states */
            state_t state_f;
            state_t state_b ;
            for (auto i = 0; i < proc_jacobian_t::ColsAtCompileTime; ++i) {

                state_f = state_k;
                state_b = state_k;

                /* perturb ith state element */
                const scalar_t h_half = 0.5*eps<scalar_t>(state_k[i]);
                state_f[i] += h_half;
                state_b[i] -= h_half;

                /* pass through process model */
                const auto& proc_noise_k = this->proc_noise();
                _f(state_f, control_k, proc_noise_k, del_k);
                _f(state_b, control_k, proc_noise_k, del_k);

                /* central difference */
                _F.col(i) = (state_f - state_b)/(2.0 * h_half);
            }
            return true;
        }

        inline bool _update_obs_jacobian(const Ref<const state_t> &state_k) {

            /* forward, backward observations */
            obs_t obs_f = obs_t::Zero();
            obs_t obs_b = obs_t::Zero();

            /* forward and backward states */
            state_t state_f;
            state_t state_b ;

            for (auto i = 0; i < proc_jacobian_t::ColsAtCompileTime; ++i) {

                state_f = state_k;
                state_b = state_k;

                /* perturb ith state element */
                const scalar_t h_half = 0.5*eps<scalar_t>(state_k[i]);
                state_f[i] += h_half;
                state_b[i] -= h_half;

                /* pass through observation model */
                const auto& obs_noise_k = this->obs_noise();
                _h(state_f, obs_f, obs_noise_k);
                _h(state_b, obs_b, obs_noise_k);

                /* central difference */
                _H.col(i) = (obs_f - obs_b)/(2.0 * h_half);
            }
            return true;
        }

        Process_t _f;           /* process model f() */
        Observe_t _h;           /* observation model h() */
        proc_jacobian_t _F;     /* process model Jacobian F */
        obs_jacobian_t _H;      /* observation model Jacobian H */
    };
}