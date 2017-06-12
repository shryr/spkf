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

#include "sqrt_base.h"

namespace spkf {

    /* forward declaration */
    template <class Process_t, class Observe_t>
    class SqrtUKF;

    /* UKF traits */
    template <class Process_t, class Observe_t>
    struct traits<SqrtUKF<Process_t, Observe_t>> {

        using scalar_t = typename Process_t::scalar_t;                                /* floating-point type */
        constexpr static const unsigned nx = Process_t::state_t::RowsAtCompileTime;   /* state vector dims */
        constexpr static const unsigned nu = Process_t::control_t::RowsAtCompileTime; /* control vector dims */
        constexpr static const unsigned nz = Observe_t::obs_t::RowsAtCompileTime;     /* observation vector dims */
    };

    template <class Process_t, class Observe_t>
    class SqrtUKF : public SqrtBase<SqrtUKF<Process_t, Observe_t>> {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        friend class SqrtBase<SqrtUKF<Process_t, Observe_t>>;
        using Base = SqrtBase<SqrtUKF<Process_t, Observe_t>>;
        using Base::nx;
        using Base::nz;
        using Base::L;
        using Base::r;

        using scalar_t = typename Base::scalar_t;                   /* floating-point scalar type */
        using state_t = typename Base::state_t;                     /* state vector type */
        using control_t = typename Base::control_t;                 /* control vector type */
        using obs_t = typename Base::obs_t;                         /* observation vector type */
        using covar_t = typename Base::covar_t;                     /* state covariance matrix type */
        using obs_covar_t = typename Base::obs_covar_t;             /* observation covariance matrix type */
        using cross_covar_t = typename Base::cross_covar_t;         /* cross covariance matrix type */
        using state_sigmas_t = typename Base::state_sigmas_t;       /* state sigma points matrix type */
        using obs_sigmas_t = typename Base::obs_sigmas_t;           /* observation sigma points matrix type */
        using state_aug_t = Matrix<scalar_t,nx,2*L>;                /* augmented state points matrix type */
        using obs_aug_t = Matrix<scalar_t,nz,2*L>;                  /* augmented observation points matrix type */

        /* default constructor */
        SqrtUKF() = default;

        /* constructor */
        SqrtUKF(const Ref<const state_t> &state,
                const Ref<const covar_t> &covar,
                const Ref<const covar_t> &proc_covar,
                const Ref<const obs_covar_t> & obs_covar,
                const scalar_t alpha = 1.0,
                const scalar_t beta = 0.0,
                const scalar_t kappa = 3.0) : Base(state, covar, proc_covar, obs_covar) {

            /* UKF params */
            _a = alpha;
            _b = beta;
            _k = kappa;
            _g = static_cast<scalar_t>(L) + _k;
            _l = _a*_a * (static_cast<scalar_t>(L) + _k) - static_cast<scalar_t>(L);

            /* UKF weights */
            _wm0 = _l/_g;                       /* wm0 = lambda / (L + lambda) */
            _wc0 = _wm0 + (1.0 - _a*_a + _b);   /* wc0 =  wm0 + (1 - alpha^2 + beta) */
            _wmi = 0.5 / _g;                    /* wm0 = 1 / 2(L + lambda) */
            _wci = _wmi;                        /* wci = wmi */
        }

        /* accessors */
        inline const scalar_t& wm0() const { return _wm0; }
        inline const scalar_t& wmi() const { return _wmi; }
        inline const scalar_t& wc0() const { return _wc0; }
        inline const scalar_t& wci() const { return _wci; }
        inline const scalar_t& gamma() const { return _g; }

    private:

        inline bool _process_covar_sr(Ref<covar_t> covar_k,
                                      Ref<covar_t> chol_covar_k) {

            const auto& state_k = this->state();
            const auto& state_sigmas_k = this->state_sigmas();

            /* construct augmented matrix of state sigma point deltas */
            for (unsigned i = 0; i < 2*L; ++i) {
                _state_aug.col(i) = sqrt(_wmi) * (state_sigmas_k.col(i+1) - state_k);
            }

            /* QR factorization, chol_covar will be the upper triangular Cholesky factor after this op */
            chol_covar_k = _state_aug.transpose().fullPivHouseholderQr().matrixQR().topLeftCorner(
                    nx, nx).template triangularView<Upper>();

            /* rank one Cholesky update */
            //const auto& first = sqrt(_wc0) * (state_sigmas_k.col(0) - state_k);
            const auto& first = sqrt(_wc0) * (state_sigmas_k.col(0) - state_k);
            llt_inplace<scalar_t, Upper>::rankUpdate(chol_covar_k, first, sgn(_wc0));

            /* update state covariance */
            covar_k = chol_covar_k.transpose() * chol_covar_k;

            return true;
        }

        inline bool _innovation_covar_sr(Ref<obs_covar_t> inov_covar_k) {

            const auto& obs_sigmas_k = this->obs_sigmas();
            const auto& observation_k = this->observation();

            /* construct augmented matrix of observation sigma point deltas */
            for (unsigned i = 0; i < 2*L; ++i) {
                _obs_aug.col(i) = sqrt(_wmi) * (obs_sigmas_k.col(i+1) - observation_k);
            }

            /* QR factorization, inov_covar_k will be the upper triangular Cholesky factor after this op */
            inov_covar_k = _obs_aug.transpose().householderQr().matrixQR().topLeftCorner(
                    nz, nz).template triangularView<Upper>();

            /* rank one Cholesky update */
            const auto& first = sqrt(_wc0) * (obs_sigmas_k.col(0) - observation_k);
            llt_inplace<scalar_t, Upper>::rankUpdate(inov_covar_k, first, sgn(_wc0));

            /* transpose to lower Cholesky factor */
            inov_covar_k.transposeInPlace();

            return true;
        }

        inline bool _kalman_gain_sr(Ref<cross_covar_t> kalman_gain_k,
                                    Ref<cross_covar_t> cross_covar_k) const {

            const auto& state_sigmas_k = this->state_sigmas();
            const auto& obs_sigmas_k = this->obs_sigmas();
            const auto& state_k = this->state();
            const auto& observation_k = this->observation();
            const auto& inov_covar_k = this->inov_covar();

            /* cross covariance */
            cross_covar_k = _wc0 * (state_sigmas_k.col(0) - state_k)
                            * (obs_sigmas_k.col(0) - observation_k).transpose();

            for (unsigned i = 1; i < 2*L; ++i) {
                cross_covar_k += _wci * (state_sigmas_k.col(i) - state_k)
                                 * (obs_sigmas_k.col(i) - observation_k).transpose();
            }

            /* kalman gain */
            kalman_gain_k = inov_covar_k.transpose().fullPivHouseholderQr().solve(
                    inov_covar_k.fullPivHouseholderQr().solve(
                            cross_covar_k.transpose())
            ).transpose();

            return true;
        }

        inline bool _update_covar_sr(Ref<covar_t> covar_k,
                                     Ref<covar_t> chol_covar_k) {

            const auto& inov_covar_k = this->inov_covar();
            const auto& kalman_gain_k = this->kalman_gain();

            /* Cholesky update factor */
            const auto update_factor = kalman_gain_k * inov_covar_k;

            /* series of rank one Cholesky updates */
            for (unsigned i = 0; i < nz; ++i) {
                llt_inplace<scalar_t, Upper>::rankUpdate(chol_covar_k, update_factor.col(i), scalar_t(-1.0));
            }

            /* transpose to lower */
            chol_covar_k.transposeInPlace();

            /* update state covariance */
            covar_k = chol_covar_k * chol_covar_k.transpose();

            return true;
        }


        /* UKF params*/
        scalar_t _a;      /* alpha: determines the spread of distribution around mean */
        scalar_t _b;      /* beta: incorporates prior distribution knowledge */
        scalar_t _k;      /* kappa: secondary scaling param */
        scalar_t _g;      /* gamma: _g = _a^2 * (L + _k) */
        scalar_t _l;      /* lambda: _l = _y - L */

        /* UKF weights */
        scalar_t _wm0;
        scalar_t _wmi;
        scalar_t _wc0;
        scalar_t _wci;

        state_aug_t _state_aug; /* augmented matrix of state sigma point differentials */
        obs_aug_t   _obs_aug;   /* augmented matrix of observation sigma point differentials */

        Process_t _f;
        Observe_t _h;
    };
}