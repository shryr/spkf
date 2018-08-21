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

#include "sigma_base.h"

namespace spkf {

    /* forward declaration */
    template <class Process_t, class Observe_t>
    class UKF;

    /* UKF traits */
    template <class Process_t, class Observe_t>
    struct traits<UKF<Process_t, Observe_t>> {

        using scalar_t = typename Process_t::scalar_t;                                /* floating-point type */
        constexpr static const unsigned nx = Process_t::state_t::RowsAtCompileTime;   /* state vector dims */
        constexpr static const unsigned nu = Process_t::control_t::RowsAtCompileTime; /* control vector dims */
        constexpr static const unsigned nz = Observe_t::obs_t::RowsAtCompileTime;     /* observation vector dims */
    };

    template <class Process_t, class Observe_t>
    class UKF : public SigmaBase<UKF<Process_t, Observe_t>> {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        friend class SigmaBase<UKF<Process_t, Observe_t>>;
        using Base = SigmaBase<UKF<Process_t, Observe_t>>;
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

        /* default constructor */
        UKF() = default;

        /* constructor */
        UKF(const Ref<const state_t> &state,
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

        inline bool _process_covar_sp(Ref<covar_t> covar_k,
                                      Ref<covar_t> chol_covar_k) {

            const auto& state_k = this->state();
            const auto& state_sigmas_k = this->state_sigmas();

            /* update covariance */
            covar_k = _wc0 * (state_sigmas_k.col(0) - state_k)
                      * (state_sigmas_k.col(0) - state_k).transpose();

            for (unsigned i = 1; i < r; ++i) {
                covar_k += _wci * (state_sigmas_k.col(i) - state_k)
                           * (state_sigmas_k.col(i) - state_k).transpose();
            }

            ignore(chol_covar_k);

            return true;
        }

        inline bool _innovation_covar_sp(Ref<obs_covar_t> inov_covar_k) const {

            const auto& obs_sigmas_k = this->obs_sigmas();
            const auto& observation_k = this->observation();

            /* compute UKF innovation covariance */
            inov_covar_k = _wc0 * (obs_sigmas_k.col(0) - observation_k)
                           * (obs_sigmas_k.col(0) - observation_k).transpose();

            for (unsigned i = 0; i < r; ++i) {
                inov_covar_k += _wci * (obs_sigmas_k.col(i) - observation_k)
                                * (obs_sigmas_k.col(i) - observation_k).transpose();
            }

            return true;
        }

        inline bool _kalman_gain_sp(Ref<cross_covar_t> kalman_gain_k,
                                    Ref<cross_covar_t> cross_covar_k) const {

            const auto& state_k = this->state();
            const auto& observation_k = this->observation();
            const auto& state_sigmas_k = this->state_sigmas();
            const auto& obs_sigmas_k = this->obs_sigmas();
            const auto& inov_covar_k = this->inov_covar();

            /* cross covariance */
            cross_covar_k = _wc0 * (state_sigmas_k.col(0) - state_k) *
                            (obs_sigmas_k.col(0) - observation_k).transpose();

            for (unsigned i = 0; i < r; ++i) {
                cross_covar_k += _wci * (state_sigmas_k.col(i) - state_k) *
                                        (obs_sigmas_k.col(i) - observation_k).transpose();
            }

            /* Kalman gain */
            kalman_gain_k = inov_covar_k.transpose().fullPivHouseholderQr().solve(
                    cross_covar_k.transpose()
            ).transpose();

            return true;
        }

        inline bool _update_covar_sp(Ref<covar_t> covar_k,
                                     Ref<covar_t> chol_covar_k) const {

            const auto& kalman_gain_k = this->kalman_gain();
            const auto& inov_covar_k = this->inov_covar();

            /* UKF covariance update */
            covar_k -= kalman_gain_k * inov_covar_k * kalman_gain_k.transpose();

            ignore(chol_covar_k);

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

        Process_t _f;
        Observe_t _h;
    };
}
