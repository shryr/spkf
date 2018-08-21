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

#include "common.h"

namespace spkf {

    /* forward declaration */
    template <class Derived>
    class KFBase;

    /* type traits of Derived class */
    template <class Derived>
    struct traits;

    /**
     * @brief Kalman filter base class
     * @tparam Derived Specialized Kalman filter implementation class
     */
    template <class Derived>
    class KFBase {
    public:

        using scalar_t = typename traits<Derived>::scalar_t;        /* floating-point type */
        constexpr static const unsigned nx = traits<Derived>::nx;   /* state vector dims */
        constexpr static const unsigned nu = traits<Derived>::nu;   /* control vector dims */
        constexpr static const unsigned nz = traits<Derived>::nz;   /* observation vector dims */

        using state_t = Matrix<scalar_t, nx, 1>;
        using control_t = Matrix<scalar_t, nu, 1>;
        using obs_t = Matrix<scalar_t, nz, 1>;
        using covar_t = Matrix<scalar_t, nx, nx>;
        using obs_covar_t = Matrix<scalar_t, nz, nz>;
        using cross_covar_t = Matrix<scalar_t, nx, nz>;

        /* ensure KFBase is only specialized with floating-point types */
        static_assert(std::is_floating_point<scalar_t>::value,
                      "FilterBase can only be instantiated with floating-point types");

        /* default constructor */
        KFBase() = default;

        /* constructor */
        KFBase(const Ref<const state_t> &state,
               const Ref<const covar_t> &covar,
               const Ref<const covar_t> &proc_covar,
               const Ref<const obs_covar_t> & obs_covar) : _state(state),
                                                           _covar(covar),
                                                           _proc_covar(proc_covar),
                                                           _obs_covar(obs_covar) {

        }

        inline bool predict(const Ref<const control_t> &control_k,
                            const scalar_t del_k,
                            const Ref<const state_t> &proc_noise_k) {

            /* update process noise */
            _proc_noise = proc_noise_k;

            /* predict state mean using the process model */
            derived()._process(_state, control_k, del_k);

            /* update state covariance */
            derived()._process_covar(_covar);

            return true;
        }

        inline bool innovate(const Ref<const obs_t> &observation_k,
                             const Ref<const obs_t> &obs_noise_k) {

            /* update observation noise */
            _obs_noise = obs_noise_k;

            /* predict observation from predicted state */
            derived()._observe(_observation);

            /* innovation (observation residual) */
            _innovation = observation_k - _observation;

            /* update innovation covariance */
            derived()._innovation_covar(_inov_covar);

            return true;
        }

        bool update() {

            /* compute kalman gain */
            derived()._kalman_gain(_kalman_gain);

            /* update state mean */
            _state += _kalman_gain * _innovation;

            /* update state covariance */
            derived()._update_covar(_covar);

            return true;
        }

        /* accessors */
        inline const state_t& state() const { return _state; }
        inline const covar_t& covar() const { return _covar; }
        inline const obs_t& observation() const { return _observation; }
        inline const obs_t& innovation() const { return _innovation; }
        inline const covar_t& proc_covar() const { return _proc_covar; }
        inline const obs_covar_t& obs_covar() const { return _obs_covar; }
        inline const cross_covar_t& kalman_gain() const { return _kalman_gain; }
        inline const obs_covar_t& inov_covar() const { return _inov_covar; }
        inline const state_t& proc_noise() const { return _proc_noise; }
        inline const obs_t& obs_noise() const { return _obs_noise; }

    private:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        state_t         _state;                 /* state vector */
        covar_t         _covar;                 /* covariance matrix */
        obs_t           _observation;           /* observation vector */
        obs_t           _innovation;            /* innovation vector */
        covar_t         _proc_covar;            /* process noise covariance */
        obs_covar_t     _obs_covar;             /* measurement noise covariance */
        state_t         _proc_noise;            /* process model additive noise vector */
        obs_t           _obs_noise;             /* observation model additive noise vector */
        obs_covar_t     _inov_covar;            /* innovation covariance */
        cross_covar_t   _kalman_gain;           /* kalman gain */

        /* reference to derived object */
        inline Derived& derived() {
            return *static_cast<Derived*>(this);
        }

        /* const reference to derived object */
        inline const Derived& const_derived() const {
            return *static_cast<const Derived*>(this);
        }
    };

}
