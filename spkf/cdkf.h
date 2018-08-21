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
    class CDKF;

    /* UKF traits */
    template <class Process_t, class Observe_t>
    struct traits<CDKF<Process_t, Observe_t>> {

        using scalar_t = typename Process_t::scalar_t;                                /* floating-point type */
        constexpr static const unsigned nx = Process_t::state_t::RowsAtCompileTime;   /* state vector dims */
        constexpr static const unsigned nu = Process_t::control_t::RowsAtCompileTime; /* control vector dims */
        constexpr static const unsigned nz = Observe_t::obs_t::RowsAtCompileTime;     /* observation vector dims */
    };

    /**
     *  @brief Central Difference Kalman Filter
     *
     *  @tparam Process_t Process Model
     *  @tparam Observe_t Observation Model
     */
    template <class Process_t, class Observe_t>
    class CDKF : public SigmaBase<CDKF<Process_t, Observe_t>> {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        friend class SigmaBase<CDKF<Process_t, Observe_t>>;
        using Base = SigmaBase<CDKF<Process_t, Observe_t>>;
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
        CDKF() = default;

        /* constructor */
        CDKF(const Ref<const state_t> &state,
             const Ref<const covar_t> &covar,
             const Ref<const covar_t> &proc_covar,
             const Ref<const obs_covar_t> & obs_covar,
             const scalar_t hh = sqrt(3.0)) : Base(state, covar, proc_covar, obs_covar) {

            /* CDKF params */
            _hh = hh;
            _g = _hh*_hh;   /* gamma = hh^2 */

            /* CDKF weights */
            _wm0 = (_g - static_cast<scalar_t>(L))/_g;  /* wm0 = (h^2 - L)/h^2 */
            _wmi = 0.5 / _g;                            /* wmi = 1/(2h^2) */
            _wc0 = 0.5 / hh;                            /* wc0 = 1/2h */
            _wci = 0.5 * sqrt(_g -1 ) / _g;             /* wci = (h^2 - 1)/2h^2 */
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

            const auto& state_sigmas_k = this->state_sigmas();

            /* matrix of first state sigma point as column vectors */
            Matrix<scalar_t, nx, L> center;
            for (unsigned i = 0; i < L; ++i) {
                center.col(i) = 2.0 * state_sigmas_k.col(0);
            }

            const auto& front = state_sigmas_k.template block<nx,L>(0,1);
            const auto& back = state_sigmas_k.template block<nx,L>(0,L+1);

            /* augmented matrix */
            _state_aug.template block<nx,L>(0,0) = _wc0 * (front - back);          /* 1st order diff A */
            _state_aug.template block<nx,L>(0,L) = _wci * (front + back - center); /* 2nd order diff B*/

            /* QR factorization */
            chol_covar_k = _state_aug.transpose().fullPivHouseholderQr().matrixQR().topLeftCorner(
                    nx, nx).template triangularView<Upper>();

            /* transpose to lower Cholesky factor */
            chol_covar_k.transposeInPlace();

            /* update state covariance */
            covar_k = chol_covar_k * chol_covar_k.transpose();

            return true;
        }

        inline bool _innovation_covar_sp(Ref<obs_covar_t> inov_covar_k) {

            const auto& obs_sigmas_k = this->obs_sigmas();

            /* matrix of first observation sigma point as column vectors */
            Matrix<scalar_t, nz, L> center;
            for (unsigned i = 0; i < L; ++i) {
                center.col(i) = 2.0 * obs_sigmas_k.col(0);
            }

            const auto& front = obs_sigmas_k.template block<nz,L>(0,1);
            const auto& back = obs_sigmas_k.template block<nz,L>(0,L+1);

            /* augmented matrix of differences */
            _obs_aug.template block<nz,L>(0,0) = _wc0 * (front - back);          /* 1st order diff C */
            _obs_aug.template block<nz,L>(0,L) = _wci * (front + back - center); /* 2nd order diff D */

            /* QR factorization */
            inov_covar_k = _obs_aug.transpose().householderQr().matrixQR().topLeftCorner(
                    nz, nz).template triangularView<Upper>();

            /* transpose to lower Cholesky factor */
            inov_covar_k.transposeInPlace();

            return true;
        }

        inline bool _kalman_gain_sp(Ref<cross_covar_t> kalman_gain_k,
                                    Ref<cross_covar_t> cross_covar_k) const {

            const auto& chol_covar_k = this->chol_covar();
            const auto& inov_covar_k = this->inov_covar();

            /* CDKF cross covariance */
            cross_covar_k = chol_covar_k * _obs_aug.template block<nz,nx>(0,0).transpose();

            /* CDKF kalman gain */
            kalman_gain_k = inov_covar_k.transpose().fullPivHouseholderQr().solve(
                    inov_covar_k.fullPivHouseholderQr().solve(
                            cross_covar_k.transpose())
            ).transpose();

            return true;
        }

        inline bool _update_covar_sp(Ref<covar_t> covar_k,
                                     Ref<covar_t> chol_covar_k) {

            const auto& kalman_gain_k = this->kalman_gain();

            const auto& first = chol_covar_k - (kalman_gain_k * _obs_aug.template block<nz,nx>(0,0));
            const auto& middle = kalman_gain_k * _obs_aug.template block<nz,L-nx>(0,nx);
            const auto& last = kalman_gain_k * _obs_aug.template block<nz,L>(0,L);

            /* reuse augmented matrix */
            _state_aug.template block<nx,nx>(0,0) = first;
            _state_aug.template block<nx,L-nx>(0,nx) = middle;
            _state_aug.template block<nx,L>(0,L) = last;

            chol_covar_k = _state_aug.transpose().fullPivHouseholderQr().matrixQR().topLeftCorner(
                    nx, nx).template triangularView<Upper>();

            /* transpose */
            chol_covar_k.transposeInPlace();

            /* covariance update */
            covar_k = chol_covar_k * chol_covar_k.transpose();

            return true;
        }


        /* CDKF params*/
        scalar_t    _hh;        /* cdkf spread param  */
        scalar_t    _g;         /* gamma: _g = _h*_h */

        covar_t     _chol_covar;

        /* CDKF weights */
        scalar_t    _wm0;
        scalar_t    _wmi;
        scalar_t    _wc0;
        scalar_t    _wci;

        state_aug_t _state_aug; /* augmented matrix of state sigma point differentials */
        obs_aug_t   _obs_aug;   /* augmented matrix of observation sigma point differentials */

        Process_t _f;
        Observe_t _h;
    };
}
