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
#include <iostream>
#include <type_traits>

namespace spkf {

    using Eigen::Matrix;
    using Eigen::Ref;
    using Eigen::Upper;
    using Eigen::Lower;
    using Eigen::internal::llt_inplace;
    using Eigen::LLT;

    /* pi */
    template <typename Scalar>
    constexpr Scalar const pi() { return std::acos(-Scalar(1)); }

    /* eps */
    template <typename Scalar>
    const Scalar eps(const Scalar x) {
        const Scalar sqrt_machine_eps = sqrt(std::numeric_limits<Scalar>::epsilon());
        return ( sqrt_machine_eps * std::max(std::abs(x), sqrt_machine_eps) );
    }

    /* singularity free angle addition */
    template <typename Scalar>
    Scalar add_rads(const Scalar &rad1, const Scalar &rad2) {
        auto sum = rad1 + rad2;
        if (sum > pi<Scalar>()) sum -= 2.0*pi<Scalar>();
        if (sum < -1.0*pi<Scalar>()) sum += 2.0*pi<Scalar>();
        return sum;
    }

    /* ignore unused params */
    template <typename T>
    void ignore(T &&) { }

    /* sign function */
    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
}
