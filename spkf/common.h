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
