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

#include "unicycle.h"

#include <math.h>
#include <ekf.h>
#include <ukf.h>
#include <cdkf.h>
#include <sqrt_ukf.h>
#include <sqrt_cdkf.h>

using namespace spkf;
using std::cout;

int main() {

    /* time step and no of itertions */
    const double del_k = 0.01;                                 /* time step: d_k = 10ms */
    const int iter_n = 10000;                                  /* no. of iterations */

    /* state vector: x_k = [x_k, y_k, theta_k, v_k, w_k]^T */
    StateT<double> state;
    state[0] = 0.0;                                             /* x_k = 0.0 meters */
    state[1] = -10.0;                                           /* y_k = -10.0 meters */
    state[2] = 0.0;                                             /* theta_k = 0.0 rads */
    state[3] = spkf::pi<double>();                              /* v_k = pi meters/second^2 */
    state[4] = spkf::pi<double>()/10;                           /* w_k = pi/10 rads/second^2 */

    /* constants */
    const auto covar = 20.0 * CovarT<double>::Identity();       /* state covariance: Px = 20.0*I */
    const auto proc_covar = 5.0 * CovarT<double>::Identity();   /* process noise covariance: Q = 5.0*I */
    const auto obs_covar = 3.0 * ObsCovarT<double>::Identity(); /* observation noise covariance: R = 3.0*I */

    /* variables */
    StateT<double> gt_state = state;                            /* ground truth state */
    StateT<double> proc_noise = StateT<double>::Zero();         /* process noise mean = [0]^T */
    ObsT<double> obs_noise = ObsT<double>::Zero();              /* observation noise mean = [0]^T */
    ObsT<double> gt_obs;                                        /* ground truth observation */
    ObsT<double> prt_obs;                                       /* perturbed measurement */
    ControlT<double> control;                                   /* control vector u_k = [a_k, w_t]^T */

    /* models */
    process_t<double> f;                                        /* process model f() */
    observe_t<double> h;                                        /* observation model */

    /* normal distribution */
    std::random_device rd;
    std::mt19937 twister(rd());
    std::normal_distribution<double> proc_dist(0.0, 1.0);       /* process normal rand u=0.0, s=1.0 */
    std::normal_distribution<double> obs_dist(0.0, 0.8);        /* observation normal rand u=0.0, s=0.8 */
    std::normal_distribution<double> control_dist(0.0, 2.0);    /* control normal rand u=0.0, s=2.0 */
    std::normal_distribution<double> prt_dist(0.0, 0.8);        /* control normal rand u=0.0, s=0.8 */

    /* initialize filters */
    /* EKF: extended Kalman filter */
    EKF<process_t<double>, observe_t<double>>
    ekf(state, covar, proc_covar, obs_covar);

    /* UKF: unscented Kalman filter */
    UKF<process_t<double>, observe_t<double>>
    ukf(state, covar, proc_covar, obs_covar);

    /* CDKF: central difference Kalman filter */
    CDKF<process_t<double>, observe_t<double>>
    cdkf(state, covar, proc_covar, obs_covar);

    /* SqrtUKF: square-root, unscented Kalman filter */
    SqrtUKF<process_t<double>, observe_t<double>>
    sqrt_ukf(state, covar, proc_covar, obs_covar);

    /* SqrtCDKF: square-root, central difference Kalman filter */
    SqrtCDKF<process_t<double>, observe_t<double>>
    sqrt_cdkf(state, covar, proc_covar, obs_covar);

    /* iterate for iter_n time steps */
    for (auto i = 0; i < iter_n; ++i) {

        /* generate control commands */
        control[0] = control_dist(twister);   /* a_k m/s^2 */
        control[1] = control_dist(twister);   /* w_k rads/s^2 */

        /* generate process noise */
        for (auto j = 0; j < proc_noise.rows(); ++j) {
            proc_noise[j] = proc_dist(twister);
        }

        /* generate observation noise */
        for (auto j = 0; j < obs_noise.rows(); ++j) {
            obs_noise[j] = obs_dist(twister);
        }

        /* prediction step */
        ekf.predict(control, del_k, proc_noise);
        ukf.predict(control, del_k, proc_noise);
        cdkf.predict(control, del_k, proc_noise);
        sqrt_ukf.predict(control, del_k, proc_noise);
        sqrt_cdkf.predict(control, del_k, proc_noise);

        /* get ground truth state */
        f(gt_state, control, proc_noise, del_k);

        /* propagate it through h() to get ground truth observation */
        h(gt_state, gt_obs, obs_noise);

        /* perturb ground truth measurement */
        for(auto j = 0; j < prt_obs.rows(); ++j) {
            prt_obs[j] = gt_obs[j] + prt_dist(twister);
        }

        /* innovation step with perturbed observation */
        ekf.innovate(prt_obs, obs_noise);
        ukf.innovate(prt_obs, obs_noise);
        cdkf.innovate(prt_obs, obs_noise);
        sqrt_ukf.innovate(prt_obs, obs_noise);
        sqrt_cdkf.innovate(prt_obs, obs_noise);

        /* update step */
        ekf.update();
        ukf.update();
        cdkf.update();
        sqrt_ukf.update();
        sqrt_cdkf.update();
    }

    /* errors */
    const auto& ekf_error = (ekf.state() - gt_state).norm();
    const auto& ukf_error = (ukf.state() - gt_state).norm();
    const auto& cdkf_error = (cdkf.state() - gt_state).norm();
    const auto& sqrt_ukf_error = (sqrt_ukf.state() - gt_state).norm();
    const auto& sqrt_cdkf_error = (sqrt_cdkf.state() - gt_state).norm();

    /* print state estimates */
    cout << "------------- state estimates --------------" << "\n\n";
    cout << "G.T. state:    " << "\n" << gt_state          << "\n\n";
    cout << "EKF state:     " << "\n" << ekf.state()       << "\n\n";
    cout << "UKF state:     " << "\n" << ukf.state()       << "\n\n";
    cout << "CDKF state:    " << "\n" << cdkf.state()      << "\n\n";
    cout << "SR-UKF state:  " << "\n" << sqrt_ukf.state()  << "\n\n";
    cout << "SR-CDKF state: " << "\n" << sqrt_cdkf.state() << "\n\n";

    /* print covariances */
    cout << "---------- covariance estimates ------------" << "\n\n";
    cout << "EKF covar:     " << "\n" << ekf.covar()       << "\n\n";
    cout << "UKF covar:     " << "\n" << ukf.covar()       << "\n\n";
    cout << "CDKF covar:    " << "\n" << cdkf.covar()      << "\n\n";
    cout << "SR-UKF covar:  " << "\n" << sqrt_ukf.covar()  << "\n\n";
    cout << "SR-CDKF covar: " << "\n" << sqrt_cdkf.covar() << "\n\n";

    /* print state errors */
    cout << "--------------- state errors ---------------" << "\n\n";
    cout << "EKF error:     " << "\n" << ekf_error         << "\n\n";
    cout << "UKF state:     " << "\n" << ukf_error         << "\n\n";
    cout << "CDKF state:    " << "\n" << cdkf_error        << "\n\n";
    cout << "SR-UKF state:  " << "\n" << sqrt_ukf_error    << "\n\n";
    cout << "SR-CDKF state: " << "\n" << sqrt_cdkf_error   << "\n\n";

    /* fin */
    return 0;
}