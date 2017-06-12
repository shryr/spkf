# **spkf**
Sigma-point Kalman Filters

spkf is a header only C++ library with canonical implementations of:

 - **EKF**: Extended Kalman Filter
 - **UKF**: Unscented Kalman Filter
 - **CDKF**: Central Difference Kalman Filter
 - **SqrtUKF**: Square-Root Unscented Kalman Filter
 - **SqrtCDKF**: Square-Root Central Difference Kalman Filter

## Dependencies
- Eigen 3.2.0
```
sudo apt-get install libeigen3-dev
```

- CMake 3.5.0 (optional)
```
sudo apt-get install cmake
```

## Usage
Define Process and Observation models as function object types `process_t` and `observe_t`

### `process_t`
Process model $\boldsymbol{x}_{k+1} = f(\boldsymbol{x}_k, \boldsymbol{u}_k + \boldsymbol{q}_k)$

```
/* process model: x_{k+1} = f(x_k, u_k, q) */
template <typename Scalar>
struct process_t {

    using scalar_t = Scalar;
    using state_t = StateT<scalar_t>;
    using control_t = ControlT<scalar_t>;

    inline bool operator()(Eigen::Ref<state_t> state_k,
                           const Eigen::Ref<const control_t> &control_k,
                           const Eigen::Ref<const state_t> &proc_noise_k,
                           const Scalar del_k) const {

        state_k[0] += state_k[3] * del_k * cos(state_k[2]);
        state_k[1] += state_k[3] * del_k * sin(state_k[2]);
        state_k[2] += state_k[4] * del_k;
        state_k[3] += control_k[0] * del_k;
        state_k[4] += control_k[1] * del_k;

        state_k += proc_noise_k;

        return true;
    }
};
```

Alias these types in **`process_t`**
- **`scalar_t`** scalar floating-point type (e.g. float/double)
- **`state_t`** state vector type
- **`control_t`** control vector type

### `observe_t`
Observation model $\boldsymbol{x}_{k+1} = f(\boldsymbol{x}_k, \boldsymbol{u}_k + \boldsymbol{q}_k)$

```
/* observation model: z_k = h(x_k) */
template <typename Scalar>
struct observe_t {

    using scalar_t = Scalar;
    using state_t = StateVector<Scalar>;
    using meas_t = MeasVector<Scalar>;

    inline bool operator()(const Eigen::Ref<const state_t> &state_k,
                           Eigen::Ref<meas_t> meas_k) const {
        return true;
    }
};
```

Types that need be aliased in **`observe_t`**
- **`scalar_t`** scalar floating-point type (e.g. float/double)
- **`state_t`** state vector type
- **`meas_t`** measurement vector type

`SqrtCDKF`, for instance can be specialized by using **`process_t`** and **`observe_t`** as template parameters
```
spkf::SqrtCDKF<process_t<double>, observe_t<double>>
filter(state_k, covar_k, proc_covar_k, meas_covar_k);
```

Once initialized, the filter can be run by repeatedly calling the predict `filter.predict(control_k, del_k)` and update `filter.update(meas_k)` functions

## Compiling and running the example

- Build
```
mkdir -p build
cd build
cmake ..
make -j
```

- Run
```
cd ../bin/
./unicycle
```

