#ifndef FORMINTEGS_IMPL_HPP
#define FORMINTEGS_IMPL_HPP

/// @file
/// Shared compile-time optional-member detection for domain form operations.
/// Missing fields use defaults; present fields must be usable as constant
/// expressions of the requested type. Inherited accessible members are supported.
/// Linear quadrature functions declare geometry flags directly on their type.

namespace mfem {

namespace internal {

/// Require a value that can be used as a non-type template argument.
/// @tparam T Type of the compile-time value.
/// @tparam value Constant expression being checked.
template <class T, T value> struct RequireConstexpr {};
/// Dependent false value for diagnostics in otherwise unselected templates.
template <class> constexpr bool always_false = false;

#define MFEM_STRINGIFY(x) MFEM_STRINGIFY_IMPL(x)
#define MFEM_STRINGIFY_IMPL(x) #x

#define MFEM_EXTRACT_FIELD(MFEM_EF_TYPE, MFEM_EF_NAME, MFEM_EF_DEFAULT)        \
  template <class T> struct Extract##MFEM_EF_NAME {                            \
    constexpr static MFEM_EF_TYPE value = MFEM_EF_DEFAULT;                     \
  };                                                                           \
                                                                               \
  template <class T>                                                           \
  concept Has##MFEM_EF_NAME = requires { T::MFEM_EF_NAME; };                   \
                                                                               \
  template <class T>                                                           \
  concept HasConstexpr##MFEM_EF_NAME = requires {                              \
    typename RequireConstexpr<MFEM_EF_TYPE,                                    \
                              static_cast<MFEM_EF_TYPE>(T::MFEM_EF_NAME)>;     \
  };                                                                           \
                                                                               \
  template <class T>                                                           \
    requires HasConstexpr                                                      \
  ##MFEM_EF_NAME<T> struct Extract##MFEM_EF_NAME<T> {                          \
    constexpr static MFEM_EF_TYPE value =                                      \
        static_cast<MFEM_EF_TYPE>(T::MFEM_EF_NAME);                            \
  };                                                                           \
                                                                               \
  template <class T>                                                           \
    requires(Has##MFEM_EF_NAME<T> && !HasConstexpr##MFEM_EF_NAME<T>)           \
  struct Extract##MFEM_EF_NAME<T> {                                            \
    static_assert(                                                             \
        always_false<T>,                                                       \
        "T has accessible " MFEM_STRINGIFY(                                    \
            MFEM_EF_NAME) ", however it is not convertible to "                \
                          "constexpr static " MFEM_STRINGIFY(MFEM_EF_TYPE));   \
  }

MFEM_EXTRACT_FIELD(bool, Mixed, true);
MFEM_EXTRACT_FIELD(int, trial_vdim, 1);
MFEM_EXTRACT_FIELD(int, test_vdim, 1);
MFEM_EXTRACT_FIELD(bool, trial_out_of_plane, false);
MFEM_EXTRACT_FIELD(bool, test_out_of_plane, false);
MFEM_EXTRACT_FIELD(bool, jacobian, false);
MFEM_EXTRACT_FIELD(bool, metric, false);
MFEM_EXTRACT_FIELD(bool, inverse_metric, false);
MFEM_EXTRACT_FIELD(bool, adjugate, false);
MFEM_EXTRACT_FIELD(bool, reference_output, false);

#undef MFEM_STRINGIFY
#undef MFEM_STRINGIFY_IMPL
#undef MFEM_EXTRACT_FIELD

/// @tparam T Operation; read trial_vdim if present, otherwise default to 1.
template <class T> constexpr int ExtractTrialVDim() {
  return Extracttrial_vdim<T>::value;
}

/// @tparam T Operation; read test_vdim if present, otherwise default to 1.
template <class T> constexpr int ExtractTestVDim() {
  return Extracttest_vdim<T>::value;
}

/// @tparam T Operation; read trial_out_of_plane if present, otherwise false.
template <class T> constexpr bool ExtractTrialOutOfPlane() {
  return Extracttrial_out_of_plane<T>::value;
}

/// @tparam T Operation; read test_out_of_plane if present, otherwise false.
template <class T> constexpr bool ExtractTestOutOfPlane() {
  return Extracttest_out_of_plane<T>::value;
}

} // namespace internal
} // namespace mfem

#endif
