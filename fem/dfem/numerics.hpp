// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace mfem::future
{

namespace numerics_detail
{

/// The bits of @a x as an IEEE-754 double if @a x is positive, finite and
/// normal, else 0.
constexpr std::uint64_t encode(double x)
{
   if (!(x >= std::numeric_limits<double>::min() &&
         x <= std::numeric_limits<double>::max())) { return 0; }
   // x = m 2^e with 2^52 <= m < 2^53; scaling by two is exact.
   constexpr double m_min = 4503599627370496.0; // 2^52
   int e = 0;
   while (x < m_min) { x *= 2.0; e--; }
   while (x >= 2.0 * m_min) { x /= 2.0; e++; }
   return (static_cast<std::uint64_t>(e + 1075) << 52) |
          (static_cast<std::uint64_t>(x) - (std::uint64_t(1) << 52));
}

/// The double whose bits are @a bits, the inverse of encode().
constexpr double decode(std::uint64_t bits)
{
   const std::uint64_t m_min = std::uint64_t(1) << 52;
   const int e = static_cast<int>(bits >> 52) - 1075;
   double x = static_cast<double>((bits & (m_min - 1)) | m_min);
   for (int i = e; i < 0; i++) { x /= 2.0; }
   for (int i = e; i > 0; i--) { x *= 2.0; }
   return x;
}

} // namespace numerics_detail

/// @brief Accuracy target of the q-function evaluated by an integrator's
/// action, written MFEM_NUMERICS(Tolerance, x), see Numerics.
template <std::uint64_t bits>
struct Tolerance
{
   static constexpr double value = numerics_detail::decode(bits);
};

/// @brief Accuracy target of the q-function derivative evaluated by an
/// integrator's derivative action, written MFEM_NUMERICS(DerivativeTolerance,
/// x), see Numerics.
template <std::uint64_t bits>
struct DerivativeTolerance
{
   static constexpr double value = numerics_detail::decode(bits);
};

namespace numerics_detail
{

/// True, with the encoded accuracy target in @a bits, if @a T is the
/// requirement @a item_t; false, with @a bits 0, for any other type.
template <template <std::uint64_t> class item_t, typename T>
struct item_of : std::false_type
{
   static constexpr std::uint64_t bits = 0;
};

template <template <std::uint64_t> class item_t, std::uint64_t b>
struct item_of<item_t, item_t<b>> : std::true_type
{
   static constexpr std::uint64_t bits = b;
};

/// True if @a T is a Tolerance or a DerivativeTolerance.
template <typename T>
constexpr bool is_item_v = item_of<Tolerance, T>::value ||
                           item_of<DerivativeTolerance, T>::value;

/// The encoded accuracy target of the requirement @a T, see encode().
template <typename T>
constexpr std::uint64_t bits_v = item_of<Tolerance, T>::bits +
                                 item_of<DerivativeTolerance, T>::bits;

} // namespace numerics_detail

/// @brief Numerical requirements of an integrator.
///
/// They are written with MFEM_NUMERICS and given to
/// DifferentiableOperator::AddDomainIntegrator() or
/// DifferentiableOperator::AddBoundaryIntegrator() after the attributes, the
/// derivatives or the second derivatives, e.g.
///
///    dop.AddDomainIntegrator<LocalQFBackend>(
///       qf, inputs, outputs, ir, attr, Derivatives<U> {},
///       MFEM_NUMERICS(DerivativeTolerance, 1e-6));
///
/// With MFEM_USE_POSEIDON, the q-function call of the action (Tolerance) or of
/// the derivative action (DerivativeTolerance) is a Poseidon site with the
/// accuracy target x of MFEM_NUMERICS(kind, x), see general/poseidon.hpp.
/// Without MFEM_USE_POSEIDON the requirements have no effect.
///
/// The requirements are supported by the LocalQFBackend. DerivativeTolerance
/// requires derivatives, DerivativeKernels::Action, and an integrator that is
/// not a functional. Numerics<> has no requirement.
template <typename... items_t>
struct Numerics
{
   static_assert((numerics_detail::is_item_v<items_t> && ...),
                 "the items of Numerics are MFEM_NUMERICS(Tolerance, x) and "
                 "MFEM_NUMERICS(DerivativeTolerance, x)");
   static_assert(
      (int(numerics_detail::item_of<Tolerance, items_t>::value) + ... + 0)
      <= 1 &&
      (int(numerics_detail::item_of<DerivativeTolerance, items_t>::value) + ...
       + 0) <= 1,
      "Numerics takes at most one Tolerance and one DerivativeTolerance");
   static_assert(((numerics_detail::bits_v<items_t> != 0) && ...),
                 "MFEM_NUMERICS(kind, x) requires a positive, finite and "
                 "normal x");
};

/// @brief The numerical requirement @a kind, Tolerance or DerivativeTolerance,
/// with the accuracy target @a x, a positive constant expression, see
/// Numerics. Equal targets give the same type. Requirements are combined with
/// |, e.g.
///
///    MFEM_NUMERICS(DerivativeTolerance, 1e-6) | MFEM_NUMERICS(Tolerance, 1e-9)
#define MFEM_NUMERICS(kind, x)                                                \
   (::mfem::future::Numerics<::mfem::future::kind<                            \
    ::mfem::future::numerics_detail::encode(x)>> {})

/// The requirements of both operands, see MFEM_NUMERICS.
template <typename... a_t, typename... b_t>
constexpr Numerics<a_t..., b_t...> operator|(Numerics<a_t...>,
                                             Numerics<b_t...>)
{
   return {};
}

namespace numerics_detail
{

template <template <std::uint64_t> class item_t, typename numerics_t>
struct find_item;

template <template <std::uint64_t> class item_t, typename... items_t>
struct find_item<item_t, Numerics<items_t...>>
{
   static constexpr bool found = (item_of<item_t, items_t>::value || ...);
   static constexpr std::uint64_t bits =
      (item_of<item_t, items_t>::bits + ... + std::uint64_t(0));
};

} // namespace numerics_detail

/// True if @a numerics_t is a Numerics<...>.
template <typename numerics_t>
constexpr bool is_numerics_v = false;

template <typename... items_t>
constexpr bool is_numerics_v<Numerics<items_t...>> = true;

/// True if @a numerics_t has an @a item_t, e.g.
/// numerics_has_v<Tolerance, numerics_t>.
template <template <std::uint64_t> class item_t, typename numerics_t>
constexpr bool numerics_has_v =
   numerics_detail::find_item<item_t, numerics_t>::found;

/// The @a item_t of @a numerics_t, e.g. numerics_item_t<Tolerance, numerics_t>,
/// whose value is the accuracy target. Requires numerics_has_v<item_t,
/// numerics_t>.
template <template <std::uint64_t> class item_t, typename numerics_t>
using numerics_item_t =
   item_t<numerics_detail::find_item<item_t, numerics_t>::bits>;

} // namespace mfem::future
