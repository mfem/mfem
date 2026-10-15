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

#ifndef MFEM_COMPLEX_FEM
#define MFEM_COMPLEX_FEM

#include "../linalg/complex_operator.hpp"
#include "../linalg/complex_vector.hpp"
#include "complex_coefficient.hpp"
#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilinearform.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#include "plinearform.hpp"
#include "pbilinearform.hpp"
#endif
#include <cmath>
#include <type_traits>

namespace mfem
{

namespace complex_fem_detail
{

template <typename C>
struct is_complex_coefficient
   : std::integral_constant<bool,
     std::is_base_of<ComplexCoefficient, C>::value ||
     std::is_base_of<ComplexVectorCoefficient, C>::value ||
     std::is_base_of<ComplexMatrixCoefficient, C>::value> { };

template <typename C>
struct is_linear_form_coefficient
   : std::integral_constant<bool,
     std::is_base_of<Coefficient, C>::value ||
     std::is_base_of<VectorCoefficient, C>::value ||
     std::is_base_of<ComplexCoefficient, C>::value ||
     std::is_base_of<ComplexVectorCoefficient, C>::value> { };

template <typename C>
struct is_bilinear_form_coefficient
   : std::integral_constant<bool,
     std::is_base_of<Coefficient, C>::value ||
     std::is_base_of<VectorCoefficient, C>::value ||
     std::is_base_of<MatrixCoefficient, C>::value ||
     std::is_base_of<ComplexCoefficient, C>::value ||
     std::is_base_of<ComplexVectorCoefficient, C>::value ||
     std::is_base_of<ComplexMatrixCoefficient, C>::value> { };

template <typename T, typename C>
using enable_linear_integrator_t = typename std::enable_if<
                                   std::is_base_of<LinearFormIntegrator, T>::value &&
                                   !std::is_same<LinearFormIntegrator, T>::value &&
                                   is_linear_form_coefficient<C>::value>::type;

template <typename T, typename C>
using enable_bilinear_integrator_t = typename std::enable_if<
                                     std::is_base_of<BilinearFormIntegrator, T>::value &&
                                     !std::is_same<BilinearFormIntegrator, T>::value &&
                                     is_bilinear_form_coefficient<C>::value>::type;

template <typename...>
struct dependent_false : std::false_type { };

template <typename T, typename C, typename Add>
void add_integrator(C &coef, Add &&add)
{
   if constexpr (is_complex_coefficient<C>::value)
   {
      using real_type = decltype(coef.real());
      using imag_type = decltype(coef.imag());
      if constexpr (std::is_constructible<T, real_type>::value &&
                    std::is_constructible<T, imag_type>::value)
      {
         add(new T(coef.real()), new T(coef.imag()));
      }
      else
      {
         static_assert(dependent_false<T, C>::value,
                       "The integrator must be constructible from the real "
                       "and imaginary coefficient component types.");
      }
   }
   else if constexpr (std::is_constructible<T, C &>::value)
   {
      add(new T(coef), nullptr);
   }
   else
   {
      static_assert(dependent_false<T, C>::value,
                    "The integrator must be constructible from the supplied "
                    "coefficient type.");
   }
}

} // namespace complex_fem_detail

/// Class for complex-valued grid function - real + imaginary part Vector with
/// associated FE space.
class ComplexGridFunction : public Vector
{
private:
   GridFunction * gfr;
   GridFunction * gfi;

protected:
   /// FE space on which the grid function lives. Owned if #fec_owned
   /// is not NULL.
   FiniteElementSpace *fes;

   /** @brief Used when the grid function is read from a file. It can also be
       set explicitly, see MakeOwner().

       If not NULL, this pointer is owned by the ComplexGridFunction. */
   FiniteElementCollection *fec_owned;

   long fes_sequence; // see FiniteElementSpace::sequence, Mesh::sequence

   void Destroy();

public:
   /** @brief Construct a ComplexGridFunction associated with the
       FiniteElementSpace @a *f. */
   ComplexGridFunction(FiniteElementSpace *f);

   /** @brief Construct a ComplexGridFunction on the given Mesh, using the data
       from @a input.

       The content of @a input should be in the format created by the method
       Save(). The reconstructed FiniteElementSpace and FiniteElementCollection
       are owned by the ComplexGridFunction. */
   ComplexGridFunction(Mesh *m, std::istream &input);

   void Update();

   /** Return update counter, similar to Mesh::GetSequence(). Used to
       check if it is up to date with the space. */
   long GetSequence() const { return fes_sequence; }

   /// Make the ComplexGridFunction the owner of #fec_owned and #fes.
   /** If the new FiniteElementCollection, @a fec_, is NULL, ownership
       of #fec_owned and #fes is taken away. */
   void MakeOwner(FiniteElementCollection *fec_) { fec_owned = fec_; }

   /// Returns a pointer to the FiniteElementCollection used to
   /// construct this ComplexGridFunction if this class owns that
   /// object. Otherwise this function will return NULL.
   FiniteElementCollection *OwnFEC() { return fec_owned; }

   /// Shortcut for calling FiniteElementSpace::GetVectorDim() on the
   /// underlying #fes
   int VectorDim() const;

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** It is assumed that this object and @a rhs use FiniteElementSpace%s that
       have the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   ComplexGridFunction &operator=(const ComplexGridFunction &rhs)
   { return operator=((const Vector &)rhs); }

   /// Copy the data from @a v.
   /** The size of @a v must be equal to double of the size of the associated
       FiniteElementSpace #fes. */
   ComplexGridFunction &operator=(const Vector &v)
   {
      MFEM_ASSERT(fes && v.Size() == 2*fes->GetVSize(), "");
      Vector::operator=(v);
      return *this;
   }

   /// Assign constant values to the ComplexGridFunction data.
   ComplexGridFunction &operator=(const complex_t & value)
   { *gfr = value.real(); *gfi = value.imag(); return *this; }

   virtual void ProjectCoefficient(Coefficient &real_coeff,
                                   Coefficient &imag_coeff);
   virtual void ProjectCoefficient(Coefficient &real_coeff);
   virtual void ProjectCoefficient(ComplexCoefficient &coeff);
   virtual void ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                   VectorCoefficient &imag_vcoeff);
   virtual void ProjectCoefficient(VectorCoefficient &real_vcoeff);
   virtual void ProjectCoefficient(ComplexVectorCoefficient &vcoeff);

   virtual void ProjectBdrCoefficient(Coefficient &real_coeff,
                                      Coefficient &imag_coeff,
                                      Array<int> &attr);
   virtual void ProjectBdrCoefficient(Coefficient &real_coeff,
                                      Array<int> &attr);
   virtual void ProjectBdrCoefficient(ComplexCoefficient &coeff,
                                      Array<int> &attr);
   virtual void ProjectBdrCoefficientNormal(VectorCoefficient &real_coeff,
                                            VectorCoefficient &imag_coeff,
                                            Array<int> &attr);
   virtual void ProjectBdrCoefficientNormal(VectorCoefficient &real_coeff,
                                            Array<int> &attr);
   virtual void ProjectBdrCoefficientNormal(ComplexVectorCoefficient &coeff,
                                            Array<int> &attr);
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &real_coeff,
                                             VectorCoefficient &imag_coeff,
                                             Array<int> &attr);
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &real_coeff,
                                             Array<int> &attr);
   virtual void ProjectBdrCoefficientTangent(ComplexVectorCoefficient &coeff,
                                             Array<int> &attr);

   /// @brief Returns ||u_ex - u_h||_L2 for H1 or L2 elements
   ///
   /// @param[in] exsol  ComplexCoefficient object reproducing the anticipated
   ///                   values of the scalar field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(ComplexCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const
   { return this->ComputeL2Error(exsol.real(), exsol.imag(), irs, elems); }

   /// @brief Returns ||u_ex - u_h||_L2 for H1 or L2 elements
   ///
   /// The complex error is computed by combining its real and imaginary parts:
   /// $$\|u_{ex} - u_h\|_{L^2} =
   /// \sqrt{\|\mathop{\rm Re}(u_{ex} - u_h)\|_{L^2}^2 +
   ///        \|\mathop{\rm Im}(u_{ex} - u_h)\|_{L^2}^2}.$$
   ///
   /// @param[in] re_exsol  Coefficient object reproducing the anticipated
   ///                   values of the real part of the scalar field, u_ex.
   /// @param[in] im_exsol  Coefficient object reproducing the anticipated
   ///                   values of the imaginary part of the scalar field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(Coefficient &re_exsol,
                                 Coefficient &im_exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// @brief Returns ||u_ex - u_h||_L2 for H1 or L2 elements
   ///
   /// The exact imaginary part is assumed to be zero.
   ///
   /// @param[in] re_exsol  Coefficient object reproducing the anticipated
   ///                   values of the real part of the scalar field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(Coefficient &re_exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// @brief Returns ||u_ex - u_h||_L2 for vector fields
   ///
   /// @param[in] exsol  ComplexVectorCoefficient object reproducing the
   ///                   anticipated values of the vector field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(ComplexVectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const
   { return this->ComputeL2Error(exsol.real(), exsol.imag(), irs, elems); }

   /// @brief Returns ||u_ex - u_h||_L2 for vector fields
   ///
   /// The complex error is computed by combining the real and imaginary vector
   /// errors with the same L2 norm used for each component.
   ///
   /// @param[in] re_exsol  VectorCoefficient object reproducing the
   ///                   anticipated values of the real part of the vector
   ///                   field, u_ex.
   /// @param[in] im_exsol  VectorCoefficient object reproducing the
   ///                   anticipated values of the imaginary part of the vector
   ///                   field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(VectorCoefficient &re_exsol,
                                 VectorCoefficient &im_exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// @brief Returns ||u_ex - u_h||_L2 for vector fields
   ///
   /// The exact imaginary part is assumed to be zero.
   ///
   /// @param[in] re_exsol  VectorCoefficient object reproducing the
   ///                   anticipated values of the real part of the vector
   ///                   field, u_ex.
   /// @param[in] irs    Optional pointer to an array of custom integration
   ///                   rules e.g. higher order than the default rules. If
   ///                   present the array will be indexed by Geometry::Type.
   /// @param[in] elems  Optional pointer to a marker array, with a length
   ///                   equal to the number of local elements, indicating
   ///                   which elements to integrate over. Only those elements
   ///                   corresponding to non-zero entries in @a elems will
   ///                   contribute to the computed L2 error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeL2Error(VectorCoefficient &re_exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   FiniteElementSpace *FESpace() { return fes; }
   const FiniteElementSpace *FESpace() const { return fes; }

   GridFunction & real() { return *gfr; }
   GridFunction & imag() { return *gfi; }
   const GridFunction & real() const { return *gfr; }
   const GridFunction & imag() const { return *gfi; }

   /// Update the memory location of the real and imaginary GridFunction @a gfr
   /// and @a gfi to match the ComplexGridFunction.
   void Sync() { gfr->SyncMemory(*this); gfi->SyncMemory(*this); }

   /// Update the alias memory location of the real and imaginary GridFunction
   /// @a gfr and @a gfi to match the ComplexGridFunction.
   void SyncAlias() { gfr->SyncAliasMemory(*this); gfi->SyncAliasMemory(*this); }

   /// @brief Returns Max|u_ex - u_h| error for complex-valued H1 or L2 elements
   ///
   /// Compute the $L_\infty$ error across the entire domain.
   ///
   /// @param[in] exsolr  Coefficient object reproducing the real part of the
   ///                    anticipated values of the scalar field, Re(u_ex).
   /// @param[in] exsoli  Coefficient object reproducing the imaginary part of
   ///                    the anticipated values of the scalar field, Im(u_ex).
   /// @param[in] irs        Optional pointer to an array of custom integration
   ///                       rules e.g. higher order than the default rules. If
   ///                       present the array will be indexed by
   ///                       Geometry::Type.
   ///
   /// @note Uses ComputeLpError internally. See the ComputeLpError
   ///       documentation for generalizations of this error computation.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   virtual real_t ComputeMaxError(Coefficient &exsolr,
                                  Coefficient &exsoli,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsolr, exsoli, NULL, irs);
   }

   /// @brief Returns ||u_ex - u_h||_Lp for complex-valued H1 or L2 elements
   ///
   /// Computes:
   ///    $$(\sum_{elems} \int_{elem} w \, |u_{ex} - u_h|^p)^{1/p}$$
   /// Where:
   ///    $$|u_{ex} - u_h| = \sqrt{Re(u_{ex} - u_h)^2 + Im(u_{ex} - u_h)^2}$$
   ///
   /// @param[in] p       Real value indicating the exponent of the $L^p$ norm.
   ///                    To avoid domain errors p should have a positive value,
   ///                    either finite or infinite.
   /// @param[in] exsolr  Coefficient object reproducing the real part of the
   ///                    anticipated values of the scalar field, Re(u_ex).
   /// @param[in] exsoli  Coefficient object reproducing the imaginary part of
   ///                    the anticipated values of the scalar field, Im(u_ex).
   /// @param[in] weight  Optional pointer to a Coefficient object reproducing
   ///                    a weighting function, w.
   /// @param[in] irs     Optional pointer to an array of custom integration
   ///                    rules e.g. higher order than the default rules. If
   ///                    present the array will be indexed by Geometry::Type.
   /// @param[in] elems   Optional pointer to a marker array, with a length
   ///                    equal to the number of local elements, indicating
   ///                    which elements to integrate over. Only those elements
   ///                    corresponding to non-zero entries in @a elems will
   ///                    contribute to the computed Lp error.
   ///
   /// @note If an array of integration rules is provided through @a irs, be
   ///       sure to include valid rules for each element type that may occur
   ///       in the list of elements.
   ///
   /// @note Quadratures with negative weights (as in some simplex integration
   ///       rules in MFEM) can produce negative integrals even with
   ///       non-negative integrands. To avoid returning negative errors this
   ///       function uses the absolute values of the element-wise integrals.
   ///       This may lead to results which are not entirely consistent with
   ///       such integration rules.
   virtual real_t ComputeLpError(const real_t p,
                                 Coefficient &exsolr,
                                 Coefficient &exsoli,
                                 Coefficient *weight = NULL,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// Save the ComplexGridFunction to an output stream.
   virtual void Save(std::ostream &out) const;

   /// Save the ComplexGridFunction to a file
   /** The given @a precision will be used for ASCII output. */
   virtual void Save(const char *fname, int precision=16) const;

   /// Destroys the grid function.
   virtual ~ComplexGridFunction() { Destroy(); }

};

/** Overload operator<< for std::ostream and ComplexGridFunction; not valid
    for the class ParComplexGridFunction */
std::ostream &operator<<(std::ostream &out, const ComplexGridFunction &sol);

/** Class for a complex-valued linear form

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the ComplexLinearForm either the real or
    imaginary integrator can be NULL. This indicates that the corresponding
    portion of the complex-valued field is equal to zero.

    The templated Add*Integrator methods require the concrete integrator type
    as an explicit template argument, for example
    `form.AddDomainIntegrator<DomainLFIntegrator>(coef)`. They allocate the
    real and imaginary integrators, which are then owned by this form. The
    supplied coefficient and its component views are not owned and must remain
    valid for as long as the generated integrators can be used.
 */
class ComplexLinearForm : public Vector
{
private:
   ComplexOperator::Convention conv;

protected:
   LinearForm * lfr;
   LinearForm * lfi;

public:
   ComplexLinearForm(FiniteElementSpace *fes,
                     ComplexOperator::Convention
                     convention = ComplexOperator::HERMITIAN);

   /** @brief Create a ComplexLinearForm on the FiniteElementSpace @a fes, using
       the same integrators as the LinearForms @a lf_r (real) and @a lf_i (imag).

       The pointer @a fes is not owned by the newly constructed object.

       The integrators are copied as pointers and they are not owned by the
       newly constructed ComplexLinearForm. */
   ComplexLinearForm(FiniteElementSpace *fes, LinearForm *lf_r, LinearForm *lf_i,
                     ComplexOperator::Convention
                     convention = ComplexOperator::HERMITIAN);

   virtual ~ComplexLinearForm();

   /// Assign constant values to the ComplexLinearForm data.
   ComplexLinearForm &operator=(const complex_t & value)
   { *lfr = value.real(); *lfi = value.imag(); return *this; }

   ComplexOperator::Convention GetConvention() const { return conv; }
   void SetConvention(const ComplexOperator::Convention &
                      convention) { conv = convention; }

   /// @name Adding Domain Integrators
   /// @{

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag);

   /// Adds new Domain Integrator, restricted to the given attributes.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag,
                            Array<int> &elem_attr_marker);

   /// Adds new domain integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddDomainIntegrator(real, imag); });
   }

   /// Adds new domain integrator with a real or complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef, Array<int> &elem_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &elem_marker]
                                            (T *real, T *imag)
      { this->AddDomainIntegrator(real, imag, elem_marker); });
   }

   /// @}

   /// @name Adding Boundary Integrators
   /// @{

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                              LinearFormIntegrator *lfi_imag);

   /** @brief Add new Boundary Integrator, restricted to the given boundary
       attributes.

       Assumes ownership of @a lfi_real and @a lfi_imag.

       The array @a bdr_attr_marker is stored internally as a pointer to the
       given Array<int> object. */
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                              LinearFormIntegrator *lfi_imag,
                              Array<int> &bdr_attr_marker);

   /// Adds new boundary integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag); });
   }

   /// Adds new boundary integrator with a real or complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   /// @name Adding Boundary Face Integrators
   /// @{

   /// Adds new Boundary Face Integrator. Assumes ownership of @a lfi.
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                             LinearFormIntegrator *lfi_imag);

   /** @brief Add new Boundary Face Integrator, restricted to the given boundary
       attributes.

       Assumes ownership of @a lfi_real and @a lfi_imag.

       The array @a bdr_attr_marker is stored internally as a pointer to the
       given Array<int> object. */
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                             LinearFormIntegrator *lfi_imag,
                             Array<int> &bdr_attr_marker);

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag); });
   }

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient, restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   FiniteElementSpace *FESpace() const { return lfr->FESpace(); }

   LinearForm & real() { return *lfr; }
   LinearForm & imag() { return *lfi; }
   const LinearForm & real() const { return *lfr; }
   const LinearForm & imag() const { return *lfi; }

   /// Update the memory location of the real and imaginary LinearForm @a lfr
   /// and @a lfi to match the ComplexLinearForm.
   void Sync() { lfr->SyncMemory(*this); lfi->SyncMemory(*this); }

   /// Update the alias memory location of the real and imaginary LinearForm @a
   /// lfr and @a lfi to match the ComplexLinearForm.
   void SyncAlias()
   { lfr->SyncAliasMemory(*this); lfi->SyncAliasMemory(*this); }

   void Update();
   void Update(FiniteElementSpace *f);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   complex_t operator()(const ComplexGridFunction &gf) const;
};


/** Class for sesquilinear form

    A sesquilinear form is a generalization of a bilinear form to complex-valued
    fields. Sesquilinear forms are linear in the second argument but the first
    argument involves a complex conjugate in the sense that:

                a(alpha u, beta v) = conj(alpha) beta a(u, v)

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the SesquilinearForm either the real or
    imaginary integrator can be NULL. This indicates that the corresponding
    portion of the complex-valued material coefficient is equal to zero.

    The templated Add*Integrator methods require the concrete integrator type
    as an explicit template argument, for example
    `form.AddDomainIntegrator<MassIntegrator>(coef)`. They allocate the real
    and imaginary integrators, which are then owned by this form. The supplied
    coefficient and its component views are not owned and must remain valid
    for as long as the generated integrators can be used.
*/
class SesquilinearForm
{
private:
   ComplexOperator::Convention conv;

   /** This data member allows one to specify what should be done to the
       diagonal matrix entries and corresponding RHS values upon elimination of
       the constrained DoFs. */
   mfem::Matrix::DiagonalPolicy diag_policy = mfem::Matrix::DIAG_ONE;

   BilinearForm *blfr;
   BilinearForm *blfi;

   /* These methods check if the real/imag parts of the sesquilinear form are
      not empty */
   bool RealInteg();
   bool ImagInteg();

   void BuildComplexOperator(OperatorHandle &A_r, OperatorHandle &A_i,
                             OperatorHandle &A) const;

public:
   SesquilinearForm(FiniteElementSpace *fes,
                    ComplexOperator::Convention
                    convention = ComplexOperator::HERMITIAN);
   /** @brief Create a SesquilinearForm on the FiniteElementSpace @a fes, using
       the same integrators as the BilinearForms @a bfr and @a bfi .

       The pointer @a fes is not owned by the newly constructed object.

       The integrators are copied as pointers and they are not owned by the
       newly constructed SesquilinearForm. */
   SesquilinearForm(FiniteElementSpace *fes,
                    BilinearForm *bfr, BilinearForm *bfi,
                    ComplexOperator::Convention
                    convention = ComplexOperator::HERMITIAN);

   ComplexOperator::Convention GetConvention() const { return conv; }
   void SetConvention(const ComplexOperator::Convention &
                      convention) { conv = convention; }

   /// Set the desired assembly level.
   /** Valid choices are:

       - AssemblyLevel::LEGACY (default)
       - AssemblyLevel::FULL
       - AssemblyLevel::PARTIAL
       - AssemblyLevel::ELEMENT
       - AssemblyLevel::NONE

       This method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level)
   {
      blfr->SetAssemblyLevel(assembly_level);
      blfi->SetAssemblyLevel(assembly_level);
   }

   BilinearForm & real() { return *blfr; }
   BilinearForm & imag() { return *blfi; }
   const BilinearForm & real() const { return *blfr; }
   const BilinearForm & imag() const { return *blfi; }

   /// @name Adding Domain Integrators
   /// @{

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                            BilinearFormIntegrator *bfi_imag);

   /// Adds new Domain Integrator, restricted to the given attributes.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                            BilinearFormIntegrator *bfi_imag,
                            Array<int> &elem_marker);

   /// Adds new domain integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddDomainIntegrator(real, imag); });
   }

   /// Adds new domain integrator with a real or complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef, Array<int> &elem_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &elem_marker]
                                            (T *real, T *imag)
      { this->AddDomainIntegrator(real, imag, elem_marker); });
   }

   /// @}

   /// @name Adding Boundary Integrators
   /// @{

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag);

   /// Adds new Boundary Integrator, restricted to specific boundary attributes.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag,
                              Array<int> &bdr_marker);

   /// Adds new boundary integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag); });
   }

   /// Adds new boundary integrator with a real or complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   /// @name Adding Interior Face Integrators
   /// @{

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                  BilinearFormIntegrator *bfi_imag);

   /// Adds new interior face integrator with a real or complex-valued
   /// coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddInteriorFaceIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddInteriorFaceIntegrator(real, imag); });
   }

   /// @}

   /// @name Adding Boundary Face Integrators
   /// @{

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                             BilinearFormIntegrator *bfi_imag);

   /// @brief Adds new boundary Face Integrator, restricted to specific boundary
   /// attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                             BilinearFormIntegrator *bfi_imag,
                             Array<int> &bdr_marker);

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag); });
   }

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient, restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   ComplexSparseMatrix *AssembleComplexSparseMatrix();

   /// Return the parallel FE space associated with the ParBilinearForm.
   FiniteElementSpace *FESpace() const { return blfr->FESpace(); }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   virtual void Update(FiniteElementSpace *nfes = NULL);

   /// Sets diagonal policy used upon construction of the linear system
   void SetDiagonalPolicy(mfem::Matrix::DiagonalPolicy dpolicy);

   /// Returns the diagonal policy of the sesquilinear form
   Matrix::DiagonalPolicy GetDiagonalPolicy() const {return diag_policy;}

   virtual ~SesquilinearForm();
};

/** Class for a mixed sesquilinear form

    A mixed sesquilinear form is a generalization of a mixed bilinear form to
    complex-valued fields. Mixed sesquilinear forms are linear in the second
    argument but the first argument involves a complex conjugate in the sense
    that:

                a(alpha u, beta v) = conj(alpha) beta a(u, v)

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the MixedSesquilinearForm either the real or
    imaginary integrator can be NULL. This indicates that the corresponding
    portion of the complex-valued material coefficient is equal to zero.
*/
class MixedSesquilinearForm
{
private:
   ComplexOperator::Convention conv;

   MixedBilinearForm * mblfr;
   MixedBilinearForm * mblfi;

   /* These methods check if the real/imag parts of the sesqulinear form are not
      empty */
   bool RealInteg();
   bool ImagInteg();

public:
   MixedSesquilinearForm(
      FiniteElementSpace * trial_fes,
      FiniteElementSpace * test_fes,
      ComplexOperator::Convention convention = ComplexOperator::HERMITIAN);

   /** @brief Create a MixedSesquilinearForm on the given trial and test
       FiniteElementSpaces, using the same integrators as the
       MixedBilinearForms @a bfr and @a bfi.

       The FiniteElementSpace pointers are not owned by the newly constructed
       object.

       The integrators are copied as pointers and they are not owned by the
       newly constructed MixedSesquilinearForm. */
   MixedSesquilinearForm(
      FiniteElementSpace * trial_fes,
      FiniteElementSpace * test_fes,
      MixedBilinearForm * bfr,
      MixedBilinearForm * bfi,
      ComplexOperator::Convention convention = ComplexOperator::HERMITIAN);

   ComplexOperator::Convention GetConvention() const { return conv; }
   void SetConvention(const ComplexOperator::Convention & convention) { conv = convention; }

   /// Set the desired assembly level.
   /** Valid choices are:

       - AssemblyLevel::LEGACY (default)
       - AssemblyLevel::FULL
       - AssemblyLevel::PARTIAL
       - AssemblyLevel::ELEMENT
       - AssemblyLevel::NONE

       This method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level)
   {
      mblfr->SetAssemblyLevel(assembly_level);
      mblfi->SetAssemblyLevel(assembly_level);
   }

   MixedBilinearForm & real() { return *mblfr; }
   MixedBilinearForm & imag() { return *mblfi; }
   const MixedBilinearForm & real() const { return *mblfr; }
   const MixedBilinearForm & imag() const { return *mblfi; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator * bfi_real,
                            BilinearFormIntegrator * bfi_imag);

   /// Adds new Domain Integrator, restricted to specific attributes.
   void AddDomainIntegrator(BilinearFormIntegrator * bfi_real,
                            BilinearFormIntegrator * bfi_imag,
                            Array<int> & elem_marker);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator * bfi_real,
                              BilinearFormIntegrator * bfi_imag);

   /** @brief Adds new boundary Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi.

       The mfem::array @a bdr_marker is stored internally as a pointer to the given
       mfem::Array<int> object. */
   void AddBoundaryIntegrator(BilinearFormIntegrator * bfi_real,
                              BilinearFormIntegrator * bfi_imag,
                              Array<int> & bdr_marker);

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator * bfi_real,
                                  BilinearFormIntegrator * bfi_imag);

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(BilinearFormIntegrator * bfi_real,
                             BilinearFormIntegrator * bfi_imag);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi.

       The mfem::array @a bdr_marker is stored internally as a pointer to the given
       mfem::Array<int> object. */
   void AddBdrFaceIntegrator(BilinearFormIntegrator * bfi_real,
                             BilinearFormIntegrator * bfi_imag,
                             Array<int> & bdr_marker);

   /** @brief Add a trace face integrator. Assumes ownership of @a bfi.

       This type of integrator assembles terms over all faces of the mesh using
       the face FE from the trial space and the two adjacent volume FEs from
       the test space. */
   void AddTraceFaceIntegrator(BilinearFormIntegrator * bfi_real,
                               BilinearFormIntegrator * bfi_imag);

   /// Adds a boundary trace face integrator. Assumes ownership of @a bfi.
   void AddBdrTraceFaceIntegrator(BilinearFormIntegrator * bfi_real,
                                  BilinearFormIntegrator * bfi_imag);

   /// Adds a boundary trace face integrator. Assumes ownership of @a bfi.
   void AddBdrTraceFaceIntegrator(BilinearFormIntegrator * bfi_real,
                                  BilinearFormIntegrator * bfi_imag,
                                  Array<int> &bdr_marker);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Updates the internal mixed forms with the new finite element space.
   virtual void Update();

   /** @brief Return a ComplexSparseMatrix wrapping the local (L-dof) real
       and imaginary matrices of the form.

       The returned wrapper has to be deleted by the caller, but it does not
       own the wrapped real and imaginary matrices, which remain owned by
       this form. */
   ComplexSparseMatrix *AssembleComplexSparseMatrix();

   /// Return the trial FE space associated with the MixedSesquilinearForm.
   FiniteElementSpace *TrialFESpace() { return mblfr->TrialFESpace(); }

   /// Read-only access to the associated trial FiniteElementSpace.
   const FiniteElementSpace *TrialFESpace() const { return mblfr->TrialFESpace(); }

   /// Return the test FE space associated with the MixedSesquilinearForm.
   FiniteElementSpace *TestFESpace() { return mblfr->TestFESpace(); }

   /// Read-only access to the associated test FiniteElementSpace.
   const FiniteElementSpace *TestFESpace() const { return mblfr->TestFESpace(); }


   void FormRectangularLinearSystem(const Array<int> & ess_trial_tdof_list,
                                    const Array<int> & ess_test_tdof_list,
                                    Vector & x,
                                    Vector & b,
                                    OperatorHandle & A,
                                    Vector & X,
                                    Vector & B);

   void FormRectangularSystemMatrix(const Array<int> & ess_trial_tdof_list,
                                    const Array<int> & ess_test_tdof_list,
                                    OperatorHandle & A);

   virtual ~MixedSesquilinearForm();
};

#ifdef MFEM_USE_MPI

/// Class for parallel complex-valued grid function - real + imaginary part
/// Vector with associated parallel FE space.
class ParComplexGridFunction : public Vector
{
private:
   ParGridFunction * pgfr;
   ParGridFunction * pgfi;

protected:
   /// FE space on which the grid function lives. Owned if #fec_owned
   /// is not NULL.
   ParFiniteElementSpace *pfes;

   /** @brief Used when the grid function is read from a file. It can also be
       set explicitly, see MakeOwner().

       If not NULL, this pointer is owned by the ParComplexGridFunction. */
   FiniteElementCollection *fec_owned;

   long fes_sequence; // see FiniteElementSpace::sequence, Mesh::sequence

   void Destroy();

public:

   /** @brief Construct a ParComplexGridFunction associated with the
       ParFiniteElementSpace @a *pf. */
   ParComplexGridFunction(ParFiniteElementSpace *pf);

   /** @brief Construct a ParComplexGridFunction on a given ParMesh,
       @a pmesh, reading from an std::istream.

       In the process, a ParFiniteElementSpace and a FiniteElementCollection are
       constructed. The new ParComplexGridFunction assumes ownership of both. */
   ParComplexGridFunction(ParMesh *pmesh, std::istream &input);

   void Update();

   /** Return update counter, similar to Mesh::GetSequence(). Used to
       check if it is up to date with the space. */
   long GetSequence() const { return fes_sequence; }

   /// Make the ParComplexGridFunction the owner of #fec_owned and #pfes.
   /** If the new FiniteElementCollection, @a fec_, is NULL, ownership
       of #fec_owned and #pfes is taken away. */
   void MakeOwner(FiniteElementCollection *fec_) { fec_owned = fec_; }

   /// Returns a pointer to the FiniteElementCollection used to
   /// construct this ParComplexGridFunction if this class owns that
   /// object. Otherwise this function will return NULL.
   FiniteElementCollection *OwnFEC() { return fec_owned; }

   /// Shortcut for calling FiniteElementSpace::GetVectorDim() on the
   /// underlying #pfes
   int VectorDim() const;

   /// Assign constant values to the ParComplexGridFunction data.
   ParComplexGridFunction &operator=(const complex_t & value)
   { *pgfr = value.real(); *pgfi = value.imag(); return *this; }

   virtual void ProjectCoefficient(Coefficient &real_coeff,
                                   Coefficient &imag_coeff);
   virtual void ProjectCoefficient(Coefficient &real_coeff);
   virtual void ProjectCoefficient(ComplexCoefficient &coeff)
   { this->ProjectCoefficient(coeff.real(), coeff.imag()); }
   virtual void ProjectCoefficient(VectorCoefficient &real_vcoeff,
                                   VectorCoefficient &imag_vcoeff);
   virtual void ProjectCoefficient(VectorCoefficient &real_vcoeff);
   virtual void ProjectCoefficient(ComplexVectorCoefficient &vcoeff)
   { this->ProjectCoefficient(vcoeff.real(), vcoeff.imag()); }

   virtual void ProjectBdrCoefficient(Coefficient &real_coeff,
                                      Coefficient &imag_coeff,
                                      Array<int> &attr);
   virtual void ProjectBdrCoefficient(Coefficient &real_coeff,
                                      Array<int> &attr);
   virtual void ProjectBdrCoefficient(ComplexCoefficient &coeff,
                                      Array<int> &attr)
   { this->ProjectBdrCoefficient(coeff.real(), coeff.imag(), attr); }
   virtual void ProjectBdrCoefficientNormal(VectorCoefficient &real_coeff,
                                            VectorCoefficient &imag_coeff,
                                            Array<int> &attr);
   virtual void ProjectBdrCoefficientNormal(VectorCoefficient &real_coeff,
                                            Array<int> &attr);
   virtual void ProjectBdrCoefficientNormal(ComplexVectorCoefficient &coeff,
                                            Array<int> &attr)
   { this->ProjectBdrCoefficientNormal(coeff.real(), coeff.imag(), attr); }
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &real_coeff,
                                             VectorCoefficient &imag_coeff,
                                             Array<int> &attr);
   virtual void ProjectBdrCoefficientTangent(VectorCoefficient &real_coeff,
                                             Array<int> &attr);
   virtual void ProjectBdrCoefficientTangent(ComplexVectorCoefficient &coeff,
                                             Array<int> &attr)
   { this->ProjectBdrCoefficientTangent(coeff.real(), coeff.imag(), attr); }

   void Distribute(const Vector *tv);
   void Distribute(const Vector &tv) { Distribute(&tv); }

   /// Returns the vector restricted to the true dofs.
   void ParallelProject(Vector &tv) const;

   FiniteElementSpace *FESpace() { return pfes; }
   const FiniteElementSpace *FESpace() const { return pfes; }

   ParFiniteElementSpace *ParFESpace() { return pfes; }
   const ParFiniteElementSpace *ParFESpace() const { return pfes; }

   ParGridFunction & real() { return *pgfr; }
   ParGridFunction & imag() { return *pgfi; }
   const ParGridFunction & real() const { return *pgfr; }
   const ParGridFunction & imag() const { return *pgfi; }

   /// Update the memory location of the real and imaginary ParGridFunction @a
   /// pgfr and @a pgfi to match the ParComplexGridFunction.
   void Sync() { pgfr->SyncMemory(*this); pgfi->SyncMemory(*this); }

   /// Update the alias memory location of the real and imaginary
   /// ParGridFunction @a pgfr and @a pgfi to match the ParComplexGridFunction.
   void SyncAlias()
   { pgfr->SyncAliasMemory(*this); pgfi->SyncAliasMemory(*this); }

   /// @brief Returns ||u_ex - u_h||_L2 in parallel for complex-valued
   ///        scalar fields
   ///
   /// The globally reduced real and imaginary L2 errors are combined with
   /// std::hypot. See ParGridFunction::ComputeL2Error for the integration-rule
   /// and element-marker semantics.
   virtual real_t ComputeL2Error(Coefficient &exsolr, Coefficient &exsoli,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      real_t err_r = pgfr->ComputeL2Error(exsolr, irs, elems);
      real_t err_i = pgfi->ComputeL2Error(exsoli, irs, elems);
      return std::hypot(err_r, err_i);
   }

   /// @brief Returns the global L2 error with a zero exact imaginary part.
   virtual real_t ComputeL2Error(Coefficient &exsolr,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const;

   /// @brief Convenience overload using the real and imaginary parts of @a exsol.
   virtual real_t ComputeL2Error(ComplexCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      return this->ComputeL2Error(exsol.real(), exsol.imag(), irs, elems);
   }

   /// @brief Returns ||u_ex - u_h||_L2 in parallel for complex-valued
   ///        vector fields
   ///
   /// The globally reduced real and imaginary vector L2 errors are combined
   /// with std::hypot. See ParGridFunction::ComputeL2Error for the
   /// integration-rule and element-marker semantics.
   virtual real_t ComputeL2Error(VectorCoefficient &exsolr,
                                 VectorCoefficient &exsoli,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      real_t err_r = pgfr->ComputeL2Error(exsolr, irs, elems);
      real_t err_i = pgfi->ComputeL2Error(exsoli, irs, elems);
      return std::hypot(err_r, err_i);
   }

   /// @brief Returns the global L2 error with a zero exact imaginary part.
   virtual real_t ComputeL2Error(VectorCoefficient &exsolr,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const;

   /// @brief Convenience overload using the real and imaginary parts of @a exsol.
   virtual real_t ComputeL2Error(ComplexVectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      return this->ComputeL2Error(exsol.real(), exsol.imag(), irs, elems);
   }

   /// @brief Returns the global Max|u_ex - u_h| error for complex-valued
   ///        scalar fields.
   ///
   /// The maximum complex magnitude is reduced across all MPI ranks.
   virtual real_t ComputeMaxError(Coefficient &exsolr,
                                  Coefficient &exsoli,
                                  const IntegrationRule *irs[] = NULL) const
   {
      return ComputeLpError(infinity(), exsolr, exsoli, NULL, irs);
   }

   /// @brief Returns the global ||u_ex - u_h||_Lp error for complex-valued
   ///        scalar fields.
   ///
   /// Computes the Lp norm of the complex magnitude over elements on all MPI
   /// ranks. Each rank integrates its local contribution, which is then reduced
   /// globally. For @a p equal to infinity, this returns the global maximum.
   ///
   /// @param[in] p        Positive finite exponent, or infinity.
   /// @param[in] exsolr   Exact real-part coefficient.
   /// @param[in] exsoli   Exact imaginary-part coefficient.
   /// @param[in] weight   Optional scalar weighting coefficient.
   /// @param[in] irs      Optional integration rules indexed by Geometry::Type.
   /// @param[in] elems    Optional local-element marker.
   ///
   /// @see ParGridFunction::ComputeLpError for the corresponding real-valued
   ///      reduction semantics.
   virtual real_t ComputeLpError(const real_t p,
                                 Coefficient &exsolr,
                                 Coefficient &exsoli,
                                 Coefficient *weight = NULL,
                                 const IntegrationRule *irs[] = NULL,
                                 const Array<int> *elems = NULL) const;

   /// Save the local portion of the ParComplexGridFunction
   /** This differs from the serial ComplexGridFunction::Save in that it
       takes into account the signs of the local dofs. */
   void Save(std::ostream &out) const;

   /// Save the ParComplexGridFunction to files
   /** Saves one file for each MPI rank. The files will be given suffixes
       according to the MPI rank. The given @a precision will be used for ASCII
       output. */
   void Save(const char *fname, int precision=16) const;

   /// Destroys grid function.
   virtual ~ParComplexGridFunction() { Destroy(); }

};

/** Overload operator<< for std::ostream and ParComplexGridFunction */
std::ostream &operator<<(std::ostream &out, const ParComplexGridFunction &sol);

/** Class for a complex-valued, parallel linear form

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the ParComplexLinearForm either the real or
    imaginary integrator can be NULL.  This indicates that the corresponding
    portion of the complex-valued field is equal to zero.

    The templated Add*Integrator methods require the concrete integrator type
    as an explicit template argument, for example
    `form.AddDomainIntegrator<DomainLFIntegrator>(coef)`. They allocate the
    real and imaginary integrators, which are then owned by this form. The
    supplied coefficient and its component views are not owned and must remain
    valid for as long as the generated integrators can be used.
 */
class ParComplexLinearForm : public Vector
{
private:
   ComplexOperator::Convention conv;

protected:
   ParLinearForm * plfr;
   ParLinearForm * plfi;

   HYPRE_BigInt * tdof_offsets;

public:

   ParComplexLinearForm(ParFiniteElementSpace *pf,
                        ComplexOperator::Convention
                        convention = ComplexOperator::HERMITIAN);

   /** @brief Create a ParComplexLinearForm on the ParFiniteElementSpace @a pf,
       using the same integrators as the LinearForms @a plf_r (real) and
       @a plf_i (imag).

      The pointer @a pf is not owned by the newly constructed object.

      The integrators are copied as pointers and they are not owned by the newly
      constructed ParComplexLinearForm. */
   ParComplexLinearForm(ParFiniteElementSpace *pf, ParLinearForm *plf_r,
                        ParLinearForm *plf_i,
                        ComplexOperator::Convention
                        convention = ComplexOperator::HERMITIAN);

   virtual ~ParComplexLinearForm();

   /// Assign constant values to the ParComplexLinearForm data.
   ParComplexLinearForm &operator=(const complex_t & value)
   { *plfr = value.real(); *plfi = value.imag(); return *this; }

   ComplexOperator::Convention GetConvention() const { return conv; }
   void SetConvention(const ComplexOperator::Convention &
                      convention) { conv = convention; }

   /// @name Adding Domain Integrators
   /// @{

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag);

   /// Adds new Domain Integrator, restricted to specific attributes.
   void AddDomainIntegrator(LinearFormIntegrator *lfi_real,
                            LinearFormIntegrator *lfi_imag,
                            Array<int> &elem_attr_marker);

   /// Adds new domain integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddDomainIntegrator(real, imag); });
   }

   /// Adds new domain integrator with a real or complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef, Array<int> &elem_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &elem_marker]
                                            (T *real, T *imag)
      { this->AddDomainIntegrator(real, imag, elem_marker); });
   }

   /// @}

   /// @name Adding Boundary Integrators
   /// @{

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                              LinearFormIntegrator *lfi_imag);

   /** @brief Add new Boundary Integrator, restricted to the given boundary
       attributes.

       Assumes ownership of @a lfi_real and @a lfi_imag.

       The array @a bdr_attr_marker is stored internally as a pointer to the
       given Array<int> object. */
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi_real,
                              LinearFormIntegrator *lfi_imag,
                              Array<int> &bdr_attr_marker);

   /// Adds new boundary integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag); });
   }

   /// Adds new boundary integrator with a real or complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   /// @name Adding Boundary Face Integrators
   /// @{

   /// Adds new Boundary Face Integrator. Assumes ownership of @a lfi.
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                             LinearFormIntegrator *lfi_imag);

   /** @brief Add new Boundary Face Integrator, restricted to the given boundary
       attributes.

       Assumes ownership of @a lfi_real and @a lfi_imag.

       The array @a bdr_attr_marker is stored internally as a pointer to the
       given Array<int> object. */
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi_real,
                             LinearFormIntegrator *lfi_imag,
                             Array<int> &bdr_attr_marker);

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag); });
   }

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient, restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_linear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   ParFiniteElementSpace *ParFESpace() const { return plfr->ParFESpace(); }

   ParLinearForm & real() { return *plfr; }
   ParLinearForm & imag() { return *plfi; }
   const ParLinearForm & real() const { return *plfr; }
   const ParLinearForm & imag() const { return *plfi; }

   /// Update the memory location of the real and imaginary ParLinearForm @a lfr
   /// and @a lfi to match the ParComplexLinearForm.
   void Sync() { plfr->SyncMemory(*this); plfi->SyncMemory(*this); }

   /// Update the alias memory location of the real and imaginary ParLinearForm
   /// @a plfr and @a plfi to match the ParComplexLinearForm.
   void SyncAlias()
   { plfr->SyncAliasMemory(*this); plfi->SyncAliasMemory(*this); }

   void Update(ParFiniteElementSpace *pf = NULL);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();

   complex_t operator()(const ParComplexGridFunction &gf) const;

};

/** Class for a parallel sesquilinear form

    A sesquilinear form is a generalization of a bilinear form to complex-valued
    fields. Sesquilinear forms are linear in the second argument but the
    first argument involves a complex conjugate in the sense that:

                a(alpha u, beta v) = conj(alpha) beta a(u, v)

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the ParSesquilinearForm either the real or
    imaginary integrator can be NULL. This indicates that the corresponding
    portion of the complex-valued material coefficient is equal to zero.

    The templated Add*Integrator methods require the concrete integrator type
    as an explicit template argument, for example
    `form.AddDomainIntegrator<MassIntegrator>(coef)`. They allocate the real
    and imaginary integrators, which are then owned by this form. The supplied
    coefficient and its component views are not owned and must remain valid
    for as long as the generated integrators can be used.
*/
class ParSesquilinearForm
{
private:
   ComplexOperator::Convention conv;

   ParBilinearForm *pblfr;
   ParBilinearForm *pblfi;

   /* These methods check if the real/imag parts of the sesqulinear form are not
      empty */
   bool RealInteg();
   bool ImagInteg();

   void SetImaginaryEssentialDiagonalToZero(
      const Array<int> &ess_tdof_list, OperatorHandle &A);

   void BuildComplexOperator(OperatorHandle &A_r, OperatorHandle &A_i,
                             OperatorHandle &A) const;

public:
   ParSesquilinearForm(ParFiniteElementSpace *pf,
                       ComplexOperator::Convention
                       convention = ComplexOperator::HERMITIAN);

   /** @brief Create a ParSesquilinearForm on the ParFiniteElementSpace @a pf,
       using the same integrators as the ParBilinearForms @a pbfr and @a pbfi .

       The pointer @a pf is not owned by the newly constructed object.

       The integrators are copied as pointers and they are not owned by the
       newly constructed ParSesquilinearForm. */
   ParSesquilinearForm(ParFiniteElementSpace *pf, ParBilinearForm *pbfr,
                       ParBilinearForm *pbfi,
                       ComplexOperator::Convention
                       convention = ComplexOperator::HERMITIAN);

   ComplexOperator::Convention GetConvention() const { return conv; }
   void SetConvention(const ComplexOperator::Convention &
                      convention) { conv = convention; }

   /// Set the desired assembly level.
   /** Valid choices are:

       - AssemblyLevel::LEGACY (default)
       - AssemblyLevel::FULL
       - AssemblyLevel::PARTIAL
       - AssemblyLevel::ELEMENT
       - AssemblyLevel::NONE

       This method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level)
   {
      pblfr->SetAssemblyLevel(assembly_level);
      pblfi->SetAssemblyLevel(assembly_level);
   }

   ParBilinearForm & real() { return *pblfr; }
   ParBilinearForm & imag() { return *pblfi; }
   const ParBilinearForm & real() const { return *pblfr; }
   const ParBilinearForm & imag() const { return *pblfi; }

   /// @name Adding Domain Integrators
   /// @{

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                            BilinearFormIntegrator *bfi_imag);

   /// Adds new Domain Integrator, restricted to specific attributes.
   void AddDomainIntegrator(BilinearFormIntegrator *bfi_real,
                            BilinearFormIntegrator *bfi_imag,
                            Array<int> &elem_marker);

   /// Adds new domain integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddDomainIntegrator(real, imag); });
   }

   /// Adds new domain integrator with a real or complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddDomainIntegrator(C &coef, Array<int> &elem_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &elem_marker]
                                            (T *real, T *imag)
      { this->AddDomainIntegrator(real, imag, elem_marker); });
   }

   /// @}

   /// @name Adding Boundary Integrators
   /// @{

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag);

   /** @brief Adds new boundary Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi.

       The array @a bdr_marker is stored internally as a pointer to the given
       Array<int> object. */
   void AddBoundaryIntegrator(BilinearFormIntegrator *bfi_real,
                              BilinearFormIntegrator *bfi_imag,
                              Array<int> &bdr_marker);

   /// Adds new boundary integrator with a real or complex-valued coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag); });
   }

   /// Adds new boundary integrator with a real or complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBoundaryIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBoundaryIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   /// @name Adding Interior Face Integrators
   /// @{

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                  BilinearFormIntegrator *bfi_imag);

   /// Adds new interior face integrator with a real or complex-valued
   /// coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddInteriorFaceIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddInteriorFaceIntegrator(real, imag); });
   }

   /// @}

   /// @name Adding Boundary Face Integrators
   /// @{

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                             BilinearFormIntegrator *bfi_imag);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi.

       The array @a bdr_marker is stored internally as a pointer to the given
       Array<int> object. */
   void AddBdrFaceIntegrator(BilinearFormIntegrator *bfi_real,
                             BilinearFormIntegrator *bfi_imag,
                             Array<int> &bdr_marker);

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      complex_fem_detail::add_integrator<T>(coef, [this](T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag); });
   }

   /// Adds new boundary face integrator with a real or complex-valued
   /// coefficient, restricted to specific boundary attributes.
   template <typename T, typename C,
             typename complex_fem_detail::enable_bilinear_integrator_t<T, C> * =
             nullptr>
   void AddBdrFaceIntegrator(C &coef, Array<int> &bdr_marker)
   {
      complex_fem_detail::add_integrator<T>(coef, [this, &bdr_marker]
                                            (T *real, T *imag)
      { this->AddBdrFaceIntegrator(real, imag, bdr_marker); });
   }

   /// @}

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   ComplexHypreParMatrix *ParallelAssemble();

   /// Return the parallel FE space associated with the ParBilinearForm.
   ParFiniteElementSpace *ParFESpace() const { return pblfr->ParFESpace(); }

   void FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A);

   /** Call this method after solving a linear system constructed using the
       FormLinearSystem method to recover the solution as a ParGridFunction-size
       vector in x. Use the same arguments as in the FormLinearSystem call. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   virtual void Update(FiniteElementSpace *nfes = NULL);

   virtual ~ParSesquilinearForm();
};

/** Class for a parallel mixed sesquilinear form

    A mixed sesquilinear form is a generalization of a mixed bilinear form to
    complex-valued fields. Mixed sesquilinear forms are linear in the second
    argument but the first argument involves a complex conjugate in the sense
    that:

                a(alpha u, beta v) = conj(alpha) beta a(u, v)

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the ParMixedSesquilinearForm either the real
    or imaginary integrator can be NULL. This indicates that the corresponding
    portion of the complex-valued material coefficient is equal to zero.
*/
class ParMixedSesquilinearForm
{
private:
   ComplexOperator::Convention conv;

   ParMixedBilinearForm * pmblfr;
   ParMixedBilinearForm * pmblfi;

   /* These methods check if the real/imag parts of the sesqulinear form are
    not empty */
   bool RealInteg();
   bool ImagInteg();

public:
   ParMixedSesquilinearForm(
      ParFiniteElementSpace * trial_fes,
      ParFiniteElementSpace * test_fes,
      ComplexOperator::Convention convention = ComplexOperator::HERMITIAN);

   /** @brief Create a ParMixedSesquilinearForm on the given trial and test
       ParFiniteElementSpaces, using the same integrators as the
       ParMixedBilinearForms @a pbfr and @a pbfi.

       The ParFiniteElementSpace pointers are not owned by the newly
       constructed object.

       The integrators are copied as pointers and they are not owned by the
       newly constructed ParMixedSesquilinearForm. */
   ParMixedSesquilinearForm(
      ParFiniteElementSpace * trial_fes,
      ParFiniteElementSpace * test_fes,
      ParMixedBilinearForm * pbfr,
      ParMixedBilinearForm * pbfi,
      ComplexOperator::Convention convention = ComplexOperator::HERMITIAN);

   ComplexOperator::Convention GetConvention() const { return conv; }
   void SetConvention(const ComplexOperator::Convention & convention) { conv = convention; }

   /// Set the desired assembly level.
   /** Valid choices are:

       - AssemblyLevel::LEGACY (default)
       - AssemblyLevel::FULL
       - AssemblyLevel::PARTIAL
       - AssemblyLevel::ELEMENT
       - AssemblyLevel::NONE

       This method must be called before assembly. */
   void SetAssemblyLevel(AssemblyLevel assembly_level)
   {
      pmblfr->SetAssemblyLevel(assembly_level);
      pmblfi->SetAssemblyLevel(assembly_level);
   }

   ParMixedBilinearForm & real() { return *pmblfr; }
   ParMixedBilinearForm & imag() { return *pmblfi; }
   const ParMixedBilinearForm & real() const { return *pmblfr; }
   const ParMixedBilinearForm & imag() const { return *pmblfi; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(BilinearFormIntegrator * bfi_real,
                            BilinearFormIntegrator * bfi_imag);

   /// Adds new Domain Integrator, restricted to specific attributes.
   void AddDomainIntegrator(BilinearFormIntegrator * bfi_real,
                            BilinearFormIntegrator * bfi_imag,
                            Array<int> & elem_marker);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(BilinearFormIntegrator * bfi_real,
                              BilinearFormIntegrator * bfi_imag);

   /** @brief Adds new boundary Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi.

       The mfem::array @a bdr_marker is stored internally as a pointer to the given
       mfem::Array<int> object. */
   void AddBoundaryIntegrator(BilinearFormIntegrator * bfi_real,
                              BilinearFormIntegrator * bfi_imag,
                              Array<int> & bdr_marker);

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator * bfi_real,
                                  BilinearFormIntegrator * bfi_imag);

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(BilinearFormIntegrator * bfi_real,
                             BilinearFormIntegrator * bfi_imag);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi.

       The mfem::array @a bdr_marker is stored internally as a pointer to the given
       mfem::Array<int> object. */
   void AddBdrFaceIntegrator(BilinearFormIntegrator * bfi_real,
                             BilinearFormIntegrator * bfi_imag,
                             Array<int> & bdr_marker);

   /** @brief Add a trace face integrator. Assumes ownership of @a bfi.

      This type of integrator assembles terms over all faces of the mesh using
      the face FE from the trial space and the two adjacent volume FEs from
      the test space. */
   void AddTraceFaceIntegrator(BilinearFormIntegrator * bfi_real,
                               BilinearFormIntegrator * bfi_imag);

   /// Adds a boundary trace face integrator. Assumes ownership of @a bfi.
   void AddBdrTraceFaceIntegrator(BilinearFormIntegrator * bfi_real,
                                  BilinearFormIntegrator * bfi_imag);

   /// Adds a boundary trace face integrator. Assumes ownership of @a bfi.
   void AddBdrTraceFaceIntegrator(BilinearFormIntegrator * bfi_real,
                                  BilinearFormIntegrator * bfi_imag,
                                  Array<int> &bdr_marker);

   /// Assemble the local matrix
   void Assemble(int skip_zeros = 1);

   /// Finalizes the matrix initialization.
   void Finalize(int skip_zeros = 1);

   /// Updates the internal mixed forms with the new finite element space.
   virtual void Update();

   /// Returns the matrix assembled on the true dofs, i.e. P^t A P.
   /** The returned matrix has to be deleted by the caller. */
   ComplexHypreParMatrix * ParallelAssemble();

   void FormRectangularLinearSystem(const Array<int> & ess_trial_tdof_list,
                                    const Array<int> & ess_test_tdof_list,
                                    Vector & x,
                                    Vector & b,
                                    OperatorHandle & A,
                                    Vector & X,
                                    Vector & B);

   void FormRectangularSystemMatrix(const Array<int> & ess_trial_tdof_list,
                                    const Array<int> & ess_test_tdof_list,
                                    OperatorHandle & A);

   virtual ~ParMixedSesquilinearForm();
};

#endif // MFEM_USE_MPI

}

#endif // MFEM_COMPLEX_FEM
