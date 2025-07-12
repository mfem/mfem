// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
#include <complex>

namespace mfem
{

/// Class for complex-valued grid function - real + imaginary part Vector with
/// associated FE space.
class ComplexGridFunction : public Vector
{
private:
   GridFunction * gfr;
   GridFunction * gfi;

protected:
   void Destroy() { delete gfr; delete gfi; }

public:
   /** @brief Construct a ComplexGridFunction associated with the
       FiniteElementSpace @a *f. */
   ComplexGridFunction(FiniteElementSpace *f);

   void Update();

   /// Assign constant values to the ComplexGridFunction data.
   ComplexGridFunction &operator=(const std::complex<real_t> & value)
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

   FiniteElementSpace *FESpace() { return gfr->FESpace(); }
   const FiniteElementSpace *FESpace() const { return gfr->FESpace(); }

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

   /// Destroys the grid function.
   virtual ~ComplexGridFunction() { Destroy(); }

};

/** Class for a complex-valued linear form

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the ComplexLinearForm either the real or
    imaginary integrator can be NULL. This indicates that the corresponding
    portion of the complex-valued field is equal to zero.
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
   ComplexLinearForm &operator=(const std::complex<real_t> & value)
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

   /// Adds new domain integrator with a complex-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddDomainIntegrator(C &coef)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Domain Integrator with a complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddDomainIntegrator(C &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()),
                                elem_marker);
   }

   /// Adds new domain integrator with a real-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddDomainIntegrator(C &coef)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Domain Integrator with a real-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddDomainIntegrator(C &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr, elem_marker);
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

   /// Adds new boundary integrator with a complex-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddBoundaryIntegrator(C &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()),
                                  bdr_marker);
   }

   /// Adds new boundary integrator with a real-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* = nullptr>
   void AddBoundaryIntegrator(C &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr, bdr_marker);
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

   /// Adds new Boundary Face integrator with a complex-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Face integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()),
                                 bdr_marker);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr, bdr_marker);
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
   void SyncAlias() { lfr->SyncAliasMemory(*this); lfi->SyncAliasMemory(*this); }

   void Update();
   void Update(FiniteElementSpace *f);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   std::complex<real_t> operator()(const ComplexGridFunction &gf) const;
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

public:
   SesquilinearForm(FiniteElementSpace *fes,
                    ComplexOperator::Convention
                    convention = ComplexOperator::HERMITIAN);
   /** @brief Create a SesquilinearForm on the FiniteElementSpace @a fes, using
       the same integrators as the BilinearForms @a bfr and @a bfi .

       The pointer @a fes is not owned by the newly constructed object.

       The integrators are copied as pointers and they are not owned by the
       newly constructed SesquilinearForm. */
   SesquilinearForm(FiniteElementSpace *fes, BilinearForm *bfr, BilinearForm *bfi,
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

   /// Adds new domain integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(ComplexCoefficient &coef)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Domain Integrator with a complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(ComplexCoefficient &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()),
                                elem_marker);
   }

   /// Adds new domain integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(Coefficient &coef)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Domain Integrator with a real-valued coefficient,
   /// restricted to the given attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(Coefficient &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr, elem_marker);
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

   /// Adds new boundary integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(ComplexCoefficient &coef)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(ComplexCoefficient &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()),
                                  bdr_marker);
   }

   /// Adds new boundary integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(Coefficient &coef)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(Coefficient &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr, bdr_marker);
   }

   /// @}

   /// @name Adding Interior Face Integrators
   /// @{

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                  BilinearFormIntegrator *bfi_imag);

   /// Adds new interior Face integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddInteriorFaceIntegrator(ComplexCoefficient &coef)
   {
      this->AddInteriorFaceIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new interior Face integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddInteriorFaceIntegrator(Coefficient &coef)
   {
      this->AddInteriorFaceIntegrator(new T(coef), (T*)nullptr);
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

   /// Adds new Boundary Face integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(ComplexCoefficient &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Face integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(ComplexCoefficient &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()),
                                 bdr_marker);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(Coefficient &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(Coefficient &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr, bdr_marker);
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

#ifdef MFEM_USE_MPI

/// Class for parallel complex-valued grid function - real + imaginary part
/// Vector with associated parallel FE space.
class ParComplexGridFunction : public Vector
{
private:

   ParGridFunction * pgfr;
   ParGridFunction * pgfi;

protected:
   void Destroy() { delete pgfr; delete pgfi; }

public:

   /** @brief Construct a ParComplexGridFunction associated with the
       ParFiniteElementSpace @a *pf. */
   ParComplexGridFunction(ParFiniteElementSpace *pf);

   void Update();

   /// Assign constant values to the ParComplexGridFunction data.
   ParComplexGridFunction &operator=(const std::complex<real_t> & value)
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

   FiniteElementSpace *FESpace() { return pgfr->FESpace(); }
   const FiniteElementSpace *FESpace() const { return pgfr->FESpace(); }

   ParFiniteElementSpace *ParFESpace() { return pgfr->ParFESpace(); }
   const ParFiniteElementSpace *ParFESpace() const { return pgfr->ParFESpace(); }

   ParGridFunction & real() { return *pgfr; }
   ParGridFunction & imag() { return *pgfi; }
   const ParGridFunction & real() const { return *pgfr; }
   const ParGridFunction & imag() const { return *pgfi; }

   /// Update the memory location of the real and imaginary ParGridFunction @a
   /// pgfr and @a pgfi to match the ParComplexGridFunction.
   void Sync() { pgfr->SyncMemory(*this); pgfi->SyncMemory(*this); }

   /// Update the alias memory location of the real and imaginary
   /// ParGridFunction @a pgfr and @a pgfi to match the ParComplexGridFunction.
   void SyncAlias() { pgfr->SyncAliasMemory(*this); pgfi->SyncAliasMemory(*this); }


   virtual real_t ComputeL2Error(Coefficient &exsolr, Coefficient &exsoli,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      real_t err_r = pgfr->ComputeL2Error(exsolr, irs, elems);
      real_t err_i = pgfi->ComputeL2Error(exsoli, irs, elems);
      return sqrt(err_r * err_r + err_i * err_i);
   }

   virtual real_t ComputeL2Error(Coefficient &exsolr,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const;

   virtual real_t ComputeL2Error(ComplexCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      return this->ComputeL2Error(exsol.real(), exsol.imag(), irs, elems);
   }

   virtual real_t ComputeL2Error(VectorCoefficient &exsolr,
                                 VectorCoefficient &exsoli,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      real_t err_r = pgfr->ComputeL2Error(exsolr, irs, elems);
      real_t err_i = pgfi->ComputeL2Error(exsoli, irs, elems);
      return sqrt(err_r * err_r + err_i * err_i);
   }

   virtual real_t ComputeL2Error(VectorCoefficient &exsolr,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const;

   virtual real_t ComputeL2Error(ComplexVectorCoefficient &exsol,
                                 const IntegrationRule *irs[] = NULL,
                                 Array<int> *elems = NULL) const
   {
      return this->ComputeL2Error(exsol.real(), exsol.imag(), irs, elems);
   }


   /// Destroys grid function.
   virtual ~ParComplexGridFunction() { Destroy(); }

};

/** Class for a complex-valued, parallel linear form

    The @a convention argument in the class's constructor is documented in the
    mfem::ComplexOperator class found in linalg/complex_operator.hpp.

    When supplying integrators to the ParComplexLinearForm either the real or
    imaginary integrator can be NULL.  This indicates that the corresponding
    portion of the complex-valued field is equal to zero.
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

      The pointer @a fes is not owned by the newly constructed object.

      The integrators are copied as pointers and they are not owned by the newly
      constructed ParComplexLinearForm. */
   ParComplexLinearForm(ParFiniteElementSpace *pf, ParLinearForm *plf_r,
                        ParLinearForm *plf_i,
                        ComplexOperator::Convention
                        convention = ComplexOperator::HERMITIAN);

   virtual ~ParComplexLinearForm();

   /// Assign constant values to the ParComplexLinearForm data.
   ParComplexLinearForm &operator=(const std::complex<real_t> & value)
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

   /// Adds new domain integrator with a complex-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddDomainIntegrator(C &coef)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Domain Integrator with a complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddDomainIntegrator(C &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()),
                                elem_marker);
   }

   /// Adds new domain integrator with a real-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddDomainIntegrator(C &coef)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Domain Integrator with a real-valued coefficient,
   /// restricted to the given attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddDomainIntegrator(C &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr, elem_marker);
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

   /// Adds new boundary integrator with a complex-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBoundaryIntegrator(C &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()),
                                  bdr_marker);
   }

   /// Adds new boundary integrator with a real-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBoundaryIntegrator(C &coef)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBoundaryIntegrator(C &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr, bdr_marker);
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

   /// Adds new Boundary Face integrator with a complex-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Face integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<ComplexCoefficient,
                                                      C>::value ||
                                      std::is_base_of<ComplexVectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()),
                                 bdr_marker);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T, typename C,
             typename std::enable_if<std::is_base_of<LinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<LinearFormIntegrator,
                                                   T>::value &&
                                     (std::is_base_of<Coefficient,
                                                      C>::value ||
                                      std::is_base_of<VectorCoefficient,
                                                      C>::value)>::type* =
             nullptr>
   void AddBdrFaceIntegrator(C &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr, bdr_marker);
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
   void SyncAlias() { plfr->SyncAliasMemory(*this); plfi->SyncAliasMemory(*this); }

   void Update(ParFiniteElementSpace *pf = NULL);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();

   std::complex<real_t> operator()(const ParComplexGridFunction &gf) const;

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

   /// Adds new domain integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(ComplexCoefficient &coef)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Domain Integrator with a complex-valued coefficient,
   /// restricted to the given attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(ComplexCoefficient &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef.real()), new T(coef.imag()),
                                elem_marker);
   }

   /// Adds new domain integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(Coefficient &coef)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Domain Integrator with a real-valued coefficient,
   /// restricted to the given attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddDomainIntegrator(Coefficient &coef,
                            Array<int> &elem_marker)
   {
      this->AddDomainIntegrator(new T(coef), (T*)nullptr, elem_marker);
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

   /// Adds new boundary integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(ComplexCoefficient &coef)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(ComplexCoefficient &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef.real()), new T(coef.imag()),
                                  bdr_marker);
   }

   /// Adds new boundary integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(Coefficient &coef)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBoundaryIntegrator(Coefficient &coef,
                              Array<int> &bdr_marker)
   {
      this->AddBoundaryIntegrator(new T(coef), (T*)nullptr, bdr_marker);
   }

   /// @}

   /// @name Adding Interior Face Integrators
   /// @{

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi_real,
                                  BilinearFormIntegrator *bfi_imag);

   /// Adds new interior Face integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddInteriorFaceIntegrator(Coefficient &coef)
   {
      this->AddInteriorFaceIntegrator(new T(coef), (T*)nullptr);
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

   /// Adds new Boundary Face integrator with a complex-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(ComplexCoefficient &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()));
   }

   /// Adds new Boundary Face integrator with a complex-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(ComplexCoefficient &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef.real()), new T(coef.imag()),
                                 bdr_marker);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(Coefficient &coef)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr);
   }

   /// Adds new Boundary Face integrator with a real-valued coefficient,
   /// restricted to specific boundary attributes.
   ///
   /// Assumes ownership of @a bfi.
   ///
   /// The array @a bdr_marker is stored internally as a pointer to the given
   /// Array<int> object.
   template <typename T,
             typename std::enable_if<std::is_base_of<BilinearFormIntegrator,
                                                     T>::value &&
                                     !std::is_same<BilinearFormIntegrator,
                                                   T>::value>::type* = nullptr>
   void AddBdrFaceIntegrator(Coefficient &coef,
                             Array<int> &bdr_marker)
   {
      this->AddBdrFaceIntegrator(new T(coef), (T*)nullptr, bdr_marker);
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

#endif // MFEM_USE_MPI

}

#endif // MFEM_COMPLEX_FEM
