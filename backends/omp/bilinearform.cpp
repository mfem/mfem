// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "backend.hpp"
#include "bilinearform.hpp"
#include "adiffusioninteg.hpp"

namespace mfem
{

namespace omp
{

BilinearForm::~BilinearForm()
{
   // Make sure all integrators free their data
   for (int i = 0; i < tbfi.Size(); i++) delete tbfi[i];
}

void BilinearForm::TransferIntegrators()
{
   mfem::Array<mfem::BilinearFormIntegrator*> &dbfi = *bform->GetDBFI();
   for (int i = 0; i < dbfi.Size(); i++)
   {
      std::string integ_name(dbfi[i]->Name());
      Coefficient *scal_coeff = dbfi[i]->GetScalarCoefficient();
      // ConstantCoefficient *const_coeff =
      //    dynamic_cast<ConstantCoefficient*>(scal_coeff);
      // // TODO: other types of coefficients ...
      // double val = const_coeff ? const_coeff->constant : 1.0;

      if (integ_name == "(undefined)")
      {
         MFEM_ABORT("BilinearFormIntegrator does not define Name()");
      }
      else if (integ_name == "diffusion")
      {
         switch (OmpEngine().IntegType())
         {
         case Acrotensor:
            tbfi.Append(new AcroDiffusionIntegrator(*scal_coeff, bform->FESpace()->Get_PFESpace()->As<FiniteElementSpace>()));
            break;
         default:
            mfem_error("integrator is not supported for any MultType");
            break;
         }
      }
      else
      {
         MFEM_ABORT("BilinearFormIntegrator [Name() = " << integ_name
                    << "] is not supported");
      }
   }
}

void BilinearForm::InitRHS(const mfem::Array<int> &ess_tdof_list,
                           mfem::Vector &mfem_x, mfem::Vector &mfem_b,
                           mfem::Operator *A,
                           mfem::Vector &mfem_X, mfem::Vector &mfem_B,
                           int copy_interior) const
{
   const mfem::Operator *P = GetProlongation();
   const mfem::Operator *R = GetRestriction();

   if (P)
   {
      // Variational restriction with P
      mfem_B.Resize(P->InLayout());
      P->MultTranspose(mfem_b, mfem_B);
      mfem_X.Resize(R->OutLayout());
      R->Mult(mfem_x, mfem_X);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      mfem_X.MakeRef(mfem_x);
      mfem_B.MakeRef(mfem_b);
   }

   if (!copy_interior && ess_tdof_list.Size() > 0)
   {
      Vector &X = mfem_X.Get_PVector()->As<Vector>();
      const Array &constraint_list = ess_tdof_list.Get_PArray()->As<Array>();

      double *X_data = X.GetData();
      const int* constraint_data = constraint_list.GetData<int>();

      Vector subvec(constraint_list.OmpLayout());
      double *subvec_data = subvec.GetData();

      const std::size_t num_constraint = constraint_list.Size();

      // This operation is a general version of mfem::Vector::SetSubVectorComplement()
      // {
#pragma omp target teams distribute parallel for        \
   map(to: subvec_data, constraint_data, X_data)        \
   if (target: constraint_list.ComputeOnDevice())            \
   if (parallel: num_constraint > 1000)
      for (std::size_t i = 0; i < num_constraint; i++) subvec_data[i] = X_data[constraint_data[i]];

      X.Fill(0.0);

#pragma omp target teams distribute parallel for        \
   map(to: X_data, constraint_data, subvec_data)        \
   if (target: constraint_list.ComputeOnDevice())            \
   if (parallel: num_constraint > 1000)
      for (std::size_t i = 0; i < num_constraint; i++) X_data[constraint_data[i]] = subvec_data[i];
      // }
   }

   ConstrainedOperator *A_constrained = dynamic_cast<ConstrainedOperator*>(A);
   if (A_constrained)
   {
      A_constrained->EliminateRHS(mfem_X, mfem_B);
   }
   else
   {
      mfem_error("mfem::omp::BilinearForm::InitRHS expects a ConstrainedOperator");
   }
}


bool BilinearForm::Assemble()
{
   if (!has_assembled)
   {
      TransferIntegrators();
      has_assembled = true;
   }

   return true;
}

void BilinearForm::FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                    mfem::OperatorHandle &A)
{
   if (A.Type() == mfem::Operator::ANY_TYPE)
   {
      // TODO: Support different test and trial spaces (MixedBilinearForm)
      const mfem::Operator *P = GetProlongation();

      mfem::Operator *rap = this;
      if (P != NULL) rap = new mfem::RAPOperator(*P, *this, *P);

      A.Reset(new ConstrainedOperator(rap, ess_tdof_list, (rap != this)));
   }
   else
   {
      MFEM_ABORT("Operator::Type is not supported, type = " << A.Type());
   }
}

void BilinearForm::FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                    mfem::Vector &x, mfem::Vector &b,
                                    mfem::OperatorHandle &A, mfem::Vector &X, mfem::Vector &B,
                                    int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   InitRHS(ess_tdof_list, x, b, A.Ptr(), X, B, copy_interior);
}

void BilinearForm::RecoverFEMSolution(const mfem::Vector &X, const mfem::Vector &b,
                                      mfem::Vector &x)
{
   const mfem::Operator *P = GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.Resize(P->OutLayout());
      P->Mult(X, x);
   }
   // Otherwise X and x point to the same data
}

void BilinearForm::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   trial_fes->ToEVector(x.Get_PVector()->As<Vector>(), x_local);

   y_local.Fill<double>(0.0);
   for (int i = 0; i < tbfi.Size(); i++) tbfi[i]->MultAdd(x_local, y_local);

   test_fes->ToLVector(y_local, y.Get_PVector()->As<Vector>());
}

void BilinearForm::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{ mfem_error("mfem::omp::BilinearForm::MultTranspose() is not supported!"); }


ConstrainedOperator::ConstrainedOperator(mfem::Operator *A_,
                                         const mfem::Array<int> &constraint_list_,
                                         bool own_A_)
   : Operator(A_->InLayout()->As<Layout>()),
     A(A_),
     own_A(own_A_),
     constraint_list(constraint_list_.Get_PArray()->As<Array>()),
     z(OutLayout()->As<Layout>()),
     w(OutLayout()->As<Layout>()),
     mfem_z((z.DontDelete(), z)),
     mfem_w((w.DontDelete(), w)) { }

void ConstrainedOperator::EliminateRHS(const mfem::Vector &mfem_x, mfem::Vector &mfem_b) const
{
   w.Fill<double>(0.0);

   const Vector &x = mfem_x.Get_PVector()->As<Vector>();
   Vector &b = mfem_b.Get_PVector()->As<Vector>();

   const double *x_data = x.GetData();
   double *b_data = b.GetData();
   double *w_data = w.GetData();
   const int* constraint_data = constraint_list.GetData<int>();

   const std::size_t num_constraint = constraint_list.Size();

   if (num_constraint > 0)
   {
#pragma omp target teams distribute parallel for             \
   map(to: w_data, constraint_data, x_data)                  \
   if (target: constraint_list.ComputeOnDevice())            \
   if (parallel: num_constraint > 1000)
      for (std::size_t i = 0; i < num_constraint; i++)
         w_data[constraint_data[i]] = x_data[constraint_data[i]];
   }

   A->Mult(mfem_w, mfem_z);
   std::cout << "here" << std::endl;
   z.Fill<double>(0.0);

   std::cout << b.Size() << " " << z.Size() << std::endl;
   b.Axpby<double>(1.0, b, -1.0, z);

   if (num_constraint > 0)
   {
#pragma omp target teams distribute parallel for        \
   map(to: b_data, constraint_data, x_data)             \
   if (target: constraint_list.ComputeOnDevice())       \
   if (parallel: num_constraint > 1000)
      for (std::size_t i = 0; i < num_constraint; i++)
         b_data[constraint_data[i]] = x_data[constraint_data[i]];
   }
}

void ConstrainedOperator::Mult(const mfem::Vector &mfem_x, mfem::Vector &mfem_y) const
{
   if (constraint_list.Size() == 0)
   {
      A->Mult(mfem_x, mfem_y);
      return;
   }

   const Vector &x = mfem_x.Get_PVector()->As<Vector>();
   Vector &y = mfem_y.Get_PVector()->As<Vector>();

   const double *x_data = x.GetData();
   double *y_data = y.GetData();
   double *z_data = z.GetData();
   const int* constraint_data = constraint_list.GetData<int>();

   const std::size_t num_constraint = constraint_list.Size();

   z.Assign<double>(x); // z = x

   // z[constraint_list] = 0.0
#pragma omp target teams distribute parallel for        \
   map(to: z_data, constraint_data)                     \
   if (target: constraint_list.ComputeOnDevice())            \
   if (parallel: num_constraint > 1000)
   for (std::size_t i = 0; i < num_constraint; i++)
      z_data[constraint_data[i]] = 0.0;

   // y = A * z
   A->Mult(mfem_z, mfem_y);

   // y[constraint_list] = x[constraint_list]
#pragma omp target teams distribute parallel for        \
   map(to: y_data, constraint_data, x_data)             \
   if (target: constraint_list.ComputeOnDevice())            \
   if (parallel: num_constraint > 1000)
   for (std::size_t i = 0; i < num_constraint; i++)
      y_data[constraint_data[i]] = x_data[constraint_data[i]];
}

// Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
ConstrainedOperator::~ConstrainedOperator()
{
   if (own_A) delete A;
}


} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)
