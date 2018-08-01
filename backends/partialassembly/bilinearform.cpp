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
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "backend.hpp"
#include "bilinearform.hpp"

namespace mfem
{

namespace pa
{

BilinearForm::~BilinearForm()
{
   // Make sure all integrators free their data
   for (int i = 0; i < tbfi.Size(); i++) delete tbfi[i];
}

void BilinearForm::TransferIntegrators(mfem::Array<mfem::BilinearFormIntegrator*>& bfi) {
   for (int i = 0; i < bfi.Size(); i++)
   {
      mfem::FiniteElementSpace* fes = bform->FESpace();
      const int order = fes->GetFE(0)->GetOrder();
      const int ir_order = 2 * order + 1;
      std::string integ_name(bfi[i]->Name());
      if (integ_name == "(undefined)")
      {
         MFEM_ABORT("BilinearFormIntegrator does not define Name()");
      }
      else if (integ_name == "mass")
      {
         std::cout << "=> " << integ_name << " Integrator transfered" << std::endl;
         MassIntegrator* integ = dynamic_cast<MassIntegrator*>(bfi[i]);
         Coefficient* coef;
         integ->GetParameters(coef);
         if (coef) {
            std::cout << "==> with Coefficient" << std::endl;
            typename MassEquation::ArgsCoeff args(*coef);
            AddIntegrator( new PADomainInt<MassEquation, Vector<double>>(fes, ir_order, args) );
         } else {
            std::cout << "==> without Coefficient" << std::endl;
            // typename MassEquation::ArgsEmpty args;
            // AddIntegrator( new PADomainInt<MassEquation, Vector<double>>(fes, ir_order, args) );
            HostMassEq eq(*fes, ir_order);
            // AddIntegrator( createPADomainKernel(eq) );
            AddIntegrator( createMFDomainKernel(eq) );
         }
      }
      else if (integ_name == "diffusion")
      {
         std::cout << "=> " << integ_name << " Integrator transfered" << std::endl;
         DiffusionIntegrator* integ = dynamic_cast<DiffusionIntegrator*>(bfi[i]);
         Coefficient* coef;
         integ->GetParameters(coef);
         typename DiffusionEquation::Args args(*coef);
         AddIntegrator( new PADomainInt<DiffusionEquation, Vector<double>, TensorDomainMult>(fes, ir_order, args) );
      }
      else if (integ_name == "convection")
      {
         std::cout << "=> " << integ_name << " Integrator transfered" << std::endl;
         ConvectionIntegrator* integ = dynamic_cast<ConvectionIntegrator*>(bfi[i]);
         VectorCoefficient* u;
         double* alpha;
         integ->GetParameters(u, alpha);
         typename DGConvectionEquation::Args args(*u, *alpha);
         AddIntegrator( new PADomainInt<DGConvectionEquation, Vector<double>>(fes, ir_order, args) );
      }
      else if (integ_name == "transpose")
      {
         std::cout << "=> " << integ_name << " Integrator transfered" << std::endl;
         TransposeIntegrator* transInteg = dynamic_cast<TransposeIntegrator*>(bfi[i]);
         BilinearFormIntegrator* bf;
         transInteg->GetParameters(bf);
         integ_name = bf->Name();
         if (integ_name == "dgtrace")
         {
            std::cout << "==> " << integ_name << " Integrator transfered" << std::endl;
            DGTraceIntegrator* integ = dynamic_cast<DGTraceIntegrator*>(bf);
            Coefficient* rho;
            VectorCoefficient* u;
            double* alpha;
            double* beta;
            integ->GetParameters(rho, u, alpha, beta);
            typename DGConvectionEquation::Args args(*u, -(*alpha), *beta);
            AddIntegrator( new PAFaceInt<DGConvectionEquation, Vector<double>>(fes, ir_order, args) );
         }
         else
         {
            MFEM_ABORT("Transpose BilinearFormIntegrator [Name() = " << integ_name
                       << "] is not supported");
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
                           mfem::OperatorHandle& A,
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

   if (A.Type() != mfem::Operator::ANY_TYPE)
   {
      OperatorHandle mat_e;
      A.EliminateBC(mat_e, ess_tdof_list, mfem_X, mfem_B);
   }

   if (!copy_interior && ess_tdof_list.Size() > 0)
   {
      Vector<double> &X = mfem_X.Get_PVector()->As<Vector<double>>();
      const Array &constraint_list = ess_tdof_list.Get_PArray()->As<Array>();

      double *X_data = X.GetData();
      const int* constraint_data = constraint_list.GetTypedData<int>();

      Vector<double> subvec(constraint_list.GetLayout());
      double *subvec_data = subvec.GetData();

      const std::size_t num_constraint = constraint_list.Size();

      for (std::size_t i = 0; i < num_constraint; i++) subvec_data[i] = X_data[constraint_data[i]];

      X.Fill(0.0);

      for (std::size_t i = 0; i < num_constraint; i++) X_data[constraint_data[i]] = subvec_data[i];
   }

   if (A.Type() == mfem::Operator::ANY_TYPE)
   {
      ConstrainedOperator *A_constrained = static_cast<ConstrainedOperator*>(A.Ptr());
      A_constrained->EliminateRHS(mfem_X, mfem_B);
   }
}


bool BilinearForm::Assemble()
{
   if (!has_assembled)
   {
      TransferIntegrators(*bform->GetDBFI());
      TransferIntegrators(*bform->GetFBFI());
      has_assembled = true;
   }

   return true;
}

void BilinearForm::FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                    mfem::OperatorHandle &A)
{
   if (A.Type() == mfem::Operator::ANY_TYPE)
   {
      // FIXME: Support different test and trial spaces (MixedBilinearForm)
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
   InitRHS(ess_tdof_list, x, b, A, X, B, copy_interior);
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
   trial_fes->ToEVector(x.Get_PVector()->As<Vector<double>>(), x_local);

   y_local.Fill<double>(0.0);
   for (int i = 0; i < tbfi.Size(); i++) tbfi[i]->MultAdd(x_local, y_local);
   for (int i = 0; i < pabfi.Size(); i++) pabfi[i]->MultAdd(x_local, y_local);

   test_fes->ToLVector(y_local, y.Get_PVector()->As<Vector<double>>());
}

void BilinearForm::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{ mfem_error("mfem::pa::BilinearForm::MultTranspose() is not supported!"); }


ConstrainedOperator::ConstrainedOperator(mfem::Operator *A_,
      const mfem::Array<int> &constraint_list_,
      bool own_A_)
   : Operator(A_->InLayout()->As<Layout>()),
     A(A_),
     own_A(own_A_),
     constraint_list(*InLayout()->GetEngine().MakeLayout(constraint_list_.Size()).As<Layout>(), sizeof(int)),
     z(OutLayout()->As<Layout>()),
     w(OutLayout()->As<Layout>()),
     mfem_z((z.DontDelete(), z)),
     mfem_w((w.DontDelete(), w))
{
   constraint_list.PushData(constraint_list_.GetData());
}

void ConstrainedOperator::EliminateRHS(const mfem::Vector &mfem_x, mfem::Vector &mfem_b) const
{
   w.Fill<double>(0.0);

   const Vector<double> &x = mfem_x.Get_PVector()->As<Vector<double>>();
   Vector<double> &b = mfem_b.Get_PVector()->As<Vector<double>>();

   const double *x_data = x.GetData();
   double *b_data = b.GetData();
   double *w_data = w.GetData();
   const int* constraint_data = constraint_list.GetTypedData<int>();

   const std::size_t num_constraint = constraint_list.Size();

   if (num_constraint > 0)
   {
      for (std::size_t i = 0; i < num_constraint; i++)
         w_data[constraint_data[i]] = x_data[constraint_data[i]];
   }

   A->Mult(mfem_w, mfem_z);

   b.Axpby<double>(1.0, b, -1.0, z);

   if (num_constraint > 0)
   {
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

   const Vector<double> &x = mfem_x.Get_PVector()->As<Vector<double>>();
   Vector<double> &y = mfem_y.Get_PVector()->As<Vector<double>>();

   const double *x_data = x.GetData();
   double *y_data = y.GetData();
   double *z_data = z.GetData();
   const int* constraint_data = constraint_list.GetTypedData<int>();

   const std::size_t num_constraint = constraint_list.Size();

   z.Assign<double>(x); // z = x

   // z[constraint_list] = 0.0
   for (std::size_t i = 0; i < num_constraint; i++)
      z_data[constraint_data[i]] = 0.0;

   // y = A * z
   A->Mult(mfem_z, mfem_y);

   // y[constraint_list] = x[constraint_list]
   for (std::size_t i = 0; i < num_constraint; i++)
      y_data[constraint_data[i]] = x_data[constraint_data[i]];
}

// Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
ConstrainedOperator::~ConstrainedOperator()
{
   if (own_A) delete A;
}


} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)