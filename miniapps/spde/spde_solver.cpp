// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#include <algorithm>
#include <ctime>

#include "../../examples/ex33.hpp"
#include "spde_solver.hpp"

namespace mfem
{
namespace spde
{

// Helper function that determines if output should be printed to the console.
// The output is printed if the rank is 0 and if the print level is greater than
// 0. The rank is retrieved via the fespace.
bool PrintOutput(const ParFiniteElementSpace *fespace_ptr, int print_level)
{
   return (fespace_ptr->GetMyRank() == 0 && print_level > 0);
}

void Boundary::PrintInfo(std::ostream &os) const
{
   os << "\n<Boundary Info>\n"
      << " Boundary Conditions:\n";
   for (const auto &it : boundary_attributes)
   {
      os << "  Boundary " << it.first << ": ";
      switch (it.second)
      {
         case BoundaryType::kNeumann:
            os << "Neumann";
            break;
         case BoundaryType::kDirichlet:
            os << "Dirichlet";
            break;
         case BoundaryType::kRobin:
            os << "Robin, coefficient: " << robin_coefficient;
            break;
         case BoundaryType::kPeriodic:
            os << "Periodic";
            break;
         default:
            os << "Undefined";
            break;
      }
      os << "\n";
   }
   bool first_print_statement = true;
   // If the map is not empty
   if (!dirichlet_coefficients.empty())
   {
      os << "  Inhomogeneous Dirichlet defined on ";
      for (const auto &it : dirichlet_coefficients)
      {
         if (!first_print_statement)
         {
            os << ", ";
         }
         else
         {
            first_print_statement = false;
         }
         os << it.first << "(=" << it.second << ")";
      }
      os << "\n";
   }
   os << "<Boundary Info>\n\n";
}

void Boundary::VerifyDefinedBoundaries(const Mesh &mesh) const
{
   // Verify that all defined boundaries are actually defined on the
   // mesh, i.e. if the keys of boundary attributes appear in the boundary
   // attributes of the mesh.
   mfem::out << "\n<Boundary Verify>\n";
   const Array<int> boundary(mesh.bdr_attributes);
   for (const auto &it : boundary_attributes)
   {
      if (boundary.Find(it.first) == -1)
      {
         MFEM_ABORT("  Boundary "
                    << it.first
                    << " is not defined on the mesh but in Boundary class."
                    << "Exiting...")
      }
   }

   /// Verify if all boundary attributes appear as keys in the
   /// boundary attributes, if not let the user know that we use Neumann by
   /// default.
   std::vector<int> boundary_attributes_keys;
   for (int i = 0; i < boundary.Size(); i++)
   {
      if (boundary_attributes.find(boundary[i]) == boundary_attributes.end())
      {
         boundary_attributes_keys.push_back(boundary[i]);
      }
   }
   if (!boundary_attributes_keys.empty())
   {
      mfem::out << "  Boundaries (";
      for (const auto &it : boundary_attributes_keys)
      {
         mfem::out << it << ", ";
      }
      mfem::out << ") are defined on the mesh but not in the";
      mfem::out << " boundary attributes (Use Neumann).";
   }

   /// Check if any periodic boundary is registered
   for (const auto &it : boundary_attributes)
   {
      if (it.second == BoundaryType::kPeriodic)
      {
         MFEM_ABORT("  Periodic boundaries must be defined on the mesh"
                    << ", not in Boundaries. Exiting...")
      }
   }

   mfem::out << "\n<Boundary Verify>\n\n";
}

void Boundary::ComputeBoundaryError(const ParGridFunction &solution)
{
   const ParFiniteElementSpace &fes = *solution.ParFESpace();
   const ParMesh &pmesh = *fes.GetParMesh();

   if (PrintOutput(&fes, 1))
   {
      mfem::out << "<Boundary::ComputeBoundaryError>"
                << "\n";
      mfem::out << "   GetVDim: " << fes.GetVDim() << "\n";
   }

   real_t alpha{0.0};
   real_t beta{1.0};
   real_t gamma{0.0};

   // Index i needs to be incremented by one to map to the boundary attributes
   // in the mesh.
   for (int i = 0; i < pmesh.bdr_attributes.Max(); i++)
   {
      real_t error{0};
      real_t avg{0};
      Array<int> bdr(pmesh.bdr_attributes.Max());
      bdr = 0;
      bdr[i] = 1;

      UpdateIntegrationCoefficients(i + 1, alpha, beta, gamma);
      avg = IntegrateBC(solution, bdr, alpha, beta, gamma, error);
      if (PrintOutput(&fes, 1))
      {
         mfem::out << "->Boundary " << i + 1 << "\n";
         mfem::out << "    Alpha   : " << alpha << "\n";
         mfem::out << "    Beta    : " << beta << "\n";
         mfem::out << "    Gamma   : " << gamma << "\n";
         mfem::out << "    Average : " << avg << "\n";
         mfem::out << "    Error   : " << error << "\n\n";
      }
   }

   if (PrintOutput(&fes, 1))
   {
      mfem::out << "<Boundary::ComputeBoundaryError>" << std::endl;
   }
}

void Boundary::UpdateIntegrationCoefficients(int i, real_t &alpha, real_t &beta,
                                             real_t &gamma)
{
   // Check if i is a key in boundary_attributes
   if (boundary_attributes.find(i) != boundary_attributes.end())
   {
      switch (boundary_attributes[i])
      {
         case BoundaryType::kNeumann:
            alpha = 1.0;
            beta = 0.0;
            gamma = 0.0;
            break;
         case BoundaryType::kDirichlet:
            alpha = 0.0;
            beta = 1.0;
            if (dirichlet_coefficients.find(i) != dirichlet_coefficients.end())
            {
               gamma = dirichlet_coefficients[i];
            }
            else
            {
               gamma = 0.0;
            }
            break;
         case BoundaryType::kRobin:
            alpha = 1.0;
            beta = robin_coefficient;
            gamma = 0.0;
            break;
         default:
            alpha = 1.0;
            beta = 0.0;
            gamma = 0.0;
            break;
      }
   }
   else
   {
      // If i is not a key in boundary_attributes, it corresponds to Neumann.
      alpha = 1.0;
      beta = 0.0;
      gamma = 0.0;
   }
}

void Boundary::AddHomogeneousBoundaryCondition(int boundary,
                                               BoundaryType type)
{
   boundary_attributes[boundary] = type;
}

void Boundary::AddInhomogeneousDirichletBoundaryCondition(int boundary,
                                                          real_t coefficient)
{
   boundary_attributes[boundary] = BoundaryType::kDirichlet;
   dirichlet_coefficients[boundary] = coefficient;
}

void Boundary::SetRobinCoefficient(real_t coefficient)
{
   robin_coefficient = coefficient;
}

real_t IntegrateBC(const ParGridFunction &x, const Array<int> &bdr,
                   real_t alpha, real_t beta, real_t gamma, real_t &glb_err)
{
   real_t loc_vals[3];
   real_t &nrm = loc_vals[0];
   real_t &avg = loc_vals[1];
   real_t &error = loc_vals[2];

   nrm = 0.0;
   avg = 0.0;
   error = 0.0;

   const bool a_is_zero = alpha == 0.0;
   const bool b_is_zero = beta == 0.0;

   const ParFiniteElementSpace &fes = *x.ParFESpace();
   MFEM_ASSERT(fes.GetVDim() == 1, "");
   ParMesh &mesh = *fes.GetParMesh();
   Vector shape;
   Vector loc_dofs;
   Vector w_nor;
   DenseMatrix dshape;
   Array<int> dof_ids;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (bdr[mesh.GetBdrAttribute(i) - 1] == 0)
      {
         continue;
      }

      FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
      if (FTr == nullptr)
      {
         continue;
      }

      const FiniteElement &fe = *fes.GetFE(FTr->Elem1No);
      MFEM_ASSERT(fe.GetMapType() == FiniteElement::VALUE, "");
      const int int_order = 2 * fe.GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

      fes.GetElementDofs(FTr->Elem1No, dof_ids);
      x.GetSubVector(dof_ids, loc_dofs);
      if (!a_is_zero)
      {
         const int sdim = FTr->Face->GetSpaceDim();
         w_nor.SetSize(sdim);
         dshape.SetSize(fe.GetDof(), sdim);
      }
      if (!b_is_zero)
      {
         shape.SetSize(fe.GetDof());
      }
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         IntegrationPoint eip;
         FTr->Loc1.Transform(ip, eip);
         FTr->Face->SetIntPoint(&ip);
         real_t face_weight = FTr->Face->Weight();
         real_t val = 0.0;
         if (!a_is_zero)
         {
            FTr->Elem1->SetIntPoint(&eip);
            fe.CalcPhysDShape(*FTr->Elem1, dshape);
            CalcOrtho(FTr->Face->Jacobian(), w_nor);
            val += alpha * dshape.InnerProduct(w_nor, loc_dofs) / face_weight;
         }
         if (!b_is_zero)
         {
            fe.CalcShape(eip, shape);
            val += beta * (shape * loc_dofs);
         }

         // Measure the length of the boundary
         nrm += ip.weight * face_weight;

         // Integrate alpha * n.Grad(x) + beta * x
         avg += val * ip.weight * face_weight;

         // Integrate |alpha * n.Grad(x) + beta * x - gamma|^2
         val -= gamma;
         error += (val * val) * ip.weight * face_weight;
      }
   }

   real_t glb_vals[3];
   MPI_Allreduce(loc_vals, glb_vals, 3, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 fes.GetComm());

   real_t glb_nrm = glb_vals[0];
   real_t glb_avg = glb_vals[1];
   glb_err = glb_vals[2];

   // Normalize by the length of the boundary
   if (std::abs(glb_nrm) > 0.0)
   {
      glb_err /= glb_nrm;
      glb_avg /= glb_nrm;
   }

   // Compute l2 norm of the error in the boundary condition (negative
   // quadrature weights may produce negative 'error')
   glb_err = (glb_err >= 0.0) ? sqrt(glb_err) : -sqrt(-glb_err);

   // Return the average value of alpha * n.Grad(x) + beta * x
   return glb_avg;
}

SPDESolver::SPDESolver(real_t nu, const Boundary &bc,
                       ParFiniteElementSpace *fespace, real_t l1, real_t l2,
                       real_t l3, real_t e1, real_t e2, real_t e3)
   : k_(fespace),
     m_(fespace),
     fespace_ptr_(fespace),
     bc_(bc),
     nu_(nu),
     l1_(l1),
     l2_(l2),
     l3_(l3),
     e1_(e1),
     e2_(e2),
     e3_(e3)
{
   if (PrintOutput(fespace_ptr_, print_level_))
   {
      mfem::out << "<SPDESolver> Initialize Solver .." << std::endl;
   }
   StopWatch sw;
   sw.Start();

   // Resize the marker arrays for the boundary conditions
   // Number of boundary attributes in the mesh
   int nbc{0};
   const auto &bdr_attributes = fespace_ptr_->GetParMesh()->bdr_attributes;
   if (bdr_attributes.Size() > 0)
   {
      // Assumes a contiguous range of boundary attributes (1, 2, 3, ...)
      nbc = bdr_attributes.Max() - bdr_attributes.Min() + 1;
   }
   dbc_marker_.SetSize(nbc);
   rbc_marker_.SetSize(nbc);
   dbc_marker_ = 0;
   rbc_marker_ = 0;

   // Fill the marker arrays for the boundary conditions. We decrement the number
   // it.first by one because the boundary attributes in the mesh start at 1 and
   // the marker arrays start at 0.
   for (const auto &it : bc_.boundary_attributes)
   {
      switch (it.second)
      {
         case BoundaryType::kDirichlet:
            dbc_marker_[it.first - 1] = 1;
            break;
         case BoundaryType::kRobin:
            rbc_marker_[it.first - 1] = 1;
            break;
         default:
            break;
      }
   }

   // Handle homogeneous Dirichlet boundary conditions
   // Note: for non zero DBC we usually need to project the boundary onto the
   // solution. This is not necessary in this case since the boundary is
   // homogeneous. For inhomogeneous Dirichlet we consider a lifting scheme.
   fespace_ptr_->GetEssentialTrueDofs(dbc_marker_, ess_tdof_list_);

   // Compute the rational approximation coefficients.
   int dim = fespace_ptr_->GetParMesh()->Dimension();
   int space_dim = fespace_ptr_->GetParMesh()->SpaceDimension();
   alpha_ = (nu_ + dim / 2.0) / 2.0;  // fractional exponent
   integer_order_of_exponent_ = static_cast<int>(std::floor(alpha_));
   real_t exponent_to_approximate = alpha_ - integer_order_of_exponent_;

   // Compute the rational approximation coefficients.
   ComputeRationalCoefficients(exponent_to_approximate);

   // Set the bilinear forms.

   // Assemble stiffness matrix
   auto diffusion_tensor =
      ConstructMatrixCoefficient(l1_, l2_, l3_, e1_, e2_, e3_, nu_, space_dim);
   MatrixConstantCoefficient diffusion_coefficient(diffusion_tensor);
   k_.AddDomainIntegrator(new DiffusionIntegrator(diffusion_coefficient));
   ConstantCoefficient robin_coefficient(bc_.robin_coefficient);
   k_.AddBoundaryIntegrator(new MassIntegrator(robin_coefficient), rbc_marker_);
   k_.Assemble(0);

   // Assemble mass matrix
   ConstantCoefficient one(1.0);
   m_.AddDomainIntegrator(new MassIntegrator(one));
   m_.Assemble(0);

   // Form matrices for the linear system
   Array<int> empty;
   k_.FormSystemMatrix(empty, stiffness_);
   m_.FormSystemMatrix(empty, mass_bc_);

   // Get the restriction and prolongation matrix for transformations
   restriction_matrix_ = fespace->GetRestrictionMatrix();
   prolongation_matrix_ = fespace->GetProlongationMatrix();

   // Resize the vectors B and X to the appropriate size
   if (prolongation_matrix_)
   {
      B_.SetSize(prolongation_matrix_->Width());
   }
   else
   {
      mfem::err << "<SPDESolver> prolongation matrix is not defined" << std::endl;
   }
   if (restriction_matrix_)
   {
      X_.SetSize(restriction_matrix_->Height());
   }
   else
   {
      mfem::err << "<SPDESolver> restriction matrix is not defined" << std::endl;
   }

   sw.Stop();
   if (PrintOutput(fespace_ptr_, print_level_))
   {
      mfem::out << "<SPDESolver::Timing> matrix assembly " << sw.RealTime()
                << " [s]" << std::endl;
   }
}

void SPDESolver::Solve(ParLinearForm &b, ParGridFunction &x)
{
   // ------------------------------------------------------------------------
   // Solve the PDE (A)^N g = f, i.e. compute g = (A)^{-1}^N f iteratively.
   // ------------------------------------------------------------------------

   StopWatch sw;
   sw.Start();

   // Zero initialize x to avoid touching uninitialized memory
   x = 0.0;

   ParGridFunction helper_gf(fespace_ptr_);
   helper_gf = 0.0;

   if (integer_order_of_exponent_ > 0)
   {
      if (PrintOutput(fespace_ptr_, print_level_))
      {
         mfem::out << "<SPDESolver> Solving PDE (A)^" << integer_order_of_exponent_
                   << " u = f" << std::endl;
      }
      ActivateRepeatedSolve();
      Solve(b, helper_gf, 1.0, 1.0, integer_order_of_exponent_);
      if (integer_order_)
      {
         // If the exponent is an integer, we can directly add the solution to the
         // final solution and return.
         x += helper_gf;
         if (!bc_.dirichlet_coefficients.empty())
         {
            LiftSolution(x);
         }
         return;
      }
      UpdateRHS(b);
      DeactivateRepeatedSolve();
   }

   // ------------------------------------------------------------------------
   // Solve the (remaining) fractional PDE by solving M integer order PDEs and
   // adding up the solutions.
   // ------------------------------------------------------------------------
   if (!integer_order_)
   {
      // Iterate over all expansion coefficient that contribute to the
      // solution.
      for (int i = 0; i < coeffs_.Size(); i++)
      {
         if (PrintOutput(fespace_ptr_, print_level_))
         {
            mfem::out << "\n<SPDESolver> Solving PDE -Δ u + " << -poles_[i]
                      << " u = " << coeffs_[i] << " g " << std::endl;
         }
         helper_gf = 0.0;
         Solve(b, helper_gf, 1.0 - poles_[i], coeffs_[i]);
         x += helper_gf;
      }
   }

   // Apply the inhomogeneous Dirichlet boundary conditions.
   if (!bc_.dirichlet_coefficients.empty())
   {
      LiftSolution(x);
   }

   sw.Stop();
   if (PrintOutput(fespace_ptr_, print_level_))
   {
      mfem::out << "<SPDESolver::Timing> all PCG solves " << sw.RealTime()
                << " [s]" << std::endl;
   }
}

void SPDESolver::SetupRandomFieldGenerator(int seed)
{
   delete b_wn;
   integ =
      new WhiteGaussianNoiseDomainLFIntegrator(fespace_ptr_->GetComm(), seed);
   b_wn = new ParLinearForm(fespace_ptr_);
   b_wn->AddDomainIntegrator(integ);
}

void SPDESolver::GenerateRandomField(ParGridFunction &x)
{
   if (!b_wn)
   {
      MFEM_ABORT("Need to call SPDESolver::SetupRandomFieldGenerator(...) first");
   }
   // Create stochastic load
   b_wn->Assemble();
   real_t normalization = ConstructNormalizationCoefficient(
                             nu_, l1_, l2_, l3_, fespace_ptr_->GetParMesh()->Dimension());
   (*b_wn) *= normalization;

   // Call back to solve to generate the random field
   Solve(*b_wn, x);
}

real_t SPDESolver::ConstructNormalizationCoefficient(real_t nu, real_t l1,
                                                     real_t l2, real_t l3,
                                                     int dim)
{
   // Computation considers squaring components, computing determinant, and
   // squaring
   real_t det = 0;
   if (dim == 1)
   {
      det = l1;
   }
   else if (dim == 2)
   {
      det = l1 * l2;
   }
   else if (dim == 3)
   {
      det = l1 * l2 * l3;
   }
   const real_t gamma1 = tgamma(nu + static_cast<real_t>(dim) / 2.0);
   const real_t gamma2 = tgamma(nu);
   return sqrt(pow(2 * M_PI, dim / 2.0) * det * gamma1 /
               (gamma2 * pow(nu, dim / 2.0)));
}

DenseMatrix SPDESolver::ConstructMatrixCoefficient(real_t l1, real_t l2,
                                                   real_t l3, real_t e1,
                                                   real_t e2, real_t e3,
                                                   real_t nu, int dim)
{
   if (dim == 3)
   {
      // Compute cosine and sine of the angles e1, e2, e3
      const real_t c1 = cos(e1);
      const real_t s1 = sin(e1);
      const real_t c2 = cos(e2);
      const real_t s2 = sin(e2);
      const real_t c3 = cos(e3);
      const real_t s3 = sin(e3);

      // Fill the rotation matrix R with the Euler angles.
      DenseMatrix R(3, 3);
      R(0, 0) = c1 * c3 - c2 * s1 * s3;
      R(0, 1) = -c1 * s3 - c2 * c3 * s1;
      R(0, 2) = s1 * s2;
      R(1, 0) = c3 * s1 + c1 * c2 * s3;
      R(1, 1) = c1 * c2 * c3 - s1 * s3;
      R(1, 2) = -c1 * s2;
      R(2, 0) = s2 * s3;
      R(2, 1) = c3 * s2;
      R(2, 2) = c2;

      // Multiply the rotation matrix R with the translation vector.
      Vector l(3);
      l(0) = std::pow(l1, 2);
      l(1) = std::pow(l2, 2);
      l(2) = std::pow(l3, 2);
      l *= (1 / (2.0 * nu));

      // Compute result = R^t diag(l) R
      DenseMatrix res(3, 3);
      R.Transpose();
      MultADBt(R, l, R, res);
      return res;
   }
   else if (dim == 2)
   {
      const real_t c1 = cos(e1);
      const real_t s1 = sin(e1);
      DenseMatrix Rt(2, 2);
      Rt(0, 0) =  c1;
      Rt(0, 1) =  s1;
      Rt(1, 0) = -s1;
      Rt(1, 1) =  c1;
      Vector l(2);
      l(0) = std::pow(l1, 2);
      l(1) = std::pow(l2, 2);
      l *= (1 / (2.0 * nu));
      DenseMatrix res(2, 2);
      MultADAt(Rt,l,res);
      return res;
   }
   else
   {
      DenseMatrix res(1, 1);
      res(0, 0) = std::pow(l1, 2) / (2.0 * nu);
      return res;
   }
}

void SPDESolver::Solve(const ParLinearForm &b, ParGridFunction &x, real_t alpha,
                       real_t beta, int exponent)
{
   // Form system of equations. This is less general than
   // BilinearForm::FormLinearSystem and kind of resembles the necessary subset
   // of instructions that we need in this case.
   if (prolongation_matrix_)
   {
      prolongation_matrix_->MultTranspose(b, B_);
   }
   else
   {
      B_ = b;
   }
   B_ *= beta;

   if (!apply_lift_)
   {
      // Initialize X_ to zero. Important! Might contain nan/inf -> crash.
      X_ = 0.0;
   }
   else
   {
      restriction_matrix_->Mult(x, X_);
   }

   HypreParMatrix *Op =
      Add(1.0, stiffness_, alpha, mass_bc_);  //  construct Operator
   HypreParMatrix *Ae = Op->EliminateRowsCols(ess_tdof_list_);
   Op->EliminateBC(*Ae, ess_tdof_list_, X_, B_);  // only for homogeneous BC

   for (int i = 0; i < exponent; i++)
   {
      // Solve the linear system Op X_ = B_
      SolveLinearSystem(Op);
      k_.RecoverFEMSolution(X_, b, x);
      if (repeated_solve_)
      {
         // Prepare for next iteration. X is a primal and B is a dual vector. B_
         // must be updated to represent X_ in the next step. Instead of copying
         // it, we must transform it appropriately.
         GridFunctionCoefficient gfc(&x);
         ParLinearForm previous_solution(fespace_ptr_);
         previous_solution.AddDomainIntegrator(new DomainLFIntegrator(gfc));
         previous_solution.Assemble();
         prolongation_matrix_->MultTranspose(previous_solution, B_);
         Op->EliminateBC(*Ae, ess_tdof_list_, X_, B_);
      }
   }
   delete Ae;
   delete Op;
}

void SPDESolver::LiftSolution(ParGridFunction &x)
{
   // Set lifting flag
   apply_lift_ = true;

   // Lifting of the solution takes care of inhomogeneous boundary conditions.
   // See doi:10.1016/j.jcp.2019.109009; section 2.6
   if (PrintOutput(fespace_ptr_, print_level_))
   {
      mfem::out << "\n<SPDESolver> Applying inhomogeneous DBC" << std::endl;
   }

   // Define temporary grid function for lifting.
   ParGridFunction helper_gf(fespace_ptr_);
   helper_gf = 0.0;

   // Project the boundary conditions onto the solution space.
   for (const auto &bc : bc_.dirichlet_coefficients)
   {
      Array<int> marker(fespace_ptr_->GetParMesh()->bdr_attributes.Max());
      marker = 0;
      marker[bc.first - 1] = 1;
      ConstantCoefficient cc(bc.second);
      helper_gf.ProjectBdrCoefficient(cc, marker);
   }

   // Create linear form for the right hand side.
   ParLinearForm b(fespace_ptr_);
   ConstantCoefficient zero(0.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(zero));
   b.Assemble();

   // Solve the PDE for the lifting.
   Solve(b, helper_gf, 1.0, 1.0);

   // Add the lifting to the solution.
   x += helper_gf;

   // Reset the lifting flag.
   apply_lift_ = false;
}

void SPDESolver::UpdateRHS(ParLinearForm &b) const
{
   if (!repeated_solve_)
   {
      // This function is only relevant for repeated solves.
      return;
   }
   if (restriction_matrix_)
   {
      // This effectively writes the solution of the previous iteration X_ to the
      // linear form b. Note that at the end of solve we update B_ = Mass * X_.
      restriction_matrix_->MultTranspose(B_, b);
   }
   else
   {
      b = B_;
   }
}

void SPDESolver::SolveLinearSystem(const HypreParMatrix *Op)
{
   HypreBoomerAMG prec(*Op);
   prec.SetPrintLevel(-1);
   CGSolver cg(fespace_ptr_->GetComm());
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetPreconditioner(prec);
   cg.SetOperator(*Op);
   cg.SetPrintLevel(std::max(0, print_level_ - 1));
   cg.Mult(B_, X_);
}

void SPDESolver::ComputeRationalCoefficients(real_t exponent)
{
   if (abs(exponent) > 1e-12)
   {
      if (PrintOutput(fespace_ptr_, print_level_))
      {
         mfem::out << "<SPDESolver> Approximating the fractional exponent "
                   << exponent << std::endl;
      }
      ComputePartialFractionApproximation(exponent, coeffs_, poles_);

      // If the example is build without LAPACK, the exponent
      // might be modified by the function call above.
      alpha_ = exponent + integer_order_of_exponent_;
   }
   else
   {
      integer_order_ = true;
      if (PrintOutput(fespace_ptr_, print_level_))
      {
         mfem::out << "<SPDESolver> Treating integer order PDE." << std::endl;
      }
   }
}

SPDESolver::~SPDESolver() { delete b_wn; }

}  // namespace spde
}  // namespace mfem
