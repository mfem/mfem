#include "topopt.hpp"
#include "helper.hpp"
#include <fstream>

namespace mfem
{
EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array<int> &ess_bdr_list):
   a(a), b(b), ess_bdr(1, ess_bdr_list.Size()), ess_tdof_list(0), symmetric(false)
{
   for (int i=0; i<ess_bdr_list.Size(); i++)
   {
      ess_bdr(0, i) = ess_bdr_list[i];
   }
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
   GetEssentialTrueDofs();
}

EllipticSolver::EllipticSolver(BilinearForm &a, LinearForm &b,
                               Array2D<int> &ess_bdr):
   a(a), b(b), ess_bdr(ess_bdr), ess_tdof_list(0), symmetric(false),
   iterative_mode(false)
{
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
   if (pfes) {parallel = true; comm = pfes->GetComm(); }
   else {parallel = false;}
#endif
   GetEssentialTrueDofs();
}

void EllipticSolver::GetEssentialTrueDofs()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(a.FESpace());
      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         pfes->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp, ess_bdr_list;
         ess_bdr_list.MakeRef(ess_bdr.GetRow(0),
                              ess_bdr.NumCols());
         pfes->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);

         for (int i=1; i<ess_bdr.NumRows(); i++)
         {
            ess_bdr_list.MakeRef(ess_bdr.GetRow(i),
                                 ess_bdr.NumCols());
            pfes->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i - 1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
   }
   else
   {

      if (ess_bdr.NumRows() == 1)
      {
         Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list);
      }
      else
      {
         Array<int> ess_tdof_list_comp, ess_bdr_list;
         ess_bdr_list.MakeRef(ess_bdr.GetRow(0),
                              ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, -1);
         ess_tdof_list.Append(ess_tdof_list_comp);

         for (int i=1; i<ess_bdr.NumRows(); i++)
         {
            ess_bdr_list.MakeRef(ess_bdr.GetRow(i),
                                 ess_bdr.NumCols());
            a.FESpace()->GetEssentialTrueDofs(
               ess_bdr_list, ess_tdof_list_comp, i - 1);
            ess_tdof_list.Append(ess_tdof_list_comp);
         }
      }
   }
#else
   if (ess_bdr.NumRows() == 1)
   {
      Array<int> ess_bdr_list(ess_bdr.GetRow(0), ess_bdr.NumCols());
      a.FESpace()->GetEssentialTrueDofs(
         ess_bdr_list, ess_tdof_list);
   }
   else
   {
      Array<int> ess_tdof_list_comp, ess_bdr_list;
      ess_bdr_list.MakeRef(ess_bdr.GetRow(0),
                           ess_bdr.NumCols());
      a.FESpace()->GetEssentialTrueDofs(
         ess_bdr_list, ess_tdof_list_comp, -1);
      ess_tdof_list.Append(ess_tdof_list_comp);

      for (int i=1; i<ess_bdr.NumRows(); i++)
      {
         ess_bdr_list.MakeRef(ess_bdr.GetRow(i),
                              ess_bdr.NumCols());
         a.FESpace()->GetEssentialTrueDofs(
            ess_bdr_list, ess_tdof_list_comp, i - 1);
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
   }
#endif
}

bool EllipticSolver::Solve(GridFunction &x, bool A_assembled,
                           bool b_Assembled)
{
   OperatorPtr A;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      A.Reset(new HypreParMatrix);
   }
#endif
   Vector B, X;
   if (!A_assembled) { a.Update(); a.Assemble(); }
   if (!b_Assembled) { b.Assemble(); }

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto M_ptr = new HypreBoomerAMG(static_cast<HypreParMatrix&>(*A));
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver(comm));
   }
   else
   {
      M.reset(new GSSmoother(static_cast<SparseMatrix&>(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother(static_cast<SparseMatrix&>(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->iterative_mode = iterative_mode;
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}

bool EllipticSolver::SolveTranspose(GridFunction &x, LinearForm &f,
                                    bool A_assembled, bool f_Assembled)
{
   OperatorPtr A;
   Vector B, X;

   if (!A_assembled) { a.Assemble(); }
   if (!f_Assembled) { f.Assemble(); }

   a.FormLinearSystem(ess_tdof_list, x, f, A, X, B, true);

#ifdef MFEM_USE_SUITESPARSE
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_CHOLMOD;
   umf_solver.SetOperator(*A);
   umf_solver.Mult(B, X);
   a.RecoverFEMSolution(X, *f, x);
   bool converged = true;
#else
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> M;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      M.reset(new HypreBoomerAMG);
      auto M_ptr = new HypreBoomerAMG;
      M_ptr->SetPrintLevel(0);

      M.reset(M_ptr);
      cg.reset(new CGSolver((dynamic_cast<ParFiniteElementSpace*>
                             (a.FESpace()))->GetComm()));
   }
   else
   {
      M.reset(new GSSmoother((SparseMatrix&)(*A)));
      cg.reset(new CGSolver);
   }
#else
   M.reset(new GSSmoother((SparseMatrix&)(*A)));
   cg.reset(new CGSolver);
#endif
   cg->SetRelTol(1e-14);
   cg->SetMaxIter(10000);
   cg->SetPrintLevel(0);
   cg->SetPreconditioner(*M);
   cg->SetOperator(*A);
   cg->iterative_mode = iterative_mode;
   cg->Mult(B, X);
   a.RecoverFEMSolution(X, f, x);
   bool converged = cg->GetConverged();
#endif

   return converged;
}



void IsoElasticityIntegrator::VectorGradToVoigt(DenseMatrix &vals,
                                                DenseMatrix &voigt)
{
   const int dim = vals.NumCols();
   const int dof = vals.NumRows();
   const int vdof = vals.NumRows()*dim;
   // Initialize voigt with 0s
   voigt.SetSize(vdof, dim*(dim+1)/2);
   voigt = 0.0;

   double *ptr_vals(vals.GetData()), *ptr_voigt(voigt.GetData());

   Vector src(ptr_vals, dof), trg(ptr_voigt, dof);
   // diagonal
   for (int i=0; i<dim; i++)
   {
      trg.SetData(ptr_voigt + vdof*i + dof*i);
      src.SetData(ptr_vals + dof*i);
      trg = src;
   }

   if (dim == 1) { return; }
   // upward
   int voigt_idx=dim - 1;
   for (int i=1; i<dim; i++)
   {
      voigt_idx++;
      // Right column
      trg.SetData(ptr_voigt + vdof*voigt_idx + dof*(dim - i - 1));
      src.SetData(ptr_vals + dof*(dim - 1));
      trg = src;
      // Bottom row
      trg.SetData(ptr_voigt + vdof*voigt_idx + dof*(dim - 1));
      src.SetData(ptr_vals + dof*(dim - i - 1));
      trg += src;
   }

   if (dim == 2) { return; } // no additional for 2D
   // remainder
   voigt_idx++;
   // ∂_y V_1
   trg.SetData(ptr_voigt + vdof*voigt_idx);
   src.SetData(ptr_vals + dof*1);
   trg = src;
   // ∂_x V_2
   trg.SetData(ptr_voigt + vdof*voigt_idx + dof);
   src.SetData(ptr_vals);
   trg += src;
}



void IsoStrainEnergyDensityCoefficient::VectorGradToVoigt(DenseMatrix &grad,
                                                          Vector &voigt)
{
   const int dim = grad.NumCols();
   MFEM_ASSERT(dim < 4 && dim > 0, "Dimension should be between 1 and 3.");
   // Initialize voigt with 0s
   voigt.SetSize(dim*(dim+1)/2);
   if (dim == 1)
   {
      voigt(0) = grad(0, 0); return;
   }
   else if (dim == 2)
   {
      voigt(0) = grad(0, 0); voigt(1) = grad(1, 1);
      voigt(2) = grad(0, 1) + grad(1, 0); return;
   }
   else
   {
      voigt(0) = grad(0, 0); voigt(1) = grad(1, 1); voigt(2) = grad(2, 2);
      voigt(3) = grad(2, 1) + grad(1, 2); voigt(4) = grad(2, 0) + grad(0, 2);
      voigt(5) = grad(0, 1) + grad(1, 0);
   }
}


void IsoElasticityIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Tr, DenseMatrix &elmat)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   int vdof = dim*dof;
   double w, nu_, E_;

   MFEM_ASSERT(dim == Tr.GetSpaceDim(), "");

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(dof, dim); C(dim*(dim+1)/2); CVt(dim*(dim+1)/2, vdof);
#else
   dshape.SetSize(dof, dim); C.SetSize(dim*(dim+1)/2);
   CVt.SetSize(dim*(dim+1)/2, vdof);
#endif

   elmat.SetSize(dof * dim);
   elmat = 0.0;
   C = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * Tr.OrderGrad(&el); // correct order?
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);
      VectorGradToVoigt(dshape, vshape);
      w = ip.weight * Tr.Weight();
      E_ = E->Eval(Tr, ip);
      nu_ = nu->Eval(Tr, ip);

      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<i; j++)
         {
            C(i, j) = nu_;
         }
         C(i, i) = 1.0 - nu_;
         for (int j=i + 1; j<dim; j++)
         {
            C(i, j) = nu_;
         }
      }
      for (int i=dim; i < dim*(dim+1)/2; i++)
      {
         C(i, i) = 0.5 * (1.0 - 2*nu_);
      }
      MultABt(C, vshape, CVt);
      AddMult_a(w * E_ / ((1.0 + nu_)*(1.0 - 2*nu_)), vshape, CVt, elmat);
   }
}


void VectorBdrMassIntegrator::AssembleFaceMatrix(const FiniteElement &el,
                                                 const FiniteElement &dummy,
                                                 FaceElementTransformations &Tr,
                                                 DenseMatrix &elmat)
{
   int dof = el.GetDof();
   Vector shape(dof);

   elmat.SetSize(dof*vdim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;    // <------ user control
      ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
   }

   DenseMatrix elmat_scalar(dof);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      double val = k.Eval(*Tr.Face, ip)*Tr.Face->Weight() * ip.weight;

      el.CalcShape(eip, shape);
      MultVVt(shape, elmat_scalar);
      elmat_scalar *= val;
      for (int row = 0; row < vdim; row++)
      {
         elmat.AddSubMatrix(dof*row, elmat_scalar);
      }
   }

}


void VectorBdrDirectionalMassIntegrator::AssembleFaceMatrix(
   const FiniteElement &el,
   const FiniteElement &dummy,
   FaceElementTransformations &Tr,
   DenseMatrix &elmat)
{
   int dof = el.GetDof();
   Vector shape(dof), d_val(vdim);

   elmat.SetSize(dof*vdim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;    // <------ user control
      ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
   }

   DenseMatrix elmat_scalar(dof);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      double val = k.Eval(Tr, ip)*Tr.Face->Weight() * ip.weight;
      d.Eval(d_val, Tr, ip);

      el.CalcShape(eip, shape);
      MultVVt(shape, elmat_scalar);
      elmat_scalar *= val;
      for (int row = 0; row < vdim; row++)
      {
         for (int col = 0; col < vdim; col++)
         {
            elmat.AddMatrix(d_val(row)*d_val(col), elmat_scalar, dof*row, dof*col);
         }
      }
   }

}

DesignDensity::DesignDensity(FiniteElementSpace &fes, DensityFilter &filter,
                             double target_volume_fraction,
                             double volume_tolerance)
   : filter(filter), target_volume_fraction(target_volume_fraction),
     vol_tol(volume_tolerance)
{
   x_gf.reset(MakeGridFunction(&fes));
   tmp_gf.reset(MakeGridFunction(&fes));
   {
      Mesh * mesh = fes.GetMesh();
      domain_volume = 0.0;
      for (int i=0; i<mesh->GetNE(); i++) {domain_volume += mesh->GetElementVolume(i); }
#ifdef MFEM_USE_MPI
      auto pmesh = dynamic_cast<ParMesh*>(mesh);
      if (pmesh)
      {
         MPI_Allreduce(MPI_IN_PLACE, &domain_volume, 1, MPI_DOUBLE, MPI_SUM,
                       pmesh->GetComm());
      }
#endif

   }
   *x_gf = target_volume_fraction;
   frho.reset(MakeGridFunction(&(filter.GetFESpace())));
   *frho = target_volume_fraction;
   target_volume = domain_volume * target_volume_fraction;
}

SIMPProjector::SIMPProjector(const double k, const double rho0):k(k), rho0(rho0)
{
   phys_density.reset(new MappedGridFunctionCoefficient(
   nullptr, [rho0, k](double x) {return simp(x, rho0, k);}));
   dphys_dfrho.reset(new MappedGridFunctionCoefficient(
   nullptr, [rho0, k](double x) {return der_simp(x, rho0, k);}));
}
Coefficient &SIMPProjector::GetPhysicalDensity(GridFunction &frho)
{
   phys_density->SetGridFunction(&frho);
   return *phys_density;
}
Coefficient &SIMPProjector::GetDerivative(GridFunction &frho)
{
   dphys_dfrho->SetGridFunction(&frho);
   return *dphys_dfrho;
}

ThresholdProjector::ThresholdProjector(const double beta,
                                       const double eta):beta(beta), eta(eta)
{
   const double c1 = std::tanh(beta*eta);
   const double c2 = std::tanh(beta*(1-eta));
   const double inv_denominator = 1.0 / (c1 + c2);
   phys_density.reset(new MappedGridFunctionCoefficient(
                         nullptr, [c1, c2, beta, eta](double x)
   {
      return (c1 + std::tanh(beta*(x - eta))) / (c1 + c2);
   }));
   dphys_dfrho.reset(new MappedGridFunctionCoefficient(
                        nullptr, [c1, c2, beta, eta](double x)
   {
      return beta*std::pow(1.0/std::cosh(beta*(x - eta)), 2.0) / (c1 + c2);
   }));
}
Coefficient &ThresholdProjector::GetPhysicalDensity(GridFunction &frho)
{
   phys_density->SetGridFunction(&frho);
   return *phys_density;
}
Coefficient &ThresholdProjector::GetDerivative(GridFunction &frho)
{
   dphys_dfrho->SetGridFunction(&frho);
   return *dphys_dfrho;
}
SigmoidDesignDensity::SigmoidDesignDensity(FiniteElementSpace &fes,
                                           DensityFilter &filter,
                                           double vol_frac):
   DesignDensity(fes, filter, vol_frac),
   zero_gf(MakeGridFunction(&fes))
{
   *x_gf = inv_sigmoid(vol_frac);
   rho_cf.reset(new MappedGridFunctionCoefficient(x_gf.get(), sigmoid));
   *zero_gf = 0.0;
}

void SigmoidDesignDensity::Project()
{
   ComputeVolume();
   if (current_volume > target_volume)
   {
      double inv_sig_vol_fraction = inv_sigmoid(target_volume_fraction);
      double c_l = inv_sig_vol_fraction - x_gf->Max();
      double c_r = inv_sig_vol_fraction - x_gf->Min();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      }
#endif
      double c = 0.5 * (c_l + c_r);
      double dc = 0.5 * (c_r - c_l);
      *x_gf += c;
      while (dc > 1e-09)
      {
         dc *= 0.5;
         ComputeVolume();
         if (std::fabs(current_volume - target_volume) < vol_tol) { break; }
         *x_gf += current_volume < target_volume ? dc : -dc;
      }
   }
}

double SigmoidDesignDensity::StationarityError(const GridFunction &grad,
                                               bool useL2norm)
{
   *tmp_gf = *x_gf;
   double volume_backup = current_volume;
   *x_gf -= grad;
   Project();
   double d;
   if (useL2norm)
   {
      std::unique_ptr<Coefficient> rho_diff = GetDensityDiffCoeff(*tmp_gf);
      d = zero_gf->ComputeL2Error(*rho_diff);
   }
   else
   {
      d = ComputeBregmanDivergence(*x_gf, *tmp_gf);
   }
   // Restore solution and recompute volume
   *x_gf = *tmp_gf;
   current_volume = volume_backup;
   return d;
}
double SigmoidDesignDensity::ComputeBregmanDivergence(GridFunction &p,
                                                      GridFunction &q, double epsilon)
{
   // Define safe x*log(x) to avoid log(0)
   const double log_eps = std::log(epsilon);
   auto safe_xlogy = [epsilon, log_eps](double x, double y) {return x < epsilon ? 0.0 : y < epsilon ? x*log_eps : x*std::log(y); };
   MappedPairGridFunctionCoeffitient Dh(&p, &q, [safe_xlogy](double x, double y)
   {
      const double p = sigmoid(x);
      const double q = sigmoid(y);
      return safe_xlogy(p, p) - safe_xlogy(p, q)
             + safe_xlogy(1 - p, 1 - p) - safe_xlogy(1 - p, 1 - q);
   });
   // Since Bregman divergence is always positive, ||Dh||_L¹=∫_Ω Dh.
   return std::sqrt(zero_gf->ComputeL1Error(Dh));
}

double SigmoidDesignDensity::StationarityErrorL2(GridFunction &grad)
{
   double c;
   MappedPairGridFunctionCoeffitient projected_rho(x_gf.get(),
                                                   &grad, [&c](double x, double y)
   {
      return std::min(1.0, std::max(0.0, sigmoid(x) - y + c));
   });

   double c_l = target_volume_fraction - (1.0 - grad.Min());
   double c_r = target_volume_fraction - (0.0 - grad.Max());
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
   }
#endif
   while (c_r - c_l > 1e-09)
   {
      c = 0.5 * (c_l + c_r);
      double vol = zero_gf->ComputeL1Error(projected_rho);
      if (std::fabs(vol - target_volume) < vol_tol) { break; }

      if (vol > target_volume) { c_r = c; }
      else { c_l = c; }
   }

   SumCoefficient diff_rho(projected_rho, *rho_cf, 1.0, -1.0);
   return zero_gf->ComputeL2Error(diff_rho);
}


ExponentialDesignDensity::ExponentialDesignDensity(FiniteElementSpace &fes,
                                                   DensityFilter &filter,
                                                   double vol_frac):
   DesignDensity(fes, filter, vol_frac),
   zero_gf(MakeGridFunction(&fes))
{
   *x_gf = std::log(vol_frac);
   rho_cf.reset(new MappedGridFunctionCoefficient(x_gf.get(), [](double x) {return std::exp(x);}));
   *zero_gf = 0.0;
}

void ExponentialDesignDensity::Project()
{
   ComputeVolume();
   if (current_volume > target_volume)
   {
      double log_vol_fraction = std::log(target_volume_fraction);
      double c_l = log_vol_fraction - x_gf->Max();
      double c_r = log_vol_fraction - x_gf->Min();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      }
#endif
      MappedGridFunctionCoefficient projected_rho(x_gf.get(), [](double x) {return std::min(1.0, std::exp(x));});
      double c = 0.5 * (c_l + c_r);
      double dc = 0.5 * (c_r - c_l);
      *x_gf += c;
      while (dc > 1e-09)
      {
         dc *= 0.5;
         current_volume = zero_gf->ComputeL1Error(projected_rho);
         if (std::fabs(current_volume - target_volume) < vol_tol) { break; }
         *x_gf += current_volume < target_volume ? dc : -dc;
      }
      x_gf->Clip(-infinity(), 0.0);
   }
}

double ExponentialDesignDensity::StationarityError(const GridFunction &grad,
                                                   bool useL2norm)
{
   *tmp_gf = *x_gf;
   double volume_backup = current_volume;
   *x_gf -= grad;
   Project();
   double d;
   if (useL2norm)
   {
      std::unique_ptr<Coefficient> rho_diff = GetDensityDiffCoeff(*tmp_gf);
      d = zero_gf->ComputeL2Error(*rho_diff);
   }
   else
   {
      d = ComputeBregmanDivergence(*x_gf, *tmp_gf);
   }
   // Restore solution and recompute volume
   *x_gf = *tmp_gf;
   current_volume = volume_backup;
   return d;
}
double ExponentialDesignDensity::ComputeBregmanDivergence(GridFunction &p,
                                                          GridFunction &q, double epsilon)
{
   // Define safe x*log(x) to avoid log(0)
   const double log_eps = std::log(epsilon);
   auto safe_xlogy = [epsilon, log_eps](double x, double y) {return x < epsilon ? 0.0 : y < epsilon ? x*log_eps : x*std::log(y); };
   MappedPairGridFunctionCoeffitient Dh(&p, &q, [safe_xlogy](double x, double y)
   {
      const double p = std::exp(x);
      const double q = std::exp(y);
      return safe_xlogy(p, p) - safe_xlogy(p, q);
   });
   // Since Bregman divergence is always positive, ||Dh||_L¹=∫_Ω Dh.
   return std::sqrt(zero_gf->ComputeL1Error(Dh));
}

double ExponentialDesignDensity::StationarityErrorL2(GridFunction &grad)
{
   double c;
   MappedPairGridFunctionCoeffitient projected_rho(x_gf.get(),
                                                   &grad, [&c](double x, double y)
   {
      return std::min(1.0, std::max(0.0, std::exp(x) - y + c));
   });

   double c_l = target_volume_fraction - (1.0 - grad.Min());
   double c_r = target_volume_fraction - (0.0 - grad.Max());
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
   }
#endif
   while (c_r - c_l > 1e-09)
   {
      c = 0.5 * (c_l + c_r);
      double vol = zero_gf->ComputeL1Error(projected_rho);
      if (std::fabs(vol - target_volume) < vol_tol) { break; }

      if (vol > target_volume) { c_r = c; }
      else { c_l = c; }
   }

   SumCoefficient diff_rho(projected_rho, *rho_cf, 1.0, -1.0);
   return zero_gf->ComputeL2Error(diff_rho);
}

LatentDesignDensity::LatentDesignDensity(FiniteElementSpace &fes,
                                         DensityFilter &filter, double vol_frac,
                                         std::function<double(double)> h,
                                         std::function<double(double)> primal2dual,
                                         std::function<double(double)> dual2primal,
                                         bool clip_lower, bool clip_upper):
   DesignDensity(fes, filter, vol_frac),
   h(h), p2d(primal2dual), d2p(dual2primal),
   clip_lower(clip_lower), clip_upper(clip_upper),
   zero_gf(MakeGridFunction(&fes))
{
   *x_gf = p2d(vol_frac);
   rho_cf.reset(new MappedGridFunctionCoefficient(x_gf.get(), d2p));
   *zero_gf = 0.0;
}

void LatentDesignDensity::Project()
{
   ComputeVolume();
   if (current_volume > target_volume)
   {
      double latent_vol_fraction = p2d(target_volume_fraction);
      double c_l = latent_vol_fraction - x_gf->Max();
      double c_r = latent_vol_fraction - x_gf->Min();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      }
#endif
      MappedGridFunctionCoefficient projected_rho(x_gf.get(), [](double x) {return x;});
      if (clip_lower && clip_upper)
      {
         projected_rho.SetFunction([this](double x) {return std::min(1.0, std::max(0.0, d2p(x)));});
      }
      else if (clip_lower)
      {
         projected_rho.SetFunction([this](double x) {return std::max(0.0, d2p(x));});
      }
      else if (clip_upper)
      {
         projected_rho.SetFunction([this](double x) {return std::min(1.0, d2p(x));});
      }
      else
      {
         projected_rho.SetFunction([this](double x) {return d2p(x);});
      }
      double c = 0.5 * (c_l + c_r);
      double dc = 0.5 * (c_r - c_l);
      *x_gf += c;
      while (dc > 1e-09)
      {
         dc *= 0.5;
         current_volume = zero_gf->ComputeL1Error(projected_rho);
         if (std::fabs(current_volume - target_volume) < vol_tol) { break; }
         *x_gf += current_volume < target_volume ? dc : -dc;
      }
      if (clip_lower || clip_upper)
      {
         x_gf->Clip(clip_lower ? p2d(0.0) : -infinity(),
                    clip_upper ? p2d(1.0) : infinity());
      }
   }
}

double LatentDesignDensity::StationarityError(const GridFunction &grad,
                                              bool useL2norm)
{
   *tmp_gf = *x_gf;
   double volume_backup = current_volume;
   *x_gf -= grad;
   Project();
   double d;
   if (useL2norm)
   {
      std::unique_ptr<Coefficient> rho_diff = GetDensityDiffCoeff(*tmp_gf);
      d = zero_gf->ComputeL2Error(*rho_diff);
   }
   else
   {
      d = ComputeBregmanDivergence(*x_gf, *tmp_gf);
   }
   // Restore solution and recompute volume
   *x_gf = *tmp_gf;
   current_volume = volume_backup;
   return d;
}
double LatentDesignDensity::ComputeBregmanDivergence(GridFunction &p,
                                                     GridFunction &q)
{
   MappedPairGridFunctionCoeffitient Dh(&p, &q, [this](double x, double y)
   {
      double p = d2p(x); double q = d2p(y);
      return h(p) - h(q) - y*(p-q);
   });
   // Since Bregman divergence is always positive, ||Dh||_L¹=∫_Ω Dh.
   return std::sqrt(zero_gf->ComputeL1Error(Dh));
}

double LatentDesignDensity::StationarityErrorL2(GridFunction &grad)
{
   double c;
   MappedPairGridFunctionCoeffitient projected_rho(x_gf.get(),
                                                   &grad, [&c, this](double x, double y)
   {
      return std::min(1.0, std::max(0.0, d2p(x) - y + c));
   });

   double c_l = target_volume_fraction - (1.0 - grad.Min());
   double c_r = target_volume_fraction - (0.0 - grad.Max());
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
   if (pfes)
   {
      MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
      MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
   }
#endif
   while (c_r - c_l > 1e-09)
   {
      c = 0.5 * (c_l + c_r);
      double vol = zero_gf->ComputeL1Error(projected_rho);
      if (std::fabs(vol - target_volume) < vol_tol) { break; }

      if (vol > target_volume) { c_r = c; }
      else { c_l = c; }
   }

   SumCoefficient diff_rho(projected_rho, *rho_cf, 1.0, -1.0);
   return zero_gf->ComputeL2Error(diff_rho);
}
PrimalDesignDensity::PrimalDesignDensity(FiniteElementSpace &fes,
                                         DensityFilter& filter,
                                         double vol_frac):
   DesignDensity(fes, filter, vol_frac),
   zero_gf(MakeGridFunction(&fes))
{
   rho_cf.reset(new GridFunctionCoefficient(x_gf.get()));
   *zero_gf = 0.0;
}

void PrimalDesignDensity::Project()
{
   ComputeVolume();
   if (current_volume > target_volume)
   {
      double c_l = target_volume_fraction - x_gf->Max();
      double c_r = target_volume_fraction - x_gf->Min();
#ifdef MFEM_USE_MPI
      auto pfes = dynamic_cast<ParFiniteElementSpace*>(x_gf->FESpace());
      if (pfes)
      {
         MPI_Allreduce(MPI_IN_PLACE, &c_l, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
         MPI_Allreduce(MPI_IN_PLACE, &c_r, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      }
#endif
      MappedGridFunctionCoefficient projected_rho(x_gf.get(), [](double x) {return std::min(1.0, std::max(0.0, x));});
      double c = 0.5 * (c_l + c_r);
      double dc = 0.5 * (c_r - c_l);
      *x_gf += c;
      while (dc > 1e-09)
      {
         dc *= 0.5;
         current_volume = zero_gf->ComputeL1Error(projected_rho);
         if (std::fabs(current_volume - target_volume) < vol_tol) { break; }
         *x_gf += current_volume < target_volume ? dc : -dc;
      }
      x_gf->ProjectCoefficient(projected_rho);
   }
}

double PrimalDesignDensity::StationarityError(const GridFunction &grad)
{
   // Back up current status
   *tmp_gf = *x_gf;
   double volume_backup = current_volume;

   // Project ρ + grad
   *x_gf -= grad;
   Project();

   // Compare the updated density and the original density
   double d = tmp_gf->ComputeL2Error(*rho_cf);

   // Restore solution and recompute volume
   *x_gf = *tmp_gf;
   current_volume = volume_backup;
   return d;
}
ParametrizedLinearEquation::ParametrizedLinearEquation(
   FiniteElementSpace &fes, GridFunction &filtered_density,
   DensityProjector &projector, Array2D<int> &ess_bdr):
   frho(filtered_density), projector(projector), AisStationary(false),
   BisStationary(false),
   ess_bdr(ess_bdr)
{
   a.reset(MakeBilinearForm(&fes));
   b.reset(MakeLinearForm(&fes));
}

void ParametrizedLinearEquation::SetBilinearFormStationary(bool isStationary)
{
   AisStationary = isStationary;
   if (isStationary) { a->Assemble(); }
}
void ParametrizedLinearEquation::SetLinearFormStationary(bool isStationary)
{
   BisStationary = isStationary;
   if (isStationary) { b->Assemble(); }
}
void ParametrizedLinearEquation::Solve(GridFunction &x)
{
   if (!AisStationary) { a->Update(); }
   if (!BisStationary) { b->Update(); }
   SolveSystem(x);
}
void ParametrizedLinearEquation::DualSolve(GridFunction &x, LinearForm &new_b)
{
   if (!AisStationary) { a->Update(); }
   // store current b temporarly, and assign the given linear form.
   LinearForm* b_tmp_storage = b.release();
   b.reset(&new_b);
   // Assemble and solve
   new_b.Assemble();
   SolveSystem(x);
   // Release and restore. We need to release before reset as new_b is not owned by this.
   (void) b.release();
   b.reset(b_tmp_storage);
}

TopOptProblem::TopOptProblem(LinearForm &objective,
                             ParametrizedLinearEquation &state_equation,
                             DesignDensity &density, bool solve_dual, bool apply_projection)
   :obj(objective), state_equation(state_equation), density(density),
    solve_dual(solve_dual), apply_projection(apply_projection)
{
   state.reset(MakeGridFunction(state_equation.FESpace()));
   *state = 0.0;
   if (!solve_dual)
   {
      dual_solution = state;
   }
   else
   {
      dual_solution.reset(MakeGridFunction(state_equation.FESpace()));
      *dual_solution = 0.0;
   }
   dEdfrho = state_equation.GetdEdfrho(*state, *dual_solution,
                                       density.GetFilteredDensity());
   gradF.reset(MakeGridFunction(density.FESpace()));
   *gradF = 0.0;
   if (density.FESpace() == density.FESpace_filter())
   {
      gradF_filter = gradF;
   }
   else
   {
      gradF_filter.reset(MakeGridFunction(density.FESpace_filter()));
      *gradF_filter = 0.0;

      L2projector = new L2ProjectionLFIntegrator(*gradF_filter);
      filter_to_density.reset(MakeLinearForm(density.FESpace()));
      filter_to_density->SetData(gradF->GetData());
      filter_to_density->AddDomainIntegrator(L2projector);
   }

#ifdef MFEM_USE_MPI
   auto pstate = dynamic_cast<ParGridFunction*>(state.get());
   if (pstate) { parallel = true; comm = pstate->ParFESpace()->GetComm(); }
   else { parallel = false;}
#endif
}

double TopOptProblem::Eval()
{
   if (apply_projection) { density.Project(); }
   density.UpdateFilteredDensity();
   state_equation.Solve(*state);
   val = obj(*state);
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, comm);
   }
#endif
   return val;
}

void TopOptProblem::UpdateGradient()
{
   if (solve_dual)
   {
      // state equation is assumed to be a symmetric operator
      state_equation.DualSolve(*dual_solution, obj);
   }
   density.GetFilter().Apply(*dEdfrho, *gradF_filter);
   if (gradF_filter != gradF)
   {
      filter_to_density->SetData(gradF->GetData());
      L2projector->SetGridFunction(*gradF_filter);
      filter_to_density->Assemble();
   }
}

double StrainEnergyDensityCoefficient::Eval(ElementTransformation &T,
                                            const IntegrationPoint &ip)
{
   double L = lambda.Eval(T, ip);
   double M = mu.Eval(T, ip);
   double density;
   if (&u2 == &u1)
   {
      u1.GetVectorGradient(T, grad1);
      double div_u = grad1.Trace();
      grad1.Symmetrize();
      density = L*div_u*div_u + 2*M*(grad1*grad1);
   }
   else
   {
      u1.GetVectorGradient(T, grad1);
      u2.GetVectorGradient(T, grad2);
      double div_u1 = grad1.Trace();
      double div_u2 = grad2.Trace();
      grad1.Symmetrize();

      density = L*div_u1*div_u2 + 2*M*(grad1*grad2);
   }
   return -dphys_dfrho.Eval(T, ip) * density;
}

double IsoStrainEnergyDensityCoefficient::Eval(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   double E_ = E.Eval(T, ip);
   double nu_ = nu.Eval(T, ip);
   double density = 0.0;
   const int dim = T.GetSpaceDim();
   C.SetSize(dim*(dim+1)/2);
   C = 0.0;

   for (int i=0; i<dim; i++)
   {
      for (int j=0; j<i; j++)
      {
         C(i, j) = nu_;
      }
      C(i, i) = 1.0 - nu_;
      for (int j=i + 1; j<dim; j++)
      {
         C(i, j) = nu_;
      }
   }
   for (int i=dim; i < dim*(dim+1)/2; i++)
   {
      C(i, i) = 0.5 * (1.0 - 2*nu_);
   }

   if (&u2 == &u1)
   {
      u1.GetVectorGradient(T, grad1);
      VectorGradToVoigt(grad1, voigt1);
      density = C.InnerProduct(voigt1, voigt1);
   }
   else
   {
      u1.GetVectorGradient(T, grad1);
      VectorGradToVoigt(grad1, voigt1);
      u2.GetVectorGradient(T, grad2);
      VectorGradToVoigt(grad2, voigt2);
      density = C.InnerProduct(voigt1, voigt2);
   }
   density *= E_ / ((1.0 + nu_)*(1.0 - 2*nu_));
   return -dphys_dfrho.Eval(T, ip) * density;
}

double ThermalEnergyDensityCoefficient::Eval(ElementTransformation &T,
                                             const IntegrationPoint &ip)
{
   double K = kappa.Eval(T, ip);
   double density;
   if (&u2 == &u1)
   {
      u1.GetGradient(T, grad1);
      density = K*(grad1*grad1);
   }
   else
   {
      u1.GetGradient(T, grad1);
      u2.GetGradient(T, grad2);

      density = K*(grad1*grad2);
   }
   return -dphys_dfrho.Eval(T, ip) * density;
}

ParametrizedElasticityEquation::ParametrizedElasticityEquation(
   FiniteElementSpace &fes, GridFunction &filtered_density,
   DensityProjector &projector,
   Coefficient &E, Coefficient &nu, VectorCoefficient &f,
   Array2D<int> &ess_bdr):
   ParametrizedLinearEquation(fes, filtered_density, projector, ess_bdr),
   E(E), nu(nu), filtered_density(filtered_density),
   phys_E(E, projector.GetPhysicalDensity(filtered_density)),
   f(f)
{
   a->AddDomainIntegrator(new IsoElasticityIntegrator(phys_E, nu));
   b->AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   SetLinearFormStationary();
}

ParametrizedDiffusionEquation::ParametrizedDiffusionEquation(
   FiniteElementSpace &fes,
   GridFunction &filtered_density,
   DensityProjector &projector,
   Coefficient &kappa,
   Coefficient &f, Array2D<int> &ess_bdr):
   ParametrizedLinearEquation(fes, filtered_density, projector, ess_bdr),
   kappa(kappa), filtered_density(filtered_density),
   phys_kappa(kappa, projector.GetPhysicalDensity(filtered_density)),
   f(f)
{
   a->AddDomainIntegrator(new DiffusionIntegrator(phys_kappa));
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   SetLinearFormStationary();
}

/// @brief Volumetric force for linear elasticity

VolumeForceCoefficient::VolumeForceCoefficient(double r_,Vector &  center_,
                                               Vector & force_) :
   VectorCoefficient(center_.Size()), r2(r_*r_), center(center_), force(force_) { }

void VolumeForceCoefficient::Eval(Vector &V, ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   Vector xx; xx.SetSize(T.GetDimension());
   T.Transform(ip,xx);
   double cr = xx.DistanceSquaredTo(center);
   V.SetSize(T.GetDimension());
   if (cr <= r2)
   {
      V = force;
   }
   else
   {
      V = 0.0;
   }
}

void VolumeForceCoefficient::Set(double r_,Vector & center_, Vector & force_)
{
   r2=r_*r_;
   center = center_;
   force = force_;
}
void VolumeForceCoefficient::UpdateSize()
{
   VectorCoefficient::vdim = center.Size();
}

/// @brief Volumetric force for linear elasticity
LineVolumeForceCoefficient::LineVolumeForceCoefficient(double r_,
                                                       Vector &center_, Vector & force_,
                                                       int direction_dim) :
   VectorCoefficient(center_.Size()), r2(r_*r_), center(center_), force(force_),
   direction_dim(direction_dim) { }

void LineVolumeForceCoefficient::Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   Vector xx; xx.SetSize(T.GetDimension());
   T.Transform(ip,xx);
   xx(direction_dim) = 0.0;
   center(direction_dim) = 0.0;
   double cr = xx.DistanceSquaredTo(center);
   V.SetSize(T.GetDimension());
   if (cr <= r2)
   {
      V = force;
   }
   else
   {
      V = 0.0;
   }
}

void LineVolumeForceCoefficient::Set(double r_,Vector & center_,
                                     Vector & force_)
{
   r2=r_*r_;
   center = center_;
   force = force_;
}
void LineVolumeForceCoefficient::UpdateSize()
{
   VectorCoefficient::vdim = center.Size();
}
int Step_Armijo(TopOptProblem &problem, const GridFunction &x0,
                const GridFunction &direction,
                LinearForm &diff_densityForm, const double c1,
                double &step_size, const int max_it, const double shrink_factor)
{
   // obtain current point and gradient
   GridFunction &x_gf = problem.GetGridFunction();
   GridFunction &grad = problem.GetGradient();
   const double val = problem.GetValue();
   int myrank = 0;
#ifdef MFEM_USE_MPI
   auto pgrad = dynamic_cast<ParGridFunction*>(&grad);
   MPI_Comm comm;
   if (pgrad) { comm = pgrad->ParFESpace()->GetComm(); }
   if (Mpi::IsInitialized()) { myrank = Mpi::WorldRank(); }
#endif

   double new_val, d;
   int i;
   step_size /= shrink_factor;
   for (i=0; i<max_it; i++)
   {
      if (myrank == 0) { out << i << std::flush << "\r"; }
      step_size *= shrink_factor; // reduce step size
      x_gf = x0; // restore original position
      x_gf.Add(-step_size, direction); // advance by updated step size
      new_val = problem.Eval(); // re-evaluate at the updated point
      diff_densityForm.Assemble(); // re-evaluate density difference inner-product
      d = (diff_densityForm)(grad);
#ifdef MFEM_USE_MPI
      if (pgrad)
      {
         MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_SUM, comm);
      }
#endif
      if (new_val < val + c1*d && d < 0) { break; }
   }

   return i;
}

HelmholtzFilter::HelmholtzFilter(FiniteElementSpace &fes,
                                 const double eps, Array<int> &ess_bdr):DensityFilter(fes),
   filter(MakeBilinearForm(&fes)), eps2(eps*eps), ess_bdr(ess_bdr)
{
   filter->AddDomainIntegrator(new DiffusionIntegrator(eps2));
   filter->AddDomainIntegrator(new MassIntegrator());
   filter->Assemble();
}
void HelmholtzFilter::Apply(Coefficient &rho, GridFunction &frho,
                            bool apply_bdr) const
{
   MFEM_ASSERT(frho.FESpace() != filter->FESpace(),
               "Filter is initialized with finite element space different from the given filtered density.");
   std::unique_ptr<LinearForm> rhoForm(MakeLinearForm(frho.FESpace()));
   rhoForm->AddDomainIntegrator(new DomainLFIntegrator(rho));
   Array<int> empty_bdr(ess_bdr); empty_bdr = 0;
   EllipticSolver solver(*filter, *rhoForm, apply_bdr ? ess_bdr : empty_bdr);
   solver.SetIterativeMode();
   bool converged = solver.Solve(frho, true, false);

   if (!converged)
   {
#ifdef MFEM_USE_MPI
      if (!Mpi::IsInitialized() || Mpi::Root())
      {
         out << "HelmholtzFilter::SolveSystem Failed to Converge." <<
             std::endl;
      }
#else
      out << "HelmholtzFilter::SolveSystem Failed to Converge." <<
          std::endl;
#endif
   }
}
}