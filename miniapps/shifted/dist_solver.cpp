// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "dist_solver.hpp"

namespace mfem
{

void DiffuseField(ParGridFunction &field, int smooth_steps)
{
   // Setup the Laplacian operator.
   ParBilinearForm *Lap = new ParBilinearForm(field.ParFESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();
   HypreParMatrix *A = Lap->ParallelAssemble();

   HypreSmoother *S = new HypreSmoother(*A,0,smooth_steps);
   S->iterative_mode = true;

   Vector tmp(A->Width());
   field.SetTrueVector();
   Vector fieldtrue = field.GetTrueVector();
   tmp = 0.0;
   S->Mult(tmp, fieldtrue);

   field.SetFromTrueDofs(fieldtrue);

   delete A;
   delete S;
   delete Lap;
}

double AvgElementSize(ParMesh &pmesh)
{
   // Compute average mesh size (assumes similar cells).
   double dx, loc_area = 0.0;
   for (int i = 0; i < pmesh.GetNE(); i++)
   {
      loc_area += pmesh.GetElementVolume(i);
   }
   double glob_area;
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE,
                 MPI_SUM, pmesh.GetComm());
   const int glob_zones = pmesh.GetGlobalNE();
   switch (pmesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         dx = glob_area / glob_zones; break;
      case Geometry::SQUARE:
         dx = sqrt(glob_area / glob_zones); break;
      case Geometry::TRIANGLE:
         dx = sqrt(2.0 * glob_area / glob_zones); break;
      case Geometry::CUBE:
         dx = pow(glob_area / glob_zones, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         dx = pow(6.0 * glob_area / glob_zones, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!"); dx = 0.0;
   }

   return dx;
}

ParMesh *MakeLORMesh(ParMesh &mesh_ho, int order, bool simplex)
{
   return new ParMesh(&mesh_ho, order, BasisType::GaussLobatto);
}

LOR::LOR(ParMesh &mesh_ho, int order, Coefficient &coeff, Array<int> &ess_dofs,
         bool simplex)
   : mesh(MakeLORMesh(mesh_ho, order, simplex)),
     fec(1, mesh->Dimension()),
     fes(mesh.get(), &fec),
     a(&fes),
     irs(0, Quadrature1D::GaussLobatto)
{
   DiffusionIntegrator *diff_integ = new DiffusionIntegrator(coeff);
   MassIntegrator *mass_integ = new MassIntegrator;

   // LOR system uses collocated (2-point Gauss-Lobatto) quadrature
   int dim = mesh->Dimension();
   Geometry::Type geom = mesh->GetElementBaseGeometry(0);
   const IntegrationRule &ir = irs.Get(geom, 1);
   if (geom == Geometry::SQUARE || geom == Geometry::CUBE)
   { MFEM_VERIFY(ir.Size() == pow(2,dim), "Wrong quadrature rule"); }
   diff_integ->SetIntegrationRule(ir);
   mass_integ->SetIntegrationRule(ir);

   a.SetAssemblyLevel(AssemblyLevel::LEGACYFULL);
   a.AddDomainIntegrator(diff_integ);
   a.AddDomainIntegrator(mass_integ);
   a.Assemble();
   a.FormSystemMatrix(ess_dofs, A);
}

AssemblyLevel GetAssemblyLevel(SolverConfig config)
{
   switch (config.type)
   {
      case SolverConfig::JACOBI:
      case SolverConfig::LOR_HYPRE:
      case SolverConfig::LOR_AMGX:
      case SolverConfig::ALG_CEED:
         return AssemblyLevel::PARTIAL;
      default:
         return AssemblyLevel::LEGACYFULL;
   }
}

bool NeedsLOR(SolverConfig config)
{
   switch (config.type)
   {
      case SolverConfig::LOR_HYPRE:
      case SolverConfig::LOR_AMGX:
         return true;
      default:
         return false;
   }
}

void DistanceSolver::ScalarDistToVector(ParGridFunction &dist_s,
                                        ParGridFunction &dist_v)
{
   ParFiniteElementSpace &pfes = *dist_s.ParFESpace();
   MFEM_VERIFY(pfes.GetOrdering()==Ordering::byNODES,
               "Only Ordering::byNODES is supported.");

   const int dim = pfes.GetMesh()->Dimension();
   const int size = dist_s.Size();

   dist_s.HostReadWrite();
   dist_v.HostReadWrite();
   ParGridFunction der(&pfes);
   Vector magn(size);
   magn = 0.0;
   for (int d = 0; d < dim; d++)
   {
      dist_s.GetDerivative(1, d, der);
      for (int i = 0; i < size; i++)
      {
         magn(i) += der(i) * der(i);
         // The vector must point towards the level zero set.
         dist_v(i + d*size) = (dist_s(i) > 0.0) ? -der(i) : der(i);
      }
   }

   const double eps = 1e-16;
   for (int i = 0; i < size; i++)
   {
      const double vec_magn = std::sqrt(magn(i) + eps);
      for (int d = 0; d < dim; d++)
      {
         dist_v(i + d*size) *= fabs(dist_s(i)) / vec_magn;
      }
   }
}

void DistanceSolver::ComputeVectorDistance(Coefficient &zero_level_set,
                                           ParGridFunction &distance)
{
   ParFiniteElementSpace &pfes = *distance.ParFESpace();
   MFEM_VERIFY(pfes.GetVDim() == pfes.GetMesh()->Dimension(),
               "This function expects a vector ParGridFunction!");

   ParFiniteElementSpace pfes_s(pfes.GetParMesh(), pfes.FEColl());
   ParGridFunction dist_s(&pfes_s);
   ComputeScalarDistance(zero_level_set, dist_s);
   ScalarDistToVector(dist_s, distance);
}

Solver * HeatDistanceSolver::ConfigurePreconditioner(ParBilinearForm &a,
                                                     Coefficient &coeff,
                                                     Array<int> &ess_tdof_list,
                                                     OperatorPtr * A)
{

   OperatorPtr A_prec;
   if (NeedsLOR(solverConfig))
   {
      ParMesh *mesh = a.ParFESpace()->GetParMesh();
      int order = a.ParFESpace()->GetOrder(0); // <-- Assume uniform p
      bool simplex = false; //we don't support simplex at this time.
      lor.reset(new LOR(*mesh, order, coeff, ess_tdof_list, simplex));
      A_prec = lor->A;
   }
   else
   {
      A_prec = *A;
   }

   switch (solverConfig.type)
   {

      case SolverConfig::JACOBI:
         return new OperatorJacobiSmoother(a, ess_tdof_list);
         break;
      case SolverConfig::FA_HYPRE:
      case SolverConfig::LOR_HYPRE:
      {
         const int amg_print_lvl = (print_level > 1) ? 1 : 0;
         HypreBoomerAMG *amg = new HypreBoomerAMG(*A_prec.As<HypreParMatrix>());
         amg->SetPrintLevel(amg_print_lvl);
         return amg;
      }
#ifdef MFEM_USE_AMGX
      case SolverConfig::FA_AMGX:
      case SolverConfig::LOR_AMGX:
      {
         AmgXSolver *amg = new AmgXSolver(MPI_COMM_WORLD,
                                          AmgXSolver::PRECONDITIONER,
                                          false);
         amg->SetOperator(*A_prec.As<HypreParMatrix>());
         return amg;
      }
#endif

#if defined MFEM_USE_CEED
      case SolverConfig::ALG_CEED:
      {
         return new ceed::AlgebraicSolver(a, ess_tdof_list);
      }
#endif

      default:
         MFEM_ABORT("Solver configuration not available");
   }

   return nullptr;
}

void HeatDistanceSolver::ComputeScalarDistance(Coefficient
                                               &zero_level_set,
                                               ParGridFunction &distance)
{

   ParFiniteElementSpace &pfes = *distance.ParFESpace();
   distance.Read();
   auto check_h1 = dynamic_cast<const H1_FECollection *>(pfes.FEColl());
   MFEM_VERIFY(check_h1 && pfes.GetVDim() == 1,
               "This solver supports only scalar H1 spaces.");

   // Compute average mesh size (assumes similar cells).
   ParMesh &pmesh = *pfes.GetParMesh();

   // Step 0 - transform the input level set into a source-type bump.
   ParGridFunction source(&pfes);
   source.ProjectCoefficient(zero_level_set);
   // Optional smoothing of the initial level set.
   if (smooth_steps > 0) { DiffuseField(source, smooth_steps); }
   // Transform so that the peak is at 0.
   // Assumes range [-1, 1].
   if (transform)
   {
      for (int i = 0; i < source.Size(); i++)
      {
         const double x = source(i);
         source(i) = ((x < -1.0) || (x > 1.0)) ? 0.0 : (1.0 - x) * (1.0 + x);
      }
   }

   const int cg_print_lvl  = (print_level > 0) ? 1 : 0;

   Solver *prec1 = nullptr, *prec2 = nullptr, *prec3 = nullptr;

   // Step 1 - diffuse.
   ParGridFunction diffused_source(&pfes);
   for (int i = 0; i < diffuse_iter; i++)
   {
      OperatorPtr A;
      Vector B, X;

      // Set up RHS.
      ParLinearForm b(&pfes);
      GridFunctionCoefficient src_coeff(&source);
      b.AddDomainIntegrator(new DomainLFIntegrator(src_coeff));
      b.Assemble();

      // Diffusion and mass terms in the LHS.
      ParBilinearForm a_d(&pfes);
      a_d.SetAssemblyLevel(GetAssemblyLevel(solverConfig));
      a_d.AddDomainIntegrator(new MassIntegrator);
      ConstantCoefficient t_coeff(parameter_t);
      a_d.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
      a_d.Assemble();

      // Solve with Dirichlet BC.
      Array<int> ess_tdof_list;
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      ParGridFunction u_dirichlet(&pfes);
      u_dirichlet = 0.0;
      a_d.FormLinearSystem(ess_tdof_list, u_dirichlet, b, A, X, B);
      if (prec1 == nullptr) { prec1 = ConfigurePreconditioner(a_d, t_coeff, ess_tdof_list, &A); }

      {
         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(100);
         cg.SetPrintLevel(cg_print_lvl);

         //If the preconditioner is set first
         //then A must be of the same type as prec
         cg.SetOperator(*A);
         cg.SetPreconditioner(*prec1);
         cg.Mult(B, X);
      }
      a_d.RecoverFEMSolution(X, b, u_dirichlet);

      // Diffusion and mass terms in the LHS.
      ParBilinearForm a_n(&pfes);
      a_n.SetAssemblyLevel(GetAssemblyLevel(solverConfig));
      a_n.AddDomainIntegrator(new MassIntegrator);
      a_n.AddDomainIntegrator(new DiffusionIntegrator(t_coeff));
      a_n.Assemble();

      // Solve with Neumann BC.
      ParGridFunction u_neumann(&pfes);
      ess_tdof_list.DeleteAll();
      a_n.FormLinearSystem(ess_tdof_list, u_neumann, b, A, X, B);

      if (prec2 == nullptr) { prec2 = ConfigurePreconditioner(a_n, t_coeff, ess_tdof_list, &A); }

      {
         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(2000);
         cg.SetPrintLevel(cg_print_lvl);

         //If the preconditioner is set first
         //then A must be of the same type as prec
         cg.SetOperator(*A);
         cg.SetPreconditioner(*prec2);
         cg.Mult(B, X);
      }
      a_n.RecoverFEMSolution(X, b, u_neumann);

      const double *h_u_neumann = u_neumann.HostRead();
      const double *h_u_dirichlet = u_dirichlet.HostRead();
      double *h_diffused_source = diffused_source.HostWrite();
      for (int i = 0; i < diffused_source.Size(); i++)
      {
         // This assumes that the magnitudes of the two solutions are somewhat
         // similar; otherwise one of the solutions would dominate and the BC
         // won't look correct. To avoid this, it's good to have the source
         // away from the boundary (i.e. have more resolution).
         h_diffused_source[i] = 0.5 * (h_u_neumann[i] + h_u_dirichlet[i]);
      }
      source = diffused_source;
   }

   // Step 2 - solve for the distance using the normalized gradient.
   {
      OperatorPtr A;
      Vector B, X;
      // RHS - normalized gradient.
      ParLinearForm b2(&pfes);
      NormalizedGradCoefficient grad_u(diffused_source, pmesh.Dimension());
      b2.AddDomainIntegrator(new DomainLFGradIntegrator(grad_u));
      b2.Assemble();

      // LHS - diffusion.
      ParBilinearForm a2(&pfes);
      a2.SetAssemblyLevel(GetAssemblyLevel(solverConfig));
      a2.AddDomainIntegrator(new DiffusionIntegrator);
      a2.Assemble();

      // No BC.
      Array<int> no_ess_tdofs;

      a2.FormLinearSystem(no_ess_tdofs, distance, b2, A, X, B);
      ConstantCoefficient t_coeff(1.0);
      if (prec3 == nullptr) { prec3 = ConfigurePreconditioner(a2, t_coeff, no_ess_tdofs, &A); }
      {
         //If the preconditioner is set first
         //then A must be of the same type as prec
         CGSolver cg(MPI_COMM_WORLD);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(2000);
         cg.SetPrintLevel(cg_print_lvl);

         cg.SetOperator(*A);
         cg.SetPreconditioner(*prec3);

         cg.Mult(B, X);
      }
      a2.RecoverFEMSolution(X, b2, distance);
   }

   delete prec1;
   delete prec2;
   delete prec3;

   // Shift the distance values to have minimum at zero.
   double d_min_loc = distance.Min();
   double d_min_glob;
   MPI_Allreduce(&d_min_loc, &d_min_glob, 1, MPI_DOUBLE,
                 MPI_MIN, pfes.GetComm());
   distance -= d_min_glob;

   if (vis_glvis)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      ParFiniteElementSpace fespace_vec(&pmesh, pfes.FEColl(),
                                        pmesh.Dimension());
      NormalizedGradCoefficient grad_u(diffused_source, pmesh.Dimension());
      ParGridFunction x(&fespace_vec);
      x.ProjectCoefficient(grad_u);

      socketstream sol_sock_x(vishost, visport);
      sol_sock_x << "parallel " << pfes.GetNRanks() << " "
                 << pfes.GetMyRank() << "\n";
      sol_sock_x.precision(8);
      sol_sock_x << "solution\n" << pmesh << x;
      sol_sock_x << "window_geometry " << 0 << " " << 0 << " "
                 << 500 << " " << 500 << "\n"
                 << "window_title '" << "Heat Directions" << "'\n"
                 << "keys evvRj*******A\n" << std::flush;
   }
}

void PLapDistanceSolver::ComputeScalarDistance(Coefficient &func,
                                               ParGridFunction &fdist)
{
   ParFiniteElementSpace* fesd=fdist.ParFESpace();

   auto check_h1 = dynamic_cast<const H1_FECollection *>(fesd->FEColl());
   auto check_l2 = dynamic_cast<const L2_FECollection *>(fesd->FEColl());
   MFEM_VERIFY((check_h1 || check_l2) && fesd->GetVDim() == 1,
               "This solver supports only scalar H1 or L2 spaces.");

   ParMesh* mesh=fesd->GetParMesh();
   const int dim=mesh->Dimension();

   MPI_Comm lcomm=fesd->GetComm();
   int myrank;
   MPI_Comm_rank(lcomm,&myrank);

   const int order = fesd->GetOrder(0);
   H1_FECollection fecp(order, dim);
   ParFiniteElementSpace fesp(mesh, &fecp, 1, Ordering::byVDIM);

   ParGridFunction wf(&fesp);
   wf.ProjectCoefficient(func);
   GradientGridFunctionCoefficient gf(&wf); //gradient of wf


   ParGridFunction xf(&fesp);
   HypreParVector *sv = xf.GetTrueDofs();
   *sv=1.0;

   ParNonlinearForm *nf = new ParNonlinearForm(&fesp);

   PUMPLaplacian* pint = new PUMPLaplacian(&func,&gf,false);
   nf->AddDomainIntegrator(pint);

   pint->SetPower(2);

   //define the solvers
   HypreBoomerAMG *prec = new HypreBoomerAMG();
   prec->SetPrintLevel((print_level > 1) ? 1 : 0);

   GMRESSolver *gmres;
   gmres = new GMRESSolver(lcomm);
   gmres->SetAbsTol(newton_abs_tol/10);
   gmres->SetRelTol(newton_rel_tol/10);
   gmres->SetMaxIter(100);
   gmres->SetPrintLevel((print_level > 1) ? 1 : 0);
   gmres->SetPreconditioner(*prec);

   NewtonSolver ns(lcomm);
   ns.iterative_mode = true;
   ns.SetSolver(*gmres);
   ns.SetOperator(*nf);
   ns.SetPrintLevel((print_level == 0) ? -1 : 0);
   ns.SetRelTol(newton_rel_tol);
   ns.SetAbsTol(newton_abs_tol);
   ns.SetMaxIter(newton_iter);

   Vector b; // RHS is zero
   ns.Mult(b, *sv);

   for (int pp=3; pp<maxp; pp++)
   {
      if (myrank == 0 && print_level > 0) { std::cout<<"pp="<<pp<<std::endl; }
      pint->SetPower(pp);
      ns.Mult(b, *sv);
   }

   xf.SetFromTrueDofs(*sv);
   GridFunctionCoefficient gfx(&xf);
   PProductCoefficient tsol(func,gfx);
   fdist.ProjectCoefficient(tsol);

   // (optional) Force positive distances everywhere.
   // for (int i = 0; i < fdist.Size(); i++)
   // {
   //    fdist(i) = fabs(fdist(i));
   // }

   delete gmres;
   delete prec;
   delete nf;
   delete sv;
}


double ScreenedPoisson::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &trans,
                                         const Vector &elfun)
{
   double energy = 0.0;
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   Vector shapef(ndof);
   double fval;
   double pval;
   DenseMatrix B(ndof, ndim);
   Vector qval(ndim);

   B=0.0;

   double w;
   double ngrad2;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);

      el.CalcPhysDShape(trans, B);
      el.CalcPhysShape(trans,shapef);

      B.MultTranspose(elfun,qval);

      ngrad2=0.0;
      for (int jj=0; jj<ndim; jj++)
      {
         ngrad2 = ngrad2 + qval(jj)*qval(jj);
      }

      energy = energy + w * ngrad2 * diffcoef * 0.5;

      // add the external load -1 if fval > 0.0; 1 if fval < 0.0;
      pval=shapef*elfun;

      energy = energy + w * pval * pval * 0.5;

      if (fval>0.0)
      {
         energy = energy - w*pval;
      }
      else  if (fval<0.0)
      {
         energy = energy + w*pval;
      }
   }

   return energy;
}

void ScreenedPoisson::AssembleElementVector(const FiniteElement &el,
                                            ElementTransformation &trans,
                                            const Vector &elfun,
                                            Vector &elvect)
{
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   elvect.SetSize(ndof);
   elvect=0.0;

   Vector shapef(ndof);
   double fval;
   double pval;

   DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]

   Vector qval(ndim); //[diff_x,diff_y,diff_z,u]
   Vector lvec(ndof); //residual at ip

   B=0.0;
   qval=0.0;

   double w;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);

      el.CalcPhysDShape(trans, B);
      el.CalcPhysShape(trans,shapef);

      B.MultTranspose(elfun,qval);
      B.Mult(qval,lvec);

      elvect.Add(w * diffcoef,lvec);

      pval=shapef*elfun;

      elvect.Add(w * pval, shapef);


      //add the load
      //add the external load -1 if fval > 0.0; 1 if fval < 0.0;
      pval=shapef*elfun;
      if (fval>0.0)
      {
         elvect.Add( -w , shapef);
      }
      else if (fval<0.0)
      {
         elvect.Add(  w , shapef);
      }
   }
}

void ScreenedPoisson::AssembleElementGrad(const FiniteElement &el,
                                          ElementTransformation &trans,
                                          const Vector &elfun,
                                          DenseMatrix &elmat)
{
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   elmat.SetSize(ndof,ndof);
   elmat=0.0;

   Vector shapef(ndof);

   DenseMatrix B(ndof, ndim); //[diff_x,diff_y,diff_z]
   B = 0.0;

   double w;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      el.CalcPhysDShape(trans, B);
      el.CalcPhysShape(trans,shapef);

      AddMult_a_VVt(w , shapef, elmat);
      AddMult_a_AAt(w * diffcoef, B, elmat);
   }
}


double PUMPLaplacian::GetElementEnergy(const FiniteElement &el,
                                       ElementTransformation &trans,
                                       const Vector &elfun)
{
   double energy = 0.0;
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el);
   ir = &IntRules.Get(el.GetGeomType(), order);

   Vector shapef(ndof);
   double fval;
   double pval;
   double tval;
   Vector vgrad(ndim);
   DenseMatrix dshape(ndof, ndim);
   DenseMatrix B(ndof, ndim);
   Vector qval(ndim);
   Vector tmpv(ndof);

   B=0.0;

   double w;
   double ngrad2;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);
      fgrad->Eval(vgrad,trans,ip);
      tval=fval;
      if (fval<0.0)
      {
         fval=-fval;
         vgrad*=-1.0;
      }

      el.CalcPhysDShape(trans, dshape);
      el.CalcPhysShape(trans,shapef);

      for (int jj=0; jj<ndim; jj++)
      {
         dshape.GetColumn(jj,tmpv);
         tmpv*=fval;
         tmpv.Add(vgrad[jj],shapef);
         B.SetCol(jj,tmpv);
      }
      B.MultTranspose(elfun,qval);

      ngrad2=0.0;
      for (int jj=0; jj<ndim; jj++)
      {
         ngrad2 = ngrad2 + qval(jj)*qval(jj);
      }

      energy = energy + w * std::pow(ngrad2+ee*ee,pp/2.0)/pp;

      // add the external load -1 if fval > 0.0; 1 if fval < 0.0;
      pval=shapef*elfun;
      if (tval>0.0)
      {
         energy = energy - w * pval * tval;
      }
      else  if (tval<0.0)
      {
         energy = energy + w * pval * tval;
      }
   }

   return energy;
}

void PUMPLaplacian::AssembleElementVector(const FiniteElement &el,
                                          ElementTransformation &trans,
                                          const Vector &elfun,
                                          Vector &elvect)
{
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el)+1;
   ir = &IntRules.Get(el.GetGeomType(), order);

   elvect.SetSize(ndof);
   elvect=0.0;

   Vector shapef(ndof);
   double fval;
   double tval;
   Vector vgrad(3);

   DenseMatrix dshape(ndof, ndim);
   DenseMatrix B(ndof, ndim); // [diff_x,diff_y,diff_z]

   Vector qval(ndim); // [diff_x,diff_y,diff_z,u]
   Vector lvec(ndof); // residual at ip
   Vector tmpv(ndof);

   B=0.0;
   qval=0.0;

   double w;
   double ngrad2;
   double aa;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);
      fgrad->Eval(vgrad,trans,ip);
      tval=fval;
      if (fval<0.0)
      {
         fval=-fval;
         vgrad*=-1.0;
      }

      el.CalcPhysDShape(trans, dshape);
      el.CalcPhysShape(trans,shapef);

      for (int jj=0; jj<ndim; jj++)
      {
         dshape.GetColumn(jj,tmpv);
         tmpv*=fval;
         tmpv.Add(vgrad[jj],shapef);
         B.SetCol(jj,tmpv);
      }

      B.MultTranspose(elfun,qval);

      ngrad2=0.0;
      for (int jj=0; jj<ndim; jj++)
      {
         ngrad2 = ngrad2 + qval(jj)*qval(jj);
      }

      aa = ngrad2 + ee*ee;
      aa = std::pow(aa, (pp - 2.0) / 2.0);
      B.Mult(qval,lvec);
      elvect.Add(w * aa,lvec);

      // add the load
      // add the external load -1 if tval > 0.0; 1 if tval < 0.0;
      if (tval>0.0)
      {
         elvect.Add( -w*fval , shapef);
      }
      else  if (tval<0.0)
      {
         elvect.Add(  w*fval , shapef);
      }
   }
}

void PUMPLaplacian::AssembleElementGrad(const FiniteElement &el,
                                        ElementTransformation &trans,
                                        const Vector &elfun,
                                        DenseMatrix &elmat)
{
   int ndof = el.GetDof();
   int ndim = el.GetDim();
   const IntegrationRule *ir = NULL;
   int order = 2 * el.GetOrder() + trans.OrderGrad(&el)+1;
   ir = &IntRules.Get(el.GetGeomType(), order);

   elmat.SetSize(ndof,ndof);
   elmat=0.0;

   Vector shapef(ndof);
   double fval;
   Vector vgrad(ndim);

   Vector qval(ndim); // [diff_x,diff_y,diff_z,u]
   DenseMatrix dshape(ndof, ndim);
   DenseMatrix B(ndof, ndim); // [diff_x,diff_y,diff_z]
   Vector lvec(ndof);
   Vector tmpv(ndof);

   B=0.0;

   double w;
   double ngrad2;
   double aa, aa0, aa1;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      w = trans.Weight();
      w = ip.weight * w;

      fval=func->Eval(trans,ip);
      fgrad->Eval(vgrad,trans,ip);
      if (fval<0.0)
      {
         fval=-fval;
         vgrad*=-1.0;
      }

      el.CalcPhysDShape(trans, dshape);
      el.CalcPhysShape(trans,shapef);

      for (int jj=0; jj<ndim; jj++)
      {
         dshape.GetColumn(jj,tmpv);
         tmpv*=fval;
         tmpv.Add(vgrad[jj],shapef);
         B.SetCol(jj,tmpv);
      }

      B.MultTranspose(elfun,qval);
      B.Mult(qval,lvec);

      ngrad2=0.0;
      for (int jj=0; jj<ndim; jj++)
      {
         ngrad2 = ngrad2 + qval(jj)*qval(jj);
      }

      aa = ngrad2 + ee * ee;
      aa1 = std::pow(aa, (pp - 2.0) / 2.0);
      aa0 = (pp-2.0) * std::pow(aa, (pp - 4.0) / 2.0);

      AddMult_a_VVt(w * aa0, lvec, elmat);
      AddMult_a_AAt(w * aa1, B, elmat);
   }
}


void PDEFilter::Filter(Coefficient &func, ParGridFunction &ffield)
{
   if (sint == nullptr)
   {
      sint = new ScreenedPoisson(func, rr);
      nf->AddDomainIntegrator(sint);
      *sv = 0.0;
      gmres->SetOperator(nf->GetGradient(*sv));
   }
   else { sint->SetInput(func); }

   // form RHS
   *sv = 0.0;
   Vector rhs(sv->Size());
   nf->Mult(*sv, rhs);
   // filter the input field
   gmres->Mult(rhs, *sv);

   gf.SetFromTrueDofs(*sv);
   gf.Neg();

   GridFunctionCoefficient gfc(&gf);
   ffield.ProjectCoefficient(gfc);
}

}
