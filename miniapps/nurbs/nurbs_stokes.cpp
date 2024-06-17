// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Stabilized  Navier-Stokes


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>

using namespace std;
using namespace mfem;

real_t kappa_param = 1.0;
real_t pi  = (real_t)(M_PI);

using VectorFun = std::function<void(const Vector & x, Vector & a)>;
using ScalarFun = std::function<real_t(const Vector & x)>;

void sol_fun(const Vector & x, Vector &sol)
{
   sol = 0.0;
   if ((x[1] - 0.99 > 0.0) &&
       (fabs(x[0] - 0.5) < 0.49) )
   {
      sol[0] = 1.0;
   }
}

real_t kappa_fun(const Vector & x)
{
   return kappa_param;
}

void force_fun(const Vector & x, Vector &f)
{
   f = 0.0;
}


class StokesIntegrator : public BlockNonlinearFormIntegrator
{
private:
   Coefficient *c_mu;
   VectorCoefficient *c_force;
   Vector u, f, grad_p;
   DenseMatrix flux;

   DenseMatrix elf_u, elv_u;
   DenseMatrix elf_p, elv_p;
   DenseMatrix sh_u;
   Vector sh_p;
   DenseMatrix shg_u, shh_u, shg_p, grad_u;



   int dim = -1;
   void SetDim(int dim_)
   {
      if (dim_ != dim)
      {
         dim = dim_;
         u.SetSize(dim);
         f.SetSize(dim);
         //res.SetSize(dim);
         //  up.SetSize(dim);
        // grad_u.SetSize(dim);
        // grad_p.SetSize(dim);
      }
   }

public:
   StokesIntegrator(Coefficient &mu_,
                    VectorCoefficient &force_)
      : c_mu(&mu_), c_force(&force_) { }

   virtual real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                                   ElementTransformation &Tr,
                                   const Array<const Vector *> &elfun)
   { return 0.0; }

   /// Perform the local action of the NonlinearFormIntegrator
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Tr,
                                      const Array<const Vector *> &elfun,
                                      const Array<Vector *> &elvec)
   {
      if (el.Size() != 2)
      {
         mfem_error("StokesIntegrator::AssembleElementVector"
                    " has finite element space of incorrect block number");
      }

      int dof_u = el[0]->GetDof();
      int dof_p = el[1]->GetDof();

      SetDim(el[0]->GetDim());
      int spaceDim = Tr.GetSpaceDim();
      if (dim != spaceDim)
      {
         mfem_error("StokesIntegrator::AssembleElementVector"
                    " is not defined on manifold meshes");
      }
      elvec[0]->SetSize(dof_u);
      elvec[1]->SetSize(dof_p);

      *elvec[0] = 0.0;
      *elvec[1] = 0.0;

      elf_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
      elv_u.UseExternalData(elvec[0]->GetData(), dof_u, dim);

      sh_u.SetSize(dof_u, dim);
     // shg_u.SetSize(dof_u, dim);
      sh_p.SetSize(dof_p);


      int intorder = 2*el[0]->GetOrder();
      const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         Tr.SetIntPoint(&ip);
         real_t w = ip.weight * Tr.Weight();
         real_t mu = c_mu->Eval(Tr, ip);
         c_force->Eval(f, Tr, ip);

         // Compute shape and interpolate
         el[0]->CalcPhysVShape(Tr, sh_u);
         sh_u.MultTranspose(*elfun[0], u);
        // MultAtB(elf_u, sh_u, u);
       // elf_u.MultTranspose(sh_u, u);

        // el[0]->CalcPhysDVShape(Tr, shg_u);
        // shg_u.Mult(u, ushg_u);
       //  MultAtB(elf_u, shg_u, grad_u);

         el[1]->CalcPhysShape(Tr, sh_p);
         real_t p = sh_p*(*elfun[1]);

        // el[1]->CalcPhysDShape(Tr, shg_p);
        // shg_p.MultTranspose(*elfun[1], grad_p);

         // Compute momentum weak residual
         flux.Diag(-p,dim);  // Add pressure  to flux
       //  grad_u.Symmetrize();                   // Grad to strain
       //  flux.Add(2*mu,grad_u);                 // Add stress to flux

        // AddMult_a_ABt(w, shg_u, flux, elv_u);  // Add flux term to rhs
       //  AddMult_a_VVt(w, sh_u, elv_u);  // Add mass term to rhs

        // AddMult_a_VWt(-w, sh_u, f,    elv_u);  // Add force term to rhs

         // Compute momentum weak residual
         elvec[1]->Add(w*grad_u.Trace(), sh_p); // Add Galerkin term
      }
   }

   /// Assemble the local gradient matrix
   virtual void AssembleElementGrad(const Array<const FiniteElement*> &el,
                                    ElementTransformation &Tr,
                                    const Array<const Vector *> &elfun,
                                    const Array2D<DenseMatrix *> &elmats)
   {
    /*  int dof_u = el[0]->GetDof();
      int dof_p = el[1]->GetDof();

      SetDim(el[0]->GetDim());

      elf_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);

      elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);
      elmats(0,1)->SetSize(dof_u*dim, dof_p);
      elmats(1,0)->SetSize(dof_p, dof_u*dim);
      elmats(1,1)->SetSize(dof_p, dof_p);

      *elmats(0,0) = 0.0;
      *elmats(0,1) = 0.0;
      *elmats(1,0) = 0.0;
      *elmats(1,1) = 0.0;

      sh_u.SetSize(dof_u);
      shg_u.SetSize(dof_u, dim);
      ushg_u.SetSize(dof_u);
      sh_p.SetSize(dof_p);
      shg_p.SetSize(dof_p, dim);

      int intorder = 2*el[0]->GetOrder();
      const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         Tr.SetIntPoint(&ip);
         real_t w = ip.weight * Tr.Weight();
         real_t mu = c_mu->Eval(Tr, ip);

         el[0]->CalcPhysShape(Tr, sh_u);
         elf_u.MultTranspose(sh_u, u);

         el[0]->CalcPhysDShape(Tr, shg_u);
         MultAtB(elf_u, shg_u, grad_u);

         shg_u.Mult(u, ushg_u);

         el[1]->CalcPhysShape(Tr, sh_p);
         real_t p = sh_p*(*elfun[1]);

         el[1]->CalcPhysDShape(Tr, shg_p);
         shg_p.MultTranspose(*elfun[1], grad_p);

         // u,u block
         for (int i_u = 0; i_u < dof_u; ++i_u)
         {
            for (int j_u = 0; j_u < dof_u; ++j_u)
            {
               // Diffusion
               real_t mat = 0.0;
               for (int dim_u = 0; dim_u < dim; ++dim_u)
               {
                  mat += shg_u(i_u,dim_u)*shg_u(j_u,dim_u);
               }
               mat *= mu;

               mat *= w;
               for (int dim_u = 0; dim_u < dim; ++dim_u)
               {
                  (*elmats(0,0))(i_u + dim_u*dof_u, j_u + dim_u*dof_u) += mat;
               }

               for (int i_dim = 0; i_dim < dim; ++i_dim)
               {
                  for (int j_dim = 0; j_dim < dim; ++j_dim)
                  {
                     (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                        mu*shg_u(i_u,j_dim)*shg_u(j_u,i_dim)*w;
                  }
               }
            }
         }

         // u,p and p,u blocks
         for (int i_p = 0; i_p < dof_p; ++i_p)
         {
            for (int j_u = 0; j_u < dof_u; ++j_u)
            {
               for (int dim_u = 0; dim_u < dim; ++dim_u)
               {
                  (*elmats(0,1))(j_u + dof_u * dim_u, i_p) -= shg_u(j_u,dim_u)*sh_p(i_p)*w;
                  (*elmats(1,0))(i_p, j_u + dof_u * dim_u) += shg_u(j_u,dim_u)*sh_p(i_p)*w;
               }
            }
         }

      }*/
   }

};


class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
   GeneralResidualMonitor(const std::string& prefix_, int print_lvl)
      : prefix(prefix_)
   {
      print_level = print_lvl;
      rank = 1;
   }

   GeneralResidualMonitor(MPI_Comm comm,
                          const std::string& prefix_, int print_lvl)
      : prefix(prefix_)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
#else
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         print_level = -1;
      }
#endif
   }

   virtual void MonitorResidual(int it, real_t norm, const Vector &r, bool final)
   {
      if (it == 0)
      {
         norm0 = norm;
      }

      if ((print_level > 0 &&  it%print_level == 0) || final)
      {
         mfem::out << prefix << " iteration " << std::setw(2) << it
                   << " : ||r|| = " << norm
                   << ",  ||r||/||r_0|| = " << 100*norm/norm0<<" % \n";
      }
   }

private:
   const std::string prefix;
   int rank, print_level;
   mutable real_t norm0;
};




class SystemResidualMonitor : public IterativeSolverMonitor
{
public:
   SystemResidualMonitor(const std::string& prefix_,
                         int print_lvl,
                         Array<int> &offsets,
                         DataCollection *dc_ = nullptr)
      : prefix(prefix_), bOffsets(offsets), dc(dc_)
   {
      print_level = print_lvl;
      nvar = bOffsets.Size()-1;
      norm0.SetSize(nvar);
      rank = 1;
   }

   SystemResidualMonitor(MPI_Comm comm,
                         const std::string& prefix_,
                         int print_lvl,
                         Array<int> &offsets)
      : prefix(prefix_), bOffsets(offsets), dc(nullptr), xp(nullptr)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
      rank = 1;
#else
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         print_level = -1;
      }
#endif
      nvar = bOffsets.Size()-1;
      norm0.SetSize(nvar);
   }
   SystemResidualMonitor(MPI_Comm comm,
                         const std::string& prefix_,
                         int print_lvl,
                         Array<int> &offsets,
                         DataCollection *dc_,
                         BlockVector *x,
                         Array<ParGridFunction *> pgf_)
      : prefix(prefix_), bOffsets(offsets), dc(dc_), xp(x), pgf(pgf_)
   {
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
      rank = 1;
#else
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
      {
         print_level = print_lvl;
      }
      else
      {
         print_level = -1;
      }
#endif
      nvar = bOffsets.Size()-1;
      norm0.SetSize(nvar);
   }


   virtual void MonitorResidual(int it, real_t norm, const Vector &r, bool final)
   {
      if (dc && (it > 0))
      {
         if (rank > 1)
         {
            for (int i = 0; i < nvar; ++i)
            {
               pgf[i]->Distribute(xp->GetBlock(i));
            }
         }
         dc->SetCycle(it);
         dc->Save();
      }

      Vector vnorm(nvar);

      for (int i = 0; i < nvar; ++i)
      {
         Vector r_i(r.GetData() + bOffsets[i], bOffsets[i+1] - bOffsets[i]);
         if ( rank == 1 )
         {
            vnorm[i] = r_i.Norml2();
         }
         else
         {
            vnorm[i] = sqrt(InnerProduct(MPI_COMM_WORLD, r_i, r_i));
         }
         if (it == 0) { norm0[i] = vnorm[i]; }
      }

      bool print = (print_level > 0 &&  it%print_level == 0) || final;
      if (print)
      {
         mfem::out << prefix << " iteration " << std::setw(3) << it <<"\n"
                   << " ||r||  \t"<< "||r||/||r_0||  \n";
         for (int i = 0; i < nvar; ++i)
         {
            mfem::out <<vnorm[i]<<"\t"<< 100*vnorm[i]/norm0[i]<<" % \n";
         }
      }
   }

private:
   const std::string prefix;
   int print_level, nvar, rank;
   mutable Vector norm0;
   // Offsets for extracting block vector segments
   Array<int> &bOffsets;
   DataCollection *dc;
   BlockVector *xp;
   Array<ParGridFunction *> pgf;
};




// Custom block preconditioner for the Jacobian
class JacobianPreconditioner : public
   BlockLowerTriangularPreconditioner //BlockDiagonalPreconditioner
{
protected:
   Array<Solver *> prec;
public:
   JacobianPreconditioner(Array<int> &offsets, Array<Solver *> p)
      : BlockLowerTriangularPreconditioner (offsets), prec(p)
   { MFEM_VERIFY(offsets.Size()-1 == p.Size(), ""); };

   virtual void SetOperator(const Operator &op)
   {
      BlockOperator *jacobian = (BlockOperator *) &op;

      for (int i = 0; i < prec.Size(); ++i)
      {
         prec[i]->SetOperator(jacobian->GetBlock(i,i));
         SetDiagonalBlock(i, prec[i]);
      }

      SetBlock(1,0, const_cast<Operator*>(&jacobian->GetBlock(1,0)));
   }

   virtual ~JacobianPreconditioner()
   {
      for (int i = 0; i < prec.Size(); ++i)
      {
         delete prec[i];
      }
   }
};




int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   const char *ref_file  = "";
   int problem = 0;
   int sstype = -2;
   bool static_cond = false;
   bool visualization = false;

   real_t penalty = -1;
   int order = 1;
   int ref_levels = 0;

   bool mono = true;

   OptionsParser args(argc, argv);

   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh.");
   args.AddOption(&kappa_param, "-k", "--kappa",
                  "Sets the diffusion parameters, should be positive.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);

   }
   args.PrintOptions(mfem::out);

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement and knot insertion of knots defined
   // in a refinement file. We choose 'ref_levels' to be the largest number
   // that gives a final mesh with no more than 50,000 elements.
   {
      // Mesh refinement as defined in refinement file
      if (mesh.NURBSext && (strlen(ref_file) != 0))
      {
         mesh.RefineNURBSFromFile(ref_file);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
      mesh.PrintInfo();
   }

   // Define a finite element space on the mesh. Here we use continuous
   // Lagrange finite elements of the specified order. If order < 1, we
   // instead use an isoparametric/isogeometric space.
   Array<FiniteElementCollection *> fecs(2);
   fecs[0] = new NURBS_HDivFECollection(order);
   fecs[1] = new NURBSFECollection(order);

   Array<FiniteElementSpace *> spaces(2);
   spaces[1] = new FiniteElementSpace(&mesh, mesh.NURBSext, fecs[1]);
   spaces[0] = new FiniteElementSpace(&mesh, spaces[1]->StealNURBSext(), fecs[0]);


   mfem::out << "Number of finite element unknowns:\n"
             << "\tVelocity = "<<spaces[0]->GetTrueVSize() << endl
             << "\tPressure = "<<spaces[1]->GetTrueVSize() << endl;
   // Determine the list of true (i.e. conforming) essential boundary dofs.
   // In this example, the boundary conditions are defined by marking all
   // the boundary attributes from the mesh as essential (Dirichlet) and
   // converting them to a list of true dofs.
   Array<Array<int> *> ess_bdr(2);
   Array<int> ess_tdof_list;

   Array<int> ess_bdr_u(spaces[0]->GetMesh()->bdr_attributes.Max());
   Array<int> ess_bdr_p(spaces[1]->GetMesh()->bdr_attributes.Max());

   ess_bdr_p = 0;
   ess_bdr_u = 1;

   ess_bdr[0] = &ess_bdr_u;
   ess_bdr[1] = &ess_bdr_p;

   // Set up the linear form b(.) which corresponds to the right-hand side of
   // the FEM linear system, which in this case is (1,phi_i) where phi_i are
   // the basis functions in the finite element fespace.

   // Define the solution vector xp as a finite element grid function
   Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = spaces[0]->GetTrueVSize();
   bOffsets[2] = spaces[1]->GetTrueVSize();
   bOffsets.PartialSum();

   BlockVector xp(bOffsets);

   GridFunction x_u(spaces[0]);
   GridFunction x_p(spaces[1]);

   x_u.MakeTRef(spaces[0], xp.GetBlock(0), 0);
   x_p.MakeTRef(spaces[1], xp.GetBlock(1), 0);

   VectorFunctionCoefficient sol(dim, sol_fun);

   x_u = 0.0;//.ProjectCoefficient(sol);
   x_p = 0.0;

   x_u.SetTrueVector();
   x_p.SetTrueVector();

   // Define the output
   //VisItDataCollection visit_dc("stokes", &mesh);
   //visit_dc.RegisterField("u", &x_u);
   //visit_dc.RegisterField("p", &x_p);
   //visit_dc.SetCycle(0);
   //visit_dc.Save();

   // Define the problem parameters
   FunctionCoefficient kappa(kappa_fun);
   VectorFunctionCoefficient force(dim, force_fun);

   // Define the block nonlinear form
   BlockNonlinearForm Hform(spaces);
   Hform.AddDomainIntegrator(new StokesIntegrator(kappa, force));
   Array<Vector *> rhs(2);
   rhs = nullptr; // Set all entries in the array
   Hform.SetEssentialBC(ess_bdr, rhs);

   // Set up the preconditioner
   JacobianPreconditioner jac_prec(bOffsets,
   Array<Solver *>({new GSSmoother(0,5),
            new GSSmoother(0,5)}));

   // Set up the Jacobian solver
   GeneralResidualMonitor j_monitor("\t\t\t\tFGMRES", 25);
   FGMRESSolver j_gmres;
   j_gmres.iterative_mode = false;
   j_gmres.SetRelTol(1e-2);
   j_gmres.SetAbsTol(1e-12);
   j_gmres.SetMaxIter(300);
   j_gmres.SetPrintLevel(-1);
   j_gmres.SetMonitor(j_monitor);
  // j_gmres.SetPreconditioner(jac_prec);

   // Set up the newton solver
   SystemResidualMonitor newton_monitor("Newton", 1, bOffsets, NULL);// &visit_dc);
   NewtonSolver newton_solver;
   newton_solver.iterative_mode = true;
   newton_solver.SetPrintLevel(-1);
   newton_solver.SetMonitor(newton_monitor);
   newton_solver.SetRelTol(1e-4);
   newton_solver.SetAbsTol(1e-8);
   newton_solver.SetMaxIter(25);
   newton_solver.SetSolver(j_gmres);
   newton_solver.SetOperator(Hform);

   // Solve the Newton system
   Vector zero;
   newton_solver.Mult(zero, xp);

   // Save data in the VisIt format
  // visit_dc.SetCycle(999999);
  // visit_dc.Save();

   // Free the used memory.
   for (int i = 0; i < fecs.Size(); ++i)
   {
      delete fecs[i];
   }
   for (int i = 0; i < spaces.Size(); ++i)
   {
      delete spaces[i];
   }

   return 0;
}

