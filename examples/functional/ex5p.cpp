//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "myad.hpp"
#include "elasticity.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   if (myid) { out.Disable(); }

   // 1. Parse command line options.
   int dim = 2;
   int order = 2;
   int ref_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&ref_levels, "-r", "--refine", "Mesh refinement levels");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh ser_mesh = dim == 2
                   ? Mesh::MakeCartesian2D(6, 2, Element::QUADRILATERAL, false, 3, 1)
                   : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   // Mesh ser_mesh("../../data/mobius-strip.mesh", 1, 1);
   for (int i=0; i<ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);
   ser_mesh.Clear();
   ParMesh displaced_mesh(mesh);

   Array<int> ess_bdr(4);
   ess_bdr = 0;
   ess_bdr[3] = 1; // Set left boundary as essential boundary condition

   // Setup problem
   const Vector E({0.0, 1.0, 3.0});
   const Vector nu({0.0, 0.3, 0.3});
   const Vector density({0.0, 1.0, 1.3});
   const Vector load_point({2.9, 0.5});
   const real_t load_radius = 0.05;
   const real_t target_mass_fraction = 0.3;
   VectorFunctionCoefficient load_cf(
      dim, [load_point, load_radius, dim](const Vector &x, Vector &f)
   {
      f = 0.0;
      f[dim-1] = x.DistanceTo(load_point) < load_radius ? -1.0 : 0.0;
   });
   Vector lambda(E.Size()+1), mu(E.Size()+1);
   for (int i=0; i<E.Size(); i++)
   {
      lambda[i] = (nu[i] * E[i] / ((1.0 + nu[i]) * (1.0 - 2.0 * nu[i])));
      mu[i] = (E[i] / (2.0 * (1.0 + nu[i])));
   }
   lambda[E.Size()] = 3;
   mu[E.Size()] = 3;


   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   L2_FECollection ctrl_fec(order-1, mesh.Dimension());
   ParFiniteElementSpace ctrl_fes(&mesh, &ctrl_fec, density.Size());
   ParFiniteElementSpace ctrl_fes_scalar(&mesh, &ctrl_fec);
   H1_FECollection filter_fec(order, mesh.Dimension());
   ParFiniteElementSpace filter_fes(&mesh, &ctrl_fec, density.Size());
   ParFiniteElementSpace filter_fes_scalar(&mesh, &ctrl_fec);
   real_t radius = 0.08;

   H1_FECollection state_fec(order, mesh.Dimension());
   ParFiniteElementSpace state_fes(&mesh, &state_fec, dim);
   ParGridFunction nodes(&state_fes), nodes_org(&state_fes);
   displaced_mesh.SetNodalFESpace(&state_fes);
   displaced_mesh.GetNodes(nodes);
   nodes_org = nodes;

   QuadratureSpace qspace(&mesh, order*4);
   QuadratureFunction rho_qf(qspace);
   rho_qf = 1.0;
   const real_t domain_mass = rho_qf.Integrate();
   const real_t target_mass = target_mass_fraction * domain_mass;
   ParGridFunction drho_gf(&ctrl_fes);
   VectorConstantCoefficient drho_cf(density);
   drho_gf.ProjectCoefficient(drho_cf);

   HYPRE_BigInt numElems = mesh.GetGlobalNE();
   HYPRE_BigInt numCtrlDofs = ctrl_fes.GlobalTrueVSize();
   HYPRE_BigInt numFltrDofs = filter_fes.GlobalTrueVSize();
   HYPRE_BigInt numDispDofs = state_fes.GlobalTrueVSize();
   HYPRE_BigInt numTotalDofs = numCtrlDofs + numDispDofs;

   out << "Number of mesh    elements: " << numElems << endl;
   out << "Number of control unknowns: " << numCtrlDofs << endl;
   out << "Number of filter  unknowns: " << numFltrDofs << endl;
   out << "Number of state   unknowns: " << numDispDofs << endl;
   out << "Number of total   unknowns: " << numTotalDofs << endl;


   ParGridFunction latent(&ctrl_fes);
   ParGridFunction latent_k(&ctrl_fes);
   ParGridFunction latent_diff(&ctrl_fes);
   ParGridFunction rho_gf(&ctrl_fes); // for visualization
   ParGridFunction frho(&filter_fes);
   ParGridFunction fgrad(&filter_fes);
   ParGridFunction grad(&ctrl_fes);
   ParGridFunction grad_diff(&ctrl_fes);
   ParGridFunction displacement(&state_fes);
   latent = log(density.Sum() / density.Size() * target_mass_fraction);
   displacement = 0.0;
   frho = 0.0;
   frho.GetTrueVector() = 0.0;
   fgrad = 0.0;
   fgrad.GetTrueVector() = 0.0;
   grad = 0.0;

   MappedVectorGridFunctionCoefficient mapped_rho(latent, exp_sum_to_one);
   MappedVectorGridFunctionCoefficient mapped_rho_k(latent_k, exp_sum_to_one);
   VectorSumCoefficient mapped_rho_diff_cf(mapped_rho, mapped_rho_k, 1.0, -1.0);
   VectorGridFunctionCoefficient grad_diff_cf(&grad_diff);
   VectorGridFunctionCoefficient latent_diff_cf(&latent_diff);
   VectorGridFunctionCoefficient grad_cf(&grad);
   InnerProductCoefficient diff_rho_diff_grad(mapped_rho_diff_cf, grad_diff_cf);
   InnerProductCoefficient diff_rho_diff_latent(mapped_rho_diff_cf,
         latent_diff_cf);
   InnerProductCoefficient diff_rho_grad(mapped_rho_diff_cf, grad_cf);
   QuadratureFunction qf(qspace);
   QuadratureFunction qf_vec(qspace, density.Size());
   VectorGridFunctionCoefficient fgrad_cf(&fgrad);

   SIMPFunction simp;
   ADFunctor simp_functor(simp);

   ADVecGridFuncCF lambda_cf(simp_functor);
   lambda_cf.SetParam(lambda);
   lambda_cf.SetGridFunction(frho);
   auto dlambda_deta_cf = lambda_cf.GetJacobian();

   ADVecGridFuncCF mu_cf(simp_functor);
   mu_cf.SetParam(mu);
   mu_cf.SetGridFunction(frho);
   auto dmu_deta_cf = mu_cf.GetJacobian();

   DensityDerivative dFdeta_cf(displacement, dlambda_deta_cf, dmu_deta_cf);
   ComplianceCoefficient F(displacement, lambda_cf, mu_cf);

   // rho = sum_i density_i*eta_i
   DensityFunction rho;
   SumToOnePGFunctor rho_functor(rho);
   ADVecGridFuncCF rho_cf(rho_functor);
   rho_cf.SetParam(density);
   rho_cf.SetGridFunction(latent);

   ParLinearForm load_lf(&state_fes);
   load_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));

   ParBilinearForm helmholtz(&filter_fes_scalar);
   helmholtz.AddDomainIntegrator(new MassIntegrator);
   ConstantCoefficient eps_cf(radius*radius);
   helmholtz.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
   helmholtz.Assemble();
   helmholtz.Finalize();
   std::unique_ptr<HypreParMatrix> helmholtz_matrix(helmholtz.ParallelAssemble());
   HyprePCG helmholtz_solver(comm);
   HypreBoomerAMG helmholtz_prec(*helmholtz_matrix);
   helmholtz_prec.SetPrintLevel(0);
   helmholtz_solver.SetAbsTol(1e-10);
   helmholtz_solver.SetMaxIter(1000);
   helmholtz_solver.SetPrintLevel(0);
   helmholtz_solver.iterative_mode = true;
   helmholtz_solver.SetPreconditioner(helmholtz_prec);
   helmholtz_solver.SetOperator(*helmholtz_matrix);
   ParLinearForm helmholtz_lf(&filter_fes);
   helmholtz_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(mapped_rho));
   std::unique_ptr<Vector> helmholtz_vec(filter_fes.NewTrueDofVector());
   Vector helmholtz_comp_view;
   Vector filter_comp_view;

   ParLinearForm derv_lf(&filter_fes);
   derv_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(dFdeta_cf));
   ParLinearForm grad_lf(&ctrl_fes, grad.GetData());
   grad_lf.AddDomainIntegrator(new DGRieszLFIntegrator(fgrad_cf));

   ElasticityStateSolver elasticity_solver(filter_fes, state_fes, lambda_cf,
                                           mu_cf);
   elasticity_solver.SetLoad(load_lf);
   elasticity_solver.MarkEssentialBC(ess_bdr, 0, -1);
   MassProjection(rho_cf, rho_qf, latent, drho_gf, target_mass);


   helmholtz_lf.Assemble();
   helmholtz_lf.ParallelAssemble(*helmholtz_vec);
   for (int j=0; j<filter_fes.GetVDim(); j++)
   {
      helmholtz_comp_view.SetDataAndSize(helmholtz_vec->GetData()
                                         +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
      filter_comp_view.SetDataAndSize(frho.GetTrueVector().GetData()
                                      +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
      helmholtz_solver.Mult(helmholtz_comp_view, filter_comp_view);
   }
   frho.SetFromTrueVector();
   elasticity_solver.Mult(frho, displacement);
   F.Project(qf);
   real_t obj_val = qf.Integrate();
   derv_lf.Assemble();
   derv_lf.ParallelAssemble(*helmholtz_vec);
   for (int j=0; j<filter_fes.GetVDim(); j++)
   {
      helmholtz_comp_view.SetDataAndSize(helmholtz_vec->GetData()
                                         +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
      filter_comp_view.SetDataAndSize(fgrad.GetTrueVector().GetData()
                                      +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
      helmholtz_solver.Mult(helmholtz_comp_view, filter_comp_view);
   }
   fgrad.SetFromTrueVector();
   grad_lf.Assemble();
   grad.SetTrueVector();
   SEQUENTIAL_PRINT(comm, std::cout,
                    derv_lf.Min() << ", " << derv_lf.Max() << "\n"
                    << fgrad.Min() << ", " << fgrad.Max() << "\n"
                    << grad.Min() << ", " << grad.Max() << "\n"
                   )
   real_t target_obj_val;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream rho_sock(vishost, visport);
   rho_sock.precision(8);
   add(nodes_org, 0.1, displacement, nodes);
   displaced_mesh.SetNodes(nodes);
   rho_sock << "parallel " << num_procs << " " << myid << "\n";
   rho_gf.ProjectCoefficient(rho_cf);
   rho_sock << "solution\n" << displaced_mesh << rho_gf << flush;

   real_t step_size = 0.01;
   for (int i=0; i<1e04; i++)
   {
      if (i > 0)
      {
         grad_diff -= grad;
         latent_diff -= latent;
         diff_rho_diff_latent.Project(qf);
         real_t stepsize_numer_val = qf.Integrate();
         diff_rho_diff_grad.Project(qf);
         real_t stepsize_denom_val = qf.Integrate();
         step_size = std::fabs(stepsize_numer_val / stepsize_denom_val);
         if (!std::isfinite(step_size) || step_size < 1e-10)
         { step_size = 0.01; }
      }
      step_size = std::min(step_size, 1e05);

      out << step_size << std::endl;

      grad_diff = grad;
      latent_diff = latent;
      latent_k = latent;
      bool passed = false;
      real_t old_obj_val = obj_val;
      int reeval = 0;
      while (!passed)
      {
         add(latent_k, -step_size, grad, latent);
         MassProjection(rho_cf, rho_qf, latent, drho_gf, target_mass);

         helmholtz_lf.Assemble();
         helmholtz_lf.ParallelAssemble(*helmholtz_vec);
         for (int j=0; j<filter_fes.GetVDim(); j++)
         {
            helmholtz_comp_view.SetDataAndSize(helmholtz_vec->GetData()
                                               +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
            filter_comp_view.SetDataAndSize(frho.GetTrueVector().GetData()
                                            +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
            helmholtz_solver.Mult(helmholtz_comp_view, filter_comp_view);
         }
         frho.SetFromTrueVector();
         elasticity_solver.Mult(frho, displacement);

         F.Project(qf);
         obj_val = qf.Integrate();
         diff_rho_grad.Project(qf);
         real_t delta = qf.Integrate();
         mapped_rho_diff_cf.Project(qf_vec);
         out << latent.Min() << ", " << latent.Max() << std::endl;
         out << qf.Min() << ", " << qf.Max() << std::endl;
         out << "Test " << reeval + 1 << " with alpha = " << step_size << ". ";
         out << "Current value: " << obj_val << ", target: " << old_obj_val <<
                " + 1e-04*" << delta << " = " << old_obj_val + 1e-04*delta << std::endl;
         if (obj_val < old_obj_val + 1e-04*delta)
         {
            break;
         }
         reeval++;
         step_size *= 0.5;
      }
      out << reeval << std::endl;

      // if (i % 10 == 0)
      // {
      add(nodes_org, 0.1, displacement, nodes);
      displaced_mesh.SetNodes(nodes);
      rho_sock << "parallel " << num_procs << " " << myid << "\n";
      rho_gf.ProjectCoefficient(rho_cf);
      rho_sock << "solution\n" << displaced_mesh << rho_gf << flush;
      // }
      // Evaluate gradient
      derv_lf.Assemble(); // unfiltered derivative
      derv_lf.ParallelAssemble(*helmholtz_vec);
      for (int j=0; j<filter_fes.GetVDim(); j++)
      {
         // Filter derivative to obtain gradient in filter space
         helmholtz_comp_view.SetDataAndSize(helmholtz_vec->GetData()
                                            +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
         filter_comp_view.SetDataAndSize(fgrad.GetTrueVector().GetData()
                                         +filter_fes_scalar.GetTrueVSize()*j, filter_fes_scalar.GetTrueVSize());
         helmholtz_solver.Mult(helmholtz_comp_view, filter_comp_view);
      }
      fgrad.SetFromTrueVector();
      // L2 projection
      grad_lf.Assemble();
   }
   return 0;
}
