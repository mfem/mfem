//                          MFEM Signorini Example Using Proximal Galerkin
//
// Compile with: make pg_signorini
//
// Sample runs: ./pg_signorini
//
// Serial Version
//

#include "mfem.hpp"
#include "pg_signorini.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   int ref_levels = 0;
   int max_iterations = 20;
   real_t alpha = 0.005;
   real_t tol = 1e-6;
   bool visualization = true;

   const real_t lambda = 1.0;
   const real_t mu = 1.0;
   Array<int> col_markers;
   
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--ref_levels",
                  "Number of uniform mesh refinements.");
   args.AddOption(&max_iterations, "-i", "--iterations",
                  "Maximum number of iterations.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha parameter for boundary condition.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Iteration tolerance.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.
   const char *mesh_file = "../data/wheel.msh";
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // 3. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 4. Interpolate the geometry after refinement to control geometry error.
   int curvature_order = max(order, 2);
   mesh.SetCurvature(curvature_order);

   // 5a. Mark the outer surface of the wheel as the contact boundary.
   Array<int> contact_attr(1);
   contact_attr[0] = 1;

   // 5b. Extract the submesh corresponding to the contact boundary.
   SubMesh contact_mesh(SubMesh::CreateFromBoundary(mesh, contact_attr));
   const int sub_dim = contact_mesh.Dimension();

   // 6. Define finite elements spaces for the solutions.
   H1_FECollection H1fec(order+1, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec, dim);

   L2_FECollection L2fec(order-1, sub_dim);
   FiniteElementSpace L2fes(&contact_mesh, &L2fec, 1);

   int num_dofs_H1 = H1fes.GetVSize();
   int num_dofs_L2 = L2fes.GetVSize();

   mfem::out << "----------------------------------------\n"
       << "Number of H1 finite element unknowns: " << num_dofs_H1 << "\n"
       << "Number of L2 finite element unknowns: " << num_dofs_L2 << "\n"
       << "----------------------------------------\n";

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = H1fes.GetVSize();
   offsets[2] = L2fes.GetVSize();
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   // 7. Determine the list of essential boundary dofs.
   Array<int> ess_bdr_x(mesh.bdr_attributes.Max());
   Array<int> ess_bdr_y(mesh.bdr_attributes.Max());

   ess_bdr_x = 0; ess_bdr_x[1] = 1; ess_bdr_x[2] = 1;
   ess_bdr_y = 0; ess_bdr_y[1] = 1; ess_bdr_y[2] = 1;

   Array<int> ess_tdof_list, tmp;
   H1fes.GetEssentialTrueDofs(ess_bdr_x, tmp, 0); ess_tdof_list.Append(tmp);
   H1fes.GetEssentialTrueDofs(ess_bdr_y, tmp, 1); ess_tdof_list.Append(tmp);

   col_markers.SetSize(H1fes.GetTrueVSize());
   col_markers = 0;
   for (int i=0; i<ess_tdof_list.Size(); i++)
   {
      col_markers[ess_tdof_list[i]] = 1;
   }

   // 8. Set up the coefficients.
   Vector f(dim);
   f = 0.0; f(dim-1) = -0.5;

   Vector n_tilde(dim);
   n_tilde = 0.0; n_tilde(dim-1) = -1.0;

   VectorConstantCoefficient f_coeff(f);
   VectorConstantCoefficient n_tilde_c(n_tilde);
   ConstantCoefficient zero(0.0);
   
   GridFunction u_gf, delta_psi_gf;
   u_gf.MakeRef(&H1fes,x,offsets[0]);
   delta_psi_gf.MakeRef(&L2fes,x,offsets[1]);
   delta_psi_gf = 0.0;
   
   GridFunction u_old_gf(&H1fes);
   GridFunction psi_old_gf(&L2fes);
   GridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;
   psi_old_gf = 0.0;
   
   // 9. Initialize the solutions.
   VectorFunctionCoefficient init_u_c(dim, InitialCondition);
   u_gf.ProjectCoefficient(init_u_c);
   u_old_gf = u_gf;

   // create trace grid function for computing initial psi
   H1_FECollection *trace_fec = new H1_FECollection(order+1, sub_dim);
   FiniteElementSpace *trace_fes = new FiniteElementSpace(&contact_mesh, trace_fec, dim);
   GridFunction u_trace_gf(trace_fes);
   contact_mesh.Transfer(u_gf, u_trace_gf);

   FunctionCoefficient gap_cf(GapFunction);
   LogarithmGridFunctionCoefficient psi_init_cf(u_trace_gf, gap_cf, n_tilde);
   psi_gf.ProjectCoefficient(psi_init_cf);
   psi_old_gf = psi_gf;

   delete trace_fec;
   delete trace_fes;

   // 10. Set up linear and bilinear forms.
   ParentToSubMixedBilinearForm a10(&H1fes, &L2fes);
   a10.AddBoundaryDomainIntegrator(new MixedFormIntegrator(n_tilde_c, dim));
   a10.Assemble();
   a10.Finalize();
   SparseMatrix &A10 = a10.SpMat();
   A10.EliminateCols(col_markers);

   SparseMatrix *A01 = Transpose(A10);
   (*A01) *= -1.0;

   BlockOperator A(offsets);
   A.SetBlock(1,0,&A10);
   A.SetBlock(0,1,A01);
   
   // 11. Set up GLVis visualization.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // Visualize the initial solution.
   if (visualization)
   {
      sol_sock << "solution\n" << mesh << u_gf << std::flush;
   }

   //  12. Iterate
   int newton_iterations = 10;
   real_t increment_u = 0.1;

   GMRESSolver gmres;
   gmres.SetRelTol(1e-12);
   gmres.SetKDim(500);
   gmres.SetPrintLevel(0);
   gmres.SetMaxIter(1000000);

   for (int k = 0; k <= max_iterations; k++) {
      GridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;
      
      alpha *= 2.0;
      ConstantCoefficient alpha_cf(alpha);
      ScalarVectorProductCoefficient alpha_f(alpha, f_coeff);

      mfem::out << "\nOuter iteration: " << k+1 
               << "\n----------------------------------------" << endl;

      for(int j = 0; j < newton_iterations; j++) {
         // Define coefficients
         ExponentialGridFunctionCoefficient exp_psi(psi_gf);
         FunctionCoefficient gap_cf(GapFunction);
         SumCoefficient gap_minus_exp_psi(gap_cf, exp_psi, 1.0, -1.0);
      
         LinearForm b0,b1;
         b0.Update(&H1fes,rhs.GetBlock(0),0);
         b1.Update(&L2fes,rhs.GetBlock(1),0);

         b0.AddDomainIntegrator(new VectorDomainLFIntegrator(alpha_f));
         b0.Assemble();
         A01->AddMult(psi_old_gf, b0, 1.0);
         A01->AddMult(psi_gf, b0, -1.0);
         
         b1.AddDomainIntegrator(new DomainLFIntegrator(gap_minus_exp_psi));
         b1.Assemble();

         BilinearForm a00(&H1fes);
         a00.AddDomainIntegrator(new ElasticityIntegrator(alpha_cf, lambda, mu));
         a00.Assemble();
         a00.Finalize();
         SparseMatrix &A00 = a00.SpMat();
         A00.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);

         BilinearForm a11(&L2fes);
         a11.AddDomainIntegrator(new MassIntegrator(exp_psi));
         a11.Assemble();
         a11.Finalize();
         SparseMatrix &A11 = a11.SpMat();

         A.SetBlock(0,0,&A00);
         A.SetBlock(1,1,&A11);

         // Construct the Schur complement preconditioner.
         // P =   [ diag(K)                  0          ]
         //       [  0           H + M diag(K)^(-1) M^T ]
         Vector A00_diag(A00.Height());
         A00.GetDiag(A00_diag);
         SparseMatrix KinvMt(*A01);
         for (int i=0; i<A00_diag.Size(); i++)
         {
            KinvMt.ScaleRow(i, 1.0/A00_diag(i));
         }
         SparseMatrix *MKinvMt = Mult(A10, KinvMt);
         SparseMatrix *S = Add(1.0, A11, 1.0, *MKinvMt);

         DSmoother P00(A00);
         GSSmoother P11(*S);
         P00.iterative_mode = false;
         P11.iterative_mode = false;

         BlockDiagonalPreconditioner prec(offsets);
         prec.SetDiagonalBlock(0, &P00);
         prec.SetDiagonalBlock(1, &P11);

         gmres.SetPreconditioner(prec);
         gmres.SetOperator(A);
         gmres.Mult(rhs, x);

         delete MKinvMt;
         delete S;

         u_gf.MakeRef(&H1fes, x.GetBlock(0), 0);
         delta_psi_gf.MakeRef(&L2fes, x.GetBlock(1), 0);

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         real_t gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         mfem::out << "\tNewton iteration: " << j+1;
         mfem::out << ", Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_u)
         {
            break;
         }

      }

      if (visualization)
      {
         sol_sock << "solution\n" << mesh << u_gf << std::flush;
      }
      
      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      // cin.get();

      if (increment_u < tol || k == max_iterations-1)
      {
         break;
      }

   }

   delete A01;
   
   return 0;
}

real_t LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip) 
{
   MFEM_ASSERT(u != NULL, "grid function is not set");
   const int dim = T.GetSpaceDim();

   Vector u_val(dim);
   u->GetVectorValue(T, ip, u_val);
   
   // Return ln(φ₁ - u · ñ)
   real_t val = log(gap->Eval(T, ip) - u_val * *n_tilde);
   return max(min_val, val);
}

real_t ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   // Return exp(ψ)
   return min(max_val, max(min_val, exp(psi->GetValue(T, ip))));
}
   
void InitialCondition(const Vector &x, Vector &u)
{
   const int dim = x.Size();
   const real_t disp = -0.1;

   u = 0.0;
   u(dim-1) = disp*x(dim-1);
}

real_t GapFunction(const Vector &x)
{
   const real_t plane_g = -0.1;

   const int dim = x.Size();
   return x(dim-1) - plane_g;
}
