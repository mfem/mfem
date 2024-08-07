
// Compile with: make sdf_diffusion
//
// Sample runs: sdf_diffusion -o 2
//              sdf_diffusion -o 1 -r 4
//
// Description: This code is based on Example 40 from MFEM but implements the method for computing the signed distance function of an implicit surface from a modified eikonal equation. Here, the formulation is taken with a sgn() and the full H1 norm as regularization of the divergence. 
// The problem being solved involves \Omega = [-2, 2]^2 and M being defined by the unit circle. 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ZCoefficient : public VectorCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   ZCoefficient(int vdim, GridFunction &psi_, real_t alpha_ = 1.0)
      : VectorCoefficient(vdim), psi(&psi_), alpha(alpha_) { }

   virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip);
};

class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   DZCoefficient(int height, GridFunction &psi_, real_t alpha_ = 1.0)
      : MatrixCoefficient(height, true),  psi(&psi_), alpha(alpha_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   // const char *mesh_file = "../data/star.mesh";

   // unit square, vertices (0, 0), (0, 1), (1, 1), (1, 0)
   const char *mesh_file = "../data/square-nurbs.mesh"; 

   int order = 1;
   int max_it = 5;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t tol = 1e-4;
   real_t growth_rate = 1.0;
   real_t newton_scaling = 0.9;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha, "-step", "--step",
                  "Step size alpha.");
   args.AddOption(&growth_rate, "-gr", "--growth-rate",
                  "Growth rate of the step size alpha");
   args.AddOption(&newton_scaling, "-ns", "--newton-scaling",
                  "Newton scaling");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   // if (!args.Good())
   // {
      // args.PrintUsage(cout);
      // return 1;
   // }
   // args.PrintOptions(cout);

   // 2. Read the mesh from the mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   MFEM_ASSERT(mesh.bdr_attributes.Size(),
      "This example does not currently support meshes"
      " without boundary attributes."
   )

   // 3. Postprocess the mesh.
   // 3A. Refine the mesh to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 3B. Interpolate the geometry after refinement to control geometry error.
   // NOTE: Minimum second-order interpolation is used to improve the accuracy.
   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

	// rescale domain to square with vertices 
	// (-2, -2), (-2, 2), (2, 2), (2, -2)
	GridFunction *nodes = mesh.GetNodes();
   real_t scale = 4.;
	*nodes *= scale; 
	real_t shift = 2.; 
	*nodes -= shift; 

   // 4. Define the necessary finite element spaces on the mesh.
   L2_FECollection L2fec(order, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec, sdim);

   H1_FECollection H1fec(order, dim);
   FiniteElementSpace H1fes(&mesh, &H1fec);

   // cout << "Number of L2 finite element unknowns: "
      //   << L2fes.GetTrueVSize() << endl;
   // cout << "Number of H1 finite element unknowns: "
      //   << H1fes.GetTrueVSize() << endl;

   // 5. Determine the list of true (i.e., conforming) essential boundary dofs.

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = L2fes.GetVSize();
   offsets[2] = H1fes.GetVSize();
   offsets[3] = 1; // extra offset for scalar \lambda
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0; 

   // 6. Define an initial guess for the solution.
   ConstantCoefficient one(1.0);
   ConstantCoefficient neg_one(-1.0);
   ConstantCoefficient zero(0.0);

   // 7. Define the solution vectors as a finite element grid functions
   //    corresponding to the fespaces.
   GridFunction u_gf, delta_psi_gf;

   delta_psi_gf.MakeRef(&L2fes,x,offsets[0]);
   u_gf.MakeRef(&H1fes,x,offsets[1]);
   delta_psi_gf = 0.0;

   GridFunction psi_old_gf(&L2fes);
   GridFunction psi_gf(&L2fes);
   GridFunction u_old_gf(&H1fes);
   u_old_gf = 0.0;

   GridFunction grad_u_gf(&L2fes);

   // 8. Define the function coefficients for the solution and use them to
   //    initialize the initial guess
   psi_gf = 0.0;
   psi_old_gf = psi_gf;
   u_old_gf = u_gf;
	
	auto sgn_varphi_is = [](const Vector &x)
	{
		auto x0 = x(0), x1 = x(1); 
		auto val = x0*x0+x1*x1-1.; 

		return (val < 0.) ? -1 : (val > 0.) ? 1. : 0.;  
	}; 
	FunctionCoefficient sgn_varphi(sgn_varphi_is); 

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock, gradu_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
      gradu_sock.open(vishost,visport);
      gradu_sock.precision(8);
   }

   // solve for eta from - \Delta \eta = 1 with homogeneous 0 Dirichlet BC

   GridFunction eta(&H1fes); 
   eta = 0.0; 

   Array<int> boundary_dofs;
   H1fes.GetBoundaryTrueDofs(boundary_dofs);

   LinearForm b_eta(&H1fes); 
   b_eta.AddDomainIntegrator(new DomainLFIntegrator(one)); 
   b_eta.Assemble(); 

   BilinearForm a_eta(&H1fes); 
   a_eta.AddDomainIntegrator(new DiffusionIntegrator); 
   a_eta.Assemble(); 

   SparseMatrix A_eta; 
   Vector B_eta, X_eta; 
   a_eta.FormLinearSystem(boundary_dofs, eta, b_eta, A_eta, X_eta, B_eta); 

   GSSmoother M_eta(A_eta); 
   PCG(A_eta, M_eta, B_eta, X_eta, 1, 200, 1e-12, 0.0); 
   
   a_eta.RecoverFEMSolution(X_eta, b_eta, eta); 

   GridFunctionCoefficient eta_gf(&eta); 

   LinearForm b0,b1;
   b0.MakeRef(&L2fes,rhs.GetBlock(0),0);
   b1.MakeRef(&H1fes,rhs.GetBlock(1),0);

   ConstantCoefficient alpha_cf(alpha);

   ZCoefficient Z(sdim, psi_gf, alpha);
   DZCoefficient DZ(sdim, psi_gf, alpha);

   ScalarVectorProductCoefficient neg_Z(-1.0, Z);
   b0.AddDomainIntegrator(new VectorDomainLFIntegrator(neg_Z));

   VectorGridFunctionCoefficient psi_cf(&psi_gf);
   VectorGridFunctionCoefficient psi_old_cf(&psi_old_gf);
   VectorSumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);
	ProductCoefficient neg_sgn_varphi(neg_one, sgn_varphi); 

   b1.AddDomainIntegrator(new DomainLFIntegrator(neg_sgn_varphi));

   b1.AddDomainIntegrator(new DomainLFGradIntegrator(psi_old_minus_psi));

   GradientGridFunctionCoefficient grad_u_old(&u_old_gf); 
   ScalarVectorProductCoefficient eta_grad_u_old(eta_gf, grad_u_old); 
   ScalarVectorProductCoefficient neg_eta_grad_u_old(-1.0 * 1 / alpha, eta_grad_u_old); 
   b1.AddDomainIntegrator(new DomainLFGradIntegrator(neg_eta_grad_u_old));

	GridFunctionCoefficient u_old_cf(&u_old_gf);
	ProductCoefficient neg_u_old_cf(neg_one, u_old_cf);

   BilinearForm a00(&L2fes);
   a00.AddDomainIntegrator(new VectorMassIntegrator(DZ));

   MixedBilinearForm a01(&H1fes,&L2fes);
   a01.AddDomainIntegrator(new GradientIntegrator(neg_one)); 
   a01.Assemble();
   a01.Finalize();

   SparseMatrix &A01 = a01.SpMat();
   SparseMatrix *A10 = Transpose(a01.SpMat());

   // make n x 1 column vector 
   int col_height = H1fes.GetVSize();
   Array<int> i_s(col_height+1); Array<int> j_s(H1fes.GetVSize());
   j_s = 0; i_s = 0; 
      
   for (int k=0; k<H1fes.GetVSize(); k++) { 
      i_s[k+1] = 1;
   }

   i_s.PartialSum();

   // mass term
   LinearForm col1s(&H1fes); 
   col1s.AddDomainIntegrator(new DomainLFIntegrator(one)); 
   col1s.Assemble(); 

   SparseMatrix col_sp(i_s.begin(), j_s.begin(), col1s.begin(), col_height, 1, false, false, true);
   SparseMatrix* row_sp(Transpose(col_sp)); 

   // 10. Iterate
   int k;
   int total_iterations = 0;
   real_t increment_u = 0.1;
   GridFunction u_tmp(&H1fes);
   for (k = 0; k < max_it; k++)
   {
      u_tmp = u_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 5; j++)
      {
         total_iterations++;

         b0.Assemble();   
         b1.Assemble();
         
         BilinearForm a11(&H1fes);
         ConstantCoefficient neg_one_alpha_recip(-1.0 * 1.0 / alpha); 
         ProductCoefficient neg_eta_gf(neg_one_alpha_recip, eta_gf); 
         a11.AddDomainIntegrator(new DiffusionIntegrator(neg_eta_gf)); 

         a11.Assemble(false);  // false
         a11.Finalize();
         SparseMatrix &A11 = a11.SpMat();

         a00.Assemble(false);
         a00.Finalize(false);
         SparseMatrix &A00 = a00.SpMat();

#ifndef MFEM_USE_SUITESPARSE
         BlockOperator A(offsets);

         // all the other entries default to 0! 
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,A10);

         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);

         A.SetBlock(1,2,&col_sp); 
         A.SetBlock(2,1,row_sp); 

         BlockDiagonalPreconditioner prec(offsets);

         prec.owns_blocks = 1;

         GMRES(A,prec,rhs,x,0,2000,500,1e-12,0.0);
#else
         BlockMatrix A(offsets);
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,A10);
         A.SetBlock(0,1,&A01);
         A.SetBlock(1,1,&A11);

         A.SetBlock(1,2,&col_sp); 
         A.SetBlock(2,1,row_sp); 

         SparseMatrix *A_mono = A.CreateMonolithic(); 
         UMFPackSolver umf(*A_mono); 
         umf.Mult(rhs, x);
#endif         

         u_tmp -= u_gf;
         real_t Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;

         psi_gf.Add(newton_scaling, delta_psi_gf); 
         a00.Update(); 

         if (visualization)
         {
            sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
            GradientGridFunctionCoefficient grad_u(&u_gf);
            grad_u_gf.ProjectDiscCoefficient(grad_u);
            gradu_sock << "solution\n" << mesh << grad_u_gf << "window_title 'Gradient magnitude'"
                     << flush;
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;
         }

         if (Newton_update_size < increment_u)
         {
            break;
         }
      }

      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_u << endl;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      if (increment_u < tol || k == max_it-1)
      {
         break;
      }

      alpha *= max(growth_rate, 1.0); 
   }

   // mfem::out << "\n Outer iterations: " << k+1
            //  << "\n Total iterations: " << total_iterations
            //  << "\n Total dofs:       " << L2fes.GetTrueVSize() + H1fes.GetTrueVSize()
            //  << endl;

   delete A10;

   return 0;
}

void ZCoefficient::Eval(Vector &V, ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not positive");

   Vector psi_vals(vdim);
   psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   V = psi_vals;
   V *= phi;
}

void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");
   MFEM_ASSERT(alpha > 0, "alpha is not positive");

   Vector psi_vals(height);
   psi->GetVectorValue(T, ip, psi_vals);
   real_t norm = psi_vals.Norml2();
   real_t phi = 1.0 / sqrt(1.0/alpha + norm*norm);

   K = 0.0;
   for (int i = 0; i < height; i++)
   {
    K(i,i) = phi;
    for (int j = 0; j < height; j++)
      {
        K(i,j) -= psi_vals(i) * psi_vals(j) * pow(phi, 3);
      }
   }
}
