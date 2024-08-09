
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

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t)> fun; // f:R → R
public:
   MappedGridFunctionCoefficient()
      :GridFunctionCoefficient(),
       fun([](real_t x) {return x;}) {}
   MappedGridFunctionCoefficient(const GridFunction *gf,
                                 std::function<real_t(const real_t)> fun_,
                                 int comp=1)
      :GridFunctionCoefficient(gf, comp),
       fun(fun_) {}

   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return fun(GridFunctionCoefficient::Eval(T, ip) + c);
   }
   void SetFunction(std::function<real_t(const real_t)> fun_) { fun = fun_; }
   void SetConstant(real_t new_c) { c = new_c; }
private: 
   mutable real_t c; 
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   // const char *mesh_file = "../data/star.mesh";

   // unit square, vertices (0, 0), (0, 1), (1, 1), (1, 0)
   // const char *mesh_file = "../data/square-nurbs.mesh"; 

   // this square has vertices (-1, -1), (-1, 1), (1, -1), (1, 1)
   // which changes the scaling lower in the file 
   const char *mesh_file = "../data/periodic-square.mesh"; 

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
   // mesh.Transform([](const Vector &x, Vector &y){y = x; y *= 2.0; });
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
   // int curvature_order = max(order,2);
   // mesh.SetCurvature(curvature_order);

	// rescale domain to square with vertices 
   // (this is for ordinary square mesh)
	// (-2, -2), (-2, 2), (2, 2), (2, -2)
	GridFunction *nodes = mesh.GetNodes();
   // *nodes *= 4.; 
   // *nodes -= 2.;

   // this is the scaling if the periodic square domain is used, which 
   // has different vertices (noted above when loading) 
   *nodes *= 2.; 

   // 4. Define the necessary finite element spaces on the mesh.
   L2_FECollection L2fec(order-1, dim);
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
		auto val = x*x-1.; 
		return (val < 0.) ? -1. : (val > 0.) ? 1. : 0.;  
	}; 

	FunctionCoefficient sgn_varphi(sgn_varphi_is); 

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock, gradu_sock, true_sdf_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);

      gradu_sock.open(vishost, visport); 
      gradu_sock.precision(8); 

      true_sdf_sock.open(vishost, visport); 
      true_sdf_sock.precision(8); 
   }

   // plot out the true sdf  
   // defines true sdf for square and unit circle case
   auto true_sdf = [](const Vector &x){ return sqrt(x*x)-1.; }; 
   
   FunctionCoefficient sdf_fc(true_sdf); 
   GridFunction sdf_gf(&H1fes); 
   sdf_gf.ProjectCoefficient(sdf_fc); 

   true_sdf_sock << "solution\n" << mesh << sdf_gf << "window_title 'True SDF'" << flush;

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
   ScalarVectorProductCoefficient neg_grad_u_old(-1.0 * 1 / alpha, grad_u_old); 
   b1.AddDomainIntegrator(new DomainLFGradIntegrator(neg_grad_u_old));

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
         a11.AddDomainIntegrator(new DiffusionIntegrator(neg_one_alpha_recip)); 

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

         GradientGridFunctionCoefficient grad_u(&u_gf);
         grad_u_gf.ProjectDiscCoefficient(grad_u);

         if (visualization)
         {
            sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'"
                     << flush;
            mfem::out << "Newton_update_size = " << Newton_update_size << endl;

            gradu_sock << "solution\n" << mesh << grad_u_gf << "window_title 'Gradient magnitude'"
                     << flush;
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

   // bisection method to find the optimal c for translation 
   // this is obtained by solving 
   // int f(x) dx = int g(x) dx 
   // where f(x) is the indicator function for the region {u + c <= 0}
   // and g(x) is the indicator function for the region {varphi <= 0}

   auto f = [](double x){return (x <= 0.) ? 1. : 0.;  }; // indicator function for x <= 0
   MappedGridFunctionCoefficient curr_coeff(&u_gf, f); 
   
   // endpoints for bisection search
   real_t endpoint_a = -1., endpoint_b = 1.;  

   real_t midpoint_c; 

   real_t bisection_tol = 1e-6; 
   const int N_MAX = 100; 
   
   int n_iter = 0; 

   real_t pi = 3.1415926535;

   while (n_iter <= N_MAX) { 
      midpoint_c = (endpoint_a + endpoint_b) / 2; 

      curr_coeff.SetConstant(midpoint_c); 

      LinearForm coeff_opt_c(&H1fes); 
      coeff_opt_c.AddDomainIntegrator(new DomainLFIntegrator(curr_coeff)); 
      coeff_opt_c.Assemble();

      real_t soln_c = coeff_opt_c.Sum() - pi; 

      if (soln_c == 0.) {
         break; 
      }

      // decide the next direction of search, need to recompute
      curr_coeff.SetConstant(endpoint_a); 
      LinearForm coeff_opt_a(&H1fes); 
      coeff_opt_a.AddDomainIntegrator(new DomainLFIntegrator(curr_coeff)); 
      coeff_opt_a.Assemble();

      real_t soln_a = coeff_opt_a.Sum() - pi; 

      if (soln_c * soln_a < 0) {
         endpoint_b = midpoint_c;
      }
      else { 
         endpoint_a = midpoint_c;
      }

      // mfem::out << "\ncurrent midpoint  = " << midpoint_c << endl;
      n_iter++; 
   }

   mfem::out << "\n concluded bisection method with c = " << midpoint_c << endl;

   real_t error_initial = u_gf.ComputeL2Error(sdf_fc);

   u_gf += midpoint_c;
   real_t error_final = u_gf.ComputeL2Error(sdf_fc);

   auto u_grad_exact = [](const Vector &x, Vector &u)
   {
      auto norm = sqrt(x*x);

      u(0) = x(0) / norm; 
      u(1) = x(1) / norm;  
   }; 
   
   VectorFunctionCoefficient ex_grad(dim, u_grad_exact);

   real_t error_final_h1 = u_gf.ComputeH1Error(&sdf_fc, &ex_grad);

   mfem::out << "\nL2 Error from true SDF initial = " << error_initial << endl;

   mfem::out << "\nL2 Error from true SDF final = " << error_final << endl;

   mfem::out << "\nH1 Error from true SDF final = " << error_final_h1 << endl;

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
