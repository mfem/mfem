#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ZCoefficient : public VectorCoefficient
{
protected:
   GridFunction *psi;

public:
   ZCoefficient(int vdim, GridFunction &psi_)
      : VectorCoefficient(vdim), psi(&psi_) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *psi;

public:
   DZCoefficient(int height, GridFunction &psi_)
      : MatrixCoefficient(height),  psi(&psi_) { }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
};


bool CheckVectorComponents(const mfem::GridFunction &gf, double limit)
{
    const double* data = gf.GetData();
    const int size = gf.Size();

    for (int i = 0; i < size; ++i)
    {
        if (std::abs(data[i]) > limit)
        {
            const int vdim = gf.FESpace()->GetVDim();
            int dof_index = i / vdim;
            int component_index = i % vdim;

            std::cout << "--> Condition VIOLATED at DOF #" << dof_index
                      << ", component " << component_index
                      << ". Value: " << data[i]
                      << ", Limit: " << limit << std::endl;

            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
   // const char *mesh_file = "../data/star.mesh";
   int order = 3;
   int order_l2 = 1; 
   int max_it = 10;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t growth_rate = 1.0;
   real_t newton_scaling = 0.9;
   real_t tichonov = 1e-1;
   real_t tol = 1e-4;

   bool visualization = true;

   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
                  // "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order for RT space");
   args.AddOption(&order_l2, "-o2", "--order", "FEM order for L2 vec space"); 
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&tol, "-tol", "--tol",
                  "Stopping criteria based on the difference between"
                  "successive solution updates");
   args.AddOption(&alpha, "-step", "--step",
                  "Initial size alpha");
   args.AddOption(&growth_rate, "-gr", "--growth-rate",
                  "Growth rate of the step size alpha");
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

   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::Type::QUADRILATERAL, false);
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   int curvature_order = max(order,2);
   mesh.SetCurvature(curvature_order);

   RT_FECollection RTfec(order, dim);
   FiniteElementSpace RTfes(&mesh, &RTfec);

   L2_FECollection L2fec(order_l2, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec, 2);

   cout << "Number of H(div) dofs: "
        << RTfes.GetTrueVSize() << endl;
   cout << "Number of L² dofs: "
        << L2fes.GetTrueVSize() << endl;

   Array<int> offsets({0, RTfes.GetVSize(), L2fes.GetVSize()});
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   GridFunction p_gf(&RTfes, x.GetBlock(0)), delta_psi_gf(&L2fes, x.GetBlock(1));

   GridFunction psi_old_gf(&L2fes);
   GridFunction psi_gf(&L2fes);
   GridFunction p_old_gf(&RTfes);

   delta_psi_gf = 0.0;
   psi_gf = 0.0;
   p_gf = 0.0;
   psi_old_gf = psi_gf;
   p_old_gf = p_gf;

   VectorGridFunctionCoefficient psi_old_cf(&psi_old_gf), psi_cf(&psi_gf), p_vc(&p_gf); 

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock, true_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
      
      // sol_sock  << "keys jlA\n";
       // turn off perspective & light 
      // sol_sock << "keys cmA\n"; // colorbar + mesh + anti-alias
   
      true_sock.open(vishost,visport);
      true_sock.precision(8);
   }

   ConstantCoefficient neg_one(-1.0);
   VectorConstantCoefficient zero_vec_cf(Vector({0., 0.}));
   ConstantCoefficient zero_cf(0.0);
	VectorConstantCoefficient one_vec_cf(Vector({1., 1.})); 

   ConstantCoefficient tichonov_cf(tichonov);
   ConstantCoefficient neg_tichonov_cf(-1.0*tichonov);


   ConstantCoefficient alpha_cf((real_t) alpha); 
   ProductCoefficient neg_alpha_cf(neg_one, alpha_cf); 

   ZCoefficient Z(sdim, psi_gf);
   DZCoefficient DZ(sdim, psi_gf);
   ScalarMatrixProductCoefficient neg_DZ(-1.0, DZ);

   VectorSumCoefficient psi_newton_res(psi_old_cf, psi_cf, 1., -1.);

   LinearForm b0(&RTfes, rhs.GetBlock(0).GetData()), b1(&L2fes, rhs.GetBlock(1).GetData());

   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(psi_newton_res));

   // FunctionCoefficient f([](const Vector &x) {return -0.5 * ( (1 - 2*x[0]) * x[1] * (1 - x[1]) + x[0] * (1 - x[0]) * (1-2*x[1]));  });
   // FunctionCoefficient f([](const Vector &x) {return -2. * (x[0] + x[1]);  });
   // FunctionCoefficient f([](const Vector &x) {return -2. * x[0] - pow(x[0], 2) / 2. + pow(x[0], 3) / 3. - 2 * x[1] - pow(x[1] , 2) / 2. + pow(x[1] , 3) / 3;  });
   
   // FunctionCoefficient f([](const Vector &x) { return -x[0] - x[1]  + 1; }); 

   // FunctionCoefficient f([](const Vector &x) { return  1.; });  // just because this is prettiest 

   // FunctionCoefficient f([](const Vector &x) { return 1/M_PI * sin(M_PI * x[0]) * sin(M_PI*x[1]); }); 
   // FunctionCoefficient f([](const Vector &x)) { return -1. * (pow(x[0], 2) + pow(x[1], 2));  };
   // FunctionCoefficient f([](const Vector &x) { return  0.5*x[0]; });
   // FunctionCoefficient f([](const Vector &x) { return  0.25*(pow(x[0], 2) + pow(x[1], 2)); });

   // FunctionCoefficient f([](const Vector &x) { return -1.*( 2*(x[0] + x[1]) - pow(x[0], 3) - pow(x[1], 3)); });

   // ProductCoefficient neg_alpha_f_cf(neg_alpha_cf, f); 

   // b0.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(neg_alpha_f_cf));
VectorFunctionCoefficient f_coeff(2, [](const Vector &x, Vector &u) {
      u(0) = 0.5; 
      u(1) = 0.0;
   });

   ScalarVectorProductCoefficient alpha_f_cf(alpha_cf, f_coeff); 
   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(alpha_f_cf));

   b1.AddDomainIntegrator(new VectorDomainLFIntegrator(Z));

   BilinearForm a00(&RTfes);
   a00.AddDomainIntegrator(new DivDivIntegrator(alpha_cf)); 
   a00.AddDomainIntegrator(new VectorFEMassIntegrator(alpha_cf)); 
   
   a00.Assemble();
   a00.Finalize(); 
   SparseMatrix &A00 = a00.SpMat(); 

   MixedBilinearForm a10(&RTfes, &L2fes);
   a10.AddDomainIntegrator(new VectorFEMassIntegrator());
   a10.Assemble(false);
   a10.Finalize(false);
   SparseMatrix &A10 = a10.SpMat();
   SparseMatrix *A01 = Transpose(A10);

   BilinearForm a11(&L2fes);
   a11.AddDomainIntegrator(new VectorMassIntegrator(neg_DZ)); 

   int k;
   int total_iterations = 0;
   real_t increment_p = 0.1;
   GridFunction p_tmp(&RTfes);
   for (k = 0; k < max_it; k++)
   {
      p_tmp = p_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 5; j++)
      {
         total_iterations++;

         b0.Assemble();
         b1.Assemble();

         a11.Assemble(false);
         a11.Finalize(false);
         SparseMatrix &A11 = a11.SpMat();

         BlockMatrix A(offsets); 
         A.SetBlock(0,0,&A00);
         A.SetBlock(1,0,&A10);
         A.SetBlock(0,1,A01);
         A.SetBlock(1,1,&A11);

         // SparseMatrix *A_mono = A.CreateMonolithic(); 
         // UMFPackSolver umf(*A_mono); 
         // umf.Mult(rhs, x); 

         BlockDiagonalPreconditioner prec(offsets);
         prec.SetDiagonalBlock(0,new GSSmoother(A00));
         prec.SetDiagonalBlock(1,new GSSmoother(A11));
         prec.owns_blocks = 1;

         GMRES(A,prec,rhs,x,0,10000,500,1e-12,0.0);

         p_tmp -= p_gf;
         real_t Newton_update_size = p_tmp.ComputeL2Error(zero_vec_cf);
         p_tmp = p_gf;

         mfem::out << "tag 2" << endl;

         // Damped Newton update
         psi_gf.Add(newton_scaling, delta_psi_gf);
         a11.Update();
         b0.Update();
         b1.Update();

         if (visualization)
         {
            GridFunction p_vec(&L2fes); 
            p_vec.ProjectCoefficient(p_vc); 
            // u_gf.GetNodalValues(u_x, 1);
            // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution '" << flush;

            sol_sock << "solution\n" << mesh << p_vec << "window_title 'Discrete solution '" << flush;
            
            // sol_sock << "solution\n" << mesh << psi_gf << "window_title 'Discrete solution'"
                     // << flush;
         }

         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_p)
         {
            break;
         }

         // delete A01;
      }

      p_tmp = p_gf;
      p_tmp -= p_old_gf;
      increment_p = p_tmp.ComputeL2Error(zero_vec_cf);

      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      mfem::out << "Increment (|| uₕ - uₕ_prvs||) = " << increment_p << endl;

      p_old_gf = p_gf;
      psi_old_gf = psi_gf;

      if (increment_p < tol || k == max_it-1)
      {
         break;
      }

      alpha *= max(growth_rate, 1_r);
      alpha_cf.constant = alpha; 
   }
   delete A01;

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

   VectorFunctionCoefficient exact_coeff(2, [](const Vector &x, Vector &u) {
      // double x_val = x(0);
      // double y_val = x(1);
      // double val = x_val * y_val * (1 - x_val) * (1 - y_val);

      // u(0) = pow(x(0), 2); 
      // u(1) = pow(x(1), 2); 

      
      u(0) = 0.5; 
      u(1) = 0.0; 


      // u(0) = 0.5 * pow(x(0), 2); 
      // u(1) = 0.5 * pow(x(1), 2); 

      // u(0) = x_val * (1. - x_val);
      // u(1) = y_val * (1. - y_val);

      // u(0) = -cos(M_PI * x[0]) * sin(M_PI*x[1]); 
      // u(1) = -sin(M_PI*x[0]) * cos(M_PI*x[1]); 

      // u(0) = 1.; 
      // u(1) = 1.; 

      // u(0) = 2. * x_val - 1. ; 
      // u(1) = 2. * y_val - 1.; 
      
      // u(0) = 4. * (1 - 2 * x_val) * (y_val - pow(y_val, 2)); 
      // u(1) = 4. * (x_val - pow(x_val, 2))*(1. - 2.*y_val); 
   });

   GridFunction exact_vec(&L2fes); 
   exact_vec.ProjectCoefficient(exact_coeff); 
   if (visualization) { 
      true_sock << "solution\n" << mesh <<  exact_vec << "window_title 'True solution '" << flush; 
   }

    if (CheckVectorComponents(p_gf, 1.0))
    {
        std::cout << "Result: SUCCESS. All components are within the limit." << std::endl;
    }
    else
    {
        std::cout << "Result: FAILURE. At least one component is outside the limit." << std::endl;
    }


   double l2_error = p_gf.ComputeL2Error(exact_coeff); 
   cout << "L2 error: " << l2_error << endl; 

   return 0;
}

// NOTE: 2D ONLY 
void ZCoefficient::Eval(Vector &V, ElementTransformation &T,
                        const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(2);
   psi->GetVectorValue(T, ip, psi_vals);

   V.SetSize(2); 

   for (int i = 0; i < psi_vals.Size(); ++i) { V(i) = tanh(psi_vals(i) / 2.); }
}

// NOTE: 2D ONLY 
void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(2);
   psi->GetVectorValue(T, ip, psi_vals);

   K.SetSize(2);
   K = 0.0;
   for (int i = 0; i < 2; ++i) { K(i, i) = (1. - pow(tanh(psi_vals(i) / 2.), 2)) / 2.; }
}
