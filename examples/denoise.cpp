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

class RotationCoefficient : public MatrixCoefficient {
public:
    RotationCoefficient() : MatrixCoefficient(2) {}

    virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                      const IntegrationPoint &ip) {
        M(0,0) =  0;  M(0,1) = -1;  //  [0, -1]
        M(1,0) =  1;  M(1,1) =  0;  //  [1, 0]
    }
};


int main(int argc, char *argv[])
{
   // const char *mesh_file = "../data/star.mesh";
   int order = 2;
   int order_l2 = 1; 
   int max_it = 10;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t growth_rate = 1.0;
   real_t newton_scaling = 0.9;
   real_t tichonov = 1e-1;
   real_t tol = 1e-6;

   int ex = 1; 

   bool visualization = true;

   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
                  // "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order for RT space");
   args.AddOption(&order_l2, "-o2", "--order", "FEM order for L2 vec space"); 
   args.AddOption(&ex, "-ex", "--example", "example number"); 
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

   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::Type::TRIANGLE, false);
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

   H1_FECollection h1fec(order_l2, sdim);
   FiniteElementSpace h1fes(&mesh, &h1fec);

   L2_FECollection L2fec(order_l2, dim);
   FiniteElementSpace L2fes(&mesh, &L2fec, 2);

   Array<int> ess_tdof_list_rt;
   RTfes.GetBoundaryTrueDofs(ess_tdof_list_rt);
   
   Array<int> ess_tdof_list_h1; 
   h1fes.GetBoundaryTrueDofs(ess_tdof_list_h1); 

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;


   // ess_tdof_list = 1;
   // if (mesh.bdr_attributes.Size())
   // {
      // Array<int> ess_bdr(mesh.bdr_attributes.Max());
      // ess_bdr = 1;

      // RTfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   cout << "Number of H(div) dofs: "
        << RTfes.GetTrueVSize() << endl;
   cout << "Number of L² dofs: "
        << L2fes.GetTrueVSize() << endl;
   cout << "Number of H1 dofs: " 
         << h1fes.GetTrueVSize() << endl; 

   Array<int> offsets({0, RTfes.GetVSize(), h1fes.GetVSize(), L2fes.GetVSize(), 1}); 
   offsets.PartialSum();

   BlockVector x(offsets), rhs(offsets);
   x = 0.0; rhs = 0.0;

   GridFunction p_gf(&RTfes, x.GetBlock(0)), vphi_gf(&h1fes, x.GetBlock(1)), delta_psi_gf(&L2fes, x.GetBlock(2));

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

   ConstantCoefficient one_cf(1.0); 
   ConstantCoefficient neg_one(-1.0);
   VectorConstantCoefficient zero_vec_cf(Vector({0., 0.}));
   ConstantCoefficient zero_cf(0.0);
	VectorConstantCoefficient one_vec_cf(Vector({1., 1.})); 
	VectorConstantCoefficient neg_one_vec_cf(Vector({-1., -1.})); 

   ConstantCoefficient tichonov_cf(tichonov);
   ConstantCoefficient neg_tichonov_cf(-1.0*tichonov);

   ConstantCoefficient alpha_cf((real_t) alpha); 
   ProductCoefficient neg_alpha_cf(neg_one, alpha_cf); 

   ZCoefficient Z(sdim, psi_gf);
   DZCoefficient DZ(sdim, psi_gf);
   ScalarMatrixProductCoefficient neg_DZ(-1.0, DZ);

   VectorSumCoefficient psi_newton_res(psi_old_cf, psi_cf, 1., -1.);

   LinearForm b0(&RTfes, rhs.GetBlock(0).GetData()), b2(&L2fes, rhs.GetBlock(2).GetData());

   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(psi_newton_res));

   VectorFunctionCoefficient f_coeff(2, [ex](const Vector &x, Vector &u) {
      // NOTE: constant example 
      // u(0) = 0.5; 
      // u(1) = 0.0;

      // NOTE: linear example 
      // u(0) = x(0); 
      // u(1) = -1.0 * x(1);

      if (ex == 1) {
         // NOTE: trig example
         u(0) = cosh(M_PI*x(0)) * sin(M_PI*x(1));  
         u(1) = sinh(M_PI*x(0)) * cos(M_PI*x(1)); 

         u /= cosh(M_PI);
      }
      else if (ex == 2) { 
         // NOTE: trig example 2
         u(0) = cos(M_PI * x(0)) * sin (M_PI * x(1)); 
         u(1) = cos(M_PI * x(1)) * sin (M_PI * x(0));

         u *= (1. + 2. * pow(M_PI, 2)); 
      }
   });

   ScalarVectorProductCoefficient alpha_f_cf(alpha_cf, f_coeff); 
   b0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(alpha_f_cf));
   // b0.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(alpha_f_cf)); 

   b2.AddDomainIntegrator(new VectorDomainLFIntegrator(Z));


   RotationCoefficient R;
   ScalarMatrixProductCoefficient neg_R(neg_one, R); 
   // MixedBilinearForm a10(&RTfes, &h1fes); 
   // a10.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(neg_R)); 
   // a10.Assemble(false); 
   // a10.Finalize(false); 
   // SparseMatrix &A10 = a10.SpMat();
   // SparseMatrix *A01 = Transpose(A10);

  

   // TODO remove RT dofs every iteration too
   // a20.Assemble(false);
   // a20.Finalize(false);
   // SparseMatrix &A20 = a20.SpMat();
   // SparseMatrix *A02 = Transpose(A20);
   
   // MixedBilinearForm a02(&L2fes, &RTfes); 
   // a02.AddDomainIntegrator(new VectorFEMassIntegrator()); 

   

 


   // A.SetBlock(1,0,&A10);
   // A.SetBlock(0,1,A01);



   // A.SetBlock(0, 1, &A01); 
   // A.SetBlock(1, 0, A10); 
// 
   // A.SetBlock(2,0,&A20); 
   // A.SetBlock(0,2,A02); 






   

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
         b2.Assemble();

         BlockMatrix A(offsets); 

         BilinearForm a00(&RTfes);
         a00.AddDomainIntegrator(new DivDivIntegrator(alpha_cf)); 
         a00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         a00.Assemble();
         a00.EliminateEssentialBC(ess_tdof_list_rt, x.GetBlock(0), rhs.GetBlock(0), mfem::Operator::DIAG_ONE); 
         a00.Finalize(); 

         SparseMatrix &A00 = a00.SpMat(); 

         A.SetBlock(0,0,&A00);

         BilinearForm a11(&h1fes); 
         a11.AddDomainIntegrator(new DiffusionIntegrator(neg_one)); 
         a11.Assemble(false);
         a11.Finalize(false);

         SparseMatrix &A11 = a11.SpMat();
         A.SetBlock(1,1,&A11);

         BilinearForm a22(&L2fes);
         a22.AddDomainIntegrator(new VectorMassIntegrator(neg_DZ)); 

         a22.Assemble(false);
         a22.Finalize(false);

         SparseMatrix &A22 = a22.SpMat();
         A.SetBlock(2,2,&A22);

         MixedBilinearForm a01(&h1fes, &RTfes); 
         a01.AddDomainIntegrator(new MixedVectorGradientIntegrator(R)); 

         a01.Assemble(false); 
         a01.EliminateTestDofs(ess_bdr); 
         a01.Finalize(false); 

         SparseMatrix &A01 = a01.SpMat();
         SparseMatrix *A10 = Transpose(A01);

         A.SetBlock(0, 1, &A01); 
         A.SetBlock(1, 0, A10); 

         MixedBilinearForm a20(&RTfes, &L2fes);
         a20.AddDomainIntegrator(new VectorFEMassIntegrator());

         a20.Assemble();
         a20.EliminateTrialDofs(ess_bdr, x.GetBlock(0), b0); 

         a20.Finalize();
         SparseMatrix &A20 = a20.SpMat();
         SparseMatrix *A02 = Transpose(A20);
         A.SetBlock(2,0,&A20); 
         A.SetBlock(0,2,A02); 

         // Avg 0 condition
         int dof_h1(h1fes.GetTrueVSize());
         LinearForm avg0_data(&h1fes);
         avg0_data.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
         avg0_data.Assemble();
         Array<int> avg0_i({0, dof_h1}), avg0_j(dof_h1);
         std::iota(avg0_j.begin(), avg0_j.end(), 0);
         SparseMatrix avg0(avg0_i.GetData(), avg0_j.GetData(), avg0_data.GetData(), 1,
                           dof_h1, false, false, true);
         auto avg0T = *Transpose(avg0);

         A.SetBlock(3, 1, &avg0); 
         A.SetBlock(1, 3, &avg0T); 



         // a11.EliminateEssentialBC(ess_tdof_list, x.GetBlock(1), rhs.GetBlock(1), mfem::Operator::DIAG_ONE);


         
         // a01.EliminateTestDofs(ess_bdr); 
         // a01.EliminateEssentialBCFromTrialDofs(ess_tdof_list_rt, x.GetBlock(0), rhs.GetBlock(0)); 


         // a10.EliminateEssentialBCFromTrialDofs(ess_tdof_list_rt, x.GetBlock(0), rhs.GetBlock(0)); 
         // a10.Elimi
         // a10.Assemble(false); 
         // a10.Finalize(false); 

         // SparseMatrix &A10 = a10.SpMat();
         // SparseMatrix *A01 = Transpose(A10);

// 
         // A.SetBlock(0, 1, A01); 
         // A.SetBlock(1, 0, &A10); 

         // a20.EliminateTrialDofs(ess_tdof_list_rt); 
         // a20.EliminateEssentialBCFromTrialDofs(ess_tdof_list_rt, x.GetBlock(0), rhs.GetBlock(0)); 

         // #################3
         // a20.Assemble();
         // a20.EliminateTrialDofs(ess_bdr, x.GetBlock(0), b0); 
         // cout << "elim a20" << endl; 
// 
         // a20.Finalize();
         // SparseMatrix &A20 = a20.SpMat();
         // SparseMatrix *A02 = Transpose(A20);
         // A.SetBlock(2,0,&A20); 
         // A.SetBlock(0,2,A02); 
         // ###################

         // a02.EliminateTestDofs(ess_bdr); 
         // a02.EliminateEssentialBCFromTrialDofs(ess_tdof_list_rt, x.GetBlock(0), rhs.GetBlock(0)); 

         // a02.Assemble(false); 
         // a02.Finalize(false); 

         // SparseMatrix &A02 = a02.SpMat(); 
         // SparseMatrix *A20 = Transpose(A02); 
// 
         // A.SetBlock(2,0,A20); 
         // A.SetBlock(0,2,&A02); 



// #ifndef MFEM_USE_SUITESPARSE 
         BlockDiagonalPreconditioner prec(offsets);
         prec.SetDiagonalBlock(0,new GSSmoother(A00));
         prec.SetDiagonalBlock(1,new GSSmoother(A11));
         prec.SetDiagonalBlock(2,new GSSmoother(A22));

         prec.owns_blocks = 3;

         GMRES(A,prec,rhs,x,0,10000,500,1e-12,0.0);
// #else
         // SparseMatrix *A_mono = A.CreateMonolithic(); 
         // UMFPackSolver umf(*A_mono); 
         // umf.Mult(rhs, x); 
// #endif
         // 
         p_tmp -= p_gf;
         real_t Newton_update_size = p_tmp.ComputeL2Error(zero_vec_cf);
         p_tmp = p_gf;

         // Damped Newton update
         psi_gf.Add(newton_scaling, delta_psi_gf);
         // a11.Update();
         // a22.Update(); 
         b0.Update();
         b2.Update();

         if (visualization)
         {
            GridFunction p_vec(&L2fes); 
            p_vec.ProjectCoefficient(p_vc); 

            sol_sock << "solution\n" << mesh << p_vec << "window_title 'Discrete solution '" << flush;            
         }

         mfem::out << "Newton_update_size = " << Newton_update_size << endl;

         if (Newton_update_size < increment_p)
         {
            break;
         }

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
   // delete A01;
   // delete A10; 

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

   VectorFunctionCoefficient exact_coeff(2, [ex](const Vector &x, Vector &u) {      
      // NOTE: constant example 
      // u(0) = 0.5; 
      // u(1) = 0.0; 
      
      // NOTE: linear example 
      // u(0) = x(0); 
      // u(1) = -1.0 * x(1);

      if (ex == 1) { 
      // NOTE: trig example 
         u(0) = cosh(M_PI*x(0)) * sin(M_PI*x(1));  
         u(1) = sinh(M_PI*x(0)) * cos(M_PI*x(1)); 
         
         u /= cosh(M_PI);
      }
      else if (ex == 2) { 
         // NOTE trig example 2 
         u(0) = cos(M_PI * x(0)) * sin (M_PI * x(1)); 
         u(1) = cos(M_PI * x(1)) * sin (M_PI * x(0));
      }
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

   real_t l2_error = p_gf.ComputeL2Error(exact_coeff); 

   cout << "L2 error: " << l2_error << endl; 

   mfem::Coefficient *div_u_exact = nullptr;

   // NOTE: for constant, linear, trig examples, div p = 0  
    if (ex == 1)
   {
      // For ex=1, the divergence is zero.
      div_u_exact = new mfem::ConstantCoefficient(0.0);
   }
   else if (ex == 2) { 
   // NOTE: for trig example2, div p != 0 
      div_u_exact = new mfem::FunctionCoefficient([](const mfem::Vector &x)
      {
         return -2. * M_PI * sin(M_PI * x(0)) * sin(M_PI * x(1));
      });
   }

   real_t hdiv_error = p_gf.ComputeDivError(div_u_exact);  

   cout << "div error: " << hdiv_error << endl; 

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
