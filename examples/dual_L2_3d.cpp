#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

real_t factorial(int n) {
    if (n <= 1) return 1.0;
    real_t result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

class AlphaRuleManager {
private:
    int rule_type;
    real_t C, m_real, q, r;
    int m_int;
    real_t alpha_prev;  // For rule 4
    std::function<real_t(int)> update_func;

public:
    AlphaRuleManager(int rule, real_t C_val, real_t m_r, real_t q_val, real_t r_val, int m_i)
        : rule_type(rule), C(C_val), m_real(m_r), q(q_val), r(r_val), m_int(m_i), alpha_prev(0.0) {
        setupRule();
    }

    void setupRule() {
        switch (rule_type) {
            case 0:
                break;
            case 1: // alpha_k = C * k * (k+1) * ... * (k+m)
                update_func = [this](int k) {
                    real_t product = 1.0;
                    for (int i = 0; i <= m_int; ++i) {
                        product *= (k + 1 + i);
                    }
                    return C * product;
                };
                break;
            case 2: // alpha_k = C * pow(m, k-1)
                update_func = [this](int k) {
                    return C * pow(m_real, k);
                };
                break;
            case 3: // alpha_{k+1} = C * k * k!
                update_func = [this](int k) {
                    return C * (k + 1) * factorial(k + 1);
                };
                break;
            case 4: // alpha_{k+1} = r^{1/(q-1)} * m^{q^k} - alpha_k
                update_func = [this](int k) {
                    real_t alpha_new = pow(r, 1.0 / (q - 1.0)) * pow(m_real, pow(q, k + 1)) - alpha_prev;
                    alpha_prev = alpha_new;
                    return alpha_new;
                };
                break;
            default:
                update_func = [](int k) { return 1.0; }; // fallback
                break;
        }
    }

    real_t updateAlpha(int iteration, real_t current_alpha, real_t growth_rate) {
        if (rule_type == 0) {
            return current_alpha * max(growth_rate, 1_r);
        } else {
            return update_func(iteration);
        }
    }

    real_t getInitialAlpha() {
        if (rule_type == 4) {
            return pow(r, 1.0 / (q - 1.0));
        }
        return 1.0; 
    }
};

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
   int alpha_rule = 0;  // 0 = original rule, 1-4 = new rules
   
   // Parameters for alpha update rules
   real_t C = 1.0;      // Constant for all rules
   int m_int = 2;       // Integer parameter for rules 1 and 3
   real_t m_real = 2.0; // Real parameter for rule 2
   real_t q = 2.0;      // Parameter for rule 4
   real_t r = 2.0;      // Parameter for rule 4

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
   args.AddOption(&alpha_rule, "-ar", "--alpha-rule",
                  "Alpha update rule (0=original, 1-4=new rules)");
   args.AddOption(&C, "-C", "--constant",
                  "Constant C for alpha update rules");
   args.AddOption(&m_int, "-mi", "--m-int",
                  "Integer parameter m for alpha rules 1 and 3");
   args.AddOption(&m_real, "-mr", "--m-real",
                  "Real parameter m for alpha rule 2");
   args.AddOption(&q, "-q", "--q-param",
                  "Parameter q for alpha rule 4");
   args.AddOption(&r, "-r-param", "--r-param",
                  "Parameter r for alpha rule 4");
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

   // NOTE: based on step size update rules in PG paper appendix
   if (alpha_rule < 0 || alpha_rule > 4) {
      mfem::out << "Error: alpha_rule must be between 0 and 4" << endl;
      return 1;
   }
   // if (alpha_rule == 2 && m_real <= 1.0) {
      // mfem::out << "Error: for rule 2, m_real must be > 1" << endl;
      // return 1;
   // }
   // if (alpha_rule == 4 && (q <= 1.0 || r <= 1.0)) {
      // mfem::out << "Error: for rule 4, q and r must be > 1" << endl;
      // return 1;
   // }

//    Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::Type::TRIANGLE, false);
   
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::Type::TETRAHEDRON); 
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
   FiniteElementSpace L2fes(&mesh, &L2fec, 3);

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
      // sol_sock << "keys " << "cmA" << endl; // colorbar + mesh + anti-alias
   
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

   VectorFunctionCoefficient f_coeff(3, [ex](const Vector &x, Vector &u) {

      if (ex == 1) {
         // NOTE: trig example
        real_t a = x(0), b = x(1), c = x(2); 
        u(0) = std::cos(M_PI * a) * std::sin(M_PI * b) * std::sin(M_PI * c);
        u(1) = std::sin(M_PI * a) * std::cos(M_PI * b) * std::sin(M_PI * c);
        u(2) = std::sin(M_PI * a) * std::sin(M_PI * b) * std::cos(M_PI * c); 

        u *= (1.0 + 3.0 * M_PI * M_PI);
    }
      else if (ex == 2) { 
         // NOTE: trig example 2
        //  u(0) = cos(M_PI * x(0)) * sin (M_PI * x(1)); 
        //  u(1) = cos(M_PI * x(1)) * sin (M_PI * x(0));
// 
        //  u *= (1. + 2. * pow(M_PI, 2)); 
        u(1) = cos(M_PI * x(1)) * sin (M_PI * x(2)); 
        u(2) = cos(M_PI * x(2)) * sin (M_PI * x(1));

        u /= 1 + 2. * pow(M_PI, 2); 
        u(0) = 1.; 
      }
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

   AlphaRuleManager alpha_manager(alpha_rule, C, m_real, q, r, m_int);
   alpha = alpha_manager.getInitialAlpha(); 
   alpha_cf.constant = alpha; 
   
   mfem::out << "Using alpha update rule " << alpha_rule << " with initial alpha = " << alpha << endl;

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

#ifndef MFEM_USE_SUITESPARSE 
         BlockDiagonalPreconditioner prec(offsets);
         prec.SetDiagonalBlock(0,new GSSmoother(A00));
         prec.SetDiagonalBlock(1,new GSSmoother(A11));
         prec.owns_blocks = 1;

         GMRES(A,prec,rhs,x,0,10000,500,1e-12,0.0);
#else
         SparseMatrix *A_mono = A.CreateMonolithic(); 
         UMFPackSolver umf(*A_mono); 
         umf.Mult(rhs, x); 
#endif
         
         p_tmp -= p_gf;
         real_t Newton_update_size = p_tmp.ComputeL2Error(zero_vec_cf);
         p_tmp = p_gf;

         // Damped Newton update
         psi_gf.Add(newton_scaling, delta_psi_gf);
         a11.Update();
         b0.Update();
         b1.Update();

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
      mfem::out << "Current alpha = " << alpha << " (rule " << alpha_rule << ")" << endl;

      p_old_gf = p_gf;
      psi_old_gf = psi_gf;

      if (increment_p < tol || k == max_it-1)
      {
         break;
      }

      // Update alpha according to the specified rule
      alpha = alpha_manager.updateAlpha(k, alpha, growth_rate);
      alpha_cf.constant = alpha; 
   }
   delete A01;

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n Total dofs:       " << RTfes.GetTrueVSize() + L2fes.GetTrueVSize()
             << endl;

   VectorFunctionCoefficient exact_coeff(3, [ex](const Vector &x, Vector &u) {      
      // NOTE: constant example 
      // u(0) = 0.5; 
      // u(1) = 0.0; 
      
      // NOTE: linear example 
      // u(0) = x(0); 
      // u(1) = -1.0 * x(1);

      if (ex == 1) { 
      // NOTE: trig example
        real_t a = x(0), b = x(1), c = x(2); 

        u(0) = cos(M_PI*a) * sin(M_PI * b) * sin(M_PI*c); 
        u(1) = sin(M_PI*a)*cos(M_PI*b)*sin(M_PI*c); 
        u(2) = sin(M_PI*a)*sin(M_PI*b)*cos(M_PI*c);  
      }
      else if (ex == 2) { 
         // NOTE trig example 2 
        // u(0) = cos(M_PI * x(0)) * sin (M_PI * x(1)); 
        // u(1) = cos(M_PI * x(1)) * sin (M_PI * x(0));
        u(0) = 1.; 
        u(1) = cos(M_PI * x(1)) * sin (M_PI * x(2)); 
        u(2) = cos(M_PI * x(2)) * sin (M_PI * x(1));
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
    //   div_u_exact = new mfem::ConstantCoefficient(0.0);
    div_u_exact = new mfem::FunctionCoefficient([](const mfem::Vector &x)
      {
        real_t a = x(0), b = x(1), c = x(2); 

        real_t sin_product = std::sin(M_PI * a) * std::sin(M_PI * b) * std::sin(M_PI * c);
        real_t scale_factor = -3.0 * M_PI;

        return scale_factor * sin_product;
      });
   }
   else if (ex == 2) { 
   // NOTE: for trig example2, div p != 0 
      div_u_exact = new mfem::FunctionCoefficient([](const mfem::Vector &x)
      {
         return -2. * M_PI * sin(M_PI * x(2)) * sin(M_PI * x(1));
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

   Vector psi_vals(3);
   psi->GetVectorValue(T, ip, psi_vals);

   V.SetSize(3); 

   for (int i = 0; i < psi_vals.Size(); ++i) { V(i) = tanh(psi_vals(i) / 2.); }
}

// NOTE: 2D ONLY 
void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(3);
   psi->GetVectorValue(T, ip, psi_vals);

   K.SetSize(3);
   K = 0.0;
   for (int i = 0; i < psi_vals.Size(); ++i) { K(i, i) = (1. - pow(tanh(psi_vals(i) / 2.), 2)) / 2.; }
}
