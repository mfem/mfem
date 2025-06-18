#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// NOTE: 2D ONLY 
class ZCoefficient : public VectorCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   ZCoefficient(GridFunction &psi_)
      : VectorCoefficient(2), psi(&psi_) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

// NOTE: 2D ONLY  
class DZCoefficient : public MatrixCoefficient
{
protected:
   GridFunction *psi;
   real_t alpha;

public:
   DZCoefficient(GridFunction &psi_)
      : MatrixCoefficient(2),  psi(&psi_){ }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

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
   int order = 1;
   int max_it = 10;
   int ref_levels = 3;
   real_t alpha = 1.0;
   real_t beta = 1.0; 
   real_t tol = 1e-5;
   real_t growth_rate = 2.; 
   bool visualization = true;

   OptionsParser args(argc, argv);
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
                  "Step size alpha");
   args.AddOption(&growth_rate, "-gr", "--growth-rate",
               "Geometric step size growth rate, alpha = r**k");
   args.AddOption(&beta, "-reg", "--regularization",
                  "Image regularization term beta");
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

   // NOTE : pixel-like mesh? 
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::Type::QUADRILATERAL, false, 2., 2.);
   // NOTE: shift to [-1, 1]x[-1, 1]
   mesh.Transform([](const Vector &x, Vector &newx){newx=x; newx -= 1.; }); 
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // NOTE: incorrectly set all orders to the same initially
   RT_FECollection rtfec(order, dim); 
   H1_FECollection h1fec(order, sdim);
   L2_FECollection l2fec(order, sdim);
   FiniteElementSpace rtfes(&mesh, &rtfec); 
   FiniteElementSpace h1fes(&mesh, &h1fec);
   FiniteElementSpace l2fes_vec(&mesh, &l2fec, sdim);

   // NOTE: markers for 0 normal trace BC for RT elements; 
   // this is an essential BC for H(div) 
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      rtfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   int dof_rt(rtfes.GetTrueVSize()), dof_h1(h1fes.GetTrueVSize()), dof_l2(l2fes_vec.GetTrueVSize());

   Array<int> offsets({0, dof_rt, dof_h1, dof_l2, 1});
   offsets.PartialSum();
   BlockVector x(offsets), x_old(offsets), x_old_newt(offsets), rhs(offsets);
   BlockMatrix A(offsets);

   x = 0.;

   GridFunction p(&rtfes, x.GetBlock(0)); 
   GridFunction vphi(&h1fes, x.GetBlock(1)), 
                psi(&l2fes_vec, x.GetBlock(2));

   GridFunction p_old(&rtfes, x_old.GetBlock(0)); 
   GridFunction vphi_old(&h1fes, x_old.GetBlock(1)),
                psi_old(&l2fes_vec, x_old.GetBlock(2));

   GridFunction p_old_newt(&rtfes, x_old_newt.GetBlock(0)); 
   GridFunction vphi_old_newt(&h1fes, x_old_newt.GetBlock(1)),
                psi_old_newt(&l2fes_vec, x_old_newt.GetBlock(2));

   // Discrete coefficients
   VectorGridFunctionCoefficient psi_old_cf(&psi_old), psi_cf(&psi), p_cf(&p), vphi_cf(&vphi); 
   
   // entropy coefficients 
   VectorSumCoefficient psi_newton_res(psi_old_cf, psi_cf, 1., -1.);

   ZCoefficient Z(psi);
   DZCoefficient DZ(psi);

   // Other coefficients
   ConstantCoefficient one_cf(1.), neg_one_cf(-1.);
   ConstantCoefficient beta_cf(beta); 
   ConstantCoefficient alpha_cf(alpha); 

   // TODO: meaningful RHS? 
   VectorConstantCoefficient one_vec_cf(Vector({1., 1.})); 
   VectorConstantCoefficient alpha_vec_cf(Vector({alpha, alpha})); 
   ScalarVectorProductCoefficient beta_vec_cf(beta_cf, one_vec_cf); 

   ScalarMatrixProductCoefficient negDZ(neg_one_cf, DZ); 

   // bilinear forms 
   BilinearForm p_newton(&rtfes), vphi_newtonC(&h1fes), DZform(&l2fes_vec);
   MixedBilinearForm vphi_newton(&h1fes, &rtfes), psi_newton(&l2fes_vec, &rtfes);

   p_newton.AddDomainIntegrator(new DivDivIntegrator(alpha_cf)); 

   RotationCoefficient R;
   vphi_newton.AddDomainIntegrator(new MixedVectorGradientIntegrator(R)); 

   vphi_newtonC.AddDomainIntegrator(new DiffusionIntegrator(neg_one_cf)); 

   psi_newton.AddDomainIntegrator(new VectorMassIntegrator()); 

   DZform.AddDomainIntegrator(new VectorMassIntegrator(negDZ)); 

   // apply 0 essential boundary condition to H(div) space 
   p_newton.SetDiagonalPolicy(mfem::Operator::DIAG_ONE); 
   p_newton.Assemble(); 
   p_newton.EliminateEssentialBC(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0), mfem::Operator::DIAG_ONE); 
   p_newton.Finalize(true); 
   A.SetBlock(0, 0, &p_newton.SpMat()); 

   cout << "assembled A blocK" << endl; 

   vphi_newton.Assemble(); 
   // vphi_newton.EliminateTrialDofs(ess_tdof_list, x.GetBlock(1), rhs.GetBlock(1)); 

   // NOTE: eliminate H(div) essential BC for B^T block test functions <--> trial functions for B block 
   vphi_newton.EliminateTestDofs(ess_tdof_list); 
   vphi_newton.Finalize(true); 

   cout << "assembled vphi" << endl; 

   auto vphi_newtonT = *Transpose(vphi_newton.SpMat()); 
   A.SetBlock(1, 0, &vphi_newtonT); 
   A.SetBlock(0, 1, &vphi_newton.SpMat()); 
   
   vphi_newtonC.Assemble(); 
   cout << "C block" << endl; 
   vphi_newtonC.Finalize(true); 
   cout << "C block final" << endl; 

   A.SetBlock(1, 1, &vphi_newtonC.SpMat()); 

   DZform.Assemble(); 
   cout << "DZform" << endl; 
   DZform.Finalize(true); 
   A.SetBlock(2, 2, &DZform.SpMat()); 

   // average-value 0 constraint
   LinearForm avg0_data(&h1fes); 
   avg0_data.AddDomainIntegrator(new DomainLFIntegrator(one_cf)); 
   avg0_data.Assemble(); 
   Array<int> avg0_i({0, dof_h1}), avg0_j(dof_h1); 
   std::iota(avg0_j.begin(), avg0_j.end(), 0); 
   SparseMatrix avg0(
      avg0_i.GetData(), avg0_j.GetData(), avg0_data.GetData(), 1, dof_h1, false, false, true
   );
   auto avg0T = *Transpose(avg0); 

   A.SetBlock(3, 1, &avg0); 
   A.SetBlock(1, 3, &avg0T); 

   // linear forms 
   LinearForm image(&rtfes), prox_res_lf(&l2fes_vec, rhs.GetBlock(0).GetData()), psi_newton_lf(&l2fes_vec, rhs.GetBlock(2).GetData()); 
   
   image.AddDomainIntegrator(new VectorDomainLFIntegrator(beta_vec_cf)); // NOTE: currently just beta * 1. 

   prox_res_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(psi_newton_res)); 
   psi_newton_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(Z));

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost,visport);
      sol_sock.precision(8);
   }

   int prox_it = 0;
   double prox_residual = tol + 1;

   const double newt_tol = 1e-06;
   double newt_residual = newt_tol + 1;

   const int newt_max_it = 20;
   int newt_it = 0;

   // prox loop 
   while (prox_residual > tol && prox_it < max_it) { 
      prox_it++; 
      x_old = x; 
      cout << "prox res lf assemble" << endl; 
      prox_res_lf.Assemble(); 

      prox_res_lf.Add(-alpha, image);       

      newt_residual = newt_tol + 1; 
      newt_it = 0; 

      // newton solve for \delta \psi^k 
      while (newt_residual > newt_tol && newt_it < newt_max_it) { 
            std::cout << "\tIteration " << newt_it++ << ": ";
            x_old_newt = x;

            psi_newton.Assemble(false); 

            cout << "psinewt assemble" << endl; 
            // NOTE: eliminate H(div) essential BC for D^T block test functions <--> trial functions for D block 
            psi_newton.EliminateTestDofs(ess_tdof_list);
            cout << "psi newton elim test" << endl; 
            psi_newton.Finalize(false); 

            auto psi_newtonT = *Transpose(vphi_newton.SpMat()); 

            A.SetBlock(0, 2, &psi_newton.SpMat()); 
            A.SetBlock(2, 0, &psi_newtonT); 

            // TODO: add multiplication of new stepsize for A & BT blocks 

            DZform.Assemble(false); 
            DZform.Finalize(false); 

            prox_res_lf.Assemble(); 

            A.SetBlock(2, 2, &DZform.SpMat()); 

            SparseMatrix *A_mono = A.CreateMonolithic(); 
            UMFPackSolver umf(*A_mono); 
            umf.Mult(rhs, x); 

            const double newt_residual_psi = psi_old_newt.ComputeL2Error(psi_cf);
            const double newt_residual_p = p_old_newt.ComputeL2Error(p_cf); 
            const double newt_residual_vphi = vphi_old_newt.ComputeL2Error(vphi_cf); 

            newt_residual = sqrt(  pow(newt_residual_p, 2)
                                 + pow(newt_residual_psi, 2) 
                                 + pow(newt_residual_vphi, 2));

            cout << "Newton iteration residual" << newt_residual << endl; 

            psi_newton.Update(); 
      }
      const double prox_residual_psi = psi_old.ComputeL2Error(psi_cf);
      const double prox_residual_p = p_old.ComputeL2Error(p_cf); 
      const double prox_residual_vphi = vphi_old.ComputeL2Error(vphi_cf); 

      prox_residual = sqrt(  pow(prox_residual_p, 2)
                           + pow(prox_residual_psi, 2) 
                           + pow(prox_residual_vphi, 2));

      cout << "Prox Iteration: " << prox_it << ": " << prox_residual <<
                " (" << prox_residual_psi << ", " << prox_residual_p << ", " << prox_residual_vphi << ")" << endl;
      
      if (visualization)
      {
         sol_sock << "solution\n" << mesh << p << flush;

      }

      alpha *= growth_rate; 
      growth_rate = min(alpha, 1.e09); 
   }

}

// NOTE: 2D ONLY 
void ZCoefficient::Eval(Vector &V, ElementTransformation &T,
                        const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(2);
   psi->GetVectorValue(T, ip, psi_vals);

   for (int i = 0; i < psi_vals.Size(); ++i) { V(i) = tanh(psi_vals(i)); }
}

// NOTE: 2D ONLY 
void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(2);
   psi->GetVectorValue(T, ip, psi_vals);

   K = 0.;  
   for (int i = 0; i < 2; ++i) { K(i, i) = (1. - pow(tanh(psi_vals(i)), 2)) / 2.; }
}
