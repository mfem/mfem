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

public:
   DZCoefficient(GridFunction &psi_)
      : MatrixCoefficient(2),  psi(&psi_){ }

   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

int main(int argc, char *argv[]){ 
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

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::Type::QUADRILATERAL, false);
   const int dim = mesh.Dimension();
   const int sdim = mesh.SpaceDimension();
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // NOTE: incorrectly set all orders to the same initially
   RT_FECollection rtfec(order, dim); 
   L2_FECollection l2fec(order, sdim);
   FiniteElementSpace rtfes(&mesh, &rtfec); 
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

   int dof_rt(rtfes.GetTrueVSize()), dof_l2(l2fes_vec.GetTrueVSize());
   Array<int> offsets({0, dof_rt, dof_l2});
   offsets.PartialSum();
   BlockVector x(offsets), x_old(offsets), x_old_newt(offsets), rhs(offsets);
   BlockMatrix A(offsets);

   x = 0.; rhs = 0.; 

	GridFunction p(&rtfes, x.GetBlock(0)), vphi(&l2fes_vec, x.GetBlock(1)), psi(&l2fes_vec, x.GetBlock(2));
	GridFunction p_old(&rtfes, x_old.GetBlock(0)), vphi_old(&l2fes_vec, x_old.GetBlock(1)), psi_old(&l2fes_vec, x_old.GetBlock(2)); 
	GridFunction p_old_newt(&rtfes, x_old_newt.GetBlock(0)), vphi_old_newt(&l2fes_vec, x_old_newt.GetBlock(1)), psi_old_newt(&l2fes_vec, x_old_newt.GetBlock(2)); 

	// constants
	ConstantCoefficient one_cf(1.0), neg_one_cf(-1.0);
	ConstantCoefficient beta_cf(beta), alpha_cf(alpha); 
	VectorConstantCoefficient one_vec_cf(Vector({1., 1.})); 
   VectorConstantCoefficient alpha_vec_cf(Vector({alpha, alpha})); 


	// discrete coefficients 
   VectorGridFunctionCoefficient psi_old_cf(&psi_old), psi_cf(&psi), p_cf(&p), vphi_cf(&vphi); 


	// entropy coefficients
   ZCoefficient Z(psi);
   DZCoefficient DZ(psi);
   ScalarMatrixProductCoefficient negDZ(neg_one_cf, DZ); 

	VectorSumCoefficient psi_newton_res(psi_old_cf, psi_cf, 1., -1.);


	// bilinear forms and integrators 
	BilinearForm A00(&rtfes), A11(&l2fes_vec), A22(&l2fes_vec); 
	// MixedBilinearForm A10(&rtfes, &l2fes_vec), A20(&rtfes, &l2fes_vec);
	MixedBilinearForm A01(&l2fes_vec, &rtfes), A02(&l2fes_vec, &rtfes);
	

	A00.AddDomainIntegrator(new DivDivIntegrator(alpha_cf)); 
	A11.AddDomainIntegrator(new VectorMassIntegrator(neg_one_cf)); 
	// NOTE: the only diagonal element that gets updated every newton iteration is A22 
	A22.AddDomainIntegrator(new VectorMassIntegrator(negDZ)); 

	A01.AddDomainIntegrator(new MixedVectorProductIntegrator(alpha_vec_cf)); 
	A02.AddDomainIntegrator(new MixedVectorProductIntegrator(one_vec_cf)); 


	// NOTE: impose H(div) essential BC on A00 block
	A00.SetDiagonalPolicy(mfem::Operator::DIAG_ONE); 
	A00.Assemble(); 
	A00.EliminateEssentialBC(ess_tdof_list, x.GetBlock(0), rhs.GetBlock(0), mfem::Operator::DIAG_ONE); 
	A00.Finalize(true); 

	A11.Assemble(); 
	A11.Finalize(true); 

	A.SetBlock(0, 0, &A00.SpMat()); 
	A.SetBlock(1, 1, &A11.SpMat()); 

	// NOTE: impose H(div) essential BC ON A10, A01, A20, A02
	// eliminate TestDofs on A10, A20, which transpose into TrialDofs for A01, A02 
	A01.Assemble(); 
	A01.EliminateTestDofs(ess_tdof_list); 
	A01.Finalize(true); 

	A02.Assemble(); 
	A02.EliminateTestDofs(ess_tdof_list); 
	A02.Finalize(true); 


	// set up off-diagonal terms; unchanging during newton iterations 
	auto A10 = *Transpose(A01.SpMat()); 
	auto A20 = *Transpose(A02.SpMat()); 
	// std::unique_ptr<SparseMatrix> A01(Transpose(A10.SpMat()));  
	// std::unique_ptr<SparseMatrix> A02(Transpose(A20.SpMat()));  


	A.SetBlock(1, 0, &A01.SpMat());
	A.SetBlock(0, 1, &A10);


	A.SetBlock(2, 0, &A02.SpMat());
	A.SetBlock(0, 2, &A20);

	// linear forms 
	
	// LinearForm prox_res_lf(&rtfes, rhs.GetBlock(0).GetData()), psi_newton_res_lf(&l2fes_vec, rhs.GetBlock(2).GetData());
	
	LinearForm rhs0(&rtfes,  rhs.GetBlock(0).GetData()); 
	rhs0.AddDomainIntegrator(new VectorFEDomainLFIntegrator(psi_newton_res)); 
		//TODO: add \alpha \beta f 
	// rhs0.Assemble(); 					


	LinearForm rhs2(&l2fes_vec, rhs.GetBlock(2).GetData()); 
	rhs2.AddDomainIntegrator(new VectorDomainLFIntegrator(Z)); 

	// LinearForm Z_lf(&l2fes_vec, rhs.GetBlock(2).GetData()); 
	// Z_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(Z)); 
	

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

		while (newt_residual > newt_tol && newt_it < newt_max_it) { 
            std::cout << "\tIteration " << newt_it++ << ": ";
            x_old_newt = x;
				// update and re-assemble DZ's psi^k 
				A22.Assemble(false);
				A22.Finalize(false);  
				A.SetBlock(2, 2, &A22.SpMat()); 

				// No? 
				// Z_lf.Assemble(); 
				// assemble psi^k - psi^{k-1}
				// psi_newton_res_lf.Assemble(); 
				rhs0.Assemble(); 

				rhs0.SetSubVector(ess_tdof_list, 0.); // set essential BC on RHS

				SparseMatrix *A_mono = A.CreateMonolithic();
				UMFPackSolver umf(*A_mono);
				umf.Mult(rhs, x);

				const double newt_residual_psi = psi_old_newt.ComputeL2Error(psi_cf);
            const double newt_residual_p = p_old_newt.ComputeL2Error(p_cf); 
            const double newt_residual_vphi = vphi_old_newt.ComputeL2Error(vphi_cf); 

 				newt_residual = sqrt(  pow(newt_residual_p, 2)
                                 + pow(newt_residual_psi, 2) 
                                 + pow(newt_residual_vphi, 2)
										);

            cout << "Newton iteration residual" << newt_residual << endl; 
				
				A22.Update(); 
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

   for (int i = 0; i < psi_vals.Size(); ++i) { V(i) = tanh(psi_vals(i) / 2.); }
}

// NOTE: 2D ONLY 
void DZCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                         const IntegrationPoint &ip)
{
   MFEM_ASSERT(psi != NULL, "grid function is not set");

   Vector psi_vals(2);
   psi->GetVectorValue(T, ip, psi_vals);

   K = 0.;  
   for (int i = 0; i < 2; ++i) { K(i, i) = (1. - pow(tanh(psi_vals(i) / 2.), 2)) / 2.; }
}



