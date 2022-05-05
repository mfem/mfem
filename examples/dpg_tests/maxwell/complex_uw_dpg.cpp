//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make uw_dpg
//

//      ∇×(1/μ ∇×E) - ω^2 ϵ E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// First Order System

//  i ω μ H + ∇ × E = 0,   in Ω
// -i ω ϵ E + ∇ × H = J,   in Ω
//            E × n = E_0, on ∂Ω

// note: Ĵ = -iωJ
// UW-DPG:
// 
// E,H ∈ (L^2(Ω))^3 
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γ_h)  
//  i ω μ (H,F) + (E,∇ × F) + < n × Ê, F > = 0,      ∀ F ∈ H(curl,Ω)      
// -i ω ϵ (E,G) + (H,∇ × G) + < n × Ĥ, G > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                   Ê × n = E_0     on ∂Ω 

// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |  

// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);

void H_exact_r(const Vector &x, Vector & H_r);
void H_exact_i(const Vector &x, Vector & H_r);


void  rhs_func_r(const Vector &x, Vector & J_r);
void  rhs_func_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);
void curlH_exact_r(const Vector &x,Vector &curlH_r);
void curlH_exact_i(const Vector &x,Vector &curlH_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);

void hatE_exact_r(const Vector & X, Vector & hatE_r);
void hatE_exact_i(const Vector & X, Vector & hatE_i);

void hatH_exact_r(const Vector & X, Vector & hatH_r);
void hatH_exact_i(const Vector & X, Vector & hatH_i);

void maxwell_solution(const Vector & X, 
                      std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE);

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                      Vector &curlE_r, 
                      Vector &curlcurlE_r);

void maxwell_solution_i(const Vector & X, Vector &E_i, 
                      Vector &curlE_i, 
                      Vector &curlcurlE_i);     

int dim;
double omega;
double mu = 1.0;
double epsilon = 1.0;

enum prob_type
{
   polynomial,
   plane_wave,
   fichera_oven  
};

prob_type prob;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../../data/inline-hex.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;
   double theta = 0.0;
   bool adjoint_graph_norm = false;
   bool static_cond = false;
   int iprob = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");    
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");                  
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: polynomial, 1: plane wave, 2: Gaussian beam");                     
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                    
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");                                
   args.AddOption(&ref, "-ref", "--serial_ref",
                  "Number of serial refinements.");       
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");                                            
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (iprob > 2) { iprob = 0; }
   prob = (prob_type)iprob;

   // omega = 2.*M_PI*rnum;
   omega = 1.0;


   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   
   MFEM_VERIFY(dim == 3, "Only 3D maxwell implemented for now");

   // Define spaces
   // L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *E_fes = new FiniteElementSpace(&mesh,E_fec,dim);

   // Vector L2 space for H 
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *H_fes = new FiniteElementSpace(&mesh,H_fec, dim); 

   // H^-1/2 (curl) space for Ê   
   FiniteElementCollection * hatE_fec = new ND_Trace_FECollection(order,dim);
   FiniteElementSpace *hatE_fes = new FiniteElementSpace(&mesh,hatE_fec);

   // H^-1/2 (curl) space for Ĥ  
   FiniteElementCollection * hatH_fec = new ND_Trace_FECollection(order,dim);   
   FiniteElementSpace *hatH_fes = new FiniteElementSpace(&mesh,hatH_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * F_fec = new ND_FECollection(test_order, dim);
   FiniteElementCollection * G_fec = new ND_FECollection(test_order, dim);


   mfem::out << "E_fes space true dofs = " << E_fes->GetTrueVSize() << endl;
   mfem::out << "H_fes space true dofs = " << H_fes->GetTrueVSize() << endl;
   mfem::out << "hatE_fes space true dofs = " << hatE_fes->GetTrueVSize() << endl;
   mfem::out << "hatH_fes space true dofs = " << hatH_fes->GetTrueVSize() << endl;


//    // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);

   // Normal equation weak formulation
   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);

   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   ComplexNormalEquations * a = new ComplexNormalEquations(trial_fes,test_fec);
   a->StoreMatrices();

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new CurlIntegrator(one)),nullptr,0,0);

   // -i ω ϵ (E , G)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg)),0,1);

   // i ω μ (H, F)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(muomeg)),1,0);

   //  (H,∇ × G) 
   a->AddTrialIntegrator(new TransposeIntegrator(new CurlIntegrator(one)),nullptr,1,1);

   // < n×Ê,F>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,2,0);

   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);


// test integrators 

   //space-induced norm for H(curl) × H(curl)
   // (∇×F,∇×δF)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
   // (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,1,1);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

   // additional integrators for the adjoint graph norm
   if (adjoint_graph_norm)
   {   
      // ϵ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2),nullptr,0,0);
      // -i ω ϵ (F,∇ × δG)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(negepsomeg),0,1);
      // -i ω μ  (∇ × F, δG)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(negmuomeg),0,1);
      // i ω ϵ (∇ × G,δF)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(muomeg),1,0);
      // i ω μ (G, ∇ × δF )
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(epsomeg),1,0);
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(mu2omeg2),nullptr,1,1);
   }

   // RHS
   VectorFunctionCoefficient f_rhs_r(dim,rhs_func_r);
   VectorFunctionCoefficient f_rhs_i(dim,rhs_func_i);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_r),
                            new VectorFEDomainLFIntegrator(f_rhs_i),1);
   
   
   VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
   VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);
   VectorFunctionCoefficient hatHex_r(dim,hatH_exact_r);
   VectorFunctionCoefficient hatHex_i(dim,hatH_exact_i);
   Array<int> elements_to_refine;

   socketstream E_out_r;
   socketstream E_out_i;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      E_out_r.open(vishost, visport);
      E_out_i.open(vishost, visport);
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0;
   mfem::out << " Refinement |" 
             << "    Dofs    |" 
             << "  L2 Error  |" 
             << " Relative % |" 
             << "  Rate  |" 
             << "  Residual  |" 
             << "  Rate  |" << endl;
   mfem::out << " --------------------"      
             <<  "-------------------"    
             <<  "-------------------"    
             <<  "-------------------" << endl;   



   ref = 1;
   for (int i = 0; i<ref; i++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (mesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
                           // + hatE_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = hatE_fes->GetVSize();
      offsets[4] = hatH_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;
      double * xdata = x.GetData();

      ComplexGridFunction hatE_gf(hatE_fes);
      hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
      hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);
      hatE_gf.ProjectBdrCoefficientTangent(hatEex_r,hatEex_i, ess_bdr);

      // ComplexGridFunction hatu_gf(hatH_fes);
      // hatu_gf.real().MakeRef(hatH_fes,&xdata[offsets[3]]);
      // hatu_gf.imag().MakeRef(hatH_fes,&xdata[offsets.Last()+ offsets[3]]);
      // hatu_gf.ProjectBdrCoefficientNormal(hatHex_r,hatHex_i, ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      SparseMatrix * Ar = dynamic_cast<BlockMatrix *>(&Ahc->real())->CreateMonolithic();
      SparseMatrix * Ai = dynamic_cast<BlockMatrix *>(&Ahc->imag())->CreateMonolithic();

      ComplexSparseMatrix Ac(Ar,Ai,true,true);
      SparseMatrix * A = Ac.GetSystemMatrix();

      mfem::out << "Size of the linear system: " << A->Height() << std::endl;


      UMFPackSolver umf(*A);
      umf.Mult(B,X);

      delete A;
      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);
      double residual = residuals.Norml2();


      elements_to_refine.SetSize(0);
      double max_resid = residuals.Max();
      for (int iel = 0; iel<mesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * max_resid)
         {
            elements_to_refine.Append(iel);
         }
      }


      ComplexGridFunction E(E_fes);
      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      VectorFunctionCoefficient E_ex_r(dim,E_exact_r);
      VectorFunctionCoefficient E_ex_i(dim,E_exact_i);

      int dofs = X.Size()/2;

      double E_err_r = E.real().ComputeL2Error(E_ex_r);
      double E_err_i = E.imag().ComputeL2Error(E_ex_i);

      double L2Error = sqrt(E_err_r*E_err_r + E_err_i*E_err_i);

      double rate_err = (i) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
      double rate_res = (i) ? dim*log(res0/residual)/log((double)dof0/dofs) : 0.0;

      err0 = L2Error;
      res0 = residual;
      dof0 = dofs;

      mfem::out << std::right << std::setw(11) << i << " | " 
                << std::setw(10) <<  dof0 << " | " 
                << std::setprecision(3) 
                << std::setw(10) << std::scientific <<  err0 << " | " 
                << std::setprecision(3) 
                << std::setw(10) << std::fixed <<  0.0 << " | " 
                << std::setprecision(2) 
                << std::setw(6) << std::fixed << rate_err << " | " 
                << std::setprecision(3) 
                << std::setw(10) << std::scientific <<  res0 << " | " 
                << std::setprecision(2) 
                << std::setw(6) << std::fixed << rate_res << " | " 
                << std::resetiosflags(std::ios::showbase)
                << std::setw(10) << std::scientific 
                << std::endl;

      if (visualization)
      {
         E_out_r.precision(8);
         E_out_r << "solution\n" << mesh << E.real() <<
                  "window_title 'Real Numerical Electric field' "
                  << flush;

         E_out_i.precision(8);
         E_out_i << "solution\n" << mesh << E.imag() <<
                  "window_title 'Imag Numerical Electric field' "
                  << flush;         

         ComplexGridFunction Egf_ex(E_fes);
         Egf_ex.ProjectCoefficient(E_ex_r, E_ex_i);


         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream E_r_sock(vishost, visport);
         E_r_sock.precision(8);
         E_r_sock << "solution\n" << mesh << Egf_ex.real()  
                  << "window_title 'Real Exact Electric field' " 
                  << flush;
         socketstream E_i_sock(vishost, visport);
         E_i_sock.precision(8);
         E_i_sock << "solution\n" << mesh << Egf_ex.imag()  
                  << "window_title 'Imag Exact Electric field' " 
                  << flush;


      }

      if (i == ref-1)
         break;

      mesh.GeneralRefinement(elements_to_refine,1,1);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   delete a;
   delete F_fec;
   delete G_fec;
   delete hatH_fes;
   delete hatH_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete H_fec;
   delete E_fec;
   delete H_fes;
   delete E_fes;

   return 0;
}                                       

void E_exact_r(const Vector &x, Vector & E_r)
{
   Vector curlE_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   Vector curlE_i;
   Vector curlcurlE_i;

   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   Vector E_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   Vector E_i;
   Vector curlcurlE_i;

   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{
   Vector E_r;
   Vector curlE_r;
   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{
   Vector E_i;
   Vector curlE_i;
   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}


void H_exact_r(const Vector &x, Vector & H_r)
{
   // H = i ∇ × E / ω μ  
   // H_r = - ∇ × E_i / ω μ  
   Vector curlE_i;
   curlE_exact_i(x,curlE_i);
   H_r.SetSize(3);
   for (int i = 0; i<3; i++)
   {
      H_r(i) = - curlE_i(i) / (omega * mu);
   }
}

void H_exact_i(const Vector &x, Vector & H_i)
{
   // H = i ∇ × E / ω μ  
   // H_i =  ∇ × E_r / ω μ  
   Vector curlE_r;
   curlE_exact_r(x,curlE_r);
   H_i.SetSize(3);
   for (int i = 0; i<3; i++)
   {
      H_i(i) = curlE_r(i) / (omega * mu);
   }
}


void curlH_exact_r(const Vector &x,Vector &curlH_r)
{
   // ∇ × H_r = - ∇ × ∇ × E_i / ω μ  
   Vector curlcurlE_i;
   curlcurlE_exact_i(x,curlcurlE_i);
   curlH_r.SetSize(3);
   for (int i = 0; i<3; i++)
   {
      curlH_r(i) = -curlcurlE_i(i) / (omega * mu);
   }
}

void curlH_exact_i(const Vector &x,Vector &curlH_i)
{
   // ∇ × H_i = ∇ × ∇ × E_r / ω μ  
   Vector curlcurlE_r;
   curlcurlE_exact_r(x,curlcurlE_r);
   curlH_i.SetSize(3);
   for (int i = 0; i<3; i++)
   {
      curlH_i(i) = -curlcurlE_r(i) / (omega * mu);
   }
}


void hatE_exact_r(const Vector & x, Vector & hatE_r)
{
   E_exact_r(x,hatE_r);
}

void hatE_exact_i(const Vector & x, Vector & hatE_i)
{
   E_exact_i(x,hatE_i);
}


void hatH_exact_r(const Vector & x, Vector & hatH_r)
{
   H_exact_r(x,hatH_r);
}

void hatH_exact_i(const Vector & x, Vector & hatH_i)
{
   H_exact_i(x,hatH_i);
}


// J = -i ω ϵ E + ∇ × H 
// J_r + iJ_i = -i ω ϵ (E_r + i E_i) + ∇ × (H_r + i H_i) 
void  rhs_func_r(const Vector &x, Vector & J_r)
{
   // J_r = ω ϵ E_i + ∇ × H_r
   Vector E_i, curlH_r;
   E_exact_i(x,E_i);
   curlH_exact_r(x,curlH_r);
   J_r.SetSize(3);
   for (int i = 0; i<3; i++)
   {
      J_r(i) = omega * epsilon * E_i(i) + curlH_r(i);
   }
}

void  rhs_func_i(const Vector &x, Vector & J_i)
{
   // J_i = - ω ϵ E_r + ∇ × H_i
   Vector E_r, curlH_i;
   E_exact_r(x,E_r);
   curlH_exact_i(x,curlH_i);
   J_i.SetSize(3);
   for (int i = 0; i<3; i++)
   {
      J_i(i) = -omega * epsilon * E_r(i) + curlH_i(i);
   }
}


void maxwell_solution(const Vector & X, std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE)
{
   double x = X(0);
   double y = X(1);
   double z = X(2);

   E.resize(3);
   curlE.resize(3);
   curlcurlE.resize(3);

   // E[0] = y * z * (1.0 - y) * (1.0 - z);
   // E[1] = x * y * z * (1.0 - x) * (1.0 - z);
   // E[2] = x * y * (1.0 - x) * (1.0 - y);
      
   // curlE[0] = (1.0 - x) * x * (y*(2.0*z-3.0)+1.0);
   // curlE[1] = 2.0*(1.0 - y)*y*(x-z);
   // curlE[2] = (z-1)*z*(1.0+y*(2.0*x-3.0));
      
   // curlcurlE[0] = 2.0 * y * (1.0 - y) - (2.0 * x - 3.0) * z * (1 - z);
   // curlcurlE[1] = 2.0 * y * (x * (1.0 - x) + (1.0 - z) * z);
   // curlcurlE[2] = 2.0 * y * (1.0 - y) + x * (3.0 - 2.0 * z) * (1.0 - x);

   E[0] = y;
   E[1] = z*x;
   E[2] = x;
      
   curlE[0] = -x;
   curlE[1] = -1.0;
   curlE[2] = z-1;
      
   curlcurlE[0] = 0.0;
   curlcurlE[1] = 0.0;
   curlcurlE[2] = 0.0;

   // E[0] = y*y;
   // E[1] = 0.0;
   // E[2] = 0.0;
      
   // curlE[0] =  0.0;
   // curlE[1] =  0.0;
   // curlE[2] = -2.0*y;
      
   // curlcurlE[0] = -2.0;
   // curlcurlE[1] = 0.0;
   // curlcurlE[2] = 0.0;

}


void maxwell_solution_r(const Vector & X, Vector &E_r, 
                        Vector &curlE_r, 
                        Vector &curlcurlE_r)
{
   E_r.SetSize(3);
   curlE_r.SetSize(3);
   curlcurlE_r.SetSize(3);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<3; i++)
   {
      E_r(i) = E[i].real();
      curlE_r(i) = curlE[i].real();
      curlcurlE_r(i) = curlcurlE[i].real();
   }
}


void maxwell_solution_i(const Vector & X, Vector &E_i, 
                      Vector &curlE_i, 
                      Vector &curlcurlE_i)
{
   E_i.SetSize(3);
   curlE_i.SetSize(3);
   curlcurlE_i.SetSize(3);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<3; i++)
   {
      E_i(i) = E[i].imag();
      curlE_i(i) = curlE[i].imag();
      curlcurlE_i(i) = curlcurlE[i].imag();
   }
}  
