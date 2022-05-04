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

   omega = 2.*M_PI*rnum;


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

   // H^-1/2 space for Ê   
   FiniteElementCollection * hatE_fec = new ND_Trace_FECollection(order,dim);
   FiniteElementSpace *hatE_fes = new FiniteElementSpace(&mesh,hatE_fec);

   // H^-1/2 space for Ĥ  
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
//    ConstantCoefficient one(1.0);
//    ConstantCoefficient zero(0.0);
//    Vector vec0(dim); vec0 = 0.;
//    VectorConstantCoefficient vzero(vec0);
//    ConstantCoefficient negone(-1.0);
//    ConstantCoefficient omeg(omega);
//    ConstantCoefficient omeg2(omega*omega);
//    ConstantCoefficient negomeg(-omega);

//    // Normal equation weak formulation
//    Array<FiniteElementSpace * > trial_fes; 
//    Array<FiniteElementCollection * > test_fec; 

//    trial_fes.Append(p_fes);
//    trial_fes.Append(u_fes);
//    trial_fes.Append(hatp_fes);
//    trial_fes.Append(hatu_fes);

//    test_fec.Append(q_fec);
//    test_fec.Append(v_fec);

//    ComplexNormalEquations * a = new ComplexNormalEquations(trial_fes,test_fec);
//    a->StoreMatrices();

//    // i ω (p,q)
//    a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(omeg),0,0);

// // -(u , ∇ q)
//    a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(negone)),nullptr,1,0);

// // -(p, ∇⋅v)
//    a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),nullptr,0,1);

// //  i ω (u,v)
//    a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(omeg)),1,1);

// // < p̂, v⋅n>
//    a->AddTrialIntegrator(new NormalTraceIntegrator,nullptr,2,1);

// // < û,q >
//    a->AddTrialIntegrator(new TraceIntegrator,nullptr,3,0);

//    // for impedence condition (only on the boundary)
//    // TODO
//    // a->AddTrialIntegrator(new TraceIntegrator,nullptr,2,0);


// // test integrators 

//    //space-induced norm for H(div) × H1
//    // (∇q,∇δq)
//    a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,0,0);
//    // (q,δq)
//    a->AddTestIntegrator(new MassIntegrator(one),nullptr,0,0);
//    // (∇⋅v,∇⋅δv)
//    a->AddTestIntegrator(new DivDivIntegrator(one),nullptr,1,1);
//    // (v,δv)
//    a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

//    // additional integrators for the adjoint graph norm
//    if (adjoint_graph_norm)
//    {   
//       // -i ω (∇q,δv)
//       a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negomeg),0,1);
//       // i ω (v,∇ δq)
//       a->AddTestIntegrator(nullptr,new MixedVectorWeakDivergenceIntegrator(negomeg),1,0);
//       // ω^2 (v,δv)
//       a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2),nullptr,1,1);

//       // - i ω (∇⋅v,δq)   
//       a->AddTestIntegrator(nullptr,new VectorFEDivergenceIntegrator(negomeg),1,0);
//       // i ω (q,∇⋅v)   
//       a->AddTestIntegrator(nullptr,new MixedScalarWeakGradientIntegrator(negomeg),0,1);
//       // ω^2 (q,δq)
//       a->AddTestIntegrator(new MassIntegrator(omeg2),nullptr,0,0);
//    }

//    // RHS
//    FunctionCoefficient f_rhs_r(rhs_func_r);
//    FunctionCoefficient f_rhs_i(rhs_func_i);
//    a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs_r),new DomainLFIntegrator(f_rhs_i),0);
   
   
//    FunctionCoefficient hatpex_r(hatp_exact_r);
//    FunctionCoefficient hatpex_i(hatp_exact_i);
//    VectorFunctionCoefficient hatuex_r(dim,hatu_exact_r);
//    VectorFunctionCoefficient hatuex_i(dim,hatu_exact_i);
//    Array<int> elements_to_refine;

//    socketstream p_out_r;
//    socketstream p_out_i;
//    if (visualization)
//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
//       p_out_r.open(vishost, visport);
//       p_out_i.open(vishost, visport);
//    }

//    double res0 = 0.;
//    double err0 = 0.;
//    int dof0;
//    mfem::out << " Refinement |" 
//              << "    Dofs    |" 
//              << "  L2 Error  |" 
//              << " Relative % |" 
//              << "  Rate  |" 
//              << "  Residual  |" 
//              << "  Rate  |" << endl;
//    mfem::out << " --------------------"      
//              <<  "-------------------"    
//              <<  "-------------------"    
//              <<  "-------------------" << endl;   




//    for (int i = 0; i<ref; i++)
//    {
//       if (static_cond) { a->EnableStaticCondensation(); }
//       a->Assemble();

//       Array<int> ess_tdof_list;
//       Array<int> ess_bdr;
//       if (mesh.bdr_attributes.Size())
//       {
//          ess_bdr.SetSize(mesh.bdr_attributes.Max());
//          ess_bdr = 1;
//          // ess_bdr[1] = 0;
//          // ess_bdr[2] = 1;
//          hatp_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//          // hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//       }

//       // shift the ess_tdofs
//       for (int j = 0; j < ess_tdof_list.Size(); j++)
//       {
//          ess_tdof_list[j] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
//                         //   + hatp_fes->GetTrueVSize(); 
//       }

//       Array<int> offsets(5);
//       offsets[0] = 0;
//       offsets[1] = p_fes->GetVSize();
//       offsets[2] = u_fes->GetVSize();
//       offsets[3] = hatp_fes->GetVSize();
//       offsets[4] = hatu_fes->GetVSize();
//       offsets.PartialSum();

//       Vector x(2*offsets.Last());
//       x = 0.;
//       double * xdata = x.GetData();

//       ComplexGridFunction hatp_gf(hatp_fes);
//       hatp_gf.real().MakeRef(hatp_fes,&xdata[offsets[2]]);
//       hatp_gf.imag().MakeRef(hatp_fes,&xdata[offsets.Last()+ offsets[2]]);
//       hatp_gf.ProjectBdrCoefficient(hatpex_r,hatpex_i, ess_bdr);

//       // ComplexGridFunction hatu_gf(hatu_fes);
//       // hatu_gf.real().MakeRef(hatu_fes,&xdata[offsets[3]]);
//       // hatu_gf.imag().MakeRef(hatu_fes,&xdata[offsets.Last()+ offsets[3]]);
//       // hatu_gf.ProjectBdrCoefficientNormal(hatuex_r,hatuex_i, ess_bdr);

//       OperatorPtr Ah;
//       Vector X,B;
//       a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

//       ComplexOperator * Ahc = Ah.As<ComplexOperator>();

//       SparseMatrix * Ar = dynamic_cast<BlockMatrix *>(&Ahc->real())->CreateMonolithic();
//       SparseMatrix * Ai = dynamic_cast<BlockMatrix *>(&Ahc->imag())->CreateMonolithic();

//       ComplexSparseMatrix Ac(Ar,Ai,true,true);
//       SparseMatrix * A = Ac.GetSystemMatrix();

//       mfem::out << "Size of the linear system: " << A->Height() << std::endl;


//       UMFPackSolver umf(*A);
//       umf.Mult(B,X);

//       delete A;
//       a->RecoverFEMSolution(X,x);

//       Vector & residuals = a->ComputeResidual(x);
//       double residual = residuals.Norml2();


//       elements_to_refine.SetSize(0);
//       double max_resid = residuals.Max();
//       for (int iel = 0; iel<mesh.GetNE(); iel++)
//       {
//          if (residuals[iel] > theta * max_resid)
//          {
//             elements_to_refine.Append(iel);
//          }
//       }


//       ComplexGridFunction p(p_fes);
//       p.real().MakeRef(p_fes,x.GetData());
//       p.imag().MakeRef(p_fes,&x.GetData()[offsets.Last()]);

//       ComplexGridFunction pgf_ex(p_fes);
//       FunctionCoefficient p_ex_r(p_exact_r);
//       FunctionCoefficient p_ex_i(p_exact_i);
//       pgf_ex.ProjectCoefficient(p_ex_r, p_ex_i);

//       int dofs = X.Size()/2;

//       double p_err_r = p.real().ComputeL2Error(p_ex_r);
//       double p_err_i = p.imag().ComputeL2Error(p_ex_i);

//       double L2Error = sqrt(p_err_r*p_err_r + p_err_i*p_err_i);

//       double rate_err = (i) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
//       double rate_res = (i) ? dim*log(res0/residual)/log((double)dof0/dofs) : 0.0;

//       err0 = L2Error;
//       res0 = residual;
//       dof0 = dofs;

//       mfem::out << std::right << std::setw(11) << i << " | " 
//                 << std::setw(10) <<  dof0 << " | " 
//                 << std::setprecision(3) 
//                 << std::setw(10) << std::scientific <<  err0 << " | " 
//                 << std::setprecision(3) 
//                 << std::setw(10) << std::fixed <<  0.0 << " | " 
//                 << std::setprecision(2) 
//                 << std::setw(6) << std::fixed << rate_err << " | " 
//                 << std::setprecision(3) 
//                 << std::setw(10) << std::scientific <<  res0 << " | " 
//                 << std::setprecision(2) 
//                 << std::setw(6) << std::fixed << rate_res << " | " 
//                 << std::resetiosflags(std::ios::showbase)
//                 << std::setw(10) << std::scientific 
//                 << std::endl;

//       if (visualization)
//       {
//          p_out_r.precision(8);
//          p_out_r << "solution\n" << mesh << p.real() <<
//                   "window_title 'Real Numerical presure' "
//                   << flush;

//          p_out_i.precision(8);
//          p_out_i << "solution\n" << mesh << p.imag() <<
//                   "window_title 'Imag Numerical presure' "
//                   << flush;         
//       }

//       if (i == ref)
//          break;

//       mesh.GeneralRefinement(elements_to_refine,1,1);
//       for (int i =0; i<trial_fes.Size(); i++)
//       {
//          trial_fes[i]->Update(false);
//       }
//       a->Update();
//    }

//    delete a;
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

}

void E_exact_i(const Vector &x, Vector & E_i)
{

}

void H_exact_r(const Vector &x, Vector & H_r)
{

}

void H_exact_i(const Vector &x, Vector & H_r)
{

}



void  rhs_func_r(const Vector &x, Vector & J_r)
{

}

void  rhs_func_i(const Vector &x, Vector & J_i)
{

}


void curlE_exact_r(const Vector &x, Vector &curlE_r)
{

}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{

}

void curlH_exact_r(const Vector &x,Vector &curlH_r)
{

}

void curlH_exact_i(const Vector &x,Vector &curlH_i)
{

}


void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{

}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{

}


void hatE_exact_r(const Vector & X, Vector & hatE_r)
{

}

void hatE_exact_i(const Vector & X, Vector & hatE_i)
{

}


void hatH_exact_r(const Vector & X, Vector & hatH_r)
{

}

void hatH_exact_i(const Vector & X, Vector & hatH_i)
{

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

   E[0] = y * z * (1.0 - y) * (1.0 - z);
   E[1] = x * y * z * (1.0 - x) * (1.0 - z);
   E[2] = x * y * (1.0 - x) * (1.0 - y);
      
   curlE[0] = (1.0 - x) * x * (y*(2.0*z-3.0)+1.0);
   curlE[1] = 2.0*(1.0 - y)*y*(x-z);
   curlE[2] = (z-1)*z*(1.0+y*(2.0*x-3.0));
      
   curlcurlE[0] = 2.0 * y * (1.0 - y) - (2.0 * x - 3.0) * z * (1 - z);
   curlcurlE[1] = 2.0 * y * (x * (1.0 - x) + (1.0 - z) * z);
   curlcurlE[2] = 2.0 * y * (1.0 - y) + x * (3.0 - 2.0 * z) * (1.0 - x);

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
      E_i(i) = E[i].real();
      curlE_i(i) = curlE[i].real();
      curlcurlE_i(i) = curlcurlE[i].real();
   }
}  
