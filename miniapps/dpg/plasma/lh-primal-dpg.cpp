
// srun -n 256 ./pmaxwell-primal-tokamak -o 3 -sc -rnum
// srun -n 448 ./pmaxwell-primal-tokamak -o 4 -sc -rnum 11.0 -paraview

// srun -n 448 ./pmaxwell-primal-tokamak -o 4 -do 0 -sc -paraview (with the new epsilon GridFunction coefficients)
// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω^2 ϵ₀ ϵ E = 0 ,   in Ω
//                           E×n = E_0, on ∂Ω

// The primal-DPG formulation is obtained by integration by parts
// and the introduction of trace unknowns on the mesh skeleton

// in 3D
// E ∈ H(curl)
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h)
//  1/μ (∇×E , ∇×F) + (ω^2 ϵ₀ ϵ , F) + <Ê , F × n> = 0,      ∀ F ∈ H(curl,Ω)
//                                           Ê × n = E_0     on ∂Ω

#include "mfem.hpp"
#include "../util/utils.hpp"
#include "../util/maxwell_utils.hpp"
#include "utils/lh_utils.hpp"
#include "../util/pcomplexweakform.hpp"
#include "../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/LH_hot.msh";
   int order = 2;
   int delta_order = 1;
   int par_ref_levels = 0;
   int amr_ref_levels = 0;

   // real_t rnum=4.6e9;
   // real_t mu = 1.257e-6/factor;
   // real_t epsilon_scale = 8.8541878128e-12*factor;
   real_t rnum=1.5;
   real_t mu = 1.257;
   real_t eps0 = 8.8541878128;
   bool mumps_solver = false;
   real_t theta = 0.0;

   bool visualization = false;
   bool static_cond = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinement-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&amr_ref_levels, "-amr", "--parallel-amr-refinement-levels",
                  "Number of parallel AMR refinement levels.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
#ifdef MFEM_USE_MUMPS
   args.AddOption(&mumps_solver, "-mumps", "--mumps-solver", "-no-mumps",
                  "--no-mumps-solver", "Use the MUMPS Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }


   double omega = 2.*M_PI*rnum;

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.RemoveInternalBoundaries();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
   }

   FiniteElementCollection *E_fec = new ND_FECollection(order,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec);

   // H^-1/2 (curl) space for Ê (in 2D H1 trace, in 3D ND trace)
   int test_order = order+delta_order;
   FiniteElementCollection * hatE_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementCollection * F_fec = new ND_FECollection(test_order, dim);
   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(hatE_fes);
   test_fec.Append(F_fec);

   int gdofs = 0;
   for (int i = 0; i<trial_fes.Size(); i++)
   {
      gdofs += trial_fes[i]->GlobalTrueVSize();
   }

   if (Mpi::Root())
   {
      mfem::out << "Global number of dofs = " << gdofs << endl;
   }

   // Define the coefficients 
   ConstantCoefficient muinv(1./mu);
   ConstantCoefficient one(1.0);

   int nattr = (pmesh.attributes.Size()) ? pmesh.attributes.Max() : 0;
   Array<int> attr(nattr);
   for (int i = 0; i<nattr; i++) { attr[i] = i+1; }

   Vector zero(dim); zero = 0.0;
   Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_vcf(zero);
   VectorConstantCoefficient one_x_cf(one_x);
   VectorConstantCoefficient negone_x_cf(negone_x);

   DenseMatrix Mone(dim); 
   Mone = 0.0; Mone(0,0) = Mone(1,1) = 1.0;
   MatrixConstantCoefficient Mone_cf(Mone);
   DenseMatrix Mzero(dim); Mzero = 0.0;
   MatrixConstantCoefficient Mzero_cf(Mzero);

   Array<MatrixCoefficient*> coefs_r(nattr);
   Array<MatrixCoefficient*> coefs_i(nattr);
   for (int i = 0; i < nattr-1; ++i)
   {
      coefs_r[i] = &Mone_cf;
      coefs_i[i] = &Mzero_cf;
   }

   // S(r) 
   FunctionCoefficient S_cf_r(sfunc_r), S_cf_i(sfunc_i);
   // P(r) 
   FunctionCoefficient P_cf_r(pfunc_r), P_cf_i(pfunc_i); 

   VectorFunctionCoefficient b_cf(dim,bfunc);// b
   MatrixFunctionCoefficient bb_cf(dim,bcrossb); // b⊗b

   MatrixSumCoefficient oneminusbb(Mone_cf, bb_cf, 1.0, -1.0); // 1 - b⊗b

   // S(r) (I - b⊗b)
   ScalarMatrixProductCoefficient Soneminusbb_r(S_cf_r, oneminusbb), Soneminusbb_i(S_cf_i, oneminusbb); 

   // P(r) b⊗b 
   ScalarMatrixProductCoefficient P_cf_bb_r(P_cf_r, bb_cf), P_cf_bb_i(P_cf_i, bb_cf); 

   // εᵣ = S(r) (I - b⊗b) + P(r) b⊗b 
   MatrixSumCoefficient eps_r(Soneminusbb_r, P_cf_bb_r, 1.0, 1.0); 
   MatrixSumCoefficient eps_i(Soneminusbb_i, P_cf_bb_i, 1.0, 1.0); 

   coefs_r[nattr-1] = &eps_r;
   coefs_i[nattr-1] = &eps_i;

   PWMatrixCoefficient eps_cf_r(dim, attr, coefs_r);
   PWMatrixCoefficient eps_cf_i(dim, attr, coefs_i);

   ConstantCoefficient negeps0omeg2(-eps0 * omega * omega);

   ScalarMatrixProductCoefficient m_cf_r(negeps0omeg2, eps_cf_r);
   ScalarMatrixProductCoefficient m_cf_i(negeps0omeg2, eps_cf_i);

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);

   // (1/μ₀ ∇ × E,∇ × F)
   a->AddTrialIntegrator(new CurlCurlIntegrator(muinv), nullptr,0,0);

   // -(ω^2 ϵ₀ ϵ, F)
   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + 2;
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   const IntegrationRule &ir = IntRules.Get(pmesh.GetElementGeometry(0),
                                            2*test_order + 2);
   VectorFEMassIntegrator * integ_r = new VectorFEMassIntegrator(m_cf_r);
   VectorFEMassIntegrator * integ_i = new VectorFEMassIntegrator(m_cf_i);
   integ_r->SetIntegrationRule(ir);
   integ_i->SetIntegrationRule(ir);
   a->AddTrialIntegrator(integ_r,integ_i,0,0);

   // < n×Ê,F>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,1,0);

   // test integrators
   // (∇×F ,∇× δF)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
   // (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);

   socketstream E_out_r;
   socketstream E_theta_out_r;

   ParComplexGridFunction E_gf(E_fes);
   E_gf.real() = 0.0;
   E_gf.imag() = 0.0;

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);

   ParGridFunction E_theta_r(&L2_fes);
   ParGridFunction E_theta_i(&L2_fes);
   ParGridFunction E_theta(&L2_fes);
   E_theta = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;
   // Create ParaView directory and file
   std::string output_dir = "ParaView/PrimalDPG/" + GetTimestamp();
   if (Mpi::Root())
   {
      WriteParametersToFile(args, output_dir);
   }
   if (paraview)
   {
      std::ostringstream paraview_file_name;
      std::string filename = GetFilename(mesh_file);
      paraview_file_name << filename
                         << "_par_ref_" << par_ref_levels
                         << "_order_" << order;
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath(output_dir);
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_gf.real());
      paraview_dc->RegisterField("E_i",&E_gf.imag());
      paraview_dc->RegisterField("E_theta_r",&E_theta_r);
      paraview_dc->RegisterField("E_theta_i",&E_theta_i);
   }

   if (static_cond) { a->EnableStaticCondensation(); }

   a->Assemble();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   Array<int> one_r_bdr;
   Array<int> one_i_bdr;
   Array<int> negone_r_bdr;
   Array<int> negone_i_bdr;

   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_r_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_i_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;

      E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      one_r_bdr = 0;  one_i_bdr = 0;
      negone_r_bdr = 0;  negone_i_bdr = 0;

      // attr = 30,2 (real)
      one_r_bdr[30-1] = 1;  one_r_bdr[2-1] = 1;
      // attr = 26,6 (imag)
      one_i_bdr[26-1] = 1;  one_i_bdr[6-1] = 1;
      // attr = 22,10 (real)
      negone_r_bdr[22-1] = 1; negone_r_bdr[10-1] = 1;
      // attr = 18,14 (imag)
      negone_i_bdr[18-1] = 1; negone_i_bdr[14-1] = 1;
   }


   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = E_fes->GetVSize();
   offsets[2] = hatE_fes->GetVSize();
   offsets.PartialSum();

   Vector x(2*offsets.Last());
   x = 0.;
   double * xdata = x.GetData();

   E_gf.real().MakeRef(E_fes,&xdata[0]);
   E_gf.imag().MakeRef(E_fes,&xdata[offsets.Last()]);


   E_gf.ProjectBdrCoefficientTangent(one_x_cf,zero_vcf, one_r_bdr);
   E_gf.ProjectBdrCoefficientTangent(negone_x_cf,zero_vcf, negone_r_bdr);
   E_gf.ProjectBdrCoefficientTangent(zero_vcf,one_x_cf, one_i_bdr);
   E_gf.ProjectBdrCoefficientTangent(zero_vcf,negone_x_cf, negone_i_bdr);

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

   int num_blocks = BlockA_r->NumRowBlocks();
   Array<int> tdof_offsets(2*num_blocks+1);

   tdof_offsets[0] = 0;
   for (int i=0; i<num_blocks; i++)
   {
      const int h = BlockA_r->GetBlock(i,i).Height();
      tdof_offsets[i+1] = h;
      tdof_offsets[num_blocks+i+1] = h;
   }
   tdof_offsets.PartialSum();

   BlockOperator blockA(tdof_offsets);
   for (int i = 0; i<num_blocks; i++)
   {
      for (int j = 0; j<num_blocks; j++)
      {
         blockA.SetBlock(i,j,&BlockA_r->GetBlock(i,j));
         blockA.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
         blockA.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
         blockA.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
      }
   }

#ifdef MFEM_USE_MUMPS
   if (mumps_solver)
   {
      // Monolithic real part
      Array2D<const HypreParMatrix * > Ab_r(num_blocks,num_blocks);
      // Monolithic imag part
      Array2D<const HypreParMatrix * > Ab_i(num_blocks,num_blocks);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            Ab_r(i,j) = &(HypreParMatrix &)BlockA_r->GetBlock(i,j);
            Ab_i(i,j) = &(HypreParMatrix &)BlockA_i->GetBlock(i,j);
         }
      }
      HypreParMatrix * A_r = HypreParMatrixFromBlocks(Ab_r);
      HypreParMatrix * A_i = HypreParMatrixFromBlocks(Ab_i);

      ComplexHypreParMatrix Acomplex(A_r, A_i,true,true);

      HypreParMatrix * A = Acomplex.GetSystemMatrix();

      MUMPSSolver mumps(MPI_COMM_WORLD);
      mumps.SetPrintLevel(0);
      mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps.SetOperator(*A);
      mumps.Mult(B,X);
      delete A;
   }
#else
   if (mumps_solver)
   {
      MFEM_WARNING("MFEM compiled without mumps. Switching to an iterative solver");
   }
   mumps_solver = false;
#endif
   int num_iter = -1;
   if (!mumps_solver)
   {
      BlockDiagonalPreconditioner M(tdof_offsets);
      ParFiniteElementSpace *ams_fes = nullptr;
      if (static_cond)
      {
         ams_fes = new ParFiniteElementSpace(&pmesh,
                                             E_fes->FEColl()->GetTraceCollection());
      }

      HypreAMS * solver_E = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(0,0),
                                          (static_cond) ? ams_fes : E_fes);
      solver_E->SetPrintLevel(0);
      HypreBoomerAMG * solver_hatE = new HypreBoomerAMG((HypreParMatrix &)
                                                         BlockA_r->GetBlock(1,1));
      solver_hatE->SetPrintLevel(0);
      solver_hatE->SetRelaxType(88);

      M.SetDiagonalBlock(0,solver_E);
      M.SetDiagonalBlock(1,solver_hatE);
      M.SetDiagonalBlock(2,solver_E);
      M.SetDiagonalBlock(3,solver_hatE);

      if (myid == 0)
      {
         std::cout << "PCG iterations" << endl;
      }

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-8);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);

      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
      }

      num_iter = cg.GetNumIterations();

   }
   a->RecoverFEMSolution(X,x);

   E_gf.real().MakeRef(E_fes,x.GetData());
   E_gf.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

   AzimuthalECoefficient az_e_r(&E_gf.real());
   AzimuthalECoefficient az_e_i(&E_gf.imag());

   E_theta_r.ProjectCoefficient(az_e_r);
   E_theta_i.ProjectCoefficient(az_e_i);

   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E_gf.real(),
                              "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      common::VisualizeField(E_theta_out_r,vishost, visport, E_theta_r,
                              "Numerical Electric field (Azimuthal-real)", 501, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
   }

   delete a;
   delete F_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete E_fec;
   delete E_fes;

   return 0;
}

