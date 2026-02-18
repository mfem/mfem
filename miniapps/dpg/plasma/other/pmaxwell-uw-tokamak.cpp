
// srun -n 512 ./pmaxwell-uw-tokamak -o 4 -sc -prob 2 -paraview
// srun -n 512 ./pmaxwell-uw-tokamak -o 4 -sc -prob 4 -paraview

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - (ω^2 ϵ + i ω σ) E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// The DPG UW deals with the First Order System
//        i ω μ H + ∇ × E = 0,   in Ω (Faraday's law)
//            M E + ∇ × H = J,   in Ω (Ampere's law)
//            E × n = E_0, on ∂Ω
// Note: Ĵ = -iωJ
// where M = -(i ω ϵI + σI)

// The ultraweak-DPG formulation is obtained by integration by parts of both
// equations and the introduction of trace unknowns on the mesh skeleton

// in 3D
// E,H ∈ (L^2(Ω))^3
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γ_h)
//  i ω μ (H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)
//      M (E,G) + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                   Ê × n = E_0     on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G |    (ME,G)    |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// Here we use the "Adjoint Graph" norm on the test space i.e.,
// ||(F,G)||^2_V = ||A^*(F,G)||^2 + ||(F,G)||^2 where A is the
// maxwell operator defined by (1)


#include "mfem.hpp"
#include "../../util/pcomplexweakform.hpp"
#include "../../util/preconditioners.hpp"
#include "../../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class EpsilonMatrixCoefficient : public MatrixArrayCoefficient
{
private:
   Mesh * mesh = nullptr;
   ParMesh * pmesh = nullptr;
   Array<ParGridFunction * > pgfs;
   Array<GridFunctionCoefficient * > gf_cfs;
   GridFunction * vgf = nullptr;
   ParFiniteElementSpace * pfes = nullptr;
   int dim;
public:
   EpsilonMatrixCoefficient(const char * filename, Mesh * mesh_, ParMesh * pmesh_,
                            real_t scale = 1.0)
      : MatrixArrayCoefficient(mesh_->Dimension()), mesh(mesh_), pmesh(pmesh_),
        dim(mesh_->Dimension())
   {
      std::filebuf fb;
      fb.open(filename,std::ios::in);
      std::istream is(&fb);
      vgf = new GridFunction(mesh,is);
      fb.close();
      FiniteElementSpace * vfes = vgf->FESpace();
      int vdim = vfes->GetVDim();
      const FiniteElementCollection * fec = vfes->FEColl();
      FiniteElementSpace * fes = new FiniteElementSpace(mesh, fec);
      int num_procs = Mpi::WorldSize();
      int * partitioning = mesh->GeneratePartitioning(num_procs);
      real_t *data = vgf->GetData();
      GridFunction gf;
      pfes = new ParFiniteElementSpace(pmesh, fec);
      pgfs.SetSize(vdim);
      gf_cfs.SetSize(vdim);
      for (int i = 0; i<dim; i++)
      {
         for (int j = 0; j<dim; j++)
         {
            int k = i*dim+j;
            // int k = j*dim+i;
            gf.MakeRef(fes,&data[k*fes->GetVSize()]);
            pgfs[k] = new ParGridFunction(pmesh,&gf,partitioning);
            (*pgfs[k])*=scale;
            gf_cfs[k] = new GridFunctionCoefficient(pgfs[k]);
            Set(i,j,gf_cfs[k], true);
         }
      }
      delete fes;
   }

   void Update()
   {
      pfes->Update();
      for (int k = 0; k<pgfs.Size(); k++)
      {
         pgfs[k]->Update();
      }
   }

   ~EpsilonMatrixCoefficient()
   {
      for (int i = 0; i<pgfs.Size(); i++)
      {
         delete pgfs[i];
      }
      pgfs.DeleteAll();
      delete pfes;
   }
};


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int order = 1;
   int delta_order = 1;
   bool visualization = false;
   real_t rnum=5.0;
   bool static_cond = false;
   int pr = 0;
   bool paraview = false;
   real_t mu = 1.0;
   real_t epsilon = 1.0;
   real_t sigma = 0.1;
   bool graph_norm = true;
   int prob = 0;
   bool pmg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&sigma, "-sigma", "--sigma",
                  "conductivity");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&prob, "-prob", "--problem",
                  "Choice of problem: 0: tet mesh eps = 1, 1: hex mesh eps = 1 ,  2: tet-mesh eps from file,", 
                  "3: hex-mesh eps from file, 4: 400k-tet-mesh eps from file");
   args.AddOption(&pmg, "-pmg", "--p-refinement-multigrid", "-no-pmg",
                  "--no-p-refinement-multigrid", "Enable P-Refinement Multigrid.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&graph_norm, "-graph", "--graph-norm", "-no-gn",
                  "--no-graph-norm", "Enable adjoint graph norm.");
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


   const char *mesh_file = nullptr;
   if (prob == 0 || prob == 2)
   {
      mesh_file = "meshes/tokamak-tet.mesh";
      if (prob == 2)
      {
         rnum=50.0e6;
         mu = 1.257e-6;
         sigma = 0.0;
      }
   }
   else if (prob == 1 || prob == 3)
   {
      mesh_file = "meshes/tokamak-hex.mesh";
      if (prob == 3)
      {
         rnum=50.0e6;
         mu = 1.257e-6;
         sigma = 0.0;
      }
   }
   else if (prob == 4)
   {
      mesh_file = "meshes/tokamak-400K-tet.mesh";
      rnum = 54.0e6;
      mu = 1.257e-6;
      sigma = 0.0;
   }
   else
   {
      MFEM_ABORT("No valid problem choice given");
   }


   real_t omega = 2.*M_PI*rnum;


   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   DenseMatrix mat_eps_i(dim); mat_eps_i = 0.0;
   DenseMatrix mat_eps_r(dim); mat_eps_r = 0.0;
   mat_eps_r(0,0) = epsilon;
   mat_eps_r(1,1) = epsilon;
   mat_eps_r(2,2) = epsilon;

   DenseMatrix I(dim); I = 0.0;
   I(0,0) = 1.0;
   I(1,1) = 1.0;
   I(2,2) = 1.0;
   MatrixConstantCoefficient identity_cf(I);


   MatrixCoefficient * eps_r_cf = nullptr;
   MatrixCoefficient * eps_i_cf = nullptr;
   if (prob == 0 || prob == 1)
   {
      eps_r_cf = new MatrixConstantCoefficient(mat_eps_r);
      eps_i_cf = new MatrixConstantCoefficient(mat_eps_i);
   }
   else
   {
      const char * eps_r_file = nullptr;
      const char * eps_i_file = nullptr;
      if (prob == 2)
      {
         eps_r_file = "data/tet-eps_r.gf";
         eps_i_file = "data/tet-eps_i.gf";
      }
      else if (prob == 3)
      {
         eps_r_file = "data/hex-eps_r.gf";
         eps_i_file = "data/hex-eps_i.gf";
      }
      else if (prob == 4)
      {
         eps_r_file = "data/tet-400k-eps_r.gf";
         eps_i_file = "data/tet-400k-eps_i.gf";
      }
      real_t epsilon_scale = 8.8541878128e-12;
      eps_r_cf = new EpsilonMatrixCoefficient(eps_r_file,&mesh,&pmesh, epsilon_scale);
      eps_i_cf = new EpsilonMatrixCoefficient(eps_i_file,&mesh,&pmesh, epsilon_scale);
   }
   mesh.Clear();

   for (int i = 0; i<pr; i++)
   {
      pmesh.UniformRefinement();
      dynamic_cast<EpsilonMatrixCoefficient *>(eps_r_cf)->Update();
      dynamic_cast<EpsilonMatrixCoefficient *>(eps_i_cf)->Update();
   }


   int ne = pmesh.GetNE();
   MPI_Allreduce(MPI_IN_PLACE,&ne,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   if (myid == 0)
   {
      mfem::out << "Number of elements = " << ne << endl;
   }

   // Matrix Coefficient (M = -i\omega \epsilon - \sigma I);
   // M = -i * omega * (eps_r + i eps_i) - sigmaI
   //  = omega eps_i - sigma I + i (-omega eps_r)
   MatrixSumCoefficient Mr_cf(*eps_i_cf,identity_cf,omega,-sigma);
   ScalarMatrixProductCoefficient Mi_cf(-omega,*eps_r_cf);

   // Define spaces
   // L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dim);

   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = nullptr;
   FiniteElementCollection * hatH_fec = nullptr;
   FiniteElementCollection * F_fec = nullptr;
   int test_order = order+delta_order;
   hatE_fec = new ND_Trace_FECollection(order,dim);
   hatH_fec = new ND_Trace_FECollection(order,dim);
   F_fec = new ND_FECollection(test_order, dim);

   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);
   ParFiniteElementSpace *hatH_fes = new ParFiniteElementSpace(&pmesh,hatH_fec);
   FiniteElementCollection * G_fec = new ND_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);
   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   int gdofs = 0;
   for (int i = 0; i<trial_fes.Size(); i++)
   {
      gdofs += trial_fes[i]->GlobalTrueVSize();
   }

   if (myid == 0)
   {
      mfem::out << "Global number of dofs = " << gdofs << endl;
   }

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);
   ConstantCoefficient sigma_cf(sigma);
   ConstantCoefficient negsigma_cf(-sigma);
   ConstantCoefficient sigma2_cf(sigma*sigma);

   Vector z_one(3); z_one = 0.0; z_one(2) = 1.0;
   Vector zero(3); zero = 0.0;
   Vector z_negone(3); z_negone = 0.0; z_negone(2) = -1.0;
   VectorConstantCoefficient z_one_cf(z_one);
   VectorConstantCoefficient z_negone_cf(z_negone);
   VectorConstantCoefficient zero_cf(zero);

   if (myid == 0)
   {
      std::cout << "Assembling matrix" << endl;
   }

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,0,0);


   // --------------------------------------------------------------------------
   //  (M E , G) = (M_r E, G) + i (M_i E, G)
   a->AddTrialIntegrator(
      new TransposeIntegrator(new VectorFEMassIntegrator(Mr_cf)),
      new TransposeIntegrator(new VectorFEMassIntegrator(Mi_cf)),0,1);
   // --------------------------------------------------------------------------

   //  (H,∇ × G)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,1,1);
   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);
   // test integrators
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,1,1);

   ConstantCoefficient l2weight(1.0);

   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(l2weight),nullptr,1,1);

   // i ω μ (H, F)
   a->AddTrialIntegrator(nullptr,
                         new TransposeIntegrator(new VectorFEMassIntegrator(muomeg)),1,0);
   // < n×Ê,F>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,2,0);
   // test integrators
   // (∇×F,∇×δF)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
   // (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);

   if (graph_norm)
   {
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(mu2omeg2),nullptr,0,0);
      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(negmuomeg),0,1);

      // --------------------------------------------------------------------------
      // (M ∇ × F, δG) = (M_r  ∇ × F, δG) + i (M_i  ∇ × F, δG)
      a->AddTestIntegrator(new MixedVectorCurlIntegrator(Mr_cf),
                           new MixedVectorCurlIntegrator(Mi_cf),0,1);
      // --------------------------------------------------------------------------

      // i ω μ (∇ × G,δF)
      a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(muomeg),1,0);
   }
   // --------------------------------------------------------------------------
   // (M^* G, ∇ × δF ) = (Mr^T G,  ∇ × δF) - i (Mi^T G,  ∇ × δF)
   TransposeMatrixCoefficient Mrt_cf(Mr_cf);
   TransposeMatrixCoefficient Mit_cf(Mi_cf);
   ScalarMatrixProductCoefficient negMit_cf(-1.0,Mit_cf);
   if (graph_norm)
   {
      a->AddTestIntegrator(new MixedVectorWeakCurlIntegrator(Mrt_cf),
                           new MixedVectorWeakCurlIntegrator(negMit_cf),1,0);
   }
   // --------------------------------------------------------------------------

   // M*M^*(G,δG) = (MrMr^t + MiMi^t) + i (MiMr^t - MrMi^t)
   MatrixProductCoefficient MrMrt_cf(Mr_cf,Mrt_cf);
   MatrixProductCoefficient MiMit_cf(Mi_cf,Mit_cf);
   MatrixProductCoefficient MiMrt_cf(Mi_cf,Mrt_cf);
   MatrixProductCoefficient MrMit_cf(Mr_cf,Mit_cf);

   MatrixSumCoefficient MMr_cf(MrMrt_cf,MiMit_cf);
   MatrixSumCoefficient MMi_cf(MiMrt_cf,MrMit_cf,1.0,-1.0);

   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + 2;
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   const IntegrationRule &ir = IntRules.Get(pmesh.GetElementGeometry(0),
                                            2*test_order + 2);

   VectorFEMassIntegrator * integ_r = new VectorFEMassIntegrator(MMr_cf);
   integ_r->SetIntegrationRule(ir);
   VectorFEMassIntegrator * integ_i = new VectorFEMassIntegrator(MMi_cf);
   integ_i->SetIntegrationRule(ir);

   if (graph_norm)
   {
      a->AddTestIntegrator(integ_r,integ_i,1,1);
   }
   // --------------------------------------------------------------------------


   socketstream E_out_r;
   socketstream H_out_r;

   ParComplexGridFunction E(E_fes);
   ParComplexGridFunction H(H_fes);
   E.real() = 0.0;
   E.imag() = 0.0;
   H.real() = 0.0;
   H.imag() = 0.0;

   ParGridFunction Et(E_fes);
   ParGridFunction Ht(H_fes);

   ParaViewDataCollection * paraview_dc = nullptr;
   ParaViewDataCollection * paraview_dct = nullptr;
   std::ostringstream paraview_file_name;
   paraview_file_name << "prob_" << prob
                      << "_order_" << order
                      << "_pref_" << pr
                      << "_sc_" << (int)static_cond;

   std::ostringstream paraview_file_namet;
   paraview_file_namet << "tim-harmonin-prob_" << prob
                       << "_order_" << order
                       << "_pref_" << pr
                       << "_sc_" << (int)static_cond;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(paraview_file_name.str(), &pmesh);
      paraview_dc->SetPrefixPath("ParaView/UW/3D");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E.real());
      paraview_dc->RegisterField("E_i",&E.imag());
      paraview_dc->RegisterField("H_r",&H.real());
      paraview_dc->RegisterField("H_i",&H.imag());

      paraview_dct = new ParaViewDataCollection(paraview_file_namet.str(), &pmesh);
      paraview_dct->SetPrefixPath("ParaView/UW3D");
      paraview_dct->SetLevelsOfDetail(order);
      paraview_dct->SetCycle(0);
      paraview_dct->SetDataFormat(VTKFormat::BINARY);
      paraview_dct->SetHighOrderOutput(true);
      paraview_dct->SetTime(0.0); // set the time
      paraview_dct->RegisterField("Et",&Et);
      paraview_dct->RegisterField("Ht",&Ht);
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   Array<int> one_bdr;
   Array<int> negone_bdr;

   if (myid == 0)
   {
      std::cout << "Setting up boundary attributes" << endl;
   }

   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      one_bdr.SetSize(pmesh.bdr_attributes.Max());
      negone_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      one_bdr = 0;
      negone_bdr = 0;
      if (prob == 4)
      {
         one_bdr[232-1] = 1;
         negone_bdr[231-1] = 1;
      }
      else
      {
         one_bdr[1] = 1;
         negone_bdr[2] = 1;      
      }
   }


   if (myid == 0)
   {
      std::cout << "Setting up boundary conditions" << endl;
   }
   // Set up bdr conditions
   // shift the ess_tdofs
   for (int j = 0; j < ess_tdof_list.Size(); j++)
   {
      ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
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
   real_t * xdata = x.GetData();

   ParComplexGridFunction hatE_gf(hatE_fes);
   hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
   hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);

   hatE_gf.ProjectBdrCoefficientTangent(z_one_cf,zero_cf, one_bdr);
   hatE_gf.ProjectBdrCoefficientTangent(z_negone_cf,zero_cf, negone_bdr);

   if (myid == 0)
   {
      std::cout << "Assembly started" << endl;
   }

   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

   if (myid == 0)
   {
      std::cout << "Assembly finished" << endl;
   }

   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());

   Array<ParFiniteElementSpace *> prec_fes;
   if (static_cond)
   {
      a->GetTraceFESpaces(prec_fes);
   }
   else
   {
      prec_fes = trial_fes;
   }
   Solver * cprec = nullptr;
   if (pmg)
   {
#ifdef MFEM_USE_COMPLEX_MUMPS
      bool mumps_coarse_solver = true;
#else
      bool mumps_coarse_solver = false;
#endif
      std::vector<Array<int>> ess_bdr_marker(prec_fes.Size());
      for (int b = 0; b<prec_fes.Size(); b++)
      {
         if (pmesh.bdr_attributes.Size())
         {
            ess_bdr_marker[b].SetSize(pmesh.bdr_attributes.Max());
            if (b == 2) // hatE
            {
               ess_bdr_marker[b] = ess_bdr;
            }
            else
            {
               ess_bdr_marker[b] = 0;
            }
         }
      }
      cprec = new ComplexPRefinementMultigrid(prec_fes, ess_bdr_marker, *Ahc, mumps_coarse_solver);
   }
   else
   {
      BlockDiagonalPreconditioner * real_prec = new BlockDiagonalPreconditioner(BlockA_r->RowOffsets());
      real_prec->owns_blocks = 1;
      for (int i = 0; i<BlockA_r->NumRowBlocks(); i++)
      {
         auto prec = MakeFESpaceDefaultSolver(prec_fes[i],0);
         prec->SetOperator(BlockA_r->GetBlock(i,i));
         real_prec->SetDiagonalBlock(i,prec);
      }
      cprec = new ComplexPreconditioner(real_prec, true);
   }

   if (myid == 0)
   {
      std::cout << "PCG iterations" << endl;
   }
   CGSolver cg(MPI_COMM_WORLD);
   // GMRESSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*Ahc);
   cg.SetPreconditioner(*cprec);
   cg.Mult(B, X);

   delete cprec;


   a->RecoverFEMSolution(X,x);

   E.real().MakeRef(E_fes,x.GetData());
   E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

   H.real().MakeRef(H_fes,&x.GetData()[offsets[1]]);
   H.imag().MakeRef(H_fes,&x.GetData()[offsets.Last()+offsets[1]]);

   if (visualization)
   {
      const char * keys = (dim == 2) ? "jRcml\n" : nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E.real(),
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      common::VisualizeField(H_out_r,vishost, visport, H.real(),
                             "Numerical Magnetic field (real part)", 501, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();

      Et = E.real();
      int num_frames = 32;
      for (int it = 0; it<num_frames; it++)
      {
         real_t t = (real_t)(it % num_frames) / num_frames;
         add(cos(2.0*M_PI*t), E.real(), sin(2.0*M_PI*t), E.imag(), Et);
         paraview_dct->SetCycle(it);
         paraview_dct->SetTime((real_t)it);
         paraview_dct->Save();
      }




   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete eps_r_cf;
   delete eps_i_cf;
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

