//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω² ϵ E = Ĵ ,   in Ω
//                       E×n = E₀ , on ∂Ω

// The DPG UW deals with the First Order System
//  i ω μ H + ∇ × E = 0,   in Ω
// -i ω ϵ E + ∇ × H = 0,   in Ω
//            E × n = E_0, on ∂Ω

// The ultraweak-DPG formulation is obtained by integration by parts of both
// equations and the introduction of trace unknowns on the mesh skeleton

// in 2D
// E is vector valued and H is scalar.
//    (∇ × E, F) = (E, ∇ × F) + < n × E , F>
// or (∇ ⋅ AE , F) = (AE, ∇ F) + < AE ⋅ n, F>
// where A = [0 1; -1 0];

// E ∈ (L²(Ω))² , H ∈ L²(Ω)
// Ê ∈ H^-1/2(Γₕ), Ĥ ∈ H^1/2(Γₕ)
//  i ω μ (H,F) + (E, ∇ × F) + < AÊ, F > = 0,      ∀ F ∈ H¹
// -i ω ϵ (E,G) + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)
//                                        Ê = E₀      on ∂Ω
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) |   < Ê, F >   |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < Ĥ, G × n > |  (J,G)  |
// where (F,G) ∈  H¹ × H(curl,Ω)

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class AzimuthalECoefficient : public Coefficient
{
private:
   const GridFunction * vgf;
public:
   AzimuthalECoefficient(const GridFunction * vgf_)
      : Coefficient(), vgf(vgf_) {}
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector X, E;
      vgf->GetVectorValue(T,ip,E);
      T.Transform(ip, X);
      real_t x = X(0);
      real_t y = X(1);
      real_t r = sqrt(x*x + y*y);

      real_t val = -x*E[1] + y*E[0];
      return val/r;
   }
};

class EpsilonMatrixCoefficient : public MatrixArrayCoefficient
{
private:
   Mesh * mesh = nullptr;
   ParMesh * pmesh = nullptr;
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Array<ParGridFunction * > pgfs;
   Array<GridFunctionCoefficient * > gf_cfs;
   GridFunction * vgf = nullptr;
   int dim;
   int sdim;
   bool vis=false;
public:
   EpsilonMatrixCoefficient(const char * filename, Mesh * mesh_, ParMesh * pmesh_,
                            double scale = 1.0, double vis_=false)
      : MatrixArrayCoefficient(mesh_->Dimension()), mesh(mesh_), pmesh(pmesh_),
        dim(mesh->Dimension()), vis(vis_)
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
      int * partitioning = mesh->GeneratePartitioning(num_procs);
      double *data = vgf->GetData();
      GridFunction gf;
      pgfs.SetSize(vdim);
      gf_cfs.SetSize(vdim);
      sdim = sqrt(vdim);
      for (int i = 0; i<sdim; i++)
      {
         for (int j = 0; j<sdim; j++)
         {
            int k = i*sdim+j;
            gf.MakeRef(fes,&data[k*fes->GetVSize()]);
            pgfs[k] = new ParGridFunction(pmesh,&gf,partitioning);
            (*pgfs[k])*=scale;
            gf_cfs[k] = new GridFunctionCoefficient(pgfs[k]);
            // skip if i or j > dim
            if (i<dim && j<dim)
            {
               Set(i,j,gf_cfs[k], true);
            }
         }
      }
   }

   // Visualize the components of the matrix coefficient
   // in separate GLVis windows for each component
   void VisualizeMatrixCoefficient()
   {
      Array<socketstream *> sol_sock(pgfs.Size());
      for (int k = 0; k<pgfs.Size(); k++)
      {
         if (Mpi::Root()) { mfem::out << "Visualizing component " << k << endl; }
         char vishost[] = "localhost";
         int visport = 19916;
         sol_sock[k] = new socketstream(vishost, visport);
         sol_sock[k]->precision(8);
         *sol_sock[k] << "parallel " << num_procs << " " << myid << "\n";
         int i = k/sdim;
         int j = k%sdim;
         // plot with the title "Epsilon Matrix Coefficient Component (i,j)"
         *sol_sock[k] << "solution\n" << *pmesh << *pgfs[k]
                      << "window_title 'Epsilon Matrix Coefficient Component (" << i << "," << j <<
                      ")'" << flush;
      }
   }
   void Update()
   {
      pgfs[0]->ParFESpace()->Update();
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
   }

};


int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "data/mesh2D.mesh";
   const char * eps_r_file = "data/eps2D_r.gf";
   const char * eps_i_file = "data/eps2D_i.gf";

   int order = 2;
   int delta_order = 1;
   int par_ref_levels = 0;
   bool visualization = false;
   double rnum=4.6e9;
   bool static_cond = false;
   bool paraview = false;
   double factor = 1.0;
   double mu = 1.257e-6/factor;
   double epsilon_scale = 8.8541878128e-12*factor;
   bool graph_norm = true;
   bool mumps_solver = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&par_ref_levels, "-pr", "--parallel-refinement_levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&graph_norm, "-graph", "--graph-norm", "-no-gn",
                  "--no-graph-norm", "Enable adjoint graph norm.");
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

   Array<int> int_bdr_attr;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (mesh.FaceIsInterior(mesh.GetBdrElementFaceIndex(i)))
      {
         int_bdr_attr.Append(mesh.GetBdrAttribute(i));
      }
   }

   // mesh.RemoveInternalBoundaries();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   EpsilonMatrixCoefficient eps_r_cf(eps_r_file,&mesh,&pmesh, epsilon_scale);
   EpsilonMatrixCoefficient eps_i_cf(eps_i_file,&mesh,&pmesh, epsilon_scale);

   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
      eps_r_cf.Update();
      eps_i_cf.Update();
   }

   // eps_r_cf.VisualizeMatrixCoefficient();
   // eps_i_cf.VisualizeMatrixCoefficient();

   // Matrix Coefficient (M = -i\omega \epsilon);
   // M = -i * omega * (eps_r + i eps_i)
   //  = omega eps_i + i (-omega eps_r)
   ScalarMatrixProductCoefficient Mr_cf(omega,eps_i_cf);
   ScalarMatrixProductCoefficient Mi_cf(-omega,eps_r_cf);

   ConstantCoefficient one(1.0);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient negmuomeg(-mu*omega);
   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);
   MatrixProductCoefficient Mrot_r(Mr_cf,rot);
   MatrixProductCoefficient Mrot_i(Mi_cf,rot);
   ScalarMatrixProductCoefficient negMrot_i(-1.0,Mrot_i);

   TransposeMatrixCoefficient Mrt_cf(Mr_cf);
   TransposeMatrixCoefficient Mit_cf(Mi_cf);
   MatrixProductCoefficient MrMrt_cf(Mr_cf,Mrt_cf);
   MatrixProductCoefficient MiMit_cf(Mi_cf,Mit_cf);
   MatrixProductCoefficient MiMrt_cf(Mi_cf,Mrt_cf);
   MatrixProductCoefficient MrMit_cf(Mr_cf,Mit_cf);

   MatrixSumCoefficient MMr_cf(MrMrt_cf,MiMit_cf);
   MatrixSumCoefficient MMi_cf(MiMrt_cf,MrMit_cf,1.0,-1.0);

   mesh.Clear();

   // Define spaces
   enum TrialSpace
   {
      E_space     = 0,
      H_space     = 1,
      hatE_space  = 2,
      hatH_space  = 3
   };
   enum TestSpace
   {
      F_space = 0,
      G_space = 1
   };

   int dimc = (dim == 3) ? 3 : 1;
   int test_order = order+delta_order;

   // Vector L2 L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dimc);

   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = new RT_Trace_FECollection(order-1,dim);
   FiniteElementCollection * hatH_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementCollection * F_fec = new H1_FECollection(test_order, dim);

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

   if (Mpi::Root())
   {
      mfem::out << "Global number of dofs = " << gdofs << endl;
   }

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,
                         TrialSpace::E_space,
                         TestSpace::F_space);

   //  (M E , G) = (M_r E, G) + i (M_i E, G) = (E, M_rt G) + i (E, Mit G)
   //            = (M_rt G, E)^T + i (Mit G, E)^T
   a->AddTrialIntegrator(
      new TransposeIntegrator(new VectorFEMassIntegrator(Mrt_cf)),
      new TransposeIntegrator(new VectorFEMassIntegrator(Mit_cf)),
      TrialSpace::E_space, TestSpace::G_space);

   //  (H,∇ × G)
   a->AddTrialIntegrator(new TransposeIntegrator(new MixedCurlIntegrator(one)),
                         nullptr,
                         TrialSpace::H_space, TestSpace::G_space);

   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatH_space, TestSpace::G_space);

   // i ω μ (H, F)
   a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(muomeg),
                         TrialSpace::H_space, TestSpace::F_space);

   // < n×Ê,F>
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                         TrialSpace::hatE_space, TestSpace::F_space);

   // test integrators
   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::G_space,TestSpace::G_space);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::G_space,TestSpace::G_space);
   // (∇F,∇δF)
   a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                        TestSpace::F_space, TestSpace::F_space);
   // (F,δF)
   a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                        TestSpace::F_space, TestSpace::F_space);

   if (graph_norm)
   {
      // μ^2 ω^2 (F,δF)
      a->AddTestIntegrator(new MassIntegrator(mu2omeg2),nullptr,
                           TestSpace::F_space, TestSpace::F_space);

      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      a->AddTestIntegrator(nullptr,
                           new TransposeIntegrator(new MixedCurlIntegrator(negmuomeg)),
                           TestSpace::F_space, TestSpace::G_space);

      // (M ∇ × F, δG) = (M_r ∇ × F, δG) + i (M_i ∇ × F, δG)
      //               = (M_r A ∇ F, δG) + i (M_i A ∇ F, δG), A = [0 1; -1; 0]
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(Mrot_r),
                           new MixedVectorGradientIntegrator(Mrot_i),
                           TestSpace::F_space, TestSpace::G_space);

      // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
      a->AddTestIntegrator(nullptr,new MixedCurlIntegrator(muomeg),
                           TestSpace::G_space, TestSpace::F_space);

      // (M^* G, ∇ × δF ) = (G, Mr A ∇ δF) - i (G, Mi A ∇ δF)
      a->AddTestIntegrator(new TransposeIntegrator(new MixedVectorGradientIntegrator(
                                                      Mrot_r)),
                           new TransposeIntegrator(new MixedVectorGradientIntegrator(negMrot_i)),
                           TestSpace::G_space, TestSpace::F_space);

      // M*M^*(G,δG) = (MrMr^t + MiMi^t) + i (MiMr^t - MrMi^t)
      const IntegrationRule *irs[Geometry::NumGeom];
      int order_quad = 2*order + 2;
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }
      const IntegrationRule &ir = IntRules.Get(pmesh.GetElementGeometry(0),
                                               2*test_order + 2);
      VectorFEMassIntegrator * integ_r = new VectorFEMassIntegrator(MMr_cf);
      VectorFEMassIntegrator * integ_i = new VectorFEMassIntegrator(MMi_cf);
      integ_r->SetIntegrationRule(ir);
      integ_i->SetIntegrationRule(ir);
      a->AddTestIntegrator(integ_r,
                           integ_i,
                           TestSpace::G_space,
                           TestSpace::G_space);
   }

   socketstream E_out_r;
   socketstream H_out_r;
   socketstream E_theta_out_r;

   ParComplexGridFunction E(E_fes);
   ParComplexGridFunction H(H_fes);
   E.real() = 0.0;
   E.imag() = 0.0;
   H.real() = 0.0;
   H.imag() = 0.0;

   L2_FECollection L2fec(order, dim);
   ParFiniteElementSpace L2_fes(&pmesh, &L2fec);

   ParGridFunction E_theta_r(&L2_fes);
   ParGridFunction E_theta_i(&L2_fes);
   ParGridFunction E_theta(&L2_fes);
   E_theta = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;
   ParaViewDataCollection * paraview_tdc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_dc->SetPrefixPath("ParaViewUWDPG2D");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E.real());
      paraview_dc->RegisterField("E_i",&E.imag());
      paraview_dc->RegisterField("H_r",&H.real());
      paraview_dc->RegisterField("H_i",&H.imag());
      paraview_dc->RegisterField("E_theta_r",&E_theta_r);
      paraview_dc->RegisterField("E_theta_i",&E_theta_i);

      paraview_tdc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_tdc->SetPrefixPath("ParaViewUWDPG2D/TimeHarmonic");
      paraview_tdc->SetLevelsOfDetail(order);
      paraview_tdc->SetCycle(0);
      paraview_tdc->SetDataFormat(VTKFormat::BINARY);
      paraview_tdc->SetHighOrderOutput(true);
      paraview_tdc->SetTime(0.0); // set the time
      paraview_tdc->RegisterField("E_theta_t",&E_theta);
   }

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

      // remove internal boundaries
      for (int i = 0; i<int_bdr_attr.Size(); i++)
      {
         ess_bdr[int_bdr_attr[i]-1] = 0;
      }

      hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
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
   double * xdata = x.GetData();

   ParComplexGridFunction hatE_gf(hatE_fes);
   hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
   hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);


   Vector zero(dim); zero = 0.0;
   // Vector one_x(dim); one_x = 0.0; one_x(0) = 1.0;
   // Vector negone_x(dim); negone_x = 0.0; negone_x(0) = -1.0;
   VectorConstantCoefficient zero_cf(zero);
   // VectorConstantCoefficient one_x_cf(one_x);
   // VectorConstantCoefficient negone_x_cf(negone_x);

   // rotate the vector
   // (x,y) -> (y,-x)
   Vector rot_one_x(dim); rot_one_x = 0.0; rot_one_x(1) = -1.0;
   Vector rot_negone_x(dim); rot_negone_x = 0.0; rot_negone_x(1) = 1.0;
   VectorConstantCoefficient rot_one_x_cf(rot_one_x);
   VectorConstantCoefficient rot_negone_x_cf(rot_negone_x);

   hatE_gf.ProjectBdrCoefficientNormal(rot_one_x_cf,zero_cf, one_r_bdr);
   hatE_gf.ProjectBdrCoefficientNormal(rot_negone_x_cf,zero_cf, negone_r_bdr);
   hatE_gf.ProjectBdrCoefficientNormal(zero_cf,rot_one_x_cf, one_i_bdr);
   hatE_gf.ProjectBdrCoefficientNormal(zero_cf,rot_negone_x_cf, negone_i_bdr);

   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr Ah;
   Vector X,B;
   a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

   ComplexOperator * Ahc = Ah.As<ComplexOperator>();

   BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
   BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

   int num_blocks = BlockA_r->NumRowBlocks();
   Array<int> tdof_offsets(2*num_blocks+1);

   tdof_offsets[0] = 0;
   int skip = (static_cond) ? 0 : 2;
   int k = (static_cond) ? 2 : 0;
   for (int i=0; i<num_blocks; i++)
   {
      tdof_offsets[i+1] = trial_fes[i+k]->GetTrueVSize();
      tdof_offsets[num_blocks+i+1] = trial_fes[i+k]->GetTrueVSize();
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

   X = 0.;

#ifdef MFEM_USE_MUMPS
   if (mumps_solver)
   {
      // Monolithic real part
      Array2D <HypreParMatrix * > Ab_r(num_blocks,num_blocks);
      // Monolithic imag part
      Array2D <HypreParMatrix * > Ab_i(num_blocks,num_blocks);
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
   if (!mumps_solver)
   {
      BlockDiagonalPreconditioner M(tdof_offsets);

      if (!static_cond)
      {
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         // solver_H->SetSystemsOptions(dim);
         M.SetDiagonalBlock(0,solver_E);
         M.SetDiagonalBlock(1,solver_H);
         M.SetDiagonalBlock(num_blocks,solver_E);
         M.SetDiagonalBlock(num_blocks+1,solver_H);
      }
      HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip,
                                                                                 skip), hatE_fes);
      HypreBoomerAMG * solver_hatH = new HypreBoomerAMG((HypreParMatrix &)
                                                        BlockA_r->GetBlock(
                                                           skip+1,skip+1));
      solver_hatE->SetPrintLevel(0);
      solver_hatH->SetPrintLevel(0);
      solver_hatH->SetRelaxType(88);


      M.SetDiagonalBlock(skip,solver_hatE);
      M.SetDiagonalBlock(skip+1,solver_hatH);
      M.SetDiagonalBlock(skip+num_blocks,solver_hatE);
      M.SetDiagonalBlock(skip+num_blocks+1,solver_hatH);

      if (myid == 0)
      {
         std::cout << "PCG iterations" << endl;
      }
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-7);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);

      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
      }

      int num_iter = cg.GetNumIterations();
   }
   a->RecoverFEMSolution(X,x);

   E.real().MakeRef(E_fes,x.GetData());
   E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

   H.real().MakeRef(H_fes,&x.GetData()[offsets[1]]);
   H.imag().MakeRef(H_fes,&x.GetData()[offsets.Last()+offsets[1]]);

   int dofs = 0;
   for (int i = 0; i<trial_fes.Size(); i++)
   {
      dofs += trial_fes[i]->GlobalTrueVSize();
   }

   AzimuthalECoefficient az_e_r(&E.real());
   AzimuthalECoefficient az_e_i(&E.imag());

   E_theta_r.ProjectCoefficient(az_e_r);
   E_theta_i.ProjectCoefficient(az_e_i);

   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E.real(),
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      common::VisualizeField(H_out_r,vishost, visport, H.real(),
                             "Numerical Magnetic field (real part)", 501, 0, 500, 500, keys);
      common::VisualizeField(E_theta_out_r,vishost, visport, E_theta_r,
                             "Numerical Electric field (Azimuthal-real)", 501, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
      delete paraview_dc;

      int num_frames = 32;
      for (int i = 0; i<num_frames; i++)
      {
         real_t t = (real_t)(i % num_frames) / num_frames;
         add(cos(real_t(2.0*M_PI)*t), E_theta_r,
             sin(real_t(2.0*M_PI)*t), E_theta_i, E_theta);
         paraview_tdc->SetCycle(i);
         paraview_tdc->SetTime(t);
         paraview_tdc->Save();
      }
      delete paraview_tdc;

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