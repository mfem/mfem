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
#include "../../util/pcomplexweakform.hpp"
#include "../../util/utils.hpp"
#include "../../util/maxwell_utils.hpp"
#include "../../../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <ctime>
#include <string>
#include <sstream>
#include <cstring>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   // fine mesh (trianles)
   // default mesh
   const char *mesh_file = "data/mesh-tri34K.mesh";
   // coarse mesh (triangles)
   // const char *mesh_file = "data/mesh-tri11K.mesh";
   // coarse mesh (quadrilaterals)
   // const char *mesh_file = "data/mesh-quad5K.mesh";

   // epsilon tensor
   const char * eps_r_file = nullptr;
   const char * eps_i_file = nullptr;

   int order = 2;
   int delta_order = 1;
   int par_ref_levels = 0;
   int amr_ref_levels = 0;
   // real_t rnum=4.6e9;
   // real_t mu = 1.257e-6/factor;
   // real_t epsilon_scale = 8.8541878128e-12*factor;
   real_t rnum=4.6;
   real_t mu = 1.257;
   real_t epsilon_scale = 8.8541878128;

   //  ∇×(1/μ ∇×E) - ω² ϵ E = Ĵ ,   in Ω
   //  1/1.257e-6 - 2π*2π*4.6*4.6e18*8.8541878128e-12
   // 1/1.257e-6 - 2π*2π*4.6*4.6*8.8541878128e6
   // 1e6/1.257 - 2π*2π*4.6*4.6*8.8541878128e6
   // 1/1.257 - 2π*2π*4.6*4.6*8.8541878128

   bool visualization = false;
   bool static_cond = false;
   bool graph_norm = true;
   bool mumps_solver = false;
   real_t theta = 0.0;
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

   if (strcmp(mesh_file, "data/mesh-tri34K.mesh") == 0)
   {
      eps_r_file = "data/eps-tri34K_r.gf";
      eps_i_file = "data/eps-tri34K_i.gf";
   }
   else if (strcmp(mesh_file, "data/mesh-tri11K.mesh") == 0)
   {
      eps_r_file = "data/eps-tri11K_r.gf";
      eps_i_file = "data/eps-tri11K_i.gf";
   }
   else if (strcmp(mesh_file, "data/mesh-quad5K.mesh") == 0)
   {
      eps_r_file = "data/eps-quad5K_r.gf";
      eps_i_file = "data/eps-quad5K_i.gf";
   }
   else
   {
      MFEM_ABORT("Unknown mesh file: " + string(mesh_file));
   }

   real_t omega = 2.*M_PI*rnum;

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
   mesh.EnsureNCMesh(true);
   int * partitioning = mesh.GeneratePartitioning(num_procs);

   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning);

   EpsilonMatrixCoefficient eps_r_cf(eps_r_file,&mesh,&pmesh, epsilon_scale);
   EpsilonMatrixCoefficient eps_i_cf(eps_i_file,&mesh,&pmesh, epsilon_scale);
   mesh.Clear();

   for (int i = 0; i<par_ref_levels; i++)
   {
      pmesh.UniformRefinement();
      eps_r_cf.Update();
      eps_i_cf.Update();
   }

   // eps_r_cf.VisualizeMatrixCoefficient();
   // eps_i_cf.VisualizeMatrixCoefficient();

   // return 0;

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
   a->StoreMatrices(); // needed for AMR

   const IntegrationRule *irs[Geometry::NumGeom];
   int order_quad = 2*order + 2;
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   const IntegrationRule &ir = IntRules.Get(pmesh.GetElementGeometry(0),
                                            2*test_order + 2);

   // (E,∇ × F)
   MixedCurlIntegrator * curl_integ = new MixedCurlIntegrator(one);
   curl_integ->SetIntRule(&ir);
   a->AddTrialIntegrator(new TransposeIntegrator(curl_integ), nullptr,
                         TrialSpace::E_space,
                         TestSpace::F_space);

   //  (M E , G) = (M_r E, G) + i (M_i E, G) = (E, M_rt G) + i (E, Mit G)
   //            = (M_rt G, E)^T + i (Mit G, E)^T
   VectorFEMassIntegrator * Mrt_cf_integ = new VectorFEMassIntegrator(Mrt_cf);
   VectorFEMassIntegrator * Mit_cf_integ = new VectorFEMassIntegrator(Mit_cf);
   Mrt_cf_integ->SetIntegrationRule(ir);
   Mit_cf_integ->SetIntegrationRule(ir);
   a->AddTrialIntegrator(
      new TransposeIntegrator(Mrt_cf_integ),
      new TransposeIntegrator(Mit_cf_integ),
      TrialSpace::E_space, TestSpace::G_space);

   //  (H,∇ × G)
   MixedCurlIntegrator * curl_integ_H = new MixedCurlIntegrator(one);
   curl_integ_H->SetIntRule(&ir);
   a->AddTrialIntegrator(new TransposeIntegrator(curl_integ_H), nullptr,
                         TrialSpace::H_space, TestSpace::G_space);

   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                         TrialSpace::hatH_space, TestSpace::G_space);

   // i ω μ (H, F)
   MixedScalarMassIntegrator * muomeg_integ = new MixedScalarMassIntegrator(
      muomeg);
   muomeg_integ->SetIntRule(&ir);
   a->AddTrialIntegrator(nullptr,muomeg_integ,
                         TrialSpace::H_space, TestSpace::F_space);

   // < n×Ê,F>
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                         TrialSpace::hatE_space, TestSpace::F_space);

   // test integrators
   // (∇×G ,∇× δG)
   CurlCurlIntegrator * curlcurl_integ = new CurlCurlIntegrator(one);
   curlcurl_integ->SetIntRule(&ir);
   a->AddTestIntegrator(curlcurl_integ,nullptr,
                        TestSpace::G_space,TestSpace::G_space);
   // (G,δG)
   VectorFEMassIntegrator * vfemass_integ = new VectorFEMassIntegrator(one);
   vfemass_integ->SetIntegrationRule(ir);
   a->AddTestIntegrator(vfemass_integ,nullptr,
                        TestSpace::G_space,TestSpace::G_space);
   // (∇F,∇δF)
   DiffusionIntegrator * diff_integ = new DiffusionIntegrator(one);
   diff_integ->SetIntRule(&ir);
   a->AddTestIntegrator(diff_integ,nullptr,
                        TestSpace::F_space, TestSpace::F_space);
   // (F,δF)
   MassIntegrator * mass_integ = new MassIntegrator(one);
   mass_integ->SetIntRule(&ir);
   a->AddTestIntegrator(mass_integ,nullptr,
                        TestSpace::F_space, TestSpace::F_space);

   if (graph_norm)
   {
      // μ^2 ω^2 (F,δF)
      MassIntegrator * mu2omeg2_integ = new MassIntegrator(mu2omeg2);
      mu2omeg2_integ->SetIntRule(&ir);
      a->AddTestIntegrator(mu2omeg2_integ,nullptr,
                           TestSpace::F_space, TestSpace::F_space);

      // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
      MixedCurlIntegrator * negmuomeg_integ = new MixedCurlIntegrator(negmuomeg);
      negmuomeg_integ->SetIntRule(&ir);
      a->AddTestIntegrator(nullptr, new TransposeIntegrator(negmuomeg_integ),
                           TestSpace::F_space, TestSpace::G_space);

      // (M ∇ × F, δG) = (M_r ∇ × F, δG) + i (M_i ∇ × F, δG)
      //               = (M_r A ∇ F, δG) + i (M_i A ∇ F, δG), A = [0 1; -1; 0]
      MixedVectorGradientIntegrator * Mrot_r_integ = new
      MixedVectorGradientIntegrator(Mrot_r);
      MixedVectorGradientIntegrator * Mrot_i_integ = new
      MixedVectorGradientIntegrator(Mrot_i);
      Mrot_r_integ->SetIntRule(&ir);
      Mrot_i_integ->SetIntRule(&ir);
      a->AddTestIntegrator(Mrot_r_integ, Mrot_i_integ,
                           TestSpace::F_space, TestSpace::G_space);

      // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
      MixedCurlIntegrator * muomeg_integ = new MixedCurlIntegrator(muomeg);
      muomeg_integ->SetIntRule(&ir);
      a->AddTestIntegrator(nullptr,muomeg_integ,
                           TestSpace::G_space, TestSpace::F_space);

      // (M^* G, ∇ × δF ) = (G, Mr A ∇ δF) - i (G, Mi A ∇ δF)
      MixedVectorGradientIntegrator * Mrot_r_integ2 = new
      MixedVectorGradientIntegrator(Mrot_r);
      MixedVectorGradientIntegrator * negMrot_i_integ = new
      MixedVectorGradientIntegrator(negMrot_i);
      Mrot_r_integ2->SetIntRule(&ir);
      negMrot_i_integ->SetIntRule(&ir);
      a->AddTestIntegrator(new TransposeIntegrator(Mrot_r_integ2),
                           new TransposeIntegrator(negMrot_i_integ),
                           TestSpace::G_space, TestSpace::F_space);

      // M*M^*(G,δG) = (MrMr^t + MiMi^t) + i (MiMr^t - MrMi^t)
      VectorFEMassIntegrator * MMr_integ = new VectorFEMassIntegrator(MMr_cf);
      VectorFEMassIntegrator * MMi_integ = new VectorFEMassIntegrator(MMi_cf);
      MMr_integ->SetIntegrationRule(ir);
      MMi_integ->SetIntegrationRule(ir);
      a->AddTestIntegrator(MMr_integ, MMi_integ,
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

   // Create ParaView directory and file
   std::string output_dir = "ParaView/UW/2D" + GetTimestamp();
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
      paraview_dc->RegisterField("E_r",&E.real());
      paraview_dc->RegisterField("E_i",&E.imag());
      paraview_dc->RegisterField("H_r",&H.real());
      paraview_dc->RegisterField("H_i",&H.imag());
      paraview_dc->RegisterField("E_theta_r",&E_theta_r);
      paraview_dc->RegisterField("E_theta_i",&E_theta_i);

      std::ostringstream paraview_file_name_th;
      paraview_file_name_th << filename
                            << "_par_ref_" << par_ref_levels
                            << "_order_" << order
                            << "th";
      paraview_tdc = new ParaViewDataCollection(paraview_file_name_th.str(), &pmesh);
      paraview_tdc->SetPrefixPath(output_dir);
      paraview_tdc->SetLevelsOfDetail(order);
      paraview_tdc->SetCycle(0);
      paraview_tdc->SetDataFormat(VTKFormat::BINARY);
      paraview_tdc->SetHighOrderOutput(true);
      paraview_tdc->SetTime(0.0); // set the time
      paraview_tdc->RegisterField("E_theta_t",&E_theta);
   }

   real_t res0 = 0.;
   real_t err0 = 0.;
   int dof0 = 0; // init to suppress gcc warning

   Array<int> elements_to_refine;

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=amr_ref_levels; it++)
   {
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
      real_t * xdata = x.GetData();

      ParComplexGridFunction hatE_gf(hatE_fes);
      hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
      hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);


      Vector zero(dim); zero = 0.0;
      VectorConstantCoefficient zero_cf(zero);

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
         // BlockTriangularSymmetricPreconditioner M(tdof_offsets);
         // M.SetOperator(blockA);
         // int nblocks = blockA.NumRowBlocks();
         // for (int i = 0; i<nblocks; i++)
         // {
         //    for (int j = 0; j<nblocks; j++)
         //    {
         //       if (i != j)
         //       {
         //          M.SetBlock(i,j,&blockA.GetBlock(i,j));
         //       }
         //    }
         // }


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
         cg.SetMaxIter(1500);
         cg.SetPrintLevel(3);
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

      Vector & residuals = a->ComputeResidual(x);
      real_t residual = residuals.Norml2();
      real_t maxresidual = residuals.Max();
      real_t globalresidual = residual * residual;
      MPI_Allreduce(MPI_IN_PLACE, &maxresidual, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &globalresidual, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      H.real().MakeRef(H_fes,&x.GetData()[offsets[1]]);
      H.imag().MakeRef(H_fes,&x.GetData()[offsets.Last()+offsets[1]]);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      real_t rate_res = (it) ? dim*log(res0/globalresidual)/log((
                                                                   real_t)dof0/dofs) : 0.0;

      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(1) << std::fixed
                   << std::setw(4) <<  2.0*rnum << " π  | "
                   << std::setprecision(3);
         std::cout << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << num_iter << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
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
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((real_t)it);
         paraview_dc->Save();
      }
      if (it == amr_ref_levels)
      {
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
         break;
      }

      if (theta > 0.0)
      {
         elements_to_refine.SetSize(0);
         for (int iel = 0; iel<pmesh.GetNE(); iel++)
         {
            if (residuals[iel] > theta * maxresidual)
            {
               elements_to_refine.Append(iel);
            }
         }
         pmesh.GeneralRefinement(elements_to_refine,1,1);
      }
      else
      {
         pmesh.UniformRefinement();
      }

      eps_r_cf.Update();
      eps_i_cf.Update();

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      L2_fes.Update();
      E_theta_r.Update();
      E_theta_i.Update();
      E_theta.Update();

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