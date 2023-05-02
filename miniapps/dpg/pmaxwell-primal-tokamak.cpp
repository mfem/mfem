
// srun -n 256 ./pmaxwell-tokamak -o 3 -sc -rnum
// srun -n 448 ./pmaxwell-tokamak -o 4 -sc -rnum 11.0 -sigma 2.0 -paraview

// srun -n 448 ./pmaxwell-tokamak -o 4 -do 0 -sc -paraview (with the new epsilon GridFunction coefficients)
// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - (ω^2 ϵ + i ω σ) E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// The primal-DPG formulation is obtained by integration by parts
// and the introduction of trace unknowns on the mesh skeleton

// in 3D
// E ∈ H(curl)
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h)
//  1/μ (∇×E,∇×F) + (ω^2 ϵ + i ω σ, F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)
//                                              Ê × n = E_0     on ∂Ω

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
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
   int dim;
public:
   EpsilonMatrixCoefficient(const char * filename, Mesh * mesh_, ParMesh * pmesh_,
                            double scale = 1.0)
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
      double *data = vgf->GetData();
      GridFunction gf;
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

   // const char *mesh_file = "tokamak_100k.msh";
   // const char *mesh_file = "tokamak_200k.msh";
   // const char *mesh_file = "meshes/tokamak_100k.msh";
   // const char *mesh_file = "meshes/tokamak_100k.msh";

   const char *mesh_file = "data/mesh_100k.mesh";
   const char * eps_r_file = "data/eps_r_100k.gf";
   const char * eps_i_file = "data/eps_i_100k.gf";

   // const char *mesh_file = "meshes/box.msh";

   int order = 1;
   int delta_order = 1;
   bool visualization = false;
   double rnum=50.0e6;
   int sr = 0;
   int pr = 0;
   bool paraview = false;
   double factor = 1.0;
   double mu = 1.257e-6/factor;
   // double mu = 1.0;
   double epsilon = 1.0;
   double sigma = 0.01*factor;
   double epsilon_scale = 8.8541878128e-12*factor;
   bool graph_norm = true;
   bool mumps_solver = false;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
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
   args.AddOption(&sr, "-sref", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");
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

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   EpsilonMatrixCoefficient eps_r_cf(eps_r_file,&mesh,&pmesh, epsilon_scale);
   EpsilonMatrixCoefficient eps_i_cf(eps_i_file,&mesh,&pmesh, epsilon_scale);
   mesh.Clear();


   FiniteElementCollection *E_fec = new ND_FECollection(order,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // H^-1/2 (curl) space for Ê
   int test_order = order+delta_order;
   FiniteElementCollection * hatE_fec = new ND_Trace_FECollection(order,dim);
   FiniteElementCollection * F_fec = new ND_FECollection(test_order, dim);
   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(hatE_fes);
   test_fec.Append(F_fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient invmu_cf(1./mu);

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

   // (∇ × E,∇ × F)
   a->AddTrialIntegrator(new MixedCurlCurlIntegrator(invmu_cf), nullptr,0,0);

   // -(ω^2 ϵ, F)
   ScalarMatrixProductCoefficient m_cf_r(-omega*omega, eps_r_cf);
   ScalarMatrixProductCoefficient m_cf_i(-omega*omega, eps_i_cf);
   a->AddTrialIntegrator(new VectorFEMassIntegrator(m_cf_r),
                         new VectorFEMassIntegrator(m_cf_i),0,0);

   // < n×Ê,F>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,1,0);

   // test integrators
   // (∇×F ,∇× δF)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);

   // (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);

   socketstream E_out_r;

   ParComplexGridFunction E(E_fes);
   E.real() = 0.0;
   E.imag() = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_dc->SetPrefixPath("ParaViewPrimalDPG");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E.real());
      paraview_dc->RegisterField("E_i",&E.imag());
   }

   // internal bdr attributes
   Array<int> internal_bdr({1, 3, 6, 9, 17, 157, 185, 75, 210, 211,
                            212, 213, 214, 215, 216, 217, 218, 219,
                            220, 221, 222, 223, 224, 225, 226, 227,
                            228, 229, 230, 231, 232, 233, 234, 125});

   for (int it = 0; it<=pr; it++)
   {
      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      Array<int> one_bdr;
      Array<int> negone_bdr;

      if (myid == 0)
      {
         std::cout << "Attributes" << endl;
      }

      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         one_bdr.SetSize(pmesh.bdr_attributes.Max());
         negone_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         // need to exclude these attributes
         for (int i = 0; i<internal_bdr.Size(); i++)
         {
            ess_bdr[internal_bdr[i]-1] = 0;
         }
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         one_bdr = 0;
         negone_bdr = 0;
         one_bdr[234] = 1;
         negone_bdr[235] = 1;
      }

      if (myid == 0)
      {
         std::cout << "Attributes 2" << endl;
      }

      // Set up bdr conditions
      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize();
      }

      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = hatE_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;
      double * xdata = x.GetData();

      ParComplexGridFunction hatE_gf(hatE_fes);
      hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[1]]);
      hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[1]]);

      hatE_gf.ProjectBdrCoefficientTangent(z_one_cf,zero_cf, one_bdr);
      hatE_gf.ProjectBdrCoefficientTangent(z_negone_cf,zero_cf, negone_bdr);

      if (myid == 0)
      {
         std::cout << "Assembly started" << endl;
      }

      a->Assemble();

      if (myid == 0)
      {
         std::cout << "Assembly finished" << endl;
      }

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
         tdof_offsets[i+1] = trial_fes[i]->GetTrueVSize();
         tdof_offsets[num_blocks+i+1] = trial_fes[i]->GetTrueVSize();
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

         MUMPSSolver mumps;
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

         HypreAMS * solver_E = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(0,0),
                                            E_fes);
         HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(1,1),
                                               hatE_fes);
         solver_E->SetPrintLevel(0);
         solver_hatE->SetPrintLevel(0);

         M.SetDiagonalBlock(0,solver_E);
         M.SetDiagonalBlock(1,solver_hatE);

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

         int num_iter = cg.GetNumIterations();

      }
      mfem::out << "X.Norm = " << X.Norml2() << endl;
      a->RecoverFEMSolution(X,x);
      mfem::out << "x.Norm = " << x.Norml2() << endl;

      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         common::VisualizeField(E_out_r,vishost, visport, E.real(),
                                "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((double)it);
         paraview_dc->Save();
      }

      if (it == pr)
      {
         break;
      }

      pmesh.UniformRefinement();

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete F_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete E_fec;
   delete E_fes;

   return 0;
}

