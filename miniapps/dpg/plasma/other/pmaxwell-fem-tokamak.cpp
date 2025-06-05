
// srun -n 256 ./pmaxwell-fem-tokamak -o 3 -sc -rnum
// srun -n 448 ./pmaxwell-fem-tokamak -o 4 -sc -rnum 11.0 -sigma 2.0 -paraview

// srun -n 448 ./pmaxwell-fem-tokamak -o 4 -paraview
// Description:
// This example code demonstrates the use of MFEM to define and solve
// the standard FEM formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - (ω^2 ϵ + i ω σ) E = J ,   in Ω
//                E×n = E_0, on ∂Ω

#include "mfem.hpp"
#include "../../util/pcomplexweakform.hpp"
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

   const char *mesh_file = "data/mesh_330k.mesh";
   const char * eps_r_file = "data/eps_r_330k.gf";
   const char * eps_i_file = "data/eps_i_330k.gf";

   int order = 1;
   bool visualization = false;
   double rnum=50.0e6;
   int sr = 0;
   int pr = 0;
   bool paraview = false;
   double mu = 1.257e-6;
   double epsilon = 1.0;
   double epsilon_scale = 8.8541878128e-12;

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
   args.AddOption(&sr, "-sref", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");
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

   socketstream E_out_r;


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

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh, fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);

   if (myid == 0)
   {
      std::cout << "Assembling matrix" << endl;
   }

   ScalarMatrixProductCoefficient m_cf_r(-omega*omega, eps_r_cf);
   ScalarMatrixProductCoefficient m_cf_i(-omega*omega, eps_i_cf);

   ParComplexLinearForm *b = new ParComplexLinearForm(E_fes);
   b->Vector::operator=(0.0);

   ParSesquilinearForm *a = new ParSesquilinearForm(E_fes);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv),nullptr);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(m_cf_r),
                          new VectorFEMassIntegrator(m_cf_i));

   ParComplexGridFunction E_gf(E_fes);
   E_gf.real() = 0.0;
   E_gf.imag() = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_gf.real());
      paraview_dc->RegisterField("E_i",&E_gf.imag());
   }

   // internal bdr attributes
   Array<int> internal_bdr({1, 3, 6, 9, 17, 157, 185, 75, 210, 211,
                            212, 213, 214, 215, 216, 217, 218, 219,
                            220, 221, 222, 223, 224, 225, 226, 227,
                            228, 229, 230, 231, 232, 233, 234, 125});

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
      E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      one_bdr = 0;
      negone_bdr = 0;
      one_bdr[234] = 1;
      negone_bdr[235] = 1;
   }

   if (myid == 0)
   {
      std::cout << "Attributes 2" << endl;
   }

   Vector z_one(3); z_one = 0.0; z_one(2) = 1.0;
   Vector zero(3); zero = 0.0;
   Vector z_negone(3); z_negone = 0.0; z_negone(2) = -1.0;
   VectorConstantCoefficient z_one_cf(z_one);
   VectorConstantCoefficient z_negone_cf(z_negone);
   VectorConstantCoefficient zero_cf(zero);


   E_gf.real() = 0.0;
   E_gf.imag() = 0.0;
   E_gf.ProjectBdrCoefficientTangent(z_one_cf,zero_cf, one_bdr);
   E_gf.ProjectBdrCoefficientTangent(z_negone_cf,zero_cf, negone_bdr);

   if (myid == 0)
   {
      std::cout << "Assembly started" << endl;
   }
   b->Assemble();
   a->Assemble();

   OperatorPtr Ah;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, E_gf, *b, Ah, X, B);

   if (myid == 0)
   {
      std::cout << "Assembly finished" << endl;
   }

#ifdef MFEM_USE_MUMPS
   HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
   // auto cpardiso = new CPardisoSolver(A->GetComm());
   auto solver = new MUMPSSolver(MPI_COMM_WORLD);
   solver->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   solver->SetPrintLevel(1);
   solver->SetOperator(*A);
   solver->Mult(B,X);
   delete A;
   delete solver;
#else
   MFEM_ABORT("MFEM compiled without mumps");
#endif

   a->RecoverFEMSolution(X, *b, E_gf);

   if (visualization)
   {
      const char * keys = nullptr;
      char vishost[] = "localhost";
      int  visport   = 19916;
      common::VisualizeField(E_out_r,vishost, visport, E_gf.real(),
                             "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
   }

   if (paraview)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime((double)0);
      paraview_dc->Save();
   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete b;
   delete E_fes;
   delete fec;

   return 0;
}

