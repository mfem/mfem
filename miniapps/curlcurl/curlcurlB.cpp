// 
// Put it in mfem/example and compile  
//
// we solve (M + alpha dt K1 + nu dt K2) B^{n+1} = (M +  (1-epsilon) alpha dt K1) B0
//
// Sample run: mpirun -np 4 curlcurlB -dt 1.0
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double c0=10.0;
void B_background(const Vector &, Vector &);
void OmegaFunc(const Vector &, DenseMatrix &);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int nz = 3; // nz = 3 is the minimal value for a periodic mesh
   int dim = 3;
   int order = 1;
   double alpha, dt=1.0/64, eta=1e-5, epsilon=1e-2;
   bool pm = true;
   bool visualization = false;
   bool paraview = true;

   // Parse command line
   OptionsParser args(argc, argv);
   args.AddOption(&nz, "-nz", "--num-elem", "number of elements.");
   args.AddOption(&order, "-o", "--order","Finite element order (polynomial degree).");
   args.AddOption(&pm, "-pm", "--periodic-mesh", "-no-pm",
                  "--no-periodic-mesh", "Periodic mesh (in z).");
   args.AddOption(&c0, "-c0", "--c0", "set c0 in the background B.");
   args.AddOption(&dt, "-dt", "--dt", "set dt.");
   args.AddOption(&eta, "-eta", "--eta", "set eta (i.e., 1/S).");
   args.AddOption(&epsilon, "-epsilon", "--epsilon", "set epsilon.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-para", "--para", "-no-para", "--no-para",
                  "Enable or disable Paraview visualization.");
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
   alpha = dt;  //set alpha to be dt for now

   // Generat a base mesh
   Mesh orig_mesh = Mesh::MakeCartesian3D(nz, nz, nz, Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);

   if (pm)
   {
      // Make z direction periodic
      std::vector<Vector> translations =
      {
         Vector({0.0, 0.0, 1.0})
      };
      Mesh mesh = Mesh::MakePeriodic(
                  orig_mesh,
                  orig_mesh.CreatePeriodicVertexMapping(translations));
      mesh.RemoveInternalBoundaries();

      // Refine serial mesh
      {
         int ref_levels = 3;
         for (int l = 0; l < ref_levels; l++)
         {
            mesh.UniformRefinement();
         }
      }

      // Save the final serial mesh
      ofstream mesh_ofs("periodic-cube-z.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
 
      //    Define a parallel mesh by a partitioning of the serial mesh. Refine
      //    this mesh further in parallel to increase the resolution. Once the
      //    parallel mesh is defined, the serial mesh can be deleted.
      ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
      {
         int par_ref_levels = 1;
         for (int l = 0; l < par_ref_levels; l++)
         {
            pmesh->UniformRefinement();
         }
      }

      FiniteElementCollection *fec = new ND_FECollection(order, dim);
      ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
      HYPRE_BigInt size = fespace->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "Number of finite element unknowns: " << size << endl;
      }

      /*
      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      */

      ConstantCoefficient One(1.0);
      ParBilinearForm *mass = new ParBilinearForm(fespace);
      mass->AddDomainIntegrator(new VectorFEMassIntegrator(One));
      mass->Assemble();

      MatrixFunctionCoefficient OmegaCoeff(3, OmegaFunc);
      ParBilinearForm *k1Form = new ParBilinearForm(fespace);
      k1Form->AddDomainIntegrator(new CurlCurlIntegrator(OmegaCoeff));
      k1Form->Assemble();

      ParBilinearForm *k2Form = new ParBilinearForm(fespace);
      k2Form->AddDomainIntegrator(new CurlCurlIntegrator(One));
      k2Form->Assemble();

      // Define the background B field
      ParGridFunction B0(fespace);
      VectorFunctionCoefficient Bcoeff(dim, B_background);
      B0.ProjectCoefficient(Bcoeff);

      // Compute RHS true vector
      Vector z(fespace->TrueVSize()), z1(fespace->TrueVSize());
      ParLinearForm *rhs = new ParLinearForm(fespace);
      mass->Mult(B0, *rhs);
      rhs->ParallelAssemble(z);

      delete rhs;
      rhs = new ParLinearForm(fespace);
      k1Form->Mult(B0, *rhs);
      rhs->ParallelAssemble(z1);
      z.Add((1.0-epsilon)*alpha*dt, z1);

      HypreParMatrix *M, *K1, *K2, *tmp, *A;
      mass->Finalize();
      M = mass->ParallelAssemble();

      k1Form->Finalize();
      K1 = k1Form->ParallelAssemble();

      k2Form->Finalize();
      K2 = k2Form->ParallelAssemble();

      tmp = Add(alpha*dt, *K1, eta*dt, *K2);
      A = ParAdd(tmp, M);

      Vector x(fespace->TrueVSize());
      ParFiniteElementSpace *prec_fespace = fespace;
      HypreAMS ams(*A, prec_fespace);
      HyprePCG pcg(*A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(z, x);

      // Recover FEM GridFunction for visualization
      ParGridFunction Bnext(fespace), dB(fespace);
      Bnext.SetFromTrueDofs(x);
      subtract(B0, Bnext, dB);

      if(visualization)
      {
         ostringstream mesh_name, sol_name, dB_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid;
         sol_name << "sol." << setfill('0') << setw(6) << myid;
         dB_name << "dB." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);

         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(8);
         Bnext.Save(sol_ofs);

         ofstream sol_ofs1(dB_name.str().c_str());
         sol_ofs1.precision(8);
         dB.Save(sol_ofs1);
      }

      if (paraview)
      {
         ParaViewDataCollection paraview_dc("curlcurlB", pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetCycle(0);
         paraview_dc.RegisterField("sol",&Bnext);
         paraview_dc.RegisterField("diff",&dB);
         paraview_dc.Save();
      }

      delete M;
      delete K1;
      delete K2;
      delete tmp;
      delete A;
      delete mass;
      delete k1Form;
      delete k2Form;
      delete rhs;
      delete fespace;
      delete fec;
      delete pmesh;
   }
   else
   {
      // Save the final mesh
      ofstream mesh_ofs("cube-z.mesh");
      mesh_ofs.precision(8);
      orig_mesh.Print(mesh_ofs);
   }

   return 0;
}

// Compute B0
void B_background(const Vector &x, Vector &B)
{
   double x0=0.5, y0=0.5;
   double Az=0.5*pow(x(0)-x0,2.0) + 1.0/32.0*pow(sin(2*M_PI*(x(1)-y0)),2.0);
   B(0) = M_PI/8.0*sin(2*M_PI*(x(1)-y0))*cos(2*M_PI*(x(1)-y0));
   B(1) = -x(0)+x0;
   B(2) = c0*fabs(Az);
}

// Compute Omega = |B0|^2 I - B0 B0^T
void OmegaFunc(const Vector &x, DenseMatrix &omega)
{
   omega.SetSize(3);
   Vector B0(3);

   B_background(x, B0);
   double norm2 = B0.Norml2();
   norm2 *= norm2;
   
   omega(0,0) = norm2-B0(0)*B0(0);
   omega(1,1) = norm2-B0(1)*B0(1);
   omega(2,2) = norm2-B0(2)*B0(2);
   omega(0,1) = -B0(0)*B0(1);
   omega(0,2) = -B0(0)*B0(2);
   omega(1,2) = -B0(1)*B0(2);
   omega(1,0) = omega(0,1);
   omega(2,0) = omega(0,2);
   omega(2,1) = omega(1,2);
}
