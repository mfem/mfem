#include "lpq-common.hpp"
#include <typeinfo>

using namespace std;
using namespace mfem;
using namespace lpq_common;

//int wrap_abs(int x) { return std::abs(x); }

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   // string mesh_file = "meshes/amr-quad.mesh";
   string mesh_file = "meshes/ref-square.mesh";

   Mesh *serial_mesh = new Mesh(mesh_file);
   serial_mesh->UniformRefinement();
   ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *serial_mesh);
   // mesh->UniformRefinement();
   delete serial_mesh;

   int order = 1;
   dim = mesh->Dimension();
   space_dim = mesh->SpaceDimension();

   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   fec = new H1_FECollection(order, dim);
   fespace = new ParFiniteElementSpace(mesh, fec);

   HYPRE_BigInt sys_size = fespace->GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "Number of unknowns: " << sys_size << endl;
   }

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   Array<int> ess_tdof_list;
   ess_bdr = 0; // set this to zero
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParBilinearForm *a = new ParBilinearForm(fespace);

   BilinearFormIntegrator *bfi = nullptr;

   ConstantCoefficient one(1.0);

   // These variables will define the linear system
   ParGridFunction x(fespace);
   OperatorPtr A_legacy;
   OperatorPtr A;

   bfi = new DiffusionIntegrator();

   a->AddDomainIntegrator(bfi);
   a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a->Assemble();
   a->FormSystemMatrix(ess_tdof_list, A);

   ParBilinearForm *b = new ParBilinearForm(fespace, a);
   b->SetAssemblyLevel(AssemblyLevel::LEGACY);
   b->Assemble();
   b->FormSystemMatrix(ess_tdof_list, A_legacy);

   if (Mpi::Root()) { mfem::out << "\nMesh: " << mesh_file << "\n" << endl; }

   // Set up constant vector of ones
   Vector ones(fespace->GetTrueVSize());
   Vector result(fespace->GetTrueVSize());
   // ones = 1.0;
   // ones[0] = 2.0;
   ones.Randomize();

   if (Mpi::Root())
   {
      mfem::out << "\n--- Legacy begin ---\n" << endl;
      mfem::out << "Diag:\n" << endl;
   }
   auto diag_b = new Vector(fespace->GetTrueVSize());
   b->AssembleDiagonal(*diag_b);
   if (Mpi::Root())
   {
      diag_b->Print();
   }

   if (Mpi::Root())
   {
      mfem::out << "|A|1:\n" << endl;
   }
   A_legacy->AbsMult(ones, result);
   if (Mpi::Root())
   {
      result.Print();
      mfem::out << "\n---  Legacy end  ---\n" << endl;
   }

   if (Mpi::Root())
   {
      mfem::out << "\n--- Partial begin ---\n" << endl;
      mfem::out << "Diag:\n" << endl;
   }
   auto diag_a = new Vector(fespace->GetTrueVSize());
   a->AssembleDiagonal(*diag_a);
   if (Mpi::Root())
   {
      diag_a->Print();
   }

   if (Mpi::Root())
   {
      mfem::out << "|A|1:\n" << endl;
   }

   A->AbsMult(ones, result);
   if (Mpi::Root())
   {
      result.Print();
      mfem::out << "\n--- Partial end ---\n" << endl;
   }

   {
      // Get (conforming) prolongation
      auto cp = new ConformingProlongationOperator(*fespace);
      // Get element restriction
      auto dof_order = GetEVectorOrdering(*fespace);
      auto el_rest = fespace->GetElementRestriction(dof_order);
      // Apply integrator...
      // use integ.AddMultPA(x,y), bfi->AddMultPA(x,y)
      // Assemble Diagonal with AssembleDiagonalPA(vec)
      // Define vectors
      Vector d(el_rest->Height());
      Vector ones(el_rest->Height());
      Vector Gd(el_rest->Width());
      Vector PGd(cp->Width());

      d = 0.0;
      Gd = 0.0;
      PGd = 0.0;

      bfi->AssembleDiagonalPA(d); // Gets diag(BtDB)
      el_rest->AbsMultTranspose(d, Gd);
      cp->AbsMultTranspose(Gd, PGd);
      if (Mpi::Root())
      {
         mfem::out << "PGd" << std::endl;
         PGd.Print();
      }

      ones = 1.0;
      d = 0.0;
      Gd = 0.0;
      PGd = 0.0;
      auto diff = static_cast<DiffusionIntegrator*>(bfi);
      diff->AddAbsMultPA(ones,d); // Gets |Bt|D|B|1, might be wrong
      el_rest->AbsMultTranspose(d, Gd);
      cp->AbsMultTranspose(Gd, PGd);
      if (Mpi::Root())
      {
         mfem::out << "PGBtDB1" << std::endl;
         PGd.Print();
      }

      delete cp;
   }

   delete diag_a;
   delete diag_b;
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
