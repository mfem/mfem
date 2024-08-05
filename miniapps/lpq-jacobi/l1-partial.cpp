#include "lpq-common.hpp"
#include <typeinfo>

using namespace std;
using namespace mfem;
using namespace lpq_common;

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   // string mesh_file = "meshes/amr-quad.mesh";
   string mesh_file = "meshes/cube.mesh";

   Mesh *serial_mesh = new Mesh(mesh_file);
   // serial_mesh->UniformRefinement();
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
   ess_bdr = 1; // set this to zero
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ParBilinearForm *a = new ParBilinearForm(fespace);

   LinearFormIntegrator *lfi = nullptr;
   BilinearFormIntegrator *bfi = nullptr;
   FunctionCoefficient *scalar_u = nullptr;
   FunctionCoefficient *scalar_f = nullptr;

   ConstantCoefficient one(1.0);

   // These variables will define the linear system
   ParGridFunction x(fespace);
   OperatorPtr A_legacy;
   OperatorPtr A;
   Vector B, X;

   x = 0.0;
   scalar_u = new FunctionCoefficient(diffusion_solution);
   x.ProjectBdrCoefficient(*scalar_u, ess_bdr);

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
   ones = 1.0;

   mfem::out << "\n--- Legacy begin ---\n" << endl;
   mfem::out << "Diag:\n" << endl;
   auto diag_b = new Vector(fespace->GetTrueVSize());
   b->AssembleDiagonal(*diag_b);
   diag_b->Print();

   mfem::out << "|A|1:\n" << endl;
   A_legacy.As<HypreParMatrix>()->AbsMult(1.0, ones, 0.0, result);
   result.Print();
   mfem::out << "\n---  Legacy end  ---\n" << endl;

   mfem::out << "\n--- Partial begin ---\n" << endl;
   mfem::out << "Diag:\n" << endl;
   auto diag_a = new Vector(fespace->GetTrueVSize());
   a->AssembleDiagonal(*diag_a);
   diag_a->Print();

   mfem::out << "|A|1:\n" << endl;
   A->AbsMult(ones, result); // ---
   result.Print();
   mfem::out << "\n--- Partial end ---\n" << endl;

   // mfem::out << "\n--- Partial-by-hand begin ---\n" << endl;
   // auto pro = static_cast<const HypreParMatrix*>(fespace->GetProlongationMatrix());
   // auto res = fespace->GetRestrictionOperator();
   // mfem::out << "Size pro: " << pro->Height() << " " << pro->Width() << endl;
   // mfem::out << "Size res: " << res->Height() << " " << res->Width() << endl;
   // auto gv = fespace->GlobalVSize();
   // mfem::out << "GlobalTrueVSize " << sys_size << "   GlobalVSize " << gv << endl;

   // Vector y1(fespace->GlobalVSize());
   // pro->AbsMult(1.0, ones, 0.0, y1);
   // y1.Print();

   // mfem::out << "\n--- Partial-by-hand end ---\n" << endl;

   delete diag_a;
   delete diag_b;
   delete a;
   delete b;
   if (scalar_u) { delete scalar_u; }
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
