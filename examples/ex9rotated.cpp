#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

int problem;
void velocity_function(const Vector &x, Vector &v);
double u0_function(const Vector &x);
double inflow_function(const Vector &x);

Vector bb_min, bb_max;

void AddDGIntegrators(BilinearForm &k, VectorCoefficient &velocity)
{
   double alpha = 1.0;
   double beta = -0.5;
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -alpha));
   k.AddDomainIntegrator(new MassIntegrator);
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, alpha, beta)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, alpha, beta)));
   // k.AddInteriorFaceIntegrator(new DGTraceIntegrator(velocity, alpha, beta));
   // k.AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, alpha, beta));
}

void SaveSolution(const std::string &fname, GridFunction &gf)
{
   ofstream osol(fname);
   osol.precision(16);
   gf.Save(osol);
}

Mesh *oriented_mesh()
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;
   Mesh *mesh = new Mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 2.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh->AddVertex(x);

   //EAST
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   //WEST
   // x[0] = -1.0;   x[1] = 0.0;   x[2] = 0.0;
   // mesh->AddVertex(x);
   // x[0] = -1.0;   x[1] = 1.0;   x[2] = 0.0;
   // mesh->AddVertex(x);
   // x[0] = -1.0;   x[1] = 0.0;   x[2] = 1.0;
   // mesh->AddVertex(x);
   // x[0] = -1.0;   x[1] = 1.0;   x[2] = 1.0;
   // mesh->AddVertex(x);

   int el[8];
   el[0] = 0;
   el[1] = 1;
   el[2] = 2;
   el[3] = 3;
   el[4] = 4;
   el[5] = 5;
   el[6] = 6;
   el[7] = 7;
   mesh->AddHex(el);
   // ELEM1 WEST
   // orientation 3 WEST/EAST OK
   // el[0] = 8;
   // el[1] = 0;
   // el[2] = 3;
   // el[3] = 9;
   // el[4] = 10;
   // el[5] = 4;
   // el[6] = 7;
   // el[7] = 11;
   // orientation 3 WEST/SOUTH OK
   // el[0] = 0;
   // el[1] = 3;
   // el[2] = 9;
   // el[3] = 8;
   // el[4] = 4;
   // el[5] = 7;
   // el[6] = 11;
   // el[7] = 10;
   // orientation 3 WEST/NORTH OK
   // el[0] = 8;
   // el[1] = 9;
   // el[2] = 0;
   // el[3] = 3;
   // el[4] = 10;
   // el[5] = 11;
   // el[6] = 4;
   // el[7] = 7;
   // orientation 5 WEST/TOP OK
   // el[0] = 10;
   // el[1] = 8;
   // el[2] = 9;
   // el[3] = 11;
   // el[4] = 4;
   // el[5] = 0;
   // el[6] = 3;
   // el[7] = 7;
   // orientation 3 WEST/TOP OK
   // el[0] = 8;
   // el[1] = 9;
   // el[2] = 11;
   // el[3] = 10;
   // el[4] = 0;
   // el[5] = 3;
   // el[6] = 7;
   // el[7] = 4;
   // orientation 3 WEST/BOTTOM OK
   // el[0] = 4;
   // el[1] = 7;
   // el[2] = 3;
   // el[3] = 0;
   // el[4] = 10;
   // el[5] = 11;
   // el[6] = 9;
   // el[7] = 8;

   // ELEM1 EAST
   // orientation 3 EAST/WEST OK
   el[0] = 1;
   el[1] = 8;
   el[2] = 9;
   el[3] = 2;
   el[4] = 5;
   el[5] = 10;
   el[6] = 11;
   el[7] = 6;
   // orientation 1 EAST/WEST OK
   // el[0] = 5;
   // el[1] = 10;
   // el[2] = 8;
   // el[3] = 1;
   // el[4] = 6;
   // el[5] = 11;
   // el[6] = 9;
   // el[7] = 2;
   // orientation 7 EAST/WEST OK
   // el[0] = 6;
   // el[1] = 11;
   // el[2] = 10;
   // el[3] = 5;
   // el[4] = 2;
   // el[5] = 9;
   // el[6] = 8;
   // el[7] = 1;
   // orientation 5 EAST/WEST OK
   // el[0] = 2;
   // el[1] = 9;
   // el[2] = 11;
   // el[3] = 6;
   // el[4] = 1;
   // el[5] = 8;
   // el[6] = 10;
   // el[7] = 5;
   // orientation 3 EAST/EAST OK
   // el[0] = 9;
   // el[1] = 2;
   // el[2] = 1;
   // el[3] = 8;
   // el[4] = 11;
   // el[5] = 6;
   // el[6] = 5;
   // el[7] = 10;
   // orientation 1 EAST/EAST OK
   // el[0] = 8;
   // el[1] = 1;
   // el[2] = 5;
   // el[3] = 10;
   // el[4] = 9;
   // el[5] = 2;
   // el[6] = 6;
   // el[7] = 11;
   // orientation 7 EAST/EAST OK
   // el[0] = 10;
   // el[1] = 5;
   // el[2] = 6;
   // el[3] = 11;
   // el[4] = 8;
   // el[5] = 1;
   // el[6] = 2;
   // el[7] = 9;
   // orientation 5 EAST/EAST OK
   // el[0] = 11;
   // el[1] = 6;
   // el[2] = 2;
   // el[3] = 9;
   // el[4] = 10;
   // el[5] = 5;
   // el[6] = 1;
   // el[7] = 8;
   // orientation 3 EAST/TOP OK
   // el[0] = 9;
   // el[1] = 8;
   // el[2] = 10;
   // el[3] = 11;
   // el[4] = 2;
   // el[5] = 1;
   // el[6] = 5;
   // el[7] = 6;
   // orientation 1 EAST/TOP OK
   // el[0] = 8;
   // el[1] = 10;
   // el[2] = 11;
   // el[3] = 9;
   // el[4] = 1;
   // el[5] = 5;
   // el[6] = 6;
   // el[7] = 2;
   // orientation 7 EAST/TOP OK
   // el[0] = 10;
   // el[1] = 11;
   // el[2] = 9;
   // el[3] = 8;
   // el[4] = 5;
   // el[5] = 6;
   // el[6] = 2;
   // el[7] = 1;
   // orientation 5 EAST/TOP OK
   // el[0] = 11;
   // el[1] = 9;
   // el[2] = 8;
   // el[3] = 10;
   // el[4] = 6;
   // el[5] = 2;
   // el[6] = 1;
   // el[7] = 5;
   // orientation 5 EAST/BOTTOM OK
   // el[0] = 5;
   // el[1] = 1;
   // el[2] = 2;
   // el[3] = 6;
   // el[4] = 10;
   // el[5] = 8;
   // el[6] = 9;
   // el[7] = 11;
   // orientation 7 EAST/BOTTOM OK
   // el[0] = 1;
   // el[1] = 2;
   // el[2] = 6;
   // el[3] = 5;
   // el[4] = 8;
   // el[5] = 9;
   // el[6] = 11;
   // el[7] = 10;
   // orientation 1 EAST/BOTTOM OK
   // el[0] = 2;
   // el[1] = 6;
   // el[2] = 5;
   // el[3] = 1;
   // el[4] = 9;
   // el[5] = 11;
   // el[6] = 10;
   // el[7] = 8;
   // orientation 3 EAST/BOTTOM OK
   // el[0] = 6;
   // el[1] = 5;
   // el[2] = 1;
   // el[3] = 2;
   // el[4] = 11;
   // el[5] = 10;
   // el[6] = 8;
   // el[7] = 9;
   // orientation 3 EAST/SOUTH OK
   // el[0] = 2;
   // el[1] = 1;
   // el[2] = 8;
   // el[3] = 9;
   // el[4] = 6;
   // el[5] = 5;
   // el[6] = 10;
   // el[7] = 11;
   // orientation 5 EAST/SOUTH OK
   // el[0] = 6;
   // el[1] = 2;
   // el[2] = 9;
   // el[3] = 11;
   // el[4] = 5;
   // el[5] = 1;
   // el[6] = 8;
   // el[7] = 10;
   // orientation 7 EAST/SOUTH OK
   // el[0] = 5;
   // el[1] = 6;
   // el[2] = 11;
   // el[3] = 10;
   // el[4] = 1;
   // el[5] = 2;
   // el[6] = 9;
   // el[7] = 8;
   // orientation 1 EAST/SOUTH OK
   // el[0] = 1;
   // el[1] = 5;
   // el[2] = 10;
   // el[3] = 8;
   // el[4] = 2;
   // el[5] = 6;
   // el[6] = 11;
   // el[7] = 9;
   // orientation 3 EAST/NORTH OK
   // el[0] = 8;
   // el[1] = 9;
   // el[2] = 2;
   // el[3] = 1;
   // el[4] = 10;
   // el[5] = 11;
   // el[6] = 6;
   // el[7] = 5;
   // orientation 5 EAST/NORTH OK
   // el[0] = 9;
   // el[1] = 11;
   // el[2] = 6;
   // el[3] = 2;
   // el[4] = 8;
   // el[5] = 10;
   // el[6] = 5;
   // el[7] = 1;
   // orientation 7 EAST/NORTH OK
   // el[0] = 11;
   // el[1] = 10;
   // el[2] = 5;
   // el[3] = 6;
   // el[4] = 9;
   // el[5] = 8;
   // el[6] = 1;
   // el[7] = 2;
   // orientation 1 EAST/NORTH OK
   // el[0] = 10;
   // el[1] = 8;
   // el[2] = 1;
   // el[3] = 5;
   // el[4] = 11;
   // el[5] = 9;
   // el[6] = 2;
   // el[7] = 6;
   mesh->AddHex(el);

   mesh->FinalizeHexMesh(true);
   mesh->GenerateBoundaryElements();
   mesh->Finalize();
   return mesh;
}

Mesh *skewed_mesh_2d()
{
   static const int dim = 2;
   static const int nv = 4;
   static const int nel = 1;
   Mesh *mesh = new Mesh(dim, nv, nel);
   double x[2];
   x[0] = 0.0;   x[1] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;
   mesh->AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 2.0;
   mesh->AddVertex(x);
   int el[4];
   el[0] = 0;
   el[1] = 1;
   el[2] = 2;
   el[3] = 3;
   mesh->AddQuad(el);
   mesh->FinalizeQuadMesh(true);
   mesh->GenerateBoundaryElements();
   mesh->Finalize();
   return mesh;
}

Mesh *skewed_mesh_3d()
{
   static const int dim = 3;
   static const int nv = 8;
   static const int nel = 1;
   Mesh *mesh = new Mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 1.0;   x[1] = 2.0;   x[2] = 1.0;
   mesh->AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh->AddVertex(x);

   int el[8];
   el[0] = 0;
   el[1] = 1;
   el[2] = 2;
   el[3] = 3;
   el[4] = 4;
   el[5] = 5;
   el[6] = 6;
   el[7] = 7;
   mesh->AddHex(el);

   mesh->FinalizeHexMesh(true);
   mesh->GenerateBoundaryElements();
   mesh->Finalize();
   return mesh;
}

Mesh rotated_2dmesh(int face_perm_1, int face_perm_2)
{
   static const int dim = 2;
   static const int nv = 6;
   static const int nel = 2;
   Mesh mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;
   mesh.AddVertex(x);
   int el[4];
   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   std::rotate(&el[0], &el[face_perm_1], &el[3] + 1);

   mesh.AddQuad(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   std::rotate(&el[0], &el[face_perm_2], &el[3] + 1);
   mesh.AddQuad(el);

   mesh.FinalizeQuadMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

void rotate_3d_vertices(int *v, int ref_face, int rot)
{
   std::vector<int> face_1, face_2;

   switch (ref_face/2)
   {
      case 0:
         face_1 = {v[0], v[1], v[2], v[3]};
         face_2 = {v[4], v[5], v[6], v[7]};
         break;
      case 1:
         face_1 = {v[1], v[5], v[6], v[2]};
         face_2 = {v[0], v[4], v[7], v[3]};
         break;
      case 2:
         face_1 = {v[4], v[5], v[1], v[0]};
         face_2 = {v[7], v[6], v[2], v[3]};
         break;
   }
   if (ref_face % 2 == 0)
   {
      std::reverse(face_1.begin(), face_1.end());
      std::reverse(face_2.begin(), face_2.end());
      std::swap(face_1, face_2);
   }

   std::rotate(face_1.begin(), face_1.begin() + rot, face_1.end());
   std::rotate(face_2.begin(), face_2.begin() + rot, face_2.end());

   for (int i=0; i<4; ++i)
   {
      v[i] = face_1[i];
      v[i+4] = face_2[i];
   }
}

Mesh rotated_3dmesh(int face_perm_1, int face_perm_2)
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;
   Mesh mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 3.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);

   int el[8];

   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   rotate_3d_vertices(el, face_perm_1/4, face_perm_1%4);
   mesh.AddHex(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   rotate_3d_vertices(el, face_perm_2/4, face_perm_2%4);
   mesh.AddHex(el);

   mesh.FinalizeHexMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

bool test_mesh(const int order, const int dim, const int pi, const int pj,
   const int ref_levels)
{
   Mesh mesh = dim == 2 ? rotated_2dmesh(pi,pj) : rotated_3dmesh(pi,pj);
   Array<Refinement> refs(1);
   refs[0].index = 0;
   refs[0].ref_type = 7; // Magic magic
   mesh.GeneralRefinement(refs);

   mesh.EnsureNodes();
   mesh.SetCurvature(3, true);

   // mesh.UniformRefinement();
   for (int lev = 0; lev < ref_levels; lev++)
   {
      // mesh.UniformRefinement();
      mesh.RandomRefinement(0.6, false, 1, 4);
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   // H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   // cout << "Number of unknowns: " << fes.GetVSize() << endl;

   Vector velocity_vector(dim);
   for (int i = 0; i < dim; ++i)
   {
      velocity_vector[i] = -1.0;
   }
   // velocity_vector[1] = 1.0;
   VectorConstantCoefficient velocity(velocity_vector);
   // VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm k_test(&fes), k_ref(&fes);
   // k_ref.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // k_test.SetAssemblyLevel(AssemblyLevel::FULL);
   k_test.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // k_test.SetAssemblyLevel(AssemblyLevel::ELEMENT);

   AddDGIntegrators(k_test, velocity);
   AddDGIntegrators(k_ref, velocity);

   tic_toc.Clear();
   tic_toc.Start();
   k_test.Assemble();
   k_test.Finalize();
   tic_toc.Stop();
   // cout << "test assembly time: " << tic_toc.RealTime() << " sec." << endl;
   tic_toc.Clear();
   tic_toc.Start();
   k_ref.Assemble();
   k_ref.Finalize();
   tic_toc.Stop();
   // cout << "ref assembly time: " << tic_toc.RealTime() << " sec." << endl;

   // std::cout << "FA matrix:" << std::endl;
   // std::cout  << k_ref.SpMat() << std::endl;

   GridFunction u(&fes), r_test(&fes), r_ref(&fes), diff(&fes);
   // u.ProjectCoefficient(u0);
   u.Randomize(1);
   // u = 1.0;

   k_test.Mult(u, r_test);
   k_ref.Mult(u, r_ref);

   diff = r_test;
   diff -= r_ref;

   bool success = diff.Norml2() < 1e-12;
   return success;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   int ref_levels = 0;
   int order = 3;
   const char *device_config = "cpu";

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   int dim = 2;
   args.AddOption(&dim, "-dim", "--dimension", "dimension tested.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   int perm_dim = dim==2 ? 4 : 24;
   int cpt_success = 0;
   int cpt_failed = 0;
   for (int pi = 0; pi < perm_dim; pi++)
   {
      for (int pj = 0; pj < perm_dim; pj++)
      {
         bool success = test_mesh(order,dim,pi,pj,ref_levels);
         std::cout << (success?"SUCCESS:":"FAILED:") <<
         " pi = " << pi << ", pj = " << pj << std::endl << std::endl;
         success ? cpt_success++ : cpt_failed++;
      }
   }
   std::cout << std::endl;
   std::cout << "Successful tests: " << cpt_success << std::endl;
   std::cout << "Failed tests: " << cpt_failed << std::endl;

   return 0;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
