#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "euler.hpp"
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/square-disc.mesh";
   //const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = -1;
   int order = 1;
   int N = 5;
   // Equation constant parameters.
   const int num_states = 4;
   const double specific_heat_ratio = 1.4;
   const double R = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&N, "-n", "--#elements",
                  "number of mesh elements.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
   //                       1, 1, true);
   int dim = mesh->Dimension();
   cout << "number of elements " << mesh->GetNE() << endl;
   ofstream sol_ofv("square_disc_mesh.vtk");
   sol_ofv.precision(14);
   mesh->PrintVTK(sol_ofv, 1);
   delete mesh;
   return 0;
}
void EulerDomainIntegrator::AssembleElementVector(const FiniteElement &el,
                                                  ElementTransformation &Tr,
                                                  const Vector &elfun, Vector &elvect)
{
   const int num_nodes = el.GetDof();
   elvect.SetSize(num_states * num_nodes);
   elvect = 0.0;
   u.SetSize(num_states);
   DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   DenseMatrix flux(num_states, dim);
   shape.SetSize(num_nodes);
   dshape.SetSize(num_nodes, dim);
   int intorder = 2 * el.GetOrder();
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      // Calculate the shape function
      el.CalcShape(ip, shape);
      // Compute the physical gradient
      Tr.SetIntPoint(&ip);
      el.CalcDShape(ip, dshape);
      Mult(dshape, Tr.AdjugateJacobian(), dshapedx);
      u_mat.MultTranspose(shape, u);
      computeFlux(u, dim, flux);
      AddMult_a_ABt(-ip.weight, dshapedx, flux, res);
   }
}