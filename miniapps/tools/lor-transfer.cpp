#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int problem = 1; // problem type

int Wx = 0, Wy = 0; // window position
int Ww = 286, Wh = 286; // window size
int offx = Ww+2, offy = Wh+25; // window offsets

// Exact functions to project
double RHO_exact(const Vector &x);

// Helper functions
void visualize(VisItDataCollection &, string, int, int, int, int);
double compute_mass(FiniteElementSpace *, double, VisItDataCollection &,
                    string);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 4;
   int lref = order;
   int lorder = 0;
   bool visualization = true;
   bool useH1 = false;
   bool use_transfer = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the *_exact functions.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lorder, "-lo", "--lor-order",
                  "LOR refinement order (polynomial degree, zero by default).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&use_transfer, "-t", "--use-pointwise-transfer", "-no-t",
                  "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   Mesh mesh_lor(&mesh, lref, basis_lor);

   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      fec = new H1_FECollection(order-1, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      fec = new L2_FECollection(order-1, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }

   FiniteElementSpace fespace(&mesh, fec);
   FiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   GridFunction rho(&fespace);
   GridFunction rho_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc("HO", &mesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc("LOR", &mesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);

   // HO projections
   FunctionCoefficient RHO(RHO_exact);
   rho.ProjectCoefficient(RHO);
   double ho_mass = compute_mass(&fespace, -1.0, HO_dc, "HO       ");

   L2Projection R_L2(fespace, fespace_lor);
   L2Prolongation P_L2(R_L2);
   OperatorHandle R_transfer, P_transfer;
   fespace_lor.GetTransferOperator(fespace, R_transfer);
   fespace_lor.GetReverseTransferOperator(*(new MassIntegrator),
                                          fespace, P_transfer);
   Operator *R = use_transfer ? R_transfer.Ptr() : &R_L2;
   Operator *P = use_transfer ? P_transfer.Ptr() : &P_L2;

   // HO->LOR restriction
   R->Mult(rho, rho_lor);
   compute_mass(&fespace_lor, ho_mass, LOR_dc, "R(HO)    ");

   // LOR-HO prolongation
   GridFunction rho_prev = rho;
   P->Mult(rho_lor, rho);
   compute_mass(&fespace, ho_mass, HO_dc, "P(R(HO)) ");

   rho_prev -= rho;
   cout.precision(12);
   cout << "HO - P(R(HO))       = " << rho_prev.Normlinf() << endl << endl;

   // LOR projections
   rho_lor.ProjectCoefficient(RHO);
   GridFunction rho_lor_prev = rho_lor;
   double lor_mass = compute_mass(&fespace_lor, -1.0, LOR_dc, "LOR      ");

   // Prolongate to HO space
   P->Mult(rho_lor, rho);
   compute_mass(&fespace, lor_mass, HO_dc, "P(LOR)   ");

   // Restrict back to LOR space. This won't give the original function
   // because the rho_lor doesn't necessarily live in the range of R.
   R->Mult(rho, rho_lor);
   rho_lor_prev -= rho_lor;
   cout.precision(12);
   cout << "LOR - R(P(LOR))     = " << rho_lor_prev.Normlinf() << endl;

   // Visualization:
   if (visualization)
   {
      visualize(HO_dc, "HO", Wx, Wy, Ww, Wh);
      Wx += offx;
      visualize(LOR_dc, "R(HO)", Wx, Wy, Ww, Wh);
      Wx += offx;
      visualize(HO_dc, "P(R(HO))", Wx, Wy, Ww, Wh);

      Wx = 0;
      Wy += offy;

      visualize(LOR_dc, "LOR", Wx, Wy, Ww, Wh);
      Wx += offx;
      visualize(HO_dc, "P(LOR)", Wx, Wy, Ww, Wh);
   }

   delete fec;
   delete fec_lor;

   return 0;
}


double RHO_exact(const Vector &x)
{
   switch (problem)
   {
      case 1: // smooth field
         return x(1)+0.25*cos(2*M_PI*x.Norml2());
      case 2: // cubic function
         return x(1)*x(1)*x(1) + 2*x(0)*x(1) + x(0);
      case 3: // sharp gradient
         return M_PI/2-atan(5*(2*x.Norml2()-1));
      default:
         return 1.0;
   }
}


void visualize(VisItDataCollection &dc, string prefix,
               int x, int y, int w, int h)
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("density")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption 'L2 " << prefix << " Density'"
              << "window_title 'L2 " << prefix << " Density'" << flush;
}


double compute_mass(FiniteElementSpace *L2, double massL2,
                    VisItDataCollection &dc, string prefix)
{
   ConstantCoefficient one(1.0);
   BilinearForm ML2(L2);
   ML2.AddDomainIntegrator(new MassIntegrator(one));
   ML2.Assemble();

   GridFunction rhoone(L2);
   rhoone = 1.0;

   double newmass = ML2.InnerProduct(*dc.GetField("density"),rhoone);
   cout.precision(12);
   cout << "L2 " << prefix << " mass   = " << newmass;
   if (massL2 >= 0)
   {
      cout.precision(4);
      cout << " ("  << fabs(newmass-massL2)*100/massL2 << "%)";
   }
   cout << endl;
   return newmass;
}