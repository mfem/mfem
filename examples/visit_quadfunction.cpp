#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//setup intial condition
int problem = 3;

// Initial condition
real_t my_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;



int main(int argc, char *argv[])
{

   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int order = 2;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }


   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
     int ref_levels = 1; //(int)floor(log(50000./mesh.GetNE())/log(2.)/dim);

      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   pmesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   const int elem_index = 0;
   mfem::Element *el = pmesh.GetElement(elem_index);
   mfem::Geometry::Type geom = el->GetGeometryType();

   //Create finite element space
   DG_FECollection fec(order, pmesh.Dimension());
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(&pmesh, &fec);


   //specify integration rule
   int intorder = (2*order-1);
   //IntegrationRules irGL(geom, Quadrature1D::GaussLobatto);
   const IntegrationRule *ir = &IntRules.Get(geom, 2*order);

   //Create QuadratureSpace
   QuadratureSpace qspace(pmesh, *ir);

   QuadratureFunction quadfun(qspace);

   std::cout<<"quadfun.Size() = "<<quadfun.Size()<<std::endl;
   std::cout<<"pmesh.GetNE() * qpts "<<pmesh.GetNE() * ir->GetNPoints()<<std::endl;

   //Loop over elements and quadrature points
   int qf_idx = 0;
   for (int el = 0; el < pmesh.GetNE(); el++)
   {
     ElementTransformation *trans = pmesh.GetElementTransformation(el);
     for (int q = 0; q < ir->GetNPoints(); q++)
     {
       const IntegrationPoint &ip = ir->IntPoint(q);
       Vector phys_point(pmesh.Dimension());
       trans->Transform(ip, phys_point); // Map to physical coordinates

       quadfun(qf_idx++) = my_function(phys_point);
     }
   }


   {
      int precision = 8;
      cout.precision(precision);

      ofstream omesh("main.mesh");
      omesh.precision(precision);
      pmesh.Print(omesh);
      ofstream oquad("intial.qf");
      oquad.precision(precision);
      quadfun.Save(oquad);

   }


   //
   //Create visit file
   //
   {
     VisItDataCollection *pd = NULL;
     pd = new VisItDataCollection("quad_fun", &pmesh);
     pd->SetPrefixPath("Visit");
     pd->RegisterQField("quad_field", &quadfun);
     pd->SetLevelsOfDetail(order);
     //pd->SetDataFormat(VTKFormat::BINARY);
     //pd->SetHighOrderOutput(true);
     pd->SetCycle(0);
     pd->SetTime(0.0);
     pd->Save();
     delete pd;
   }


   // 14. Send the solution by socket to a GLVis server.
   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << quadfun << flush;
   }

   // 15. Free the used memory.


   return 0;
}

// Initial condition
real_t my_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
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
               real_t rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const real_t s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                        std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const real_t f = M_PI;
         return 2 * sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}
