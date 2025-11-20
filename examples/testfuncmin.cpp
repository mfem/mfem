//                                MFEM Example 0
//
// make testfuncmin -j4 && ./testfuncmin -n 1000 -d 1 -l2 -bt 2 -bref 2 -seed 1134 -o 3 -c 1

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   int order = 2;
   int dim = 1;
   int b_type = 1;
   int nbrute = 100;
   int bound_ref = 4;
   int seed = 0;
   bool continuous = true;
   int case_num = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dim, "-d", "--dim", "Dimension of the problem");
   args.AddOption(&nbrute, "-n", "--nbrute", "Number of brute force iterations");
   args.AddOption(&bound_ref, "-bref", "--bound-ref-factor",
                  "Refinement factor for piecewise linear bounds");
   args.AddOption(&seed, "-seed", "--seed", "Random seed for GridFunction");
   args.AddOption(&b_type, "-bt", "--basis-type",
                  "Project input function to a different bases. "
                  "0 = Gauss-Legendre nodes. "
                  "1 = Gauss-Lobatto nodes. "
                  "2 = uniformly spaced nodes. ");
   args.AddOption(&continuous, "-h1", "--h1", "-l2", "--l2",
                  "Use continuous or discontinuous space.");
   args.AddOption(&case_num, "-c", "--case", "Case number for testing.");

   args.ParseCheck();

   if (continuous && b_type == 0)
   {
      MFEM_ABORT("Continuous space do not support GL nodes. "
                 "Please use basis type: 1 for Lagrange interpolants on GLL "
                 " nodes 2 for positive bases on uniformly spaced nodes.");
   }

   Mesh *mesh = nullptr;
   if (dim == 1)
   {
      mesh = new Mesh(Mesh::MakeCartesian1D(1));
   }
   else if (dim == 2)
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL, 1.0, 1.0));
   }
   else if (dim == 3)
   {
      mesh = new Mesh(Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON, 1.0, 1.0,
                                            1.0));
   }

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   FiniteElementCollection *fec = nullptr;
   if (continuous)
   {
      fec = new H1_FECollection(order, dim, b_type);
   }
   else
   {
      fec = new L2_FECollection(order, dim, b_type);
   }

   FiniteElementSpace fespace(mesh, fec);
   GridFunction x(&fespace);
   x.Randomize(seed);
   int vdim = fespace.GetVDim();
   MFEM_VERIFY(order > 1, "Order must be greater than 1")
   if (x.Min() == x(0))
   {
      real_t temp = x(2);
      x(2) = x(0);
      x(0) = temp;
   }
   else if (x.Min() == x(1))
   {
      real_t temp = x(2);
      x(2) = x(1);
      x(1) = temp;
   }

   if (case_num == 1)
   {
      // Setup a step function with Bernstein.
      MFEM_VERIFY(dim == 1 && !continuous && b_type == 2,
                  "For case 1, use Bernstein, order odd, discontinuous, and "
                  "basis type 2.");

      DenseMatrix projmat;
      NodalTensorFiniteElement *ntfe = new L2_SegmentElement(order,
                                                             BasisType::ClosedUniform);
      Vector xdummy = x;
      int ns = floor(xdummy.Size()/2);
      int diff = xdummy.Size()%2;
      for (int i = 0; i < ns; i++)
      {
         xdummy(i) = 2.0;
         xdummy(2*ns-i-(diff == 0? 1 : 0)) = 6.0;
      }
      if (xdummy.Size() % 2 == 1)
      {
         xdummy(ns) = 4.0;
      }
      xdummy.Print();
      const FiniteElement *fe = fespace.GetFE(0);
      ElementTransformation *eltran = fespace.GetElementTransformation(0);
      fe->Project(*ntfe, *eltran, projmat);
      projmat.Mult(xdummy, x);
      delete ntfe;
   }

   std::cout << " Print x: " << std::endl;
   x.Print();
   Vector xlex = x;

   {
      const FiniteElement *fe = fespace.GetFE(0);
      const TensorBasisElement *tbe =
         dynamic_cast<const TensorBasisElement *>(fe);
      MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
      const Array<int> &dof_map = tbe->GetDofMap();
      for (int i = 0; i < dof_map.Size(); i++)
      {
         xlex(i) = x(dof_map[i]);
      }
   }

   Vector global_min(vdim), global_max(vdim);
   if (nbrute > 0)
   {
      global_min = numeric_limits<real_t>::max();
      global_max = numeric_limits<real_t>::min();
      // search for the minimum value of pfunc_proj in each element at
      // an array of integration points
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         IntegrationPoint ip;
         for (int k = 0; k < (dim > 2 ? nbrute : 1); k++)
         {
            ip.z = k/(nbrute-1.0);
            for (int j = 0; j < (dim > 1 ? nbrute : 1); j++)
            {
               ip.y = j/(nbrute-1.0);
               for (int i = 0; i < nbrute; i++)
               {
                  ip.x = i/(nbrute-1.0);
                  for (int d = 0; d < vdim; d++)
                  {
                     real_t val = x.GetValue(e, ip, d+1);
                     global_min(d) = min(global_min(d), val);
                     global_max(d) = max(global_max(d), val);
                  }
               }
            }
         }
      }
   }

   // Compute minima using recursion
   Vector lower, upper;
   // PLBound plb = x.GetElementBounds(lower, upper, bound_ref);
   PLBound plb(order + 1, bound_ref*(order+1), b_type, 0, 0.0);
   plb.SetProjectionFlagForBounding(false);
   x.GetElementBoundsAtControlPoints(0, plb, lower, upper);
   auto minima = x.GetElementMinima(0, plb);

   cout << "Brute force minima " << global_min(0) << endl;
   cout << "Computed minima using one-shot bounds: " << lower.Min() << " " <<
        upper.Min() << std::endl;
   cout << "Computed minima using recursion: " << minima.first << " " <<
        minima.second << std::endl;
   cout << "Minima estimates using coefficients/brute force/bounding(1)/bounding(recurse): "
        << std::endl;
   cout <<  x.Min() << " " << global_min(0) << " " << lower.Min() << " " <<
        minima.first << " " << std::endl;

   cout << "PlottingData: ";
   cout << order << "," << continuous << "," << b_type << ","  << dim << ",";
   for (int i = 0; i < x.Size(); i++)
   {
      cout << xlex(i) << ",";
   }
   cout << global_min(0) << "," << lower.Min() << "," << upper.Min() << "," <<
        minima.first << "," << minima.second << endl;

   // Check bounding matrix:
   DenseMatrix lboundmat = plb.GetLowerBoundMatrix(dim);
   DenseMatrix uboundmat = plb.GetUpperBoundMatrix(dim);
   Vector lbcheck(lower.Size()), ubcheck(upper.Size());
   lbcheck = 0.0;
   ubcheck = 0.0;
   lboundmat.Mult(xlex, lbcheck);
   uboundmat.Mult(xlex, ubcheck);
   std::cout << lbcheck.Min() << " " << lower.Min() << std::endl;
   std::cout << ubcheck.Min() << " " << upper.Min() << std::endl;


   delete fec;
   delete mesh;

   return 0;
}
