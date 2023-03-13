//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"

#include "mfem.hpp"
#include <fstream>
#include <iostream>
//#include "nodepair.hpp"

using namespace std;
using namespace mfem;

Vector GetNormalVector(Mesh & mesh, const int elem, const double *ref)
{
   ElementTransformation *trans = mesh.GetElementTransformation(elem);
   const int dim = mesh.Dimension();
   const int spaceDim = trans->GetSpaceDim();

   MFEM_VERIFY(spaceDim == 3, "");

   Vector n(spaceDim);

   IntegrationPoint ip;
   ip.Set(ref, dim);

   trans->SetIntPoint(&ip);
   //CalcOrtho(trans->Jacobian(), n);  // Works only for face transformations
   const DenseMatrix jac = trans->Jacobian();

   int dimNormal = -1;
   int normalSide = -1;

   const double tol = 1.0e-8;
   for (int i=0; i<dim; ++i)
   {
      const double d0 = std::abs(ref[i]);
      const double d1 = std::abs(ref[i] - 1.0);

      const double d = std::min(d0, d1);
      // TODO: this works only for hexahedral meshes!

      if (d < tol)
      {
         MFEM_VERIFY(dimNormal == -1, "");
         dimNormal = i;

         if (d0 < tol)
         {
            normalSide = 0;
         }
         else
         {
            normalSide = 1;
         }
      }
   }

   MFEM_VERIFY(dimNormal >= 0 && normalSide >= 0, "");

   MFEM_VERIFY(dim == 3, "");

   std::vector<Vector> tang(2);

   int tangDir[2] = {-1, -1};
   {
      int t = 0;
      for (int i=0; i<dim; ++i)
      {
         if (i != dimNormal)
         {
            tangDir[t] = i;
            t++;
         }
      }

      MFEM_VERIFY(t == 2, "");
   }

   for (int i=0; i<2; ++i)
   {
      tang[i].SetSize(3);

      Vector tangRef(3);
      tangRef = 0.0;
      tangRef[tangDir[i]] = 1.0;

      jac.Mult(tangRef, tang[i]);
   }

   Vector c(3);  // Cross product

   c[0] = (tang[0][1] * tang[1][2]) - (tang[0][2] * tang[1][1]);
   c[1] = (tang[0][2] * tang[1][0]) - (tang[0][0] * tang[1][2]);
   c[2] = (tang[0][0] * tang[1][1]) - (tang[0][1] * tang[1][0]);

   c /= c.Norml2();

   Vector nref(3);
   nref = 0.0;
   nref[dimNormal] = 1.0;

   Vector ndir(3);
   jac.Mult(nref, ndir);

   ndir /= ndir.Norml2();

   const double dp = ndir * c;

   // TODO: eliminate c?
   n = c;
   if (dp < 0.0)
   {
      n *= -1.0;
   }

   return n;
}

// Coordinates in xyz are assumed to be ordered as [X, Y, Z]
// where X is the list of x-coordinates for all points and so on.
void FindPointsInMesh(Mesh & mesh, Vector const& xyz)
{
   const int dim = mesh.Dimension();
   const int np = xyz.Size() / dim;

   MFEM_VERIFY(np * dim == xyz.Size(), "");

   mesh.EnsureNodes();

   //FindPointsGSLIB finder(MPI_COMM_WORLD);
   FindPointsGSLIB finder;
   finder.Setup(mesh);

   finder.FindPoints(xyz);

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   Array<unsigned int> codes = finder.GetCode();

   /// Return element number for each point found by FindPoints.
   Array<unsigned int> elems = finder.GetElem();

   /// Return reference coordinates for each point found by FindPoints.
   Vector refcrd = finder.GetReferencePosition();

   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   Vector dist = finder.GetDist();

   MFEM_VERIFY(dist.Size() == np, "");
   MFEM_VERIFY(refcrd.Size() == np * dim, "");
   MFEM_VERIFY(elems.Size() == np, "");
   MFEM_VERIFY(codes.Size() == np, "");

   bool allfound = true;
   for (auto code : codes)
      if (code == 2) { allfound = false; }

   MFEM_VERIFY(allfound, "A point was not found");

   cout << "Maximum distance of projected points: " << dist.Max() << endl;

   for (int i=0; i<np; ++i)
   {
      cout << "Point " << i << ": (";
      for (int j=0; j<dim; ++j)
      {
         cout << xyz[i + (j*np)];
         if (j == dim-1)
         {
            cout << ")" << endl;
         }
         else
         {
            cout << ", ";
         }
      }

      cout << "  element: " << elems[i] << endl;
      cout << "  element " << elems[i] << " vertices:" << endl;
      Array<int> vert;
      mesh.GetElementVertices(elems[i], vert);
      for (auto v : vert)
      {
         cout << "    " << v << endl;
      }

      cout << "  reference coordinates: (";
      for (int j=0; j<dim; ++j)
      {
         cout << refcrd[(i*dim) + j];
         if (j == dim-1)
         {
            cout << ")" << endl;
         }
         else
         {
            cout << ", ";
         }
      }

      Vector normal = GetNormalVector(mesh, elems[i], refcrd.GetData() + (i*dim));
      cout << "  normal vector: ";
      normal.Print();

      IntegrationPoint ip;
      ip.Set(refcrd.GetData() + (i*dim), dim);
      ElementTransformation *trans = mesh.GetElementTransformation(elems[i]);
      Vector phys(trans->GetSpaceDim());
      trans->Transform(ip, phys);
      cout << "  physical coordinates: ";
      phys.Print();
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file1 = "block1.mesh";
   const char *mesh_file2 = "block2.mesh";

   Array<int> attr;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file1, "-m1", "--mesh1",
                  "First mesh file to use.");
   args.AddOption(&mesh_file2, "-m2", "--mesh2",
                  "Second mesh file to use.");
   args.AddOption(&attr, "-at", "--attributes-surf",
                  "Attributes of boundary faces on contact surface for mesh 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh1(mesh_file1, 1, 1);
   Mesh mesh2(mesh_file2, 1, 1);

   const int dim = mesh1.Dimension();
   MFEM_VERIFY(dim == mesh2.Dimension(), "");


   //  Define a finite element space on the mesh. Here we use vector finite
   //  elements, i.e. dim copies of a scalar finite element space. The vector
   //  dimension is specified by the last argument of the FiniteElementSpace
   //  constructor.
   FiniteElementCollection *fec1;
   FiniteElementSpace *fespace1;
   fec1 = new H1_FECollection(1, dim);
   fespace1 = new FiniteElementSpace(&mesh1, fec1, dim);
   cout << "Number of finite element unknowns for mesh1: " << fespace1->GetTrueVSize()
        << endl;

   FiniteElementCollection *fec2 = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace2 = new FiniteElementSpace(&mesh2, fec2, dim);
   cout << "Number of finite element unknowns for mesh2: " << fespace2->GetTrueVSize()
        << endl;

   // Determine the list of true (i.e. conforming) essential boundary dofs.
   // In this example, the boundary conditions are defined by marking only
   // boundary attribute 1 from the mesh as essential and converting it to a
   // list of true dofs.
   Array<int> ess_tdof_list1, ess_bdr1(mesh1.bdr_attributes.Max());
   ess_bdr1 = 0;
   ess_bdr1[0] = 1; 
   // Not ready to be passed on yet
   // fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   Array<int> ess_tdof_list2, ess_bdr2(mesh2.bdr_attributes.Max());
   ess_bdr2 = 0; 
   ess_bdr2[0] = 1; 

   // Define the displacement vector x as a finite element grid function
   // corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x1(fespace1);
   x1 = 0.0;

   GridFunction x2(fespace2);
   x2 = 0.0;

   attr.Sort();
   cout << "Boundary attributes for contact surface faces in mesh 2" << endl;
   for (auto a : attr)
   {
      cout << a << endl;
   }

   Array<int> bdryFaces2;  // TODO: remove this?

   std::set<int> bdryVerts2;
   for (int b=0; b<mesh2.GetNBE(); ++b)
   {
      if (attr.FindSorted(mesh2.GetBdrAttribute(b)) >= 0)
      {
         bdryFaces2.Append(b);
         Array<int> vert;
         mesh2.GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            bdryVerts2.insert(v);
         }
      }
   }

   int npoints = bdryVerts2.size();
   Vector xyz(dim * npoints);
   xyz = 0.0;

   cout << "Boundary vertices for contact surface vertices in mesh 2" << endl;

   int count = 0;
   for (auto v : bdryVerts2)
   {
      cout << v << ": " << mesh2.GetVertex(v)[0] << ", "
           << mesh2.GetVertex(v)[1] << ", "
           << mesh2.GetVertex(v)[2] << endl;

      for (int i=0; i<dim; ++i)
      {
         xyz[count + (i * npoints)] = mesh2.GetVertex(v)[i];
      }

      count++;
   }

   MFEM_VERIFY(count == npoints, "");

   FindPointsInMesh(mesh1, xyz);

   // testing functions
   /*
   Vector x1(3);
   Vector xi(2);
   DenseMatrix coords2(4,3);
   x1[0] = 0.138357116570237; x1[1] =0.781266036110795; x1[2] = 0.781266036110795;
   xi[0] = -0.340604159745217; xi[1] = -0.340604159745216;
   coords2(0,0) =0.028711699111254;  coords2(0,1) =  0.671052640577229; coords2(0,2) =  0.671052640577229;
   coords2(1,0) =0.040247874087262; coords2(1,1) = 1.012958013114749; coords2(1,2) =   0.670855581724106;
   coords2(2,0) = 0.041207251768282; coords2(2,1) = 1.012615442973163; coords2(2,2) =  1.012615442973163;
   coords2(3,0) =0.040247874087262; coords2(3,1) = 0.670855581724106;  coords2(3,2) =  1.012958013114749;

   double gap = 0.; Vector dg(15); dg = 0.0; 
   DenseMatrix d2g;
   NodeSegConPairs(x1, xi, coords2, gap, dg, d2g);
   cout << gap <<endl;
   //d2g.Print();
   */
   return 0;
}
