// Compile with: make nurbs_naca_cmesh
//
// Sample run:   ./nurbs_naca_cmesh -ntail 80 -nrad 80 -ntip 20 -nwake 40 -sw 2.0
//                 -srad 2.5 -stip 1.1 -aoa 3
//
// Description:  This example code demonstrates the use of MFEM to create a C-mesh
//               around a NACA-foil section. The foil section is defined in the class
//               NACA4, which can be easily replaced with any other description of a
//               foil section. To apply an angle of attack, we rotate the domain
//               around the origin.
//
//               The mesh employs five patches of which two describe the domain behind the
//               foil section (wake). The boundary describing the foil section is divided
//               over three patches. One patch describes the domain adjacent to the boundary
//               which describes the tip / leading edge of the foil section and two patches
//               describe the domain which is adjacent to the two boundaries describing the
//               remainder of the foil section. The aim is to create a mesh with the highest
//               quality close to the boundary of the foil section and the wake.
//
//               The example returns a visit data structure for visualisation. Note that
//               one will need to increase the multiress-control to inspect the shape of the
//               NURBS.
//
//               Possible improvements:
//               - Implement optimization with TMOP
//               - Streamline GetTipXY() for two options

#include "mfem.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

// Object that discribes a symmetric NACA foil section
class NACA4
{
protected:
   // Constants describing the thickness profile
   double A, B, C, D, E;
   // Thickness of the foil section
   double t;
   // Chord of the foil section: foil length
   double c;
   // Maximum number of iterations for Newton solver
   int iter_max;
   // Tolerance for Newton solver
   double epsilon;
public:
   NACA4(double t_, double c_);
   // Returns the coordinate y corresponding to coordinate @a xi
   double y(double xi);
   // Returns the derivative of the curve at location coordinate @a xi
   double dydx(double xi);
   // Returns the curve length at coordinate @a xi
   double len(double xi);
   // Returns the derivative of the curve length at coordinate @a xi
   double dlendx(double xi);
   // Returns the coordinate x corresponding to the curve length @a l from
   // the tip of the foil section
   double xl(double l);
   // Get the chord of the foil_section
   const double GetChord() {return c;}
};

NACA4::NACA4(double t_, double c_)
{
   t = t_;
   c = c_;
   A = 0.2969, B = 0.1260, C = 0.3516, D = 0.2843, E = 0.1036;
   iter_max = 1000;
   epsilon = 1e-8;
}

double NACA4::y(double x)
{
   double y = 5*t*(A*sqrt(x/c) - B*x/c - C*pow(x/c,2) + D*pow(x/c,3) - E*pow(x/c,
                                                                             4));
   return y*c;
}

double NACA4::dydx(double x)
{
   double y = 5*t*(0.5 * A/sqrt(x/c) - B - 2*C*x/c + 3* D*pow(x/c,
                                                              2) - 4* E*pow(x/c,3));
   return y*c;
}

double NACA4::len(double x)
{
   double l = 5 * t * (A*sqrt(x/c) - B*x - C*pow(x/c,2) + D*pow(x/c,
                                                                3) - E * pow(x/c,4)) + x/c;
   return l*c;
}

double NACA4::dlendx(double xi)
{
   return 1 + dydx(xi);
}

double NACA4::xl(double l)
{
   double x = l; // Initial guess, length should be a good one
   double h;
   int i = 0;
   do
   {
      x = abs(x); // The function and its derivative do not exist for x < 0
      // Newton step: x(i+1) = x(i) - f(x) / f'(x)
      h = (len(x) - l)/dlendx(x);
      x = x - h;
   }
   while (abs(h) >= epsilon && i++ < iter_max);

   if (i >= iter_max) { mfem_error("Did not find root"); }
   return x;
}

// Function that finds the coordinates of the control points of the tip of the foil section @a xy
// based on the @a foil_section, knot vector @a kv and tip fraction @a tf
// We have two cases, with an odd number of control points and with an even number of
// control points. These may be streamlined in the future.
void GetTipXY(NACA4 foil_section, KnotVector *kv, double tf, Array<Vector*> &xy)
{
   int ncp = kv->GetNCP();
   // Length of halve the curve: the boundary covers both sides of the tip
   double l = foil_section.len(tf * foil_section.GetChord());

   // Find location of maxima of knot vector
   Array<int> i_args;
   Vector xi_args, u_args;
   kv->FindMaxima(i_args,xi_args, u_args);

   // We have two cases: one with an odd number of control points and one
   // with an even number of control points.
   int n = ncp/2;
   if (ncp % 2)
   {
      // Find arc lengths to control points on upperside of foil section
      // then find x-coordinates.
      Vector xcp(n+1);
      for (int i = 0; i < n+1; i++)
      {
         double u = 2*(u_args[n+i]-0.5);
         double lcp = u * l;
         xcp[i] = foil_section.xl(lcp);
      }

      // Find corresponding xy vector
      xy[0]->SetSize(2*n+1); xy[1]->SetSize(2*n+1);
      xy[0]->Elem(n) = 0; xy[1]->Elem(n) = 0; // Foil section tip
      for (int i = 0; i < n; i++)
      {
         // Lower halve
         xy[0]->Elem(i) = xcp[n-i];     xy[1]->Elem(i) = -foil_section.y(xcp[n-i]);

         // Upper halve
         xy[0]->Elem(n+1+i) = xcp[i+1]; xy[1]->Elem(n+1+i) = foil_section.y(xcp[i+1]);
      }
   }
   else
   {
      // Find arc lengths to control points on upperside of foil section then find x-coordinates
      Vector xcp(n);
      for (int i = 0; i < n; i++)
      {
         double u = 2*(u_args[n+i]-0.5);
         double lcp = u * l;
         xcp[i] = foil_section.xl(lcp);
      }

      // Find corresponding xy vector
      xy[0]->SetSize(2*n); xy[1]->SetSize(2*n);
      for (int i = 0; i < n; i++)
      {
         // Lower halve
         xy[0]->Elem(i) = xcp[n-1-i]; xy[1]->Elem(i) = -foil_section.y(xcp[n-1-i]);

         // Upper halve
         xy[0]->Elem(n+i) = xcp[i];   xy[1]->Elem(n+i) = foil_section.y(xcp[i]);
      }
   }
}

// Function that returns a uniform knot vector based on the @a order and the
// number of control points @a ncp.
KnotVector *UniformKnotVector(int order, int ncp)
{
   KnotVector *kv = new KnotVector(order, ncp);

   for (int i = 0; i < order+1; i++)
   {
      (*kv)[i] = 0.0;
   }
   for (int i = order+1; i < ncp; i++)
   {
      (*kv)[i] = (i-order)/double(ncp-order);
   }
   for (int i = ncp ; i < ncp + order + 1; i++)
   {
      (*kv)[i] = 1.0;
   }
   return kv;
}

// Function that returns a knot vector which is stretched with stretch @s
// with the form x^s based on the @a order and the number of control points @a ncp.
// Special case @a s = 0 will give a uniform knot vector.
KnotVector *PowerStretchKnotVector(int order, int ncp, double s = 0.0)
{
   KnotVector *kv = new KnotVector(order, ncp);

   for (int i = 0; i < order+1; i++)
   {
      (*kv)[i] = 0.0;
   }
   for (int i = order+1; i < ncp; i++)
   {
      (*kv)[i] = (i-order)/double(ncp-order);
      if (s > 0) { (*kv)[i] = pow((*kv)[i], s); }
      if (s < 0) { (*kv)[i] = 1.0 - pow(1.0-(*kv)[i], -s); }
   }
   for (int i = ncp ; i < ncp + order + 1; i++)
   {
      (*kv)[i] = 1.0;
   }
   return kv;
}

// Function that returns a knot vector with a hyperbolic tangent spacing
// with a cut-off @c using the @a order and the number of control points @a ncp.
KnotVector *TanhKnotVector(int order, int ncp, double c)
{
   KnotVector *kv = new KnotVector(order, ncp);

   for (int i = 0; i < order+1; i++)
   {
      (*kv)[i] = 0.0;
   }
   for (int i = order+1; i < ncp; i++)
   {
      (*kv)[i] = (i-order)/double(ncp-order);
      (*kv)[i] = 1 + tanh(c * ((*kv)[i]-1))/tanh(c);
   }
   for (int i = ncp ; i < ncp + order + 1; i++)
   {
      (*kv)[i] = 1.0;
   }
   return kv;
}

// Function that returns a knot vector with a hyperbolic tangent spacing from both sides
// of the knot vector with a cut-off @c using the @a order and the number of control points @a ncp.
KnotVector *DoubleTanhKnotVector(int order, int ncp, double c)
{
   KnotVector *kv = UniformKnotVector(order, ncp);

   for (int i = 0; i < order+1; i++)
   {
      (*kv)[i] = 0.0;
   }
   for (int i = order+1; i < ncp; i++)
   {
      if ((*kv)[i] < 0.5)
      {
         (*kv)[i] = -1 + 2*( 1 - (i-order)/double(ncp-order));
         (*kv)[i] = 0.5 * abs((tanh(c * ((*kv)[i]-1))/tanh(c)));
      }
      else
      {
         (*kv)[i] = 2*((i-order)/double(ncp-order) - 0.5);
         (*kv)[i] = 0.5 +(1 + tanh(c * ((*kv)[i]-1))/tanh(c))/2;
      }
   }
   for (int i = ncp ; i < ncp + order + 1; i++)
   {
      (*kv)[i] = 1.0;
   }
   return kv;
}

// Function that evaluates a linear function which describes the boundary distance
// based on the flair angle @a flair, smallest boundary bistance @a bd and
// coordinate @a x.
// This function / the flair angle is mainly used to be able to enforce inflow on
// the top and bottom boundary and to create an elegant mesh.
double FlairBoundDist(double flair, double bd, double x)
{
   double b = sin(flair);
   double c = bd*cos(flair) + bd * sin(flair) * sin(flair);
   return b * x + c;
}

int main(int argc, char *argv[])
{
   int mdim = 2;
   int order = 2;

   //
   // 1. Parse command-line options.
   //
   OptionsParser args(argc, argv);
   const char *msh_path = "";
   const char *msh_filename = "naca-cmesh";
   args.AddOption(&msh_path, "-p", "--mesh-path",
                  "Path in which the generated mesh is saved.");
   args.AddOption(&msh_filename, "-m", "--mesh-file",
                  "File where the generated mesh is written to.");

   // Foil section options
   double foil_length = 1.0;
   double foil_thickness = 0.12;
   double aoa = 0.0;
   args.AddOption(&foil_length, "-l", "--foil-length",
                  "Length of the used foil in the mesh. ");
   args.AddOption(&foil_thickness, "-t", "--foil-thickness",
                  "Thickness of the used NACA foil in the mesh as a fraction of length.");
   args.AddOption(&aoa, "-aoa", "--angle-of-attack",
                  "Angle of attack of the foil. ");

   // Mesh options
   double boundary_dist = 3.0;
   double wake_length  = 3.0;
   double tip_fraction = 0.05;
   double flair = -999;
   args.AddOption(&boundary_dist, "-b", "--boundary-distance",
                  "Radius of the c-mesh, distance between the foil and the boundary");
   args.AddOption(&wake_length, "-w", "--wake_length",
                  "Length of the mesh after the foil");
   args.AddOption(&tip_fraction, "-tf", "--tip-fraction",
                  "Fraction of the length of the foil that will be in tip patch");
   args.AddOption(&flair, "-f", "--flair-angle",
                  "Flair angle of the top and bottom boundary to enforce inflow. If left \
                at default, the flair angle is determined automatically to create an \
                elegant mesh.");

   int ncp_tip  = 3;
   int ncp_tail = 3;
   int ncp_wake = 3;
   int ncp_bnd = 3;
   args.AddOption(&ncp_tip, "-ntip", "--ncp-tip",
                  "Number of control points used over the tip of the foil.");
   args.AddOption(&ncp_tail, "-ntail", "--ncp-tail",
                  "Number of control points used over the tail of the foil.");
   args.AddOption(&ncp_wake, "-nwake", "--ncp-wake",
                  "Number of control points used over the wake behind the foil.");
   args.AddOption(&ncp_bnd, "-nbnd", "--ncp-circ",
                  "Number of control points used between the foil and the boundary.");

   double str_tip = 1;
   double str_wake = 1;
   double str_bnd = 1;
   double str_tail = 1;
   args.AddOption(&str_tip, "-stip", "--str-tip",
                  "Stretch of the knot vector of the tip.");
   args.AddOption(&str_tail, "-stail", "--str-tail",
                  "Stretch of the knot vector of the tail.");
   args.AddOption(&str_wake, "-sw", "--str-wake",
                  "Stretch of the knot vector of the wake.");
   args.AddOption(&str_bnd, "-srad", "--str-circ",
                  "Stretch of the knot vector of the circle.");

   // Parse and print commandline options
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Convert fraction
   double tail_fraction = 1.0 - tip_fraction;

   // Convert angles to radians
   double deg2rad = M_PI/180;
   aoa = aoa*deg2rad;

   //
   // 2. Create knot vectors
   //
   KnotVector *kv0 = TanhKnotVector(order, ncp_wake, str_wake);
   kv0->Flip();
   KnotVector *kv4 = new KnotVector(*kv0);
   kv4->Flip();

   KnotVector *kv1 = PowerStretchKnotVector(order, ncp_tail, -str_tail);
   KnotVector *kv3 = PowerStretchKnotVector(order, ncp_tail, str_tail);
   KnotVector *kv2 = DoubleTanhKnotVector(order, ncp_tip, str_tip);
   KnotVector *kvr = TanhKnotVector(order, ncp_bnd, str_bnd);

   KnotVector *kv_o1 = UniformKnotVector(1, 2);
   KnotVector *kv_o2 = UniformKnotVector(2, 3);

   // Variables required for curve interpolation
   Vector xi_args, u_args;
   Array<int> i_args;
   Array<Vector*> xyf(2);
   xyf[0] = new Vector();
   xyf[1] = new Vector();

   //
   // 3. Create required (variables for) curves: foil_section and flair
   //
   NACA4 foil_section(foil_thickness, foil_length);

   // The default flair angle is defined to be the same as the angle of the curve of the
   // foil section to create an elegant mesh.
   if (flair == -999)
   {
      flair = atan(foil_section.dydx(tip_fraction*foil_length));
   }

   //
   // 4. We map coordinates in patches, apply refinement and interpolate the foil section in
   //    patches 1, 2 and 3. Note the case of non-unity weights in patch 2 to create a circular
   //    shape: its coordinates are converted to homogeneous coordinates. This is not needed
   //    for other patches as homogeneous coordinates and cartesian coordinates are the same
   //    for patches with unity weight.
   //

   // Patch 0: lower wake part behind foil section.
   NURBSPatch patch0(kv_o1, kv_o1, 3);
   {
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            patch0(i,j,2) = 1.0;
         }

      // Define points
      patch0(0,0,0) = foil_length + wake_length;
      patch0(0,0,1) = 0.0;

      patch0(1,0,0) = foil_length;
      patch0(1,0,1) = 0.0;

      patch0(0,1,0) = foil_length + wake_length;
      patch0(0,1,1) = -FlairBoundDist(flair, boundary_dist, patch0(0,1,0));

      patch0(1,1,0) = foil_length;
      patch0(1,1,1) = -FlairBoundDist(flair, boundary_dist, patch0(1,1,0));

      // Refine
      patch0.DegreeElevate(0, order-1);
      patch0.KnotInsert(0, *kv0);
      patch0.DegreeElevate(1, order-1);
      patch0.KnotInsert(1, *kvr);
   }

   // Patch 1: Lower tail of foil
   NURBSPatch patch1(kv_o1, kv_o1, 3);;
   {
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            patch1(i,j,2) = 1.0;
         }

      // Define points
      patch1(0,0,0) = foil_length;
      patch1(0,0,1) = 0.0;

      patch1(1,0,0) = tip_fraction*foil_length;
      patch1(1,0,1) = -foil_section.y(patch1(1,0,0));

      patch1(0,1,0) = foil_length;
      patch1(0,1,1) = -FlairBoundDist(flair, boundary_dist, patch1(0,1,0));;

      patch1(1,1,0) = -boundary_dist*sin(flair) + tip_fraction*foil_length;
      patch1(1,1,1) = -boundary_dist*cos(flair);

      // Refine
      patch1.DegreeElevate(0, order-1);
      patch1.KnotInsert(0, *kv1);

      int ncp = kv1->GetNCP();
      xyf[0]->SetSize(ncp); xyf[1]->SetSize(ncp);

      // Project foil
      kv1->FindMaxima(i_args,xi_args, u_args);
      for (int i = 0; i < ncp; i++)
      {
         (*xyf[0])[i] = foil_length*(1.0 - tail_fraction*u_args[i]);
         (*xyf[1])[i] = -foil_section.y((*xyf[0])[i]);
      }

      kv1->FindInterpolant(xyf);
      for (int i = 0; i < ncp; i++)
      {
         patch1(i,0,0) = (*xyf[0])[i];
         patch1(i,0,1) = (*xyf[1])[i];
      }

      patch1.DegreeElevate(1, order-1);
      patch1.KnotInsert(1, *kvr);
   }

   // Patch 2: Tip of foil section
   NURBSPatch patch2(kv_o2, kv_o1, 3);
   {
      // Define weights
      for (int i = 0; i < 3; i++)
         for (int j = 0; j < 2; j++)
         {
            patch2(i,j,2) = 1.0;
         }

      // Define points
      patch2(2,0,0) = tip_fraction*foil_length;
      patch2(2,0,1) = foil_section.y(patch2(2,0,0));

      patch2(1,0,0) = 0.0;
      patch2(1,0,1) = 0.0;
      patch2(1,0,2) = cos((180*deg2rad-2*flair)/2);

      patch2(0,0,0) = tip_fraction*foil_length;
      patch2(0,0,1) = -foil_section.y(patch2(0,0,0));


      patch2(2,1,0) = -boundary_dist*cos(90*deg2rad-flair) + tip_fraction*foil_length;
      patch2(2,1,1) = boundary_dist*sin(90*deg2rad-flair);

      patch2(1,1,0) = -boundary_dist/sin(flair);
      patch2(1,1,1) = 0.0;
      patch2(1,1,2) = cos((180*deg2rad-2*flair)/2);

      patch2(0,1,0) = -boundary_dist*cos(90*deg2rad-flair) + tip_fraction*foil_length;
      patch2(0,1,1) = -boundary_dist*sin(90*deg2rad-flair);

      // Deal with non-uniform weight: convert to homogeneous coordinates
      patch2(1,0,0) *= patch2(1,0,2);
      patch2(1,0,1) *= patch2(1,0,2);
      patch2(1,1,0) *= patch2(1,1,2);
      patch2(1,1,1) *= patch2(1,1,2);

      // Refine
      patch2.DegreeElevate(0, order-2);
      patch2.KnotInsert(0, *kv2);

      // Project foil
      int ncp = kv2->GetNCP();
      xyf[0]->SetSize(ncp); xyf[1]->SetSize(ncp);

      GetTipXY(foil_section, kv2, tip_fraction, xyf);

      kv2->FindInterpolant(xyf);
      for (int i = 0; i < ncp; i++)
      {
         // Also deal with non-uniform weights here: convert to homogeneous coordinates
         patch2(i,0,0) = (*xyf[0])[i]*patch2(i,0,2);
         patch2(i,0,1) = (*xyf[1])[i]*patch2(i,0,2);
      }

      // Project circle
      patch2.DegreeElevate(1, order-1);
      patch2.KnotInsert(1, *kvr);
   }

   // Patch 3: Upper part of trailing part foil section
   NURBSPatch patch3(kv_o1, kv_o1, 3);;
   {
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            patch3(i,j,2) = 1.0;
         }

      // Define points
      patch3(0,0,0) = tip_fraction*foil_length;
      patch3(0,0,1) = foil_section.y(patch3(0,0,0));

      patch3(1,0,0) = foil_length;
      patch3(1,0,1) = 0.0;

      patch3(0,1,0) = -boundary_dist*sin(flair) + tip_fraction*foil_length;
      patch3(0,1,1) = boundary_dist*cos(flair);

      patch3(1,1,0) = foil_length;
      patch3(1,1,1) = FlairBoundDist(flair, boundary_dist, patch3(1,1,0));;

      // Refine
      patch3.DegreeElevate(0, order-1);
      patch3.KnotInsert(0, *kv3);

      int ncp = kv3->GetNCP();
      xyf[0]->SetSize(ncp); xyf[1]->SetSize(ncp);

      // Project foil
      kv3->FindMaxima(i_args,xi_args, u_args);
      for (int i = 0; i < ncp; i++)
      {
         (*xyf[0])[i] = foil_length*(tip_fraction + tail_fraction*u_args[i]);
         (*xyf[1])[i] = foil_section.y((*xyf[0])[i]);
      }

      kv3->FindInterpolant(xyf);
      for (int i = 0; i < ncp; i++)
      {
         patch3(i,0,0) = (*xyf[0])[i];
         patch3(i,0,1) = (*xyf[1])[i];
      }

      patch3.DegreeElevate(1, order-1);
      patch3.KnotInsert(1, *kvr);
   }

   // Patch 4: Upper trailing wake part
   NURBSPatch patch4(kv_o1, kv_o1, 3);;
   {
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            patch4(i,j,2) = 1.0;
         }

      // Define points
      patch4(0,0,0) = foil_length;
      patch4(0,0,1) = 0.0;

      patch4(1,0,0) = foil_length+ wake_length;
      patch4(1,0,1) = 0.0;

      patch4(0,1,0) = foil_length;
      patch4(0,1,1) = FlairBoundDist(flair, boundary_dist, patch4(0,1,0));;

      patch4(1,1,0) = foil_length+ wake_length;
      patch4(1,1,1) = FlairBoundDist(flair, boundary_dist, patch4(1,1,0));;

      // Refine
      patch4.DegreeElevate(0, order-1);
      patch4.KnotInsert(0, *kv4);
      patch4.DegreeElevate(1, order-1);
      patch4.KnotInsert(1, *kvr);
   }

   // Apply angle of attack
   patch0.Rotate2D(-aoa);
   patch1.Rotate2D(-aoa);
   patch2.Rotate2D(-aoa);
   patch3.Rotate2D(-aoa);
   patch4.Rotate2D(-aoa);

   //
   // 5. Print mesh to file
   //

   // Open mesh output file
   string mesh_file;
   mesh_file.append(msh_path);
   mesh_file.append(msh_filename);
   mesh_file.append(".mesh");
   ofstream output(mesh_file.c_str());

   // File header
   output<<"MFEM NURBS mesh v1.0"<<endl;
   output<< endl << "# " << mdim << "D C-mesh around a symmetric NACA foil section"
         << endl << endl;
   output<< "dimension"<<endl;
   output<< mdim <<endl;
   output<< endl;

   // NURBS patches defined as elements
   output << "elements"<<endl;
   output << "5"<<endl;
   output << "1 3 0 1 5 4" << endl;   // Lower wake
   output << "1 3 1 2 6 5" << endl;   // Lower tail
   output << "1 3 2 3 7 6" << endl;   // Tip
   output << "1 3 3 1 8 7" << endl;   // Upper tail
   output << "1 3 1 0 9 8" << endl;   // Upper wake
   output << endl;

   // Boundaries
   output << "boundary" <<endl;
   output << "10" <<endl;
   output << "1 1 5 4" << endl;   // Bottom
   output << "1 1 6 5" << endl;   // Bottom
   output << "2 1 7 6" << endl;   // Inflow
   output << "3 1 8 7" << endl;   // Top
   output << "3 1 9 8" << endl;   // Top
   output << "4 1 4 0" << endl;   // Outflow
   output << "4 1 0 9" << endl;   // Outflow
   output << "5 1 1 2" << endl;   // Foil section
   output << "5 1 2 3" << endl;   // Foil section
   output << "5 1 3 1" << endl;   // Foil section
   output << endl;

   // Edges
   output <<"edges"<<endl;
   output <<"15"<<endl;
   output << "0 0 1"<<endl;
   output << "1 1 2"<<endl;
   output << "2 2 3"<<endl;
   output << "3 3 1"<<endl;

   output << "0 4 5"<<endl;
   output << "1 5 6"<<endl;
   output << "2 6 7"<<endl;
   output << "3 7 8"<<endl;
   output << "0 9 8"<<endl;

   output << "4 0 4"<<endl;
   output << "4 1 5"<<endl;
   output << "4 2 6"<<endl;
   output << "4 3 7"<<endl;
   output << "4 1 8"<<endl;
   output << "4 0 9"<<endl;
   output << endl;

   // Vertices
   output << "vertices" << endl;
   output << 10 << endl;
   output << endl;

   // Patches
   output<<"patches"<<endl;
   output<<endl;

   output << "# Patch 0 " << endl;
   patch0.Print(output); output<<endl;
   output << "# Patch 1 " << endl;
   patch1.Print(output); output<<endl;
   output << "# Patch 2 " << endl;
   patch2.Print(output); output<<endl;
   output << "# Patch 3 " << endl;
   patch3.Print(output); output<<endl;
   output << "# Patch 4 " << endl;
   patch4.Print(output); output<<endl;

   // Close
   output.close();
   delete kv0;
   delete kv1;
   delete kv2;
   delete kv3;
   delete kv4;
   delete kvr;
   delete kv_o1;
   delete kv_o2;
   delete xyf[0], xyf[1];

   cout << endl << "Boundary identifiers:" << endl;
   cout << "   1   Bottom" << endl;
   cout << "   2   Inflow" << endl;
   cout << "   3   Top" << endl;
   cout << "   4   Outflow" << endl;
   cout << "   5   Foil section" << endl;
   cout << "=========================================================="<< endl;
   cout << "  "<< mdim <<"D mesh generated: " <<mesh_file.c_str()<< endl ;
   cout << "=========================================================="<< endl;

   // Print mesh info to screen
   cout << "=========================================================="<< endl;
   cout << " Attempting to read mesh: " <<mesh_file.c_str()<< endl ;
   cout << "=========================================================="<< endl;
   Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
   mesh->PrintInfo();

   // Print mesh to file for visualisation
   VisItDataCollection dc = VisItDataCollection("mesh", mesh);
   dc.SetPrefixPath("solution");
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   // Close
   delete mesh;
   return 0;
}
