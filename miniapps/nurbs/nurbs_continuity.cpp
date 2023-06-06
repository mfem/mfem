#include <iostream>
#include "mfem.hpp"

/**
In this miniapp the continuity of a square shaped mesh is reduced using
8 options. Cases 1-4 reduce the continuity of a NURBSPatch. Cases 5-7
reduce the continuity on a full reread NURBS mesh. Case 8 always fails as
the continuity is reduced twice: on the NURBSPatch and on the reread mesh.
The miniapp prints two knot ectors and the knot repetitions in the knot
vector for reference.

CASE 0: No continuity reduction
CASE 1: NURBSPatch: Reduction to C^0-continuity via dedicated function
CASE 2: NURBSPatch: Reduction to C^n-continuity
CASE 3: NURBSPatch: Continuity reduced with n
CASE 4: NURBSPatch: Continuity of one KnotVector in 1 direction reduced with n
CASE 5: NURBSMesh: Reduction to C^0-continuity via dedicated function
CASE 6: NURBSMesh: Reduction to C^n-continuity via general function
CASE 7: NURBSMesh: Continuity reduced with n
CASE 8: NURBSMesh: ALWAYS FAILS. Continuity reduced in the NURBSPatch. Mesh
         printed, reread and continuity reduced again.

SAMPLE RUNS
RUN 1, CASE 0: No continuity reduction of 2nd order mesh
   ./nurbs_continuity -c 1 -o 3 -n 6 -nc 0

   Resulting knot vectors:
   3 6 0 0 0 0 0.333333 0.666667 1 1 1 1
   3 6 0 0 0 0 0.333333 0.666667 1 1 1 1

RUN 2, CASE 1: Reduction to C^0-continuity of 3rd order NURBSPatch
   ./nurbs_continuity -c 1 -o 3 -n 6 -nc 1

   Resulting knot vectors:
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1

RUN 3, CASE 2: Reduction to C^0 continuity via general method of 3rd order NURBSPatch
   ./nurbs_continuity -c 0 -o 3 -n 6 -nc 2

   Resulting knot vectors:
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1

RUN 4, CASE 2: Reduction to C^1 continuity of 3rd order NURBSPatch
   ./nurbs_continuity -c 1 -o 3 -n 6 -nc 2

   Resulting knot vectors:
   3 8 0 0 0 0 0.333333 0.333333 0.666667 0.666667 1 1 1 1
   3 8 0 0 0 0 0.333333 0.333333 0.666667 0.666667 1 1 1 1

RUN 5, CASE 3: 4th order NURBSPatch with continuity reduced with 2
   ./nurbs_continuity -c 2 -o 4 -n 6 -nc 3

   Resulting knot vectors:
   4 8 0 0 0 0 0 0.5 0.5 0.5 1 1 1 1 1
   4 8 0 0 0 0 0 0.5 0.5 0.5 1 1 1 1 1

RUN 6, CASE 4: 4th order NURBSPatch with continuity of knotvector 1 reduced with 2
   ./nurbs_continuity -c 2 -o 4 -n 6 -nc 4

   Resulting knot vectors:
   4 8 0 0 0 0 0 0.5 0.5 0.5 1 1 1 1 1
   4 6 0 0 0 0 0 0.5 1 1 1 1 1

RUN 7, CASE 5: Similar to run 2, but then applied to a mesh.
   ./nurbs_continuity -c 1 -o 3 -n 6 -nc 1

   Knot vectors:
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1

RUN 8, CASE 6: Similar to run 3, but then applied to a mesh.
   ./nurbs_continuity -c 0 -o 3 -n 6 -nc 6

   Knot vectors:
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1
   3 10 0 0 0 0 0.333333 0.333333 0.333333 0.666667 0.666667 0.666667 1 1 1 1

RUN 9, CASE 7: Similar to run 5, but then applied to a mesh.
   ./nurbs_continuity -c 2 -o 4 -n 6 -nc 7

   Knot vectors:
   4 8 0 0 0 0 0 0.5 0.5 0.5 1 1 1 1 1
   4 8 0 0 0 0 0 0.5 0.5 0.5 1 1 1 1 1

THESE TESTS SHOULD FAIL
Unable to increase continuity
   ./nurbs_continuity -c 3 -o 3 -n 6 -nc 3

Continuity can not be lower than 0
   ./nurbs_continuity -c -1 -o 3 -n 6 -nc 3
   ./nurbs_continuity -c 3 -o 3 -n 6 -nc 4

CASE 8 should always fail: Continuity reduced twice, specific continuity can
   not be guaranteed.
   ./nurbs_continuity -nc 8 -o 4 -c 2 -n 8

*/

using namespace std;
using namespace mfem;

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

   // Count number of elements. Required for patch0.DegreeElevate().
   kv->GetElements();
   return kv;
}

int main(int argc, char *argv[])
{
   // Parse commandline options
   OptionsParser args(argc, argv);

   double boxwidth = 1.0;
   double boxheight = 1.0;

   int ncp = 4;
   int order = 1;

   int cfactor = -1;
   int ncase = 0;

   const char *msh_filename = "square_continuity";

   args.AddOption(&boxwidth, "-b", "--boxwidth",
                  "Width of one of the boxes");
   args.AddOption(&boxheight, "-g", "--boxheight",
                  "Heigth of one of the boxes");

   args.AddOption(&ncp, "-n", "--ncp",
                  "Number of controlpoints");
   args.AddOption(&order, "-o", "--order",
                  "Order of NURBS");

   args.AddOption(&cfactor, "-c", "--continuity",
                  "C^n-continuity of NURBS");
   args.AddOption(&ncase, "-nc", "--n-case",
                  "Number of test case.");

   args.AddOption(&msh_filename, "-m", "--mesh-file",
                  "File where the generated mesh is written to.");

   // Parse and print commandline options
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (ncase < 0 && ncase > 8)
   {
      mfem_error("Test case does not exist");
   }

   KnotVector *kv_o1 = UniformKnotVector(1, 2);
   KnotVector *kv0 = UniformKnotVector(order, ncp);
   const KnotVector *knotv0, *knotv1; // For printing

   NURBSPatch patch0(kv_o1, kv_o1, 3);
   {
      // define weights
      for (int j = 0; j < 2; j++)
         for (int i = 0; i < 2; i++)
         {
            patch0(i,j,2) = 1.0;
         }

      // Map points
      patch0(0,0,0) = 0;
      patch0(0,0,1) = 0;

      patch0(1,0,0) = boxwidth;
      patch0(1,0,1) = 0;

      patch0(1,1,0) = boxwidth;
      patch0(1,1,1) = boxheight;

      patch0(0,1,0) = 0;
      patch0(0,1,1) = boxheight;

      // p- and h- refinement
      patch0.DegreeElevate(order-1);
      patch0.KnotInsert(0, *kv0);
      patch0.KnotInsert(1, *kv0);
   }

   if (ncase == 0)
   {
      // No continuity reduction to patch
   }
   else if (ncase == 1)
   {
      patch0.ReduceToC0Continuity();
   }
   else if (ncase == 2)
   {
      patch0.ReduceToCnContinuity(cfactor);
   }
   else if (ncase == 3)
   {
      patch0.ReduceContinuity(cfactor);
   }
   else if (ncase == 4)
   {
      patch0.ReduceContinuity(cfactor, 0);
   }
   else if (ncase == 8)
   {
      patch0.ReduceToCnContinuity(cfactor);
   }

   // Open mesh output file
   string mesh_file;
   mesh_file.append(msh_filename);

   mesh_file.append(".mesh");
   ofstream output(mesh_file.c_str());


   // Print Mesh
   {
      // File header
      output<<"MFEM NURBS mesh v1.0"<<endl;
      output<< endl << "# " << 2 << "D square mesh" << endl << endl;
      output<< "dimension"<<endl;
      output<< "2" <<endl;
      output<< endl;

      // Elements
      output<<"elements"<<endl;
      output<<"1"<<endl;
      //domain geomtype nodalnrs
      output<<"1 3 0 1 3 2"<<endl;

      // Boundaries
      output<<"boundary"<<endl;
      output<<"4"<<endl;
      //bndNr GeomType FromNode ToNode
      output<<"1 1 0 1"<< endl;   // Bottom
      output<<"2 1 1 3"<< endl;   // Right side
      output<<"3 1 3 2"<< endl;   // Top side
      output<<"4 1 2 0"<< endl;   // Left side
      output<<endl;


      // Edges
      output<<"edges"<<endl;
      output<<"4"<<endl;
      //KnotvectorNr FromNode ToNode
      output<<"0 0 1"<<endl;
      output<<"1 1 3"<<endl;
      output<<"0 2 3"<<endl;
      output<<"1 0 2"<<endl;
      output<<endl;

      // Vertices
      output << "vertices" << endl;
      output << 4 << endl;
      output << endl;

      // Print to file
      output<<"patches"<<endl;
      output<<endl;
      patch0.Print(output); output<<endl; //write additional lines for mesh file
      output.close();
   }

   delete kv_o1;
   delete kv0;

   // Print mesh info to screen
   {
      cout << "=========================================================="<< endl;
      cout << " Attempting to read mesh: " << mesh_file.c_str() << endl;
      cout << "=========================================================="<< endl;
      Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
      mesh->PrintInfo();

      // Print mesh to file for visualisation
      VisItDataCollection dc = VisItDataCollection("mesh", mesh);
      dc.SetPrefixPath("solution");
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
      delete mesh;
   }

   if (ncase < 5) // Stop test for specific cases
   {
      cout << "=========================================================="<< endl;
      cout << " Printing info for reference " << endl;
      cout << "=========================================================="<< endl;
      knotv0 = patch0.GetKV(0);
      knotv1 = patch0.GetKV(1);

      cout << "Knot vectors:" << endl;
      knotv0->Print(cout);
      knotv1->Print(cout);
      cout << endl;

      cout << "Knot repetitions: " << endl;
      cout << knotv0->GetNKR() << endl;
      cout << knotv1->GetNKR() << endl;
   }
   else // More testing continuity, print KV for reference later
   {
      Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);

      if (ncase == 5)
      {
         mesh->NURBSReduceToC0Continuity();
      }
      else if (ncase == 6)
      {
         mesh->NURBSReduceToCnContinuity(cfactor);
      }
      else if (ncase == 7)
      {
         mesh->NURBSReduceContinuity(cfactor);
      }
      else if (ncase == 8)
      {
         mesh->NURBSReduceToCnContinuity(cfactor+1);
      }

      mesh_file = "";
      mesh_file.append(msh_filename);
      mesh_file.append("_reduced.mesh");

      ofstream output_continuity(mesh_file.c_str());
      mesh->Print(output_continuity); // Old/non-patch NURBS mesh format
      output_continuity << endl;
      output_continuity.close();

      // Print mesh info to screen
      {
         cout << "=========================================================="<< endl;
         cout << " Attempting to read mesh with continuity reduction: " << endl;
         cout << "  " << mesh_file.c_str() << endl;
         cout << "=========================================================="<< endl;
         Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
         mesh->PrintInfo();

         // Print mesh to file for visualisation
         VisItDataCollection dc = VisItDataCollection("mesh_continuity_reduced", mesh);
         dc.SetPrefixPath("solution");
         dc.SetCycle(0);
         dc.SetTime(0.0);
         dc.Save();

         cout << "=========================================================="<< endl;
         cout << " Printing info for reference " << endl;
         cout << "=========================================================="<< endl;
         knotv0 = mesh->NURBSext->GetKnotVector(0);
         knotv1 = mesh->NURBSext->GetKnotVector(1);

         cout << "Knot vectors:" << endl;
         knotv0->Print(cout);
         knotv1->Print(cout);
         cout << endl;

         cout << "Knot repetitions: " << endl;
         cout << knotv0->GetNKR() << endl;
         cout << knotv1->GetNKR() << endl;

         delete mesh;
      }
   }
}
