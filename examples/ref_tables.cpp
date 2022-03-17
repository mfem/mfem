
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;




int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   bool visualization = true;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   Mesh mesh = Mesh::MakeCartesian2D(2,2,mfem::Element::QUADRILATERAL);
   mesh.EnsureNCMesh();

   Array<Table * > map_table;
   // in case of a refinement we need the Transpose of the coarse2fine Table
   // in case of a derefinement we need the coarse2fine table

   int ref = 2;
   char vishost[] = "localhost";
   int  visport   = 19916;

   if (visualization)
   {
      socketstream sout(vishost, visport);
      sout.precision(precision);
      sout << "mesh\n" << mesh << flush;
   }

   Table temp;
   for (int i = 0; i<ref; i++)
   {
      mesh.RandomRefinement(0.5);
      if (visualization)
      {
         socketstream sout(vishost, visport);
         sout.precision(precision);
         sout << "mesh\n" << mesh << flush;
      }
      const CoarseFineTransformations & tr = mesh.GetRefinementTransforms();
      tr.MakeCoarseToFineTable(temp);
      map_table.Append(Transpose(temp));
   }

   // derefine
   Vector errors(mesh.GetNE());
   errors = 1;
   errors[8] = 0.;  errors[7] = 0.;
   errors[6] = 0.;  errors[5] = 0.;
   errors[21] = 0.; errors[18] = 0.;
   errors[19] = 0.; errors[20] = 0.;
   mesh.DerefineByError(errors,0.3);

   if (visualization)
   {
      socketstream sout1(vishost, visport);
      sout1.precision(precision);
      sout1 << "mesh\n" << mesh << flush;
   }

   const CoarseFineTransformations &tr = mesh.ncmesh->GetDerefinementTransforms();
   map_table.Append(new Table);
   tr.MakeCoarseToFineTable(*map_table.Last());

   for (int j = 0; j<mesh.GetNE(); j++)
   {
      // while the element depth > 0 for the element
      mfem::out << "Refinement history for element: " << std::setw(4) << j << ": ";
      int row = j;
      for (int i = map_table.Size()-1 ; i>=0; i--)
      {
         row = map_table[i]->GetRow(row)[0];
         if (i == 0)
         {
            mfem::out << std::setw(4) << row << endl;
         }
         else
         {
            mfem::out << std::setw(4) << row ;
         }
      }
   }

   return 0;
}

