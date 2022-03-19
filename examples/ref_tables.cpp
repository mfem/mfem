
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

   Array<int> refinements;
   refinements.Append(0);
   refinements.Append(1);
   refinements.Append(3);
   mesh.GeneralRefinement(refinements);
   if (visualization)
   {
      socketstream sout(vishost, visport);
      sout.precision(precision);
      sout << "mesh\n" << mesh << flush;
   }
   const CoarseFineTransformations & tr = mesh.GetRefinementTransforms();
   tr.MakeCoarseToFineTable(temp);
   map_table.Append(Transpose(temp));

   refinements.SetSize(0);
   refinements.Append(7);
   mesh.GeneralRefinement(refinements);
   if (visualization)
   {
      socketstream sout(vishost, visport);
      sout.precision(precision);
      sout << "mesh\n" << mesh << flush;
   }
   const CoarseFineTransformations & tr1 = mesh.GetRefinementTransforms();
   tr1.MakeCoarseToFineTable(temp);
   map_table.Append(Transpose(temp));


   // // for (int i = 0; i<ref; i++)
   // // {
   // //    mesh.RandomRefinement(0.5);
   // //    if (visualization)
   // //    {
   // //       socketstream sout(vishost, visport);
   // //       sout.precision(precision);
   // //       sout << "mesh\n" << mesh << flush;
   // //    }
   // //    const CoarseFineTransformations & tr = mesh.GetRefinementTransforms();
   // //    tr.MakeCoarseToFineTable(temp);
   // //    map_table.Append(Transpose(temp));
   // // }

   // // derefine
   Vector errors(mesh.GetNE());
   errors = 1;
   errors[12] = 0.;  errors[14] = 0.;
   errors[13] = 0.;  errors[15] = 0.;
   // errors[21] = 0.; errors[18] = 0.;
   // errors[19] = 0.; errors[20] = 0.;
   mesh.DerefineByError(errors,0.3);

   if (visualization)
   {
      socketstream sout1(vishost, visport);
      sout1.precision(precision);
      sout1 << "mesh\n" << mesh << flush;
   }

   const CoarseFineTransformations &tr2 = mesh.ncmesh->GetDerefinementTransforms();
   map_table.Append(new Table);
   tr2.MakeCoarseToFineTable(*map_table.Last());

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

   // example ...
   // create the maps from mesh 1 to mesh 3 (after one ref and 1 dref)
   // 1. specify the newly created elements (either from ref or from dref)
   // 2. mark the elements that are deleted
   // 3. provide the map from the unmodified elements to their new numbers

   // for the above scenario
   Table & ref_table = *Transpose(*map_table[1]);
   Table & dref_table = *Transpose(*map_table[2]);


   Table * T = Mult(ref_table,dref_table);


   mfem::out << "ref_table = " << endl;
   ref_table.Print();

   mfem::out << "dref_table = " << endl;
   dref_table.Print();

   mfem::out << "combined_table = " << endl;
   T->Print();


   Table * Tt = Transpose(*T);

   Array<int> old_elems_map(T->Size()); // -1 if is to be deleted
   Array<int> new_elems;

   // loop though the old elements
   for (int i = 0; i<T->Size(); i++)
   {
      // check row size
      int n = T->RowSize(i);
      int * row = T->GetRow(i);
      if (n == 1)
      {
         // check the size of the transpose row
         int m = Tt->RowSize(row[0]);
         if (m == 1)
         {
            // the element is left unchanged
            mfem::out << "Element number = " << i << " is mapped to element number = " <<
                      row[0] << endl;
            old_elems_map[i] = row[0];
         }
         else
         {
            mfem::out << "Element number = " << i <<
                      " is derefined (deleted). Create new element: " << row[0] << endl;
            old_elems_map[i] = -1;
            new_elems.Append(row[0]);
         }
      }
      else
      {
         mfem::out << "Element number = " << i <<
                   " is refined (deleted). Create new elements = " ;
         old_elems_map[i] = -1;
         for (int j = 0; j<n; j++)
         {
            mfem::out << row[j];
            new_elems.Append(row[j]);
            if (j == n-1)
            {
               mfem::out << endl;
            }
            else
            {
               mfem::out << ", ";
            }
         }
      }
   }

   // new_elems.Sort();
   new_elems.Unique();

   mfem::out << "elements map = " ; old_elems_map.Print(cout,old_elems_map.Size());
   mfem::out << "new_elements = " ; new_elems.Print(cout,new_elems.Size());


   return 0;
}

