#include <iostream>
#include "mfem.hpp"

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
   return kv;
}




// Function for creating a mesh based on given orientations of two cubes
void mesher(int opatch0, int opatch1)
{

   // Parse commandline options
   double boxwidth = 1.0;
   double boxheight = 1.0;
   double boxdepth = 1.0;
   int ncp = 4;
   //int opatch0 = 1;
   //int opatch1 = 1;
   const char *msh_filename = "twocubes";
   double offset = 0.0;
   bool uniformknots = false;
   

   int order = 1;

   KnotVector *kv_o1 = UniformKnotVector(1, 2);


   NURBSPatch patch0(kv_o1, kv_o1, kv_o1, 4);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         for (int k = 0; k < 2; k++)
         {
            patch0(i,j,k,3) = 1.0;
         }


   // Define points
   switch(opatch0)
   {
      case 1:
         patch0(0,0,0,0) = 0;
         patch0(0,0,0,1) = 0;
         patch0(0,0,0,2) = 0;

         patch0(1,0,0,0) = boxwidth;
         patch0(1,0,0,1) = 0;
         patch0(1,0,0,2) = 0;

         patch0(1,1,0,0) = boxwidth;
         patch0(1,1,0,1) = boxheight;
         patch0(1,1,0,2) = 0;

         patch0(0,1,0,0) = 0;
         patch0(0,1,0,1) = boxheight;
         patch0(0,1,0,2) = 0;


         patch0(0,0,1,0) = 0;
         patch0(0,0,1,1) = 0;
         patch0(0,0,1,2) = boxdepth;

         patch0(1,0,1,0) = boxwidth;
         patch0(1,0,1,1) = 0;
         patch0(1,0,1,2) = boxdepth;

         patch0(1,1,1,0) = boxwidth;
         patch0(1,1,1,1) = boxheight;
         patch0(1,1,1,2) = boxdepth;

         patch0(0,1,1,0) = 0;
         patch0(0,1,1,1) = boxheight;
         patch0(0,1,1,2) = boxdepth;
         break;

      case 2:
         patch0(0,0,0,0) = 0;
         patch0(0,0,0,1) = boxheight;
         patch0(0,0,0,2) = 0;

         patch0(1,0,0,0) = 0;
         patch0(1,0,0,1) = 0;
         patch0(1,0,0,2) = 0;

         patch0(1,1,0,0) = boxwidth;
         patch0(1,1,0,1) = 0;
         patch0(1,1,0,2) = 0;

         patch0(0,1,0,0) = boxwidth;
         patch0(0,1,0,1) = boxheight;
         patch0(0,1,0,2) = 0;



         patch0(0,0,1,0) = 0;
         patch0(0,0,1,1) = boxheight;
         patch0(0,0,1,2) = boxdepth;

         patch0(1,0,1,0) = 0;
         patch0(1,0,1,1) = 0;
         patch0(1,0,1,2) = boxdepth;

         patch0(1,1,1,0) = boxwidth;
         patch0(1,1,1,1) = 0;
         patch0(1,1,1,2) = boxdepth;

         patch0(0,1,1,0) = boxwidth;
         patch0(0,1,1,1) = boxheight;
         patch0(0,1,1,2) = boxdepth;
         break;

      case 3:
         patch0(0,0,0,0) = boxwidth;
         patch0(0,0,0,1) = boxheight;
         patch0(0,0,0,2) = 0;

         patch0(1,0,0,0) = 0;
         patch0(1,0,0,1) = boxheight;
         patch0(1,0,0,2) = 0;

         patch0(1,1,0,0) = 0;
         patch0(1,1,0,1) = 0;
         patch0(1,1,0,2) = 0;

         patch0(0,1,0,0) = boxwidth;
         patch0(0,1,0,1) = 0;
         patch0(0,1,0,2) = 0;


         patch0(0,0,1,0) = boxwidth;
         patch0(0,0,1,1) = boxheight;
         patch0(0,0,1,2) = boxdepth;

         patch0(1,0,1,0) = 0;
         patch0(1,0,1,1) = boxheight;
         patch0(1,0,1,2) = boxdepth;

         patch0(1,1,1,0) = 0;
         patch0(1,1,1,1) = 0;
         patch0(1,1,1,2) = boxdepth;

         patch0(0,1,1,0) = boxwidth;
         patch0(0,1,1,1) = 0;
         patch0(0,1,1,2) = boxdepth;
         break;


      case 4:
         patch0(0,0,0,0) = boxwidth;
         patch0(0,0,0,1) = 0;
         patch0(0,0,0,2) = 0;

         patch0(1,0,0,0) = boxwidth;
         patch0(1,0,0,1) = boxheight;
         patch0(1,0,0,2) = 0;

         patch0(1,1,0,0) = 0;
         patch0(1,1,0,1) = boxheight;
         patch0(1,1,0,2) = 0;

         patch0(0,1,0,0) = 0;
         patch0(0,1,0,1) = 0;
         patch0(0,1,0,2) = 0;



         patch0(0,0,1,0) = boxwidth;
         patch0(0,0,1,1) = 0;
         patch0(0,0,1,2) = boxdepth;

         patch0(1,0,1,0) = boxwidth;
         patch0(1,0,1,1) = boxheight;
         patch0(1,0,1,2) = boxdepth;

         patch0(1,1,1,0) = 0;
         patch0(1,1,1,1) = boxheight;
         patch0(1,1,1,2) = boxdepth;

         patch0(0,1,1,0) = 0;
         patch0(0,1,1,1) = 0;
         patch0(0,1,1,2) = boxdepth;
         break;
   }





   NURBSPatch patch1(kv_o1, kv_o1, kv_o1, 4);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         for (int k = 0; k < 2; k++)
         {
            patch1(i,j,k,3) = 1.0;
         }


   patch1(0,0,0,0) = offset+boxwidth;
   patch1(0,0,0,1) = offset;
   patch1(0,0,0,2) = offset;

   patch1(1,0,0,0) = offset+boxwidth;
   patch1(1,0,0,1) = offset;
   patch1(1,0,0,2) = offset;

   patch1(1,1,0,0) = offset+boxwidth;
   patch1(1,1,0,1) = offset;
   patch1(1,1,0,2) = offset;

   patch1(0,1,0,0) = offset+boxwidth;
   patch1(0,1,0,1) = offset;
   patch1(0,1,0,2) = offset;


   patch1(0,0,1,0) = offset+boxwidth;
   patch1(0,0,1,1) = offset;
   patch1(0,0,1,2) = offset;

   patch1(1,0,1,0) = offset+boxwidth;
   patch1(1,0,1,1) = offset;
   patch1(1,0,1,2) = offset;

   patch1(1,1,1,0) = offset+boxwidth;
   patch1(1,1,1,1) = offset;
   patch1(1,1,1,2) = offset;

   patch1(0,1,1,0) = offset+boxwidth;
   patch1(0,1,1,1) = offset;
   patch1(0,1,1,2) = offset;





   switch(opatch1)
   {
      case 1:
         patch1(0,0,0,0) += 0;
         patch1(0,0,0,1) += 0;
         patch1(0,0,0,2) += 0;

         patch1(1,0,0,0) += boxwidth;
         patch1(1,0,0,1) += 0;
         patch1(1,0,0,2) += 0;

         patch1(1,1,0,0) += boxwidth;
         patch1(1,1,0,1) += boxheight;
         patch1(1,1,0,2) += 0;

         patch1(0,1,0,0) += 0;
         patch1(0,1,0,1) += boxheight;
         patch1(0,1,0,2) += 0;


         patch1(0,0,1,0) += 0;
         patch1(0,0,1,1) += 0;
         patch1(0,0,1,2) += boxdepth;

         patch1(1,0,1,0) += boxwidth;
         patch1(1,0,1,1) += 0;
         patch1(1,0,1,2) += boxdepth;

         patch1(1,1,1,0) += boxwidth;
         patch1(1,1,1,1) += boxheight;
         patch1(1,1,1,2) += boxdepth;

         patch1(0,1,1,0) += 0;
         patch1(0,1,1,1) += boxheight;
         patch1(0,1,1,2) += boxdepth;
         break;

      case 2:
         patch1(0,0,0,0) += 0;
         patch1(0,0,0,1) += boxheight;
         patch1(0,0,0,2) += 0;

         patch1(1,0,0,0) += 0;
         patch1(1,0,0,1) += 0;
         patch1(1,0,0,2) += 0;

         patch1(1,1,0,0) += boxwidth;
         patch1(1,1,0,1) += 0;
         patch1(1,1,0,2) += 0;

         patch1(0,1,0,0) += boxwidth;
         patch1(0,1,0,1) += boxheight;
         patch1(0,1,0,2) += 0;



         patch1(0,0,1,0) += 0;
         patch1(0,0,1,1) += boxheight;
         patch1(0,0,1,2) += boxdepth;

         patch1(1,0,1,0) += 0;
         patch1(1,0,1,1) += 0;
         patch1(1,0,1,2) += boxdepth;

         patch1(1,1,1,0) += boxwidth;
         patch1(1,1,1,1) += 0;
         patch1(1,1,1,2) += boxdepth;

         patch1(0,1,1,0) += boxwidth;
         patch1(0,1,1,1) += boxheight;
         patch1(0,1,1,2) += boxdepth;
         break;

      case 3:
         patch1(0,0,0,0) += boxwidth;
         patch1(0,0,0,1) += boxheight;
         patch1(0,0,0,2) += 0;

         patch1(1,0,0,0) += 0;
         patch1(1,0,0,1) += boxheight;
         patch1(1,0,0,2) += 0;

         patch1(1,1,0,0) += 0;
         patch1(1,1,0,1) += 0;
         patch1(1,1,0,2) += 0;

         patch1(0,1,0,0) += boxwidth;
         patch1(0,1,0,1) += 0;
         patch1(0,1,0,2) += 0;


         patch1(0,0,1,0) += boxwidth;
         patch1(0,0,1,1) += boxheight;
         patch1(0,0,1,2) += boxdepth;

         patch1(1,0,1,0) += 0;
         patch1(1,0,1,1) += boxheight;
         patch1(1,0,1,2) += boxdepth;

         patch1(1,1,1,0) += 0;
         patch1(1,1,1,1) += 0;
         patch1(1,1,1,2) += boxdepth;

         patch1(0,1,1,0) += boxwidth;
         patch1(0,1,1,1) += 0;
         patch1(0,1,1,2) += boxdepth;
         break;


      case 4:
         patch1(0,0,0,0) += boxwidth;
         patch1(0,0,0,1) += 0;
         patch1(0,0,0,2) += 0;

         patch1(1,0,0,0) += boxwidth;
         patch1(1,0,0,1) += boxheight;
         patch1(1,0,0,2) += 0;

         patch1(1,1,0,0) += 0;
         patch1(1,1,0,1) += boxheight;
         patch1(1,1,0,2) += 0;

         patch1(0,1,0,0) += 0;
         patch1(0,1,0,1) += 0;
         patch1(0,1,0,2) += 0;



         patch1(0,0,1,0) += boxwidth;
         patch1(0,0,1,1) += 0;
         patch1(0,0,1,2) += boxdepth;

         patch1(1,0,1,0) += boxwidth;
         patch1(1,0,1,1) += boxheight;
         patch1(1,0,1,2) += boxdepth;

         patch1(1,1,1,0) += 0;
         patch1(1,1,1,1) += boxheight;
         patch1(1,1,1,2) += boxdepth;

         patch1(0,1,1,0) += 0;
         patch1(0,1,1,1) += 0;
         patch1(0,1,1,2) += boxdepth;
         break;
   }






   // Open mesh output file
   string mesh_file;
   mesh_file.append(msh_filename);
   mesh_file.append(to_string(opatch0));
   mesh_file.append(to_string(opatch1));
   

   mesh_file.append(".mesh");
   ofstream output(mesh_file.c_str());


   // File header
   output<<"MFEM NURBS mesh v1.0"<<endl;
   output<< endl << "# " << 3 << "D Two cubes mesh" << endl << endl;
   output<< "dimension"<<endl;
   output<< "3" <<endl;
   output<< endl;
   


   // Elements

   output<<"elements"<<endl;
   output<<"2"<<endl;

   // Patch 1 flipped around z-axis: does not work1

   switch(opatch0)
   {
      case 1: // Original
         output<<"1 5 0 1 4 3 6 7 10 9"<<endl;
         break;

      case 2: 
         output<<"1 5 3 0 1 4 9 6 7 10 "<<endl;
         break;

      case 3: 
         output<<"1 5 4 3 0 1 10 9 6 7"<<endl;
         break;

      case 4:
         output<<"1 5 1 4 3 0 7 10 9 6"<<endl;
         break;

   }


   switch(opatch1)
   {
      case 1: 
         output<<"1 5 1 2 5 4 7 8 11 10"<<endl;
         break;
      case 2: 
         output<<"1 5 4 1 2 5 10 7 8 11"<<endl;
         break;

      case 3:
         output<<"1 5 5 4 1 2 11 10 7 8"<<endl; 
         break;

      case 4: 
         output<<"1 5 2 5 4 1 8 11 10 7"<<endl; 
         break;
   }

   output<<endl;

   // Boundaries
   output<<"boundary"<<endl;
   output<<"0"<<endl;

   output<<endl;

   // Edges
   output<<"edges"<<endl;
   output<<"20"<<endl;
   output<<"0 0 1"<<endl;
   output<<"1 1 4"<<endl;
   output<<"0 3 4"<<endl;
   output<<"1 0 3"<<endl;
   output<<"2 1 2"<<endl;
   output<<"1 2 5"<<endl;
   output<<"2 4 5"<<endl;

   output<<"0 6 7"<<endl;
   output<<"1 7 10"<<endl;
   output<<"0 9 10"<<endl;
   output<<"1 6 9"<<endl;
   output<<"2 7 8"<<endl;
   output<<"1 8 11"<<endl;
   output<<"2 10 11"<<endl;

   output<<"3 0 6"<<endl;
   output<<"3 1 7"<<endl;
   output<<"3 2 8"<<endl;
   output<<"3 3 9"<<endl;
   output<<"3 4 10"<<endl;
   output<<"3 5 11"<<endl;


   output<<endl;


   // Vertices
   output << "vertices" << endl;
   output << 12 << endl;
   output << endl;

   // Print to file
   output<<"patches"<<endl;
   output<<endl;
   patch0.Print(output); output<<endl;
   patch1.Print(output); output<<endl;
   output.close();

   delete kv_o1;


   // Print mesh info to screen
   cout << "=========================================================="<< endl;
   cout << " Created: " <<mesh_file.c_str()<< endl ;
   cout << "=========================================================="<< endl;
   Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
   mesh->PrintInfo();
  
   // Close
   delete mesh;

}


int main(int argc, char *argv[])
{
   for (int i = 1; i < 5; i++)
      for (int j = 1; j < 5; j++)
      {
         mesher(i, j);
      }
}