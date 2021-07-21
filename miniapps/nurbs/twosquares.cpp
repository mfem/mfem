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

KnotVector *PowerStretchKnotVector(int order, int ncp, double stretch = 0.0)
{
   KnotVector *kv = new KnotVector(order, ncp);

   for (int i = 0; i < order+1; i++)
   {
      (*kv)[i] = 0.0;
   }
   for (int i = order+1; i < ncp; i++)
   {
      (*kv)[i] = (i-order)/double(ncp-order);
      if (stretch > 0) { (*kv)[i] = pow((*kv)[i],stretch); }
      if (stretch < 0) { (*kv)[i] = 1.0 - pow(1.0-(*kv)[i],-stretch); }
   }
   for (int i = ncp ; i < ncp + order + 1; i++)
   {
      (*kv)[i] = 1.0;
   }
   return kv;
}




// Function for creating a mesh based on given orientations of two squares
void mesher(int opatch0, int opatch1)
{
   double boxwidth = 1.0;
   double boxheight = 1.0;
   int ncp = 4;
   const char *msh_filename = "twosquares";
   double offset = 0.0;
   bool uniformknots = false;
   int order = 1;


   KnotVector *kv0 = PowerStretchKnotVector(order, ncp, 1.5);
   KnotVector *kv1 = PowerStretchKnotVector(order, ncp, 1.5);
   KnotVector *kv2 = PowerStretchKnotVector(order, ncp, 1.5);
  


   KnotVector *kv_o1 = UniformKnotVector(1, 2);

   NURBSPatch patch0(kv_o1, kv_o1, 3);
   for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++)
      {
         patch0(i,j,2) = 1.0;
      }

   // Define points

   switch(opatch0)
   {
      case 1:
         // Define points
         patch0(0,0,0) = 0;
         patch0(0,0,1) = 0;

         patch0(1,0,0) = boxwidth;
         patch0(1,0,1) = 0;

         patch0(1,1,0) = boxwidth;
         patch0(1,1,1) = boxheight;

         patch0(0,1,0) = 0;
         patch0(0,1,1) = boxheight;
         break;

      case 2:
         // Define points
         patch0(0,0,0) = 0;
         patch0(0,0,1) = boxheight;

         patch0(1,0,0) = 0;
         patch0(1,0,1) = 0;

         patch0(1,1,0) = boxwidth;
         patch0(1,1,1) = 0;

         patch0(0,1,0) = boxwidth;
         patch0(0,1,1) = boxheight;
         break;

      case 3:
         patch0(0,0,0) = boxwidth;
         patch0(0,0,1) = boxheight;

         patch0(1,0,0) = 0;
         patch0(1,0,1) = boxheight;

         patch0(1,1,0) = 0;
         patch0(1,1,1) = 0;

         patch0(0,1,0) = boxwidth;
         patch0(0,1,1) = 0;
         break;

      case 4:

         patch0(0,0,0) = boxwidth;
         patch0(0,0,1) = 0;

         patch0(1,0,0) = boxwidth;
         patch0(1,0,1) = boxheight;

         patch0(1,1,0) = 0;
         patch0(1,1,1) = boxheight;

         patch0(0,1,0) = 0;
         patch0(0,1,1) = 0;
         break;
   }



   NURBSPatch patch1(kv_o1, kv_o1, 3);
   for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++)
      {
         patch1(i,j,2) = 1.0;
      }

   patch1(0,0,0) = offset+boxwidth;
   patch1(0,0,1) = offset;

   patch1(1,0,0) = offset+boxwidth;
   patch1(1,0,1) = offset;

   patch1(0,1,0) = offset+boxwidth;
   patch1(0,1,1) = offset;

   patch1(1,1,0) = offset+boxwidth;
   patch1(1,1,1) = offset;

   switch(opatch1)
   {
      case 1:
         // Define points
         patch1(0,0,0) += 0;
         patch1(0,0,1) += 0;

         patch1(1,0,0) += boxwidth;
         patch1(1,0,1) += 0;

         patch1(1,1,0) += boxwidth;
         patch1(1,1,1) += boxheight;

         patch1(0,1,0) += 0;
         patch1(0,1,1) += boxheight;
         break;

      case 2:
         // Define points
         patch1(0,0,0) += 0;
         patch1(0,0,1) += boxheight;

         patch1(1,0,0) += 0;
         patch1(1,0,1) += 0;

         patch1(1,1,0) += boxwidth;
         patch1(1,1,1) += 0;

         patch1(0,1,0) += boxwidth;
         patch1(0,1,1) += boxheight;
         break;

      case 3:
         patch1(0,0,0) += boxwidth;
         patch1(0,0,1) += boxheight;

         patch1(1,0,0) += 0;
         patch1(1,0,1) += boxheight;

         patch1(1,1,0) += 0;
         patch1(1,1,1) += 0;

         patch1(0,1,0) += boxwidth;
         patch1(0,1,1) += 0;
         break;

      case 4:
         // Define points
         patch1(0,0,0) += boxwidth;
         patch1(0,0,1) += 0;

         patch1(1,0,0) += boxwidth;
         patch1(1,0,1) += boxheight;

         patch1(1,1,0) += 0;
         patch1(1,1,1) += boxheight;

         patch1(0,1,0) += 0;
         patch1(0,1,1) += 0;
         break;
   }


   // Open mesh output file
   string mesh_file;
   mesh_file.append(msh_filename);

   mesh_file.append(to_string(opatch0));
   mesh_file.append(to_string(opatch1));
   

   mesh_file.append(".mesh");
   ofstream output(mesh_file.c_str());


   // Print

   // File header
   output<<"MFEM NURBS mesh v1.0"<<endl;
   output<< endl << "# " << 2 << "D Two cubes mesh" << endl << endl;
   output<< "dimension"<<endl;
   output<< "2" <<endl;
   output<< endl;
   


   // Elements

   output<<"elements"<<endl;
   output<<"2"<<endl;

   switch(opatch0)
   {
      case 1:
         output<<"1 3 0 1 4 3"<<endl; 
         break;
      case 2:
         output<<"1 3 3 0 1 4"<<endl; 

         break;
      case 3:
         output<<"1 3 4 3 0 1"<<endl;  
         break;
      case 4:
         output<<"1 3 1 4 3 0"<<endl;   
         break;
   }


   switch(opatch1)
   {
      case 1: 
         output<<"1 3 1 2 5 4"<<endl;
         break;
      case 2:
         output<<"1 3 4 1 2 5"<<endl;
         break;

      case 3:
         output<<"1 3 5 4 1 2"<<endl;
         break;

      case 4:
         output<<"1 3 2 5 4 1"<<endl;
         break;
   }
   output<<endl;

   // Boundaries
   output<<"boundary"<<endl;
   output<<"0"<<endl;
   output<<endl;

   // Edges
   output<<"edges"<<endl;
   output<<"7"<<endl;
   output<<"0 0 1"<<endl;
   output<<"1 1 4"<<endl;
   output<<"0 3 4"<<endl;
   output<<"1 0 3"<<endl;
   output<<"2 1 2"<<endl;
   output<<"1 2 5"<<endl;
   output<<"2 4 5"<<endl;
   output<<endl;


   // Vertices
   output << "vertices" << endl;
   output << 6 << endl;
   output << endl;

   // Print to file
   output<<"patches"<<endl;
   output<<endl;
   patch0.Print(output); output<<endl;
   patch1.Print(output); output<<endl;
   output.close();

   delete kv0, kv1, kv_o1;


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