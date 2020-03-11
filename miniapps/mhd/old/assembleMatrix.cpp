//                                MFEM test matrices
//example run: ./assembleMatrix -m ./Meshes/xperiodic-new.mesh -local -lr
//We have checked that all the mass and stiffness matrices are (almost) symmetric

#include "mfem.hpp"
#include "myCoefficient.hpp"
#include "myIntegrator.hpp"
#include "ResistiveMHDOperator.hpp"
#include "PCSolver.hpp"
#include "InitialConditions.hpp"
#include <memory>
#include <iostream>
#include <fstream>

double yrefine=0.2;
bool region(const Vector &p, const int lev)
{
   const double region_eps = 1e-8;
   const double x = p(0), y = p(1);
   //return std::max(std::max(std::max(x - yrefine, -y-yrefine), y - yrefine), -x-yrefine);
   if(lev==0)
      return std::max(-y-yrefine, y - yrefine)<region_eps;
   else
   {
      double ynew=0.8*yrefine;
      double xcenter=0.2, xedge=0.9;
      return (fabs(y)<ynew+region_eps && (fabs(x)<xcenter+region_eps || fabs(x)>xedge-region_eps) );
   }
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "./Meshes/xperiodic-square.mesh";
   int ref_levels = 2;
   int order = 2;
   bool visit = false;
   int precision = 8;
   int icase = 1;
   bool local_refine=false;
   int local_refine_levels=1;
   beta = 0.001; 
   Lx=3.0;
   lambda=5.0;

   bool visualization = true;
   int vis_steps = 100;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&local_refine, "-local", "--local-refine", "-no-local",
                  "--no-local-refine",
                  "Enable or disable local refinement before unifrom refinement.");
   args.AddOption(&local_refine_levels, "-lr", "--local-refine",
                  "Number of levels to refine locally.");
   args.Parse();
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file.    
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //++++++Refine the mesh to increase the resolution.    
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   //++++++Refine locally first    
   if (local_refine)
   {
      for(int lev=0; lev<local_refine_levels; lev++)
      {
        Vector pt;
        Array<int> marked_elements;
        for (int i = 0; i < mesh->GetNE(); i++)
        {
           // check all nodes of the element
           IsoparametricTransformation T;
           mesh->GetElementTransformation(i, &T);
           for (int j = 0; j < T.GetPointMat().Width(); j++)
           {
              T.GetPointMat().GetColumnReference(j, pt);
              if (region(pt, lev))
              {
                 marked_elements.Append(i);
                 break;
              }
           }
        }
        mesh->GeneralRefinement(marked_elements);
      }
   }

   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll); 
   SparseMatrix Mmat, Kmat;
   BilinearForm *K, *M;

   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   Array<int> ess_tdof_list;
   ess_bdr = 0;
   ess_bdr[0] = 1;  //set attribute 1 to Direchlet boundary fixed
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //stiffness matrix
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   ofstream myfile ("Kmat.m");
   Kmat.PrintMatlab(myfile);

   ofstream myfile2 ("Mmat.m");
   Mmat.PrintMatlab(myfile2);

   ofstream omesh("refined.mesh");
   omesh.precision(8);
   mesh->Print(omesh);

   // 10. Free the used memory.
   delete mesh;
   delete K;
   delete M;
   return 0;
}



