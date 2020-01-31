#include <mfem.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace mfem;

static void mu_vec_fn(const Vector& x,Vector& V)
{
   V = 0.0;
   for (int i = 0; i < std::min(V.Size(),x.Size()); i++)
   {
      V(i) = std::sqrt(x(i)*x(i));
   }
}

int main(int argc, char *argv[])
{
   int rank, num_procs;
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   //Mesh mesh("./test.mesh.txt", 1, 1);
   Mesh mesh(32, 32, Element::QUADRILATERAL, true, 1.0, 1.0);
   //Mesh mesh(16, 16, 16, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0);
   for (int i = 0; i < 4; i++)
   {
      //mesh.UniformRefinement();
   }
   // if EnsureNCMesh is not called, everything is ok
   mesh.EnsureNCMesh();
   //mesh.RandomRefinement(0.5);

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   //pmesh->UniformRefinement();

   VectorFunctionCoefficient mu_vec(pmesh->SpaceDimension(), mu_vec_fn);

   int mu_ord = 1;
   FiniteElementCollection *mu_fec = NULL;
   ParFiniteElementSpace *mu_fes = NULL;
   // Both ND and RT fails using EnsureNCMesh
   //mu_fec = new ND_FECollection(mu_ord, pmesh->Dimension());
   mu_fec = new RT_FECollection(mu_ord, pmesh->Dimension());
   mu_fes = new ParFiniteElementSpace(pmesh, mu_fec, 1, Ordering::byVDIM);

   ParGridFunction gf(mu_fes);
   gf = 0.0;
   gf.ProjectDiscCoefficient(mu_vec, GridFunction::ARITHMETIC);
   // Using this is always fine
   //gf.ProjectCoefficient(mu_vec);

   std::ostringstream oname;
   oname << "out." << std::setfill('0') << std::setw(6) << rank;

#if 0
   std::ofstream gfofs(oname.str().c_str());
   gfofs.precision(8);
   pmesh->Print(gfofs);
   gf.Save(gfofs);
#else
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << rank << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh << gf << std::flush;
#endif

   delete mu_fes;
   delete mu_fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}
