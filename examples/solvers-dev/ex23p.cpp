//                       MFEM Example 22 - Parallel Version
//
// Compile with: make ex22p
//
// Sample runs:  mpirun -np 4 ex22p -m ../../data/square-disc.mesh -o 2
//               mpirun -np 4 ex22p -m ../../data/beam-tet.mesh
//               mpirun -np 4 ex22p -m ../../data/beam-hex.mesh
//               mpirun -np 4 ex22p -m ../../data/fichera.mesh
//               mpirun -np 4 ex22p -m ../../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex22p -m ../../data/amr-hex.mesh
//               mpirun -np 4 ex22p -m ../../hexa728.mesh
//               mpirun -np 4 ex22p -m ../../data/rectwhole7_2attr.e
// Description:  This example code solves a simple electromagnetic wave
//               propagation problem corresponding to the second order indefinite
//               Maxwell equation curl curl E - \omega^2 E = f with a PML
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example also demonstrates the use complex valued bilear and linear forms.
//               We recommend viewing examples 22 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
using namespace std;
using namespace mfem;

#ifndef MFEM_USE_SUPERLU
#error This example requires that MFEM is built with MFEM_USE_SUPERLU=YES
#endif

// Exact solution, E, and r.h.s., f. See below for implementation.
void compute_pml_mesh_data(Mesh * mesh);
void maxwell_ess_data(const Vector &x, std::vector<std::complex<double>> &Eval);
void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);
double pml_detJ_inv_Re(const Vector &x);
double pml_detJ_inv_Im(const Vector &x);
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M);
void pml_detJ_inv_JT_J_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_inv_JT_J_Im(const Vector &x, DenseMatrix &M);



double omega;
int dim;
int src = 2;
Array2D<double> domain_bdr;
Array2D<double> pml_lngth;
Array2D<double> comp_domain_bdr;

enum prob_type 
{
   scatter,
   waveguide
};

prob_type prob = scatter;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/beam-tet.mesh";
   int order = 1;
   bool visualization = 1;
   double freq = 1.0;
   int ref_levels = 1;
   int par_ref_levels = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                                              " solution.");
   args.AddOption(&ref_levels, "-rs", "--refinements-serial", "Number of serial refinements");                                           
   args.AddOption(&par_ref_levels, "-rp", "--refinements-parallel", "Number of parallel refinements");                                           
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&src, "-src", "--source wave", "Source wave flag -" 
                  "1: plane wave, 2: Point source, 3: sin in x direction");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   omega = 2.0 * M_PI * freq;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   

   if (strcmp(mesh_file, "../../data/beam-hex.mesh") == 0 || strcmp(mesh_file, "../../data/beam-quad.mesh") == 0)
   {
      mfem::out << "Problem type: Waveguide" << endl; 
      prob = prob_type::waveguide;
   }
   else
   {
      mfem::out << "Problem type: Scattering" << endl; 
   }
   
   dim = mesh->Dimension();
   compute_pml_mesh_data(mesh);

   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
         //  (int)floor(log(1000. / mesh->GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   // pmesh->ReorientTetMesh();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   // Right hand side is zero
   ParComplexLinearForm b(fespace, ComplexOperator::HERMITIAN);
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);
   b.Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParComplexGridFunction x(fespace);
   VectorFunctionCoefficient E_Re(sdim, E_bdr_data_Re);
   // VectorFunctionCoefficient E_Im(sdim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Re, ess_bdr);

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   ConstantCoefficient muinv(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));
   FunctionCoefficient det_inv_Re(pml_detJ_inv_Re);
   FunctionCoefficient det_inv_Im(pml_detJ_inv_Im);
   MatrixFunctionCoefficient temp_c1_Re(dim, pml_detJ_inv_JT_J_Re);
   MatrixFunctionCoefficient temp_c1_Im(dim, pml_detJ_inv_JT_J_Im);
   MatrixFunctionCoefficient temp_c2_Re(dim, pml_detJ_JT_J_inv_Re);
   MatrixFunctionCoefficient temp_c2_Im(dim, pml_detJ_JT_J_inv_Im);

   ScalarMatrixProductCoefficient pml_c1_Re(muinv, temp_c1_Re);
   ScalarMatrixProductCoefficient pml_c1_Im(muinv, temp_c1_Im);
   ScalarMatrixProductCoefficient pml_c2_Re(sigma, temp_c2_Re);
   ScalarMatrixProductCoefficient pml_c2_Im(sigma, temp_c2_Im);


   ParSesquilinearForm a(fespace, ComplexOperator::HERMITIAN);
   
   if (dim == 3)
   {
      a.AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                            new CurlCurlIntegrator(pml_c1_Im));
   }
   else
   {
      a.AddDomainIntegrator(new CurlCurlIntegrator(det_inv_Re),
                            new CurlCurlIntegrator(det_inv_Im));
   }
   a.AddDomainIntegrator(new VectorFEMassIntegrator(pml_c2_Re),
                         new VectorFEMassIntegrator(pml_c2_Im));
   a.Assemble();


   OperatorHandle Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   // Transform to monolithic HypreParMatrix
   HypreParMatrix * A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();

   if (myid == 0)
   {
      cout << "Size of linear system: " << A->GetGlobalNumRows() << endl;
   }

   // SuperLU direct solver
   SuperLURowLocMatrix *SA = new SuperLURowLocMatrix(*A);
   SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
   superlu->SetPrintStatistics(false);
   superlu->SetSymmetricPattern(false);
   superlu->SetColumnPermutation(superlu::PARMETIS);
   superlu->SetOperator(*SA);
   superlu->Mult(B, X);

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);


   ParComplexGridFunction x_gf(fespace);
   // x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);
   // x_gf.ProjectCoefficient(E_Re, E_Im);
   x_gf.ProjectCoefficient(E_Re, E_Re);
   // x_gf.ProjectBdrCoefficientTangent(E_Re, E_Re, ess_bdr);

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string keys;
      if (dim == 3)
      {
         keys = "keys mF\n";
      }
      else
      {
         keys = "keys arRljcUU\n";
      }
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *pmesh << x.real() << keys << flush;

      MPI_Barrier(MPI_COMM_WORLD);
      socketstream src_sock(vishost, visport);
      src_sock << "parallel " << num_procs << " " << myid << "\n";
      src_sock.precision(8);
      src_sock << "solution\n" << *pmesh << x_gf.real() << keys << flush;

   }

   // 17. Free the used memory.
   // delete superlu;
   // delete SA;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}


void compute_pml_mesh_data(Mesh * mesh)
{
   mesh->EnsureNodes();
   GridFunction * nodes = mesh->GetNodes();
   // Assuming square/cubic domain 
   double min_coord =  nodes->Min();
   double max_coord =  nodes->Max();
   double domain_length = abs(max_coord-min_coord); 
   // shift to zero
   // *nodes -= min_coord;
   // // scale to one
   // *nodes *= 1./domain_length;

   int ndofs = nodes->FESpace()->GetNDofs();
   Vector xcoords(ndofs), ycoords(ndofs), zcoords(ndofs);
   

   for (int comp = 0; comp < nodes->FESpace()->GetVDim(); comp++)
   {
      // cout << comp << endl;
      for (int i = 0; i < ndofs; i++)
      {
         if (comp == 0)
         {
            xcoords(i) = (*nodes)[nodes->FESpace()->DofToVDof(i, comp)];
         }
         else if (comp == 1)
         {
            ycoords(i) = (*nodes)[nodes->FESpace()->DofToVDof(i, comp)];
         }
         else if (comp == 2)
         {
            zcoords(i) = (*nodes)[nodes->FESpace()->DofToVDof(i, comp)];
         }
      }
   }

   domain_bdr.SetSize(3,2);
   domain_bdr(0,0) = xcoords.Min();
   domain_bdr(0,1) = xcoords.Max();
   domain_bdr(1,0) = ycoords.Min();
   domain_bdr(1,1) = ycoords.Max();
   domain_bdr(2,0) = zcoords.Min();
   domain_bdr(2,1) = zcoords.Max();

   pml_lngth.SetSize(dim,2);
   comp_domain_bdr.SetSize(dim,2);
   for (int i=0; i<dim; i++)
   {
      for (int j=0; j<2; j++)
      {
         pml_lngth(i,j) = 0.125 * (domain_bdr(i,1) - domain_bdr(i,0));
      }
      comp_domain_bdr(i,0) = domain_bdr(i,0) + pml_lngth(i,0);
      comp_domain_bdr(i,1) = domain_bdr(i,1) - pml_lngth(i,1);
   }

   if (prob == waveguide)
   {
      for (int i=0; i<dim; i++)
      {
         comp_domain_bdr(i,0) = domain_bdr(i,0);
         comp_domain_bdr(i,1) = domain_bdr(i,1);
      }   
      comp_domain_bdr(0,1) = domain_bdr(0,1) - pml_lngth(0,1);
   }

}



void maxwell_ess_data(const Vector &x, std::vector<std::complex<double>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i)
      E[i] = complex<double>(0., 0.);

   std::complex<double> zi = std::complex<double>(0., 1.);

   double alpha, beta;
   if (src == 1) // planewave
   {
      if (dim == 2)
      {
         alpha = omega / sqrt(2);
         beta = x(0) + x(1);
      }
      else
      {
         alpha = omega / sqrt(3);
         beta = x(0) + x(1) + x(2);
      }
      if (prob == waveguide) 
      {
         beta = x(0);
      }
      E[0] = cos(alpha * beta) + zi * sin(alpha * beta);


   }
   else if (src == 2) // point source
   {
      double r;
      double x0 = x(0) - 0.5 * (domain_bdr(0,0)+domain_bdr(0,1));
      double x1 = x(1) - 0.5 * (domain_bdr(1,0)+domain_bdr(1,1));
      beta = x0 * x0 + x1 * x1;
      if (dim == 3)
      {
         double x2 = x(2) - 0.5*(domain_bdr(2,0)+domain_bdr(2,1));
         beta += x2 * x2;
      }
      r = sqrt(beta);
      E[0] = cos(omega * r) + zi * sin(omega * r);
   }
   else if (src == 3) // sin i one direction
   {
      E[0] = 1.0; 
      E[1] = 0.0;
   }
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   // Initialize
   E = 0.0;
   bool in_pml = false;
   for (int i = 0; i < dim; ++i)
   {
      // check if x(i) is in the computational domain or not
      if (x(i) < comp_domain_bdr(i,0) || x(i) > comp_domain_bdr(i,1))
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      std::vector<std::complex<double>> Eval(E.Size());
      if (prob != waveguide)
      {
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i) E[i] = Eval[i].real();
      }
      else
      {
         if (abs(x(0) < 1e-13)) 
         {
            maxwell_ess_data(x, Eval);
            for (int i = 0; i < dim; ++i) E[i] = Eval[i].real();
         }
      }
   }
}

//define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   // Initialize
   E = 0.0;
   bool in_pml = false;
   for (int i = 0; i < dim; ++i)
   {
      if (abs(x(i)-domain_bdr(i,0)) < 1e-13 || abs(x(i) - domain_bdr(i,1)) < 1e-13)
      {
         in_pml = true;
         break;
      }
   }
   if (!in_pml)
   {
      std::vector<std::complex<double>> Eval(E.Size());
      maxwell_ess_data(x, Eval);
      for (int i = 0; i < dim; ++i)
         E[i] = Eval[i].imag();
   }
}

// PML
void pml_function(const Vector &x, std::vector<std::complex<double>> &dxs)
{
   std::complex<double> zi  = std::complex<double>(0., 1.);
   std::complex<double> one = std::complex<double>(1., 0.);

   double n = 2.0;
   double c = 10.0;
   double coeff;

   // initialize to one
   for (int i = 0; i < dim; ++i) dxs[i] = one;

   // Stretch in each direction independenly
   for (int i = 0; i < dim; ++i)
   {
      for (int j=0; j<2; ++j)
      if (x(i) >= comp_domain_bdr(i,1))
      {
         coeff = n * c / omega / pow(pml_lngth(i,1), n);
         dxs[i] = one + zi * coeff * abs(pow(x(i) - comp_domain_bdr(i,1), n - 1.0));
      }
      if (x(i) <= comp_domain_bdr(i,0))
      {
         coeff = n * c / omega / pow(pml_lngth(i,0), n);
         dxs[i] = one + zi * coeff * abs(pow(x(i) - comp_domain_bdr(i,0), n - 1.0));
      }
   }
}

double pml_detJ_inv_Re(const Vector &x)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);
   for (int i = 0; i < dim; ++i)
      det *= dxs[i];

   complex<double> det_inv = one / det;
   return det_inv.real();
}
double pml_detJ_inv_Im(const Vector &x)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);
   for (int i = 0; i < dim; ++i)
      det *= dxs[i];

   complex<double> det_inv = one / det;
   return det_inv.imag();
}
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = one / pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;

   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = det * diag[i];
      M(i, i) = temp.real();
   }
}
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<std::complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = one / pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;

   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = det * diag[i];
      M(i, i) = temp.imag();
   }
}
void pml_detJ_inv_JT_J_Re(const Vector &x, DenseMatrix &M)
{
   std::vector<complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;

   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = diag[i] / det;
      M(i, i) = temp.real();
   }
}
void pml_detJ_inv_JT_J_Im(const Vector &x, DenseMatrix &M)
{
   std::vector<std::complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;

   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = diag[i] / det;
      M(i, i) = temp.imag();
   }
}
