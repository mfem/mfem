//                       MFEM Example mode port - Parallel Version
//
// Compile with: make portmodetest
//
// Sample runs:  mpirun -np 4 portmodetest
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               eigenvalue problem -Delta u = lambda u with homogeneous
//               Dirichlet boundary conditions.
//
//               We compute a number of the lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order, or an isoparametric/isogeometric space if
//               order < 1 (quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of the LOBPCG eigenvalue solver
//               together with the BoomerAMG preconditioner in HYPRE, as well as
//               optionally the SuperLU or STRUMPACK parallel direct solvers.
//               Reusing a single GLVis visualization window for multiple
//               eigenfunctions is also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void SetPortBC(int prob, int dim, int mode, ParGridFunction &port_bc);
void source(const Vector &x, Vector &f, GridFunction &gf);
int dim;
class GridFunction_VectorFunctionCoefficient : public VectorCoefficient
{
private:
   GridFunction mode_gf;
   std::function<void(const Vector &, Vector &, GridFunction &)> Function;
   Coefficient *Q;
public:
   GridFunction_VectorFunctionCoefficient(int dim,
                             std::function<void(const Vector &, Vector &, GridFunction &)> F,
                             GridFunction gf,
                             Coefficient *q = nullptr)
                             : VectorCoefficient(dim),mode_gf(gf),Function(std::move(F)),Q(q)
                             {};
   using VectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int port_order = 1;
   Mesh port_mesh(10, 1.0);

   for (int l = 0; l <= 2; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
      {
         port_mesh.UniformRefinement();
      }
   }
   int port_dim = port_mesh.Dimension();
   dim = port_dim+1;
   int nrelem = port_mesh.GetNE();
   // Initialize bdr attributes
   for (int i = 0; i < port_mesh.GetNBE(); ++i)
   {
      port_mesh.GetBdrElement(i)->SetAttribute(i+1);
   }
   // Loop through the elements is waveguide or not
   for (int i = 0; i < nrelem; ++i)
   {
      bool is_waveguide = false;
      Element *el = port_mesh.GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();
      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = port_mesh.GetVertex(vert_idx);
         for (int comp = 0; comp < port_dim; ++comp)
         {
            if ((coords[comp] < 0.55) && (coords[comp] > 0.45))
            {
               is_waveguide = true;
               break;
            }
         }
      }
      if (is_waveguide)
      {
         el->SetAttribute(2);
      }
   }
   port_mesh.SetAttributes();
   ParMesh *port_pmesh = new ParMesh(MPI_COMM_WORLD, port_mesh);
   FiniteElementCollection *port_fec;
   port_fec = new H1_FECollection(port_order, port_dim);
   ParFiniteElementSpace *port_fespace = new ParFiniteElementSpace(port_pmesh, port_fec);
   HYPRE_BigInt size = port_fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }
   ParGridFunction x(port_fespace);
   int prob = 0;
   int mode = 0;
   SetPortBC(prob, port_dim, mode, x);

   cout << "main row 130 dim: " << port_dim << endl;

   Vector phy_point(1);
   phy_point(0) = 0.85;
   //phy_point(1) = 0.25;
   phy_point.Print(cout << "physical point: ");
   
   IntegrationPoint ip;
   int elem_idx;
   ElementTransformation* tran;
   for (int i=0; i<port_pmesh->GetNE(); ++i)
   {  
      tran = port_pmesh->GetElementTransformation(i);
      InverseElementTransformation invtran(tran);
      int ret = invtran.Transform(phy_point, ip);
      if (ret == 0)
      {
         elem_idx = i;
         break;
      }
   }

   cout << elem_idx << "-th element\n"
      << "reference point: " << ip.x << endl;

   cout << "GridFunction value: " << x.GetValue(elem_idx, ip) << endl;


   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("portmode", port_pmesh);
   pd->SetPrefixPath("./PortModeParaView");
   pd->RegisterField("solution_portmode", &x);
   pd->SetLevelsOfDetail(port_order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
   delete port_fespace;
   delete port_fec;
   delete port_pmesh;
   return 0;
}

/**
   Solves the eigenvalue problem -Div(Grad x) = lambda x with homogeneous
   Dirichlet boundary conditions on the boundary of the domain. Returns mode
   number "mode" (counting from zero) in the ParGridFunction "x".
*/
void ScalarWaveGuide(int mode, ParGridFunction &x)
{
   int nev = std::max(mode + 2, 5);
   int seed = 75;

   ParFiniteElementSpace &port_fespace = *x.ParFESpace();
   ParMesh &pmesh = *port_fespace.GetParMesh();

   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   Vector mu(pmesh.attributes.Max());
   cout<<"pmesh.attributes.Max(): "<<pmesh.attributes.Max()<<std::endl;
   mu = 1.0;
   mu(1) = mu(0)*10;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm a(&port_fespace);

   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   a.EliminateEssentialBCDiag(ess_bdr, 1.0);
   a.Finalize();

   ParBilinearForm m(&port_fespace);
   m.AddDomainIntegrator(new MassIntegrator(mu_func));
   m.Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreBoomerAMG amg(*A);
   amg.SetPrintLevel(0);

   HypreLOBPCG lobpcg(MPI_COMM_WORLD);
   lobpcg.SetNumModes(nev);
   lobpcg.SetRandomSeed(seed);
   lobpcg.SetPreconditioner(amg);
   lobpcg.SetMaxIter(200);
   lobpcg.SetTol(1e-8);
   lobpcg.SetPrecondUsageMode(1);
   lobpcg.SetPrintLevel(1);
   lobpcg.SetMassMatrix(*M);
   lobpcg.SetOperator(*A);
   lobpcg.Solve();

   x = lobpcg.GetEigenvector(mode);

   delete A;
   delete M;
}

/**
   Solves the eigenvalue problem -Curl(Curl x) = lambda x with homogeneous
   Dirichlet boundary conditions, on the tangential component of x, on the
   boundary of the domain. Returns mode number "mode" (counting from zero) in
   the ParGridFunction "x".
*/
void VectorWaveGuide(int mode, ParGridFunction &x)
{
   int nev = std::max(mode + 2, 5);

   ParFiniteElementSpace &port_fespace = *x.ParFESpace();
   ParMesh &pmesh = *port_fespace.GetParMesh();

   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm a(&port_fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator);
   a.Assemble();
   a.EliminateEssentialBCDiag(ess_bdr, 1.0);
   a.Finalize();

   ParBilinearForm m(&port_fespace);
   m.AddDomainIntegrator(new VectorFEMassIntegrator);
   m.Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreAMS ams(*A,&port_fespace);
   ams.SetPrintLevel(0);
   ams.SetSingularProblem();

   HypreAME ame(MPI_COMM_WORLD);
   ame.SetNumModes(nev);
   ame.SetPreconditioner(ams);
   ame.SetMaxIter(100);
   ame.SetTol(1e-8);
   ame.SetPrintLevel(1);
   ame.SetMassMatrix(*M);
   ame.SetOperator(*A);
   ame.Solve();

   x = ame.GetEigenvector(mode);

   delete A;
   delete M;
}

/**
   Solves the eigenvalue problem -Div(Grad x) = lambda x with homogeneous
   Neumann boundary conditions on the boundary of the domain. Returns mode
   number "mode" (counting from zero) in the ParGridFunction "x_l2". Note that
   mode 0 is a constant field so higher mode numbers are often more
   interesting. The eigenmode is solved using continuous H1 basis of the
   appropriate order and then projected onto the L2 basis and returned.
*/
void PseudoScalarWaveGuide(int mode, ParGridFunction &x_l2)
{
   int nev = std::max(mode + 2, 5);
   int seed = 75;

   ParFiniteElementSpace &port_fespace_l2 = *x_l2.ParFESpace();
   ParMesh &pmesh = *port_fespace_l2.GetParMesh();
   int order_l2 = port_fespace_l2.FEColl()->GetOrder();

   H1_FECollection port_fec(order_l2+1, pmesh.Dimension());
   ParFiniteElementSpace port_fespace(&pmesh, &port_fec);
   ParGridFunction x(&port_fespace);
   x = 0.0;

   GridFunctionCoefficient xCoef(&x);

   if (mode == 0)
   {
      x = 1.0;
      x_l2.ProjectCoefficient(xCoef);
      return;
   }

   ParBilinearForm a(&port_fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator); // Shift eigenvalues by 1
   a.Assemble();
   a.Finalize();

   ParBilinearForm m(&port_fespace);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreBoomerAMG amg(*A);
   amg.SetPrintLevel(0);

   HypreLOBPCG lobpcg(MPI_COMM_WORLD);
   lobpcg.SetNumModes(nev);
   lobpcg.SetRandomSeed(seed);
   lobpcg.SetPreconditioner(amg);
   lobpcg.SetMaxIter(200);
   lobpcg.SetTol(1e-8);
   lobpcg.SetPrecondUsageMode(1);
   lobpcg.SetPrintLevel(1);
   lobpcg.SetMassMatrix(*M);
   lobpcg.SetOperator(*A);
   lobpcg.Solve();

   x = lobpcg.GetEigenvector(mode);

   x_l2.ProjectCoefficient(xCoef);

   delete A;
   delete M;
}

// Compute eigenmode "mode" of either a Dirichlet or Neumann Laplacian or of a
// Dirichlet curl curl operator based on the problem type and dimension of the
// domain.
void SetPortBC(int prob, int dim, int mode, ParGridFunction &port_bc)
{
   switch (prob)
   {
      case 0:
         ScalarWaveGuide(mode, port_bc);
         break;
      case 1:
         if (dim == 3)
         {
            VectorWaveGuide(mode, port_bc);
         }
         else
         {
            PseudoScalarWaveGuide(mode, port_bc);
         }
         break;
      case 2:
         PseudoScalarWaveGuide(mode, port_bc);
         break;
   }
}

void GridFunction_VectorFunctionCoefficient::Eval(Vector &V, ElementTransformation &T, 
                                                   const IntegrationPoint &ip)
{
   real_t x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   V.SetSize(vdim);
   if (Function)
   {
      Function(transip, V, mode_gf);
   }
   if (Q)
   {
      V *= Q->Eval(T, ip, GetTime());
   }
}

void source(const Vector &x, Vector &f, GridFunction &gf)
{
   //for dim = 2
   Vector phy_point(1);
   phy_point(0) = x(1);
   real_t position_mode_x = 0.3;
   real_t position_mode_y = 0.5;
   //phy_point(1) = 0.25;
   //phy_point.Print(cout << "physical point: ");
   
   IntegrationPoint ip;
   int elem_idx;
   ElementTransformation* tran;
   for (int i=0; i<gf.FESpace()->GetNE(); ++i)
   {  
      tran = gf.FESpace()->GetElementTransformation(i);
      InverseElementTransformation invtran(tran);
      int ret = invtran.Transform(phy_point, ip);
      if (ret == 0)
      {
         elem_idx = i;
         break;
      }
   }

   // cout << elem_idx << "-th element\n"
   //    << "reference point: " << ip.x << endl;
   real_t gf_value = 0;
   //cout << "GridFunction value: " << gf.GetValue(elem_idx, ip) << endl;
   gf_value = gf.GetValue(elem_idx, ip);
   real_t r = 0.0;
   r = pow(x[0] - position_mode_x, 2.);
   double n = 20 / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   f = 0.0;
   f[0] = gf_value * coeff * exp(alpha);
}
