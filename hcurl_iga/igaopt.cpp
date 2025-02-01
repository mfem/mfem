//                                MFEM Example 3 -- modified for NURBS FE
//
// Compile with: make igaopt
//
// Sample runs:  mpirun -np 4 igaopt -m ../../data/square-nurbs.mesh
//               mpirun -np 4 igaopt -m ../../data/square-nurbs.mesh -o 2
//               mpirun -np 4 igaopt -m ../../data/cube-nurbs.mesh
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               NURBS-based H(curl) spaces only implemented for meshes
//               consisting of a single patch.
//
//               We recommend viewing examples 1-2 before viewing this example.
#pragma GCC push_options
#pragma GCC optimize ("O0")


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace mfem;


class MyParGridFunction : public ParGridFunction
{
public:
   // MyParGridFunction() : ParGridFunction() {}
   MyParGridFunction(ParFiniteElementSpace *_f) : ParGridFunction(_f) {}
   GridFunction *Globalize(Mesh *, const int *);
   ~MyParGridFunction(){ }
};

GridFunction *MyParGridFunction::Globalize(Mesh *mesh, const int *partitioning)
{
   ParFiniteElementSpace *par_fes = this->ParFESpace();
   // duplicate the FiniteElementCollection from 'this'
   FiniteElementCollection *fec;
   fec = FiniteElementCollection::New(par_fes->FEColl()->Name());
   // create a global FiniteElementSpace from the local one:
   FiniteElementSpace *fes;
   fes = new FiniteElementSpace(mesh, fec);
   // create GridFunction
   GridFunction *gf;
   gf = new GridFunction(fes);
   const int myid = par_fes->GetMyRank();
   const int ne = fes->GetNE();

   Array<int> gvdofs, lvdofs;
   Vector lnodes;
   int element_counter = 0;
   MPI_Status status;
   // data to send and received
   int *index;
   double *nodes;
   int size;
   for (int i = 0; i < ne ; i++)
   {
      if (partitioning[i] == myid)
      {
         par_fes->GetElementVDofs(element_counter, lvdofs);
         this->GetSubVector(lvdofs, lnodes);
         fes->GetElementVDofs(i, gvdofs);
         // process 0, set values
         if (myid == 0)
            gf->SetSubVector(gvdofs, lnodes);
         // other process, send values to process 0
         else
         {
            size = gvdofs.Size();
            index = new int[size];
            index = gvdofs.GetData();
            nodes = new double[size];
            nodes = lnodes.GetData();
            MPI_Send(&size, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
            MPI_Send(index, size, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(nodes, size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
         }
         element_counter++;
      }
      if (myid == 0 && partitioning[i]!= 0)
      {
         MPI_Recv(&size, 1, MPI_INT, partitioning[i], 10, MPI_COMM_WORLD,&status);
         index = new int[size];
         nodes = new double[size];
         MPI_Recv(index, size, MPI_INT, partitioning[i], 0, MPI_COMM_WORLD, &status );
         MPI_Recv(nodes, size, MPI_DOUBLE, partitioning[i] ,1, MPI_COMM_WORLD, &status);
         gvdofs = Array<int>(index,size);
         lnodes = Vector(nodes,size);
         gf->SetSubVector(gvdofs, lnodes);
      }
   }

   MPI_Bcast(gf->GetData(), gf->Size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

   return gf;
}



// Class for setting up a simple Cartesian PML region
class PML
{
private:
   Mesh *mesh;

   int dim;

   // Length of the PML Region in each direction
   Array2D<real_t> length;

   // Computational Domain Boundary
   Array2D<real_t> comp_dom_bdr;

   // Domain Boundary
   Array2D<real_t> dom_bdr;

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   PML(Mesh *mesh_,Array2D<real_t> length_);

   // Return Computational Domain Boundary
   Array2D<real_t> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<real_t> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(ParMesh *mesh_);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<real_t>> &dxs, real_t eps);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML * pml = nullptr;
   PWConstCoefficient epsc;
   void (*Function)(const Vector &, PML *, Vector &, real_t);
public:
   PMLDiagMatrixCoefficient(int dim, void(*F)(const Vector &, PML *,
                                              Vector &, real_t),
                            PML * pml_,
                            PWConstCoefficient epsc)
      : VectorCoefficient(dim), pml(pml_), Function(F), epsc(epsc)
   {}

   using VectorCoefficient::Eval;

   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   { 
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      real_t eps = epsc.Eval(T,ip);
      //mfem::out << "MAIN.cpp row 108: " << eps <<std::endl;
      (*Function)(transip, pml, K, eps);
   }
};

class GridFunction_VectorFunctionCoefficient : public VectorCoefficient
{
private:
   GridFunction mode_gf;
   std::function<void(const Vector &, Vector &, GridFunction &, real_t)> Function;
   Coefficient *Q;
   PWConstCoefficient pweps;
public:
   GridFunction_VectorFunctionCoefficient(int dim,
                             std::function<void(const Vector &, Vector &, GridFunction &, real_t )> F,
                             GridFunction gf, PWConstCoefficient pweps,
                             Coefficient *q = nullptr)
                             : VectorCoefficient(dim),mode_gf(gf),Function(std::move(F)),pweps(pweps),Q(q)
                             {};
   using VectorCoefficient::Eval;
   /// Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
};

template <typename T> T pow2(const T &x) { return x*x; }
void SetPortBC(int prob, int dim, int mode, ParGridFunction &port_bc, real_t &neff);
void source_mode_im(const Vector &x, Vector &f, GridFunction &gf, real_t eps);
void source_mode_re(const Vector &x, Vector &f, GridFunction &gf, real_t eps);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D, real_t eps);

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D, real_t eps);
void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D, real_t eps);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

double mu = 1.0;
//double epsilon = 1.0;
double omega;
int dim;
bool exact_known = false;


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   // Parse command-line options.
   const char *mesh_file = "../../data/square-nurbs.mesh";
   int ref_levels = -1;
   bool NURBS = true;
   int order = 1;
   bool static_cond = false;
   const char *device_config = "cuda";
   bool visualization = 1;
   double freq = 5.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&NURBS, "-n", "--nurbs", "-nn","--no-nurbs",
                  "NURBS.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if(myid == 0 )
   {
      args.PrintOptions(cout);
   }



   //set the waveguide mode port
   int port_order = 1;
   Mesh port_mesh(10, 0.2);

   for (int l = 0; l <= 4; l++)
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
            if ((coords[comp] < 0.12) && (coords[comp] > 0.08))
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
   //constexpr MemoryType mt = MemoryType::HOST;
   int *partitioning = port_mesh.GeneratePartitioning(num_procs, 1);
   ParMesh *port_pmesh = new ParMesh(MPI_COMM_WORLD, port_mesh, partitioning);
   FiniteElementCollection *port_fec;
   port_fec = new H1_FECollection(port_order, port_dim);
   ParFiniteElementSpace *port_fespace = new ParFiniteElementSpace(port_pmesh, port_fec);
   HYPRE_BigInt size = port_fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }
   MyParGridFunction port_x(port_fespace);
   int prob = 0;
   int mode = 0;
   real_t neff;
   SetPortBC(prob, port_dim, mode, port_x, neff);

   GridFunction *global_port_x = port_x.Globalize(&port_mesh, partitioning);

   real_t min_gf = 0, max_gf = 0;
   min_gf = global_port_x->Min();
   max_gf = global_port_x->Max();
   for(int i = 0; i < global_port_x->Size(); i ++)
   {
      (global_port_x->GetData())[i] =1 - (((global_port_x->GetData())[i])-min_gf)/(max_gf - min_gf);
   }

   // cout << "main.cpp row 359: " << neff << endl;

   // int k = 1;
   // while(k)
   // {
   //  sleep(5);
   // }

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Read the mesh from the given mesh file. We can handle triangular,
   // quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   // the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   //int sdim = mesh->SpaceDimension();
   // Setup PML length
   Array2D<real_t> length(dim, 2); length = 0.2;
   PML * pml = new PML(mesh,length);
   // Angular frequency
   omega = real_t(2.0 * M_PI) * freq;

   comp_domain_bdr = pml->GetCompDomainBdr();
   domain_bdr = pml->GetDomainBdr();

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 9; l++)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   //mfem::out<<"main.cpp: test row 304"<<std::endl;
   delete mesh;
   pml->SetAttributes(pmesh);

   int nelem = pmesh->GetNE();
   for (int i = 0; i < nelem; ++i)
   {
      bool is_waveguide = false;
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;

      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = pmesh->GetVertex(vert_idx);    
         if ((coords[1] < 0.52) && (coords[1] > 0.48))
         {
            is_waveguide = true;
            break;
         }
      }
      if (is_waveguide && (el->GetAttribute() != 2))
      {
         el->SetAttribute(3);
      }
      else if (is_waveguide && (el->GetAttribute() == 2))
      {
         el->SetAttribute(4);
      }
   }
   pmesh->SetAttributes();

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;

   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);                                                                              
   mfem::out<<"ID "<<myid<<" "<<getpid()<<" Create NURBS fec and ext"<<std::endl;

   // int k = 1;
   // while(k)
   // {
   //  sleep(5);
   // }

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   mfem::out<<"main.cpp: test row 243"<<std::endl;
   mfem::out << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;


   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   cout << "Number of knowns in essential BCs: "
        << ess_tdof_list.Size() << endl;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   Vector epsilon(pmesh->attributes.Max());
   epsilon = 1;
   epsilon(2) = epsilon(0)*20;
   epsilon(3) = epsilon(0)*20;
   PWConstCoefficient pweps(epsilon);

   Vector mepsilon(pmesh->attributes.Max());
   mepsilon = -pow2(omega);
   mepsilon(2) = mepsilon(0)*20;
   mepsilon(3) = mepsilon(0)*20;
   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;
   //VectorFunctionCoefficient f(dim, source);
   GridFunction_VectorFunctionCoefficient port_Coefficient_im(dim, source_mode_im, *global_port_x, pweps);
   //GridFunction_VectorFunctionCoefficient port_Coefficient_re(dim, source_mode_re, *global_port_x, pweps);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(port_Coefficient_im));
   b.Vector::operator=(0.0);
   b.Assemble();
   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParComplexGridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators.
   Array<int> attr;
   Array<int> attrPML;
   cout<<"pmesh->attributes.Max(): "<<pmesh->attributes.Max()<<std::endl;
   if (pmesh->attributes.Size())
   {
      attr.SetSize(pmesh->attributes.Max());
      attrPML.SetSize(pmesh->attributes.Max());
      attr = 0;   attr[0] = 1;  attr[2] = 1;
      attrPML = 0;
      if (pmesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
         attrPML[3] = 1;
      }
   }

   ConstantCoefficient muinv(1_r / mu);
   PWConstCoefficient omeg(mepsilon);
   RestrictedCoefficient restr_muinv(muinv,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

   // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(fespace, conv);
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),NULL);
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);
   
   int cdim = (dim == 2) ? 1 : dim;
   PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml, pweps);
   PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml, pweps);
   ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
   ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
   VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re, pml, pweps);
   PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im, pml, pweps);
   ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
   VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   // Integrators inside the PML region
   a.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                         new CurlCurlIntegrator(restr_c1_Im));
   a.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                         new VectorFEMassIntegrator(restr_c2_Im));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.

   a.Assemble(0);
   
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   mfem::out<<"main.cpp: test row 449"<<std::endl;
   cout << "Size of linear system: " << A->Height() << endl;

   Vector prec_epsilon(pmesh->attributes.Max());
   prec_epsilon = pow2(omega);
   prec_epsilon(2) = prec_epsilon(0)*20;
   prec_epsilon(3) = prec_epsilon(0)*20;

   PWConstCoefficient absomeg(prec_epsilon);
   RestrictedCoefficient restr_absomeg(absomeg,attr);

   ParBilinearForm prec(fespace);
   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));

   PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml, pweps);
   ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
   VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);
   
   PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs,pml, pweps);
   ScalarVectorProductCoefficient c2_abs(absomeg,pml_c2_abs);
   VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);
   
   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));
   
   prec.Assemble();
   
   // 14b. Define and apply a GMRES solver for AU=B with a block diagonal
   //      preconditioner based on the Gauss-Seidel or Jacobi sparse smoother.
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fespace->GetTrueVSize();
   offsets[2] = fespace->GetTrueVSize();
   offsets.PartialSum();
   
   std::unique_ptr<Operator> pc_r;
   std::unique_ptr<Operator> pc_i;
   real_t s = (conv == ComplexOperator::HERMITIAN) ? -1_r : 1_r;
   
   OperatorPtr PCOpAh;
   prec.SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
   prec.FormSystemMatrix(ess_tdof_list, PCOpAh);
   mfem::out<<"main.cpp: test row 497"<<std::endl;

   // Gauss-Seidel Smoother

   //pc_r.reset(new HypreSmoother(*PCOpAh.As<HypreParMatrix>()));
   pc_r.reset(new HypreBoomerAMG(*PCOpAh.As<HypreParMatrix>()));
   pc_i.reset(new ScaledOperator(pc_r.get(), s));
   
   BlockDiagonalPreconditioner BlockDP(offsets);
   BlockDP.SetDiagonalBlock(0, pc_r.get());
   BlockDP.SetDiagonalBlock(1, pc_i.get());
   
   GMRESSolver gmres(MPI_COMM_WORLD);
   // HypreSolver *amg = new HypreBoomerAMG;
   // HypreGMRES gmres(A);
   gmres.SetPrintLevel(1);
   gmres.SetKDim(200);
   gmres.SetMaxIter(10000);
   gmres.SetRelTol(1e-5);
   gmres.SetAbsTol(0.0);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(BlockDP);
   gmres.Mult(B, X);
   
   // mfem::out<<"ID "<<myid<<" "<<getpid()<<std::endl;
   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);
   // mfem::out<<"main.cpp: test row 405"<<std::endl;
   // int k = 1;
   // while(k)
   // {
   //  sleep(5);
   // }

   ofstream mesh_ofs("pndnurbs.mesh");
   mesh_ofs.precision(8);
   
   pmesh->Print(mesh_ofs);
   
   ofstream sol_r_ofs("pndnurbs-sol_r.gf");
   ofstream sol_i_ofs("pndnurbs-sol_i.gf");
   sol_r_ofs.precision(8);
   sol_i_ofs.precision(8);
   x.real().Save(sol_r_ofs);
   x.imag().Save(sol_i_ofs);
   
   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./ParaView");
   pd->RegisterField("solution_real", &(x.real()));
   pd->RegisterField("solution_imag", &(x.imag()));
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;


   // Free the used memory.
   delete pml;
   delete fespace;
   delete fec;
   delete pmesh;
   //delete NURBSext;// NURBSext have been destoryed when construct the ParNURBSext!!!
   if(myid == 0)
   {
      mfem::out<<"****** FINISH ******"<<std::endl;
   }
   return 0;
}

void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).real();
   }
   
}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow(dxs[i], 2));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D, real_t eps)
{  
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs, eps);
   
   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   
   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {  
      D = (1.0 / det).real();
   }
   else
   {  
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = (1.0 / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).imag();
      }
   }
}

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D, real_t eps)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, eps);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = abs(1.0 / det);
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = abs(pow(dxs[i], 2) / det);
      }
   }
}

PML::PML(Mesh *mesh_, Array2D<real_t> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void PML::SetBoundaries()
{  
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   for (int i = 0; i < dim; i++)
   {  
      dom_bdr(i, 0) = pmin(i);
      dom_bdr(i, 1) = pmax(i);
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void PML::SetAttributes(ParMesh *mesh_)
{
   // Initialize bdr attributes
   for (int i = 0; i < mesh_->GetNBE(); ++i)
   {
      mesh_->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = mesh_->GetNE();

   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         real_t *coords = mesh_->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void PML::StretchFunction(const Vector &x,
                          vector<complex<real_t>> &dxs, real_t eps)
{
   constexpr complex<real_t> zi = complex<real_t>(0., 1.);

   real_t n = 2.0;
   real_t c = 5.0;
   real_t coeff;
   real_t k = omega * sqrt(eps * mu);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1_r));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1_r + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1_r));
      }
   }
}

/**
   Solves the eigenvalue problem -Div(Grad x) = lambda x with homogeneous
   Dirichlet boundary conditions on the boundary of the domain. Returns mode
   number "mode" (counting from zero) in the ParGridFunction "x".
*/
void ScalarWaveGuide(int mode, ParGridFunction &x, real_t &neff)
{
   int nev = std::max(mode + 2, 5);
   int seed = 75;

   ParFiniteElementSpace &fespace = *x.ParFESpace();
   ParMesh &pmesh = *fespace.GetParMesh();

   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   Vector mu(pmesh.attributes.Max());
   cout<<"pmesh.attributes.Max(): "<<pmesh.attributes.Max()<<std::endl;
   mu = 1.0;
   mu(1) = mu(0)*400;
   PWConstCoefficient mu_func(mu);

   ParBilinearForm a(&fespace);

   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();
   a.EliminateEssentialBCDiag(ess_bdr, 1.0);
   a.Finalize();

   ParBilinearForm m(&fespace);
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
   Array<real_t> eigenvalues;
   lobpcg.GetEigenvalues(eigenvalues);
   neff = eigenvalues[mode];

   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("portmode", &pmesh);
   pd->SetPrefixPath("./PortModeParaView");
   pd->RegisterField("solution_portmode", &x);
   pd->SetLevelsOfDetail(1);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;

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

   ParFiniteElementSpace &fespace = *x.ParFESpace();
   ParMesh &pmesh = *fespace.GetParMesh();

   Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new CurlCurlIntegrator);
   a.Assemble();
   a.EliminateEssentialBCDiag(ess_bdr, 1.0);
   a.Finalize();

   ParBilinearForm m(&fespace);
   m.AddDomainIntegrator(new VectorFEMassIntegrator);
   m.Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m.EliminateEssentialBCDiag(ess_bdr, numeric_limits<real_t>::min());
   m.Finalize();

   HypreParMatrix *A = a.ParallelAssemble();
   HypreParMatrix *M = m.ParallelAssemble();

   HypreAMS ams(*A,&fespace);
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

   ParFiniteElementSpace &fespace_l2 = *x_l2.ParFESpace();
   ParMesh &pmesh = *fespace_l2.GetParMesh();
   int order_l2 = fespace_l2.FEColl()->GetOrder();

   H1_FECollection fec(order_l2+1, pmesh.Dimension());
   ParFiniteElementSpace fespace(&pmesh, &fec);
   ParGridFunction x(&fespace);
   x = 0.0;

   GridFunctionCoefficient xCoef(&x);

   if (mode == 0)
   {
      x = 1.0;
      x_l2.ProjectCoefficient(xCoef);
      return;
   }

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddDomainIntegrator(new MassIntegrator); // Shift eigenvalues by 1
   a.Assemble();
   a.Finalize();

   ParBilinearForm m(&fespace);
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
void SetPortBC(int prob, int dim, int mode, ParGridFunction &port_bc, real_t &neff)
{
   switch (prob)
   {
      case 0:
         ScalarWaveGuide(mode, port_bc, neff);
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
   real_t eps = pweps.Eval(T,ip);
   //mfem::out<<"main.cpp: test row 1092"<<std::endl;
   V.SetSize(vdim);
   if (Function)
   {  //mfem::out<<"main.cpp: test row 1095"<<std::endl;
      Function(transip, V, mode_gf, eps);//mfem::out<<"main.cpp: test row 1096"<<std::endl;
   }
   if (Q)
   {
      V *= Q->Eval(T, ip, GetTime());
   }
}

void source_mode_re(const Vector &x, Vector &f, GridFunction &gf, real_t eps)
{  
   //for dim = 2
   if (x(1)<0.4 || x(1)>0.6)
   {
      f = 0.0;
   }
   else
   {
      Vector phy_point(1);
      phy_point(0) = x(1)-0.4;
      real_t position_mode_x = 0.5;
      real_t position_mode_y = 0.5;
      //phy_point(1) = 0.25;
      //phy_point.Print(cout << "physical point: ");
      IntegrationPoint ip;
      int elem_idx;
      ElementTransformation* tran;
      //mfem::out << "main.cpp row 1054 gf.FESpace()->GetNE(): " <<gf.FESpace()->GetNE()<<endl;
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
      f = 0.0;
      // cout << elem_idx << "-th element\n"
      //    << "reference point: " << ip.x << endl;
      real_t gf_value = 0;
      // cout << "GridFunction value: " << gf.GetValue(elem_idx, ip) << endl;
      gf_value = gf.GetValue(elem_idx, ip);
      real_t r = 0.0;
      r = pow(x[0] - position_mode_x, 2.);
      double n = real_t(5) * omega * sqrt(eps * mu) / real_t(M_PI);
      double coeff = pow(n, 2) / M_PI;
      double alpha = -pow(n, 4) * r;
      //gf_value = pow(gf_value, 2);
      f[1] = gf_value * sin(coeff * (x[0] - position_mode_x)); //* exp(alpha);
   }
}

void source_mode_im(const Vector &x, Vector &f, GridFunction &gf, real_t eps)
{  
   //for dim = 2
   if (x(1)<0.4 || x(1)>0.6)
   {
      f = 0.0;
   }
   else
   {
      Vector phy_point(1);
      phy_point(0) = x(1)-0.4;
      real_t position_mode_x = 0.5;
      real_t position_mode_y = 0.5;
      //phy_point(1) = 0.25;
      //phy_point.Print(cout << "physical point: ");
      IntegrationPoint ip;
      int elem_idx;
      ElementTransformation* tran;
      //mfem::out << "main.cpp row 1054 gf.FESpace()->GetNE(): " <<gf.FESpace()->GetNE()<<endl;
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
      f = 0.0;
      // cout << elem_idx << "-th element\n"
      //    << "reference point: " << ip.x << endl;
      real_t gf_value = 0;
      // cout << "GridFunction value: " << gf.GetValue(elem_idx, ip) << endl;
      gf_value = gf.GetValue(elem_idx, ip);
      real_t r = 0.0;
      r = pow(x[0] - position_mode_x, 2.);
      real_t n = real_t(5) * omega * sqrt(2.9 * mu) / real_t(M_PI);
      real_t coeff = pow(n, 2) / M_PI;
      real_t alpha = -pow(n, 2) * r;
      //gf_value = pow(gf_value, 2);
      f[1] = eps * gf_value * coeff * exp(alpha);
   }
}


#pragma GCC pop_options
