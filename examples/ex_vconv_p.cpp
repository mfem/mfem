#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static double freq = 0.5, kappa;
static int dim;

void u_func(const Vector &, Vector &);

enum VEC_TYPE {INVALID_VEC_TYPE = -1,
               H1V_TYPE = 0,
               ND_TYPE,
               RT_TYPE,
               L2V_TYPE,
               NUM_VEC_TYPES
              };
enum CONV_TYPE {INVALID_CONV_TYPE = -1,
                PROJECTION = 0,
                INTERPOLATION_OP,
                SOLVE,
                SOLVE_W_DBC,
                NUM_CONV_TYPES
               };

FiniteElementCollection * GetFECollection(VEC_TYPE type, int p);
ParFiniteElementSpace * GetFESpace(VEC_TYPE type, ParMesh &pmesh,
                                   FiniteElementCollection &fec);
string GetTypeName(VEC_TYPE type);
string GetConvTypeName(CONV_TYPE type);

void Projection(const ParGridFunction &v0, ParGridFunction &v1);
void InterpolationOp(const ParGridFunction &v0, ParGridFunction &v1);
void LeastSquares(VEC_TYPE t0, const ParGridFunction &v0,
                  VEC_TYPE t1, ParGridFunction &v1);
void LeastSquaresBC(VEC_TYPE t0, const ParGridFunction &v0,
                    VEC_TYPE t1, ParGridFunction &v1,
                    VectorCoefficient &vc);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order0 = 1;
   int order1 = 1;
   int type0 = 0;
   int type1 = 1;
   int conv_type = -1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order0, "-o0", "--initial-order",
                  "Finite element order (polynomial degree) "
                  "for initial field.");
   args.AddOption(&order1, "-o1", "--final-order",
                  "Finite element order (polynomial degree) "
                  "for final field.");
   args.AddOption(&type0, "-t0", "--initial-type",
                  "Set the basis type for the initial field: "
                  "0-H1V, 1-H(Curl), 2-H(Div), 3-L2V, -1 loop over all.");
   args.AddOption(&type1, "-t1", "--final-type",
                  "Set the basis type for the final field: "
                  "0-H1V, 1-H(Curl), 2-H(Div), 3-L2V, -1 loop over all.");
   args.AddOption(&conv_type, "-c", "--conversion-type",
                  "Set the conversion scheme: "
                  "0-Projection, 1-Interpolation Op, 2-Least Squares, "
                  "3-Least Squares with BC, -1 loop over all.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }
   kappa = freq * M_PI;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (mpi.Root()) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }
   if (dim > 2)
   {
      pmesh.ReorientTetMesh();
   }

   VectorFunctionCoefficient uCoef(dim, u_func);

   int Ww = 300, Wh = 220, Fw = 3, Fh = 23, Ws = 15;

   if (mpi.Root())
   {
      cout << "L2 Errors:" << endl;
   }
   int t0a = (type0 == -1) ? 0 : type0;
   int t0b = (type0 == -1) ? NUM_VEC_TYPES : (type0+1);
   for (int t0 = t0a; t0 < t0b; t0++)
   {
      FiniteElementCollection *fec0 = GetFECollection((VEC_TYPE)t0, order0);
      ParFiniteElementSpace   *fes0 = GetFESpace((VEC_TYPE)t0, pmesh, *fec0);

      ParGridFunction x0(fes0);
      x0.ProjectCoefficient(uCoef);

      double err0 = x0.ComputeL2Error(uCoef);
      if (mpi.Root())
      {
         cout << "Initial " << GetTypeName((VEC_TYPE)t0)
              << ": \t\t" << err0 << endl;
      }

      // nn. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         ostringstream oss;
         oss << GetTypeName((VEC_TYPE)t0);
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock0(vishost, visport);
         sol_sock0 << "parallel " << pmesh.GetNRanks() << ' '
                   << pmesh.GetMyRank() << '\n';
         sol_sock0.precision(8);
         sol_sock0 << "solution\n" << pmesh << x0
                   << "keys vvv "
                   << "window_title '" << oss.str() << "'"
                   << "window_geometry "
                   << Ws * (t0 - t0a) << " " << Ws * (t0 - t0a) << " "
                   << (int)(1.5 * Ww) << " " << (int)(1.5 * Wh)
                   << flush;
      }

      int t1a = (type1 == -1) ? 0 : type1;
      int t1b = (type1 == -1) ? NUM_VEC_TYPES : (type1+1);
      for (int t1 = t1a; t1 < t1b; t1++)
      {
         FiniteElementCollection *fec1 = GetFECollection((VEC_TYPE)t1, order1);
         ParFiniteElementSpace   *fes1 = GetFESpace((VEC_TYPE)t1, pmesh, *fec1);

         ParGridFunction x1(fes1);

         if (mpi.Root())
         {
            cout << GetTypeName((VEC_TYPE)t0) << " -> "
                 << GetTypeName((VEC_TYPE)t1) << ":" << endl;
         }

         int c01a = (conv_type == -1) ? 0 : conv_type;
         int c01b = (conv_type == -1) ? NUM_CONV_TYPES : (conv_type+1);
         for (int c01 = c01a; c01 < c01b; c01++)
         {
            switch ((CONV_TYPE)c01)
            {
               case PROJECTION:
                  Projection(x0, x1);
                  break;
               case INTERPOLATION_OP:
                  // InterpolationOp(x0, x1);
                  x1 = 0.0;
                  break;
               case SOLVE:
                  LeastSquares((VEC_TYPE)t0, x0, (VEC_TYPE)t1, x1);
                  break;
               case SOLVE_W_DBC:
                  LeastSquaresBC((VEC_TYPE)t0, x0, (VEC_TYPE)t1, x1, uCoef);
                  break;
               default:
                  x1 = 0.0;
            }

            double err1 = x1.ComputeL2Error(uCoef);
            cout << GetConvTypeName((CONV_TYPE)c01)
                 << "\t\t" << err1 << endl;

            if (visualization)
            {
               ostringstream oss;
               oss << GetTypeName((VEC_TYPE)t0) << " --" << c01 << "--> "
                   << GetTypeName((VEC_TYPE)t1);
               char vishost[] = "localhost";
               int  visport   = 19916;
               socketstream sol_sock1(vishost, visport);
               sol_sock1 << "parallel " << pmesh.GetNRanks() << ' '
                         << pmesh.GetMyRank() << '\n';
               sol_sock1.precision(8);
               sol_sock1 << "solution\n" << pmesh << x1
                         << "keys vvv "
                         << "window_title '" << oss.str() << "'"
                         << "window_geometry "
                         << (int)((Ww + Fw) * (1.5 + c01 - c01a) +
                                  Ws * (t0 - t0a))
                         << " " << (Wh + Fh) * (t1 - t1a) + Ws * (t0 - t0a)
                         << " " << Ww << " " << Wh
                         << flush;
            }
         }
         if (mpi.Root())
         {
            cout << endl;
         }

         delete fes1;
         delete fec1;
      }

      delete fes0;
      delete fec0;

      if (t0 < t0b - 1)
      {
         char c;
         if (mpi.Root())
         {
            cout << "press (q)uit or (c)ontinue --> " << flush;
            cin >> c;
         }
         MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

         if (c != 'c')
         {
            break;
         }
      }
      if (mpi.Root())
      {
         cout << endl;
      }
   }

   return 0;
}

void u_func(const Vector &x, Vector &u)
{
   u.SetSize(dim);

   double kx = kappa * x[0];
   double ky = kappa * x[1];
   double kz = (dim == 3) ? (kappa * x[2]) : 0.0;

   // Add the gradient of a scalar function
   u(0) = sin(kx) * cos(ky);
   u(1) = cos(kx) * sin(ky);
   if (dim == 3)
   {
      u(0) *= cos(kz);
      u(1) *= cos(kz);
      u(2) = cos(kx) * cos(ky) * sin(kz);
   }

   // Add the curl of a vector function
   u(0) -= cos(kx) * sin(ky);
   u(1) += sin(kx) * cos(ky);
   if (dim == 3)
   {
      u(0) += cos(kx) * sin(kz);
      u(1) -= cos(ky) * sin(kz);
      u(2) += (sin(ky) - sin(kx)) * cos(kz);
   }
}

FiniteElementCollection * GetFECollection(VEC_TYPE type, int p)
{
   switch (type)
   {
      case H1V_TYPE:
         return new H1_FECollection(p, dim);
      case ND_TYPE:
         return new ND_FECollection(p, dim);
      case RT_TYPE:
         return new RT_FECollection(p-1, dim);
      case L2V_TYPE:
         return new L2_FECollection(p-1, dim);
      default:
         return NULL;
   }
}

ParFiniteElementSpace * GetFESpace(VEC_TYPE type,
                                   ParMesh &pmesh,
                                   FiniteElementCollection &fec)
{
   switch (type)
   {
      case H1V_TYPE:
      case L2V_TYPE:
         return new ParFiniteElementSpace(&pmesh, &fec, dim);
      case ND_TYPE:
      case RT_TYPE:
         return new ParFiniteElementSpace(&pmesh, &fec);
      default:
         return NULL;
   }
}

string GetTypeName(VEC_TYPE type)
{
   switch (type)
   {
      case H1V_TYPE:
         return "    H1V";
      case ND_TYPE:
         return "H(Curl)";
      case RT_TYPE:
         return " H(Div)";
      case L2V_TYPE:
         return "    L2V";
      default:
         return "--";
   }
}

string GetConvTypeName(CONV_TYPE type)
{
   switch (type)
   {
      case PROJECTION:
         return "Projection            ";
      case INTERPOLATION_OP:
         return "Interpolation Operator";
      case SOLVE:
         return "Least Squares         ";
      case SOLVE_W_DBC:
         return "Least Squares with BC ";
      default:
         return "--";
   }
}

/** Perform a naive projection from one vector field to another.

   This scheme simply evaluates v0 at the interpolation points of v1.

   If v0 has reduced continuity compared to v1 this can produce
   results that depend on the order in which the elements are
   traversed.

   Suitable conversions:
   H1V     -> H(Curl), H(Div), or L2V
   H(Curl) -> L2V
   H(Div)  -> L2V

 */
void Projection(const ParGridFunction &v0, ParGridFunction &v1)
{
   VectorGridFunctionCoefficient v0Coef(&v0);
   v1.ProjectCoefficient(v0Coef);
}

/** In theory this interpolation scheme should be equivalent to projection.

    Building an interpolastion matrix could lead to computational
    efficiency compared to simple projection if the operator will be
    used several times.

    Unfortunately this is broken for several combinations of source
    and target fields.
*/
void InterpolationOp(const ParGridFunction &v0, ParGridFunction &v1)
{
   ParDiscreteLinearOperator op(v0.ParFESpace(), v1.ParFESpace());
   op.AddDomainInterpolator(new IdentityInterpolator);
   op.Assemble();
   op.Finalize();

   op.Mult(v0, v1);
}

/** Compute a least-squares best fit using the target basis functions.

    This scheme is more difficult to setup and more computationally
    expensive but the results can be significantly better than simple
    projections.
*/
void LeastSquares(VEC_TYPE t0, const ParGridFunction &v0,
                  VEC_TYPE t1, ParGridFunction &v1)
{
   bool trans = false;
   ParFiniteElementSpace *fes0, *fes1;
   if ((t0 == H1V_TYPE || t0 == L2V_TYPE) &&
       (t1 == ND_TYPE || t1 == RT_TYPE))
   {
      fes0 = v1.ParFESpace();
      fes1 = v0.ParFESpace();
      trans = true;
   }
   else
   {
      fes0 = v0.ParFESpace();
      fes1 = v1.ParFESpace();
   }
   ParMixedBilinearForm op(fes0, fes1);
   if (t0 == ND_TYPE || t0 == RT_TYPE || t1 == ND_TYPE || t1 == RT_TYPE)
   {
      op.AddDomainIntegrator(new VectorFEMassIntegrator);
   }
   else
   {
      op.AddDomainIntegrator(new VectorMassIntegrator);
   }
   op.Assemble();
   op.Finalize();

   ParLinearForm b(v1.ParFESpace());
   if (trans)
   {
      op.MultTranspose(v0, b);
   }
   else
   {
      op.Mult(v0, b);
   }

   ParBilinearForm m(v1.ParFESpace());
   if (t1 == ND_TYPE || t1 == RT_TYPE)
   {
      m.AddDomainIntegrator(new VectorFEMassIntegrator);
   }
   else
   {
      m.AddDomainIntegrator(new VectorMassIntegrator);
   }
   m.Assemble();
   m.Finalize();

   HypreParMatrix * M = m.ParallelAssemble();

   HypreDiagScale diag(*M);
   HyprePCG pcg(*M);
   pcg.SetPreconditioner(diag);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(1000);

   Vector B, X;
   b.ParallelAssemble(B);

   X.SetSize(v1.ParFESpace()->TrueVSize()); X = 0.0;
   pcg.Mult(B, X);
   v1.Distribute(X);

   delete M;
}

/** Compute a least-squares best fit with boundary conditions.

    This scheme is virtually identical to the previous one but it
    makes use of boundary values, when available, to improve the
    accuracy.  This scheme can produce significantly better results
    when the normal derivative of the field is large near the
    boundary.  This is particularly true when the field is
    under-resolved near the boundary.
*/
void LeastSquaresBC(VEC_TYPE t0, const ParGridFunction &v0,
                    VEC_TYPE t1, ParGridFunction &v1,
                    VectorCoefficient &vc)
{
   bool trans = false;
   ParFiniteElementSpace *fes0, *fes1;
   if ((t0 == H1V_TYPE || t0 == L2V_TYPE) &&
       (t1 == ND_TYPE || t1 == RT_TYPE))
   {
      fes0 = v1.ParFESpace();
      fes1 = v0.ParFESpace();
      trans = true;
   }
   else
   {
      fes0 = v0.ParFESpace();
      fes1 = v1.ParFESpace();
   }
   ParMixedBilinearForm op(fes0, fes1);
   if (t0 == ND_TYPE || t0 == RT_TYPE || t1 == ND_TYPE || t1 == RT_TYPE)
   {
      op.AddDomainIntegrator(new VectorFEMassIntegrator);
   }
   else
   {
      op.AddDomainIntegrator(new VectorMassIntegrator);
   }
   op.Assemble();
   op.Finalize();

   ParLinearForm b(v1.ParFESpace());
   if (trans)
   {
      op.MultTranspose(v0, b);
   }
   else
   {
      op.Mult(v0, b);
   }

   ParBilinearForm m(v1.ParFESpace());
   if (t1 == ND_TYPE || t1 == RT_TYPE)
   {
      m.AddDomainIntegrator(new VectorFEMassIntegrator);
   }
   else
   {
      m.AddDomainIntegrator(new VectorMassIntegrator);
   }
   m.Assemble();
   m.Finalize();

   Array<int> ess_bdr;
   Array<int> ess_tdof_list;
   if (v1.ParFESpace()->GetParMesh()->bdr_attributes.Size())
   {
      ess_bdr.SetSize(v1.ParFESpace()->GetParMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      v1.ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   if (t1 == H1V_TYPE)
   {
      v1.ProjectBdrCoefficient(vc, ess_bdr);
   }
   if (t1 == ND_TYPE)
   {
      v1.ProjectBdrCoefficientTangent(vc, ess_bdr);
   }
   else if (t1 == RT_TYPE)
   {
      v1.ProjectBdrCoefficientNormal(vc, ess_bdr);
   }

   OperatorPtr M;
   Vector B, X;
   m.FormLinearSystem(ess_tdof_list, v1, b, M, X, B);

   HypreDiagScale diag(*M.As<HypreParMatrix>());
   HyprePCG pcg(*M.As<HypreParMatrix>());
   pcg.SetPreconditioner(diag);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(1000);

   pcg.Mult(B, X);
   v1.Distribute(X);
}
