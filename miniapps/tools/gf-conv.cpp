#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static int dim;

enum SCA_TYPE {INVALID_SCA_TYPE = -1,
               H1_TYPE = 0,
               L2_TYPE,
               L2I_TYPE,
               NUM_SCA_TYPES
              };
enum CONV_TYPE {INVALID_CONV_TYPE = -1,
                PROJECTION = 0,
                INTERPOLATION_OP,
                SOLVE,
                SOLVE_W_DBC,
                NUM_CONV_TYPES
               };

FiniteElementCollection * GetFECollection(SCA_TYPE type, int p);
ParFiniteElementSpace * GetFESpace(SCA_TYPE type, ParMesh &pmesh,
                                   FiniteElementCollection &fec);
void parseFieldNames(const char * field_name_c_str,
                     vector<string> &field_names);

string GetTypeName(SCA_TYPE type);
string GetTypeShortName(SCA_TYPE type);
string GetConvTypeName(CONV_TYPE type);
string GetConvTypeShortName(CONV_TYPE type);

void Projection(const ParGridFunction &v0, ParGridFunction &v1);
void InterpolationOp(const ParGridFunction &v0, ParGridFunction &v1);
void LeastSquares(SCA_TYPE t0, const ParGridFunction &v0,
                  SCA_TYPE t1, ParGridFunction &v1);
void LeastSquaresBC(SCA_TYPE t0, const ParGridFunction &v0,
                    SCA_TYPE t1, ParGridFunction &v1,
                    Coefficient &c);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   Mpi::Init();
   if (!Mpi::Root()) { mfem::out.Disable(); mfem::err.Disable(); }
   Hypre::Init();
#endif

   // Parse command-line options.
   const char *coll_name = NULL;
   int cycle = 0;

   const char *field_name_c_str = "ALL";

   Array<int> orders;
   Array<int> types;
   Array<int> conv_types;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&coll_name, "-r", "--root-file",
                  "Set the VisIt data collection root file prefix.", true);
   args.AddOption(&cycle, "-c", "--cycle", "Set the cycle index to read.");
   args.AddOption(&field_name_c_str, "-fn", "--field-names",
                  "List of field names to get values from.");
   args.AddOption(&orders, "-o", "--final-order",
                  "Finite element orders for each final field "
                  "(an array of integers for multiple fields).");
   args.AddOption(&types, "-t", "--final-type",
                  "Set the basis type for the final fields: "
                  "0-H1, 1-L2, 2-L2I, -1 loop over all.");
   args.AddOption(&conv_types, "-ct", "--conversion-type",
                  "Set the conversion schemes: "
                  "0-Projection, 1-Interpolation Op, 2-Least Squares, "
                  "3-Least Squares with BC, -1 loop over all.");
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
      args.PrintUsage(mfem::out);
      return 1;
   }
   args.PrintOptions(mfem::out);

#ifdef MFEM_USE_MPI
   VisItDataCollection dc(MPI_COMM_WORLD, coll_name);
#else
   VisItDataCollection dc(coll_name);
#endif

   dc.Load(cycle);

   if (dc.Error() != DataCollection::NO_ERROR)
   {
      mfem::out << "Error loading VisIt data collection: " << coll_name << endl;
      return 1;
   }

   dim = dc.GetMesh()->Dimension();
   int spaceDim = dc.GetMesh()->SpaceDimension();

   mfem::out << endl;
   mfem::out << "Collection Name:    " << dc.GetCollectionName() << endl;
   mfem::out << "Manifold Dimension: " << dim << endl;
   mfem::out << "Space Dimension:    " << spaceDim << endl;
   mfem::out << "Cycle:              " << dc.GetCycle() << endl;
   mfem::out << "Time:               " << dc.GetTime() << endl;
   mfem::out << "Time Step:          " << dc.GetTimeStep() << endl;
   mfem::out << endl;

   typedef DataCollection::FieldMapType fields_t;
   const fields_t &fields = dc.GetFieldMap();
   // Print the names of all fields.
   mfem::out << "fields: [ ";
   for (fields_t::const_iterator it = fields.begin(); it != fields.end(); ++it)
   {
      if (it != fields.begin()) { mfem::out << ", "; }
      mfem::out << it->first;
   }
   mfem::out << " ]" << endl;

   // Parsing desired field names
   vector<string> field_names;
   parseFieldNames(field_name_c_str, field_names);

   if (field_names.size() == 1)
   {
      if (field_names[0] == "ALL")
      {
         fields_t::const_iterator it = fields.begin();
         field_names[0] = it->first; it++;
         for ( ; it != fields.end(); ++it)
         {
            field_names.push_back(it->first);
         }
      }
   }

   if (orders.Size() < field_names.size())
   {
      int size = orders.Size();
      int order = (size > 0) ? orders[0] : 1;

      orders.SetSize(field_names.size());
      for (int i=size; i < field_names.size(); i++)
      {
         orders[i] = order;
      }
   }

   if (types.Size() < field_names.size())
   {
      int size = types.Size();
      int type = (size > 0) ? types[0] : 0;

      types.SetSize(field_names.size());
      for (int i=size; i < field_names.size(); i++)
      {
         types[i] = type;
      }
   }

   if (conv_types.Size() < field_names.size())
   {
      int size = conv_types.Size();
      int type = (size > 0) ? conv_types[0] : 0;

      conv_types.SetSize(field_names.size());
      for (int i=size; i < field_names.size(); i++)
      {
         conv_types[i] = type;
      }
   }

   // Print field names to be extracted
   mfem::out << "Extracting fields: ";
   for (int i=0; i < field_names.size(); i++)
   {
      mfem::out << " \"" << field_names[i] << "\"";
   }
   mfem::out << endl;

#ifdef MFEM_USE_MPI
   ParMesh *mesh = dynamic_cast<ParMesh*>(dc.GetMesh());
#else
   Mesh *mesh = dc.GetMesh();
#endif
   if (mesh == NULL)
   {
      mfem::out << "Problem with mesh\n";
      return 1;
   }

   int Ww = 300, Wh = 220, Fw = 3, Fh = 23, Ws = 15;

   // Loop over all requested fields.
   for (int i=0; i < field_names.size(); i++)
   {
#ifdef MFEM_USE_MPI
      ParGridFunction *x0 = dc.GetParField(field_names[i]);
#else
      GridFunction *x0 = dc.GetField(field_names[i]);
#endif
      if (x0 == NULL)
      {
         mfem::out << "Problem with x0 for field \"" << field_names[i] << "\"\n";
         continue;
      }

      int t0 = 0;

      // nn. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         ostringstream oss;
         oss << field_names[i];
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock0(vishost, visport);
#ifdef MFEM_USE_MPI
         sol_sock0 << "parallel " << mesh->GetNRanks() << ' '
                   << mesh->GetMyRank() << '\n';
#endif
         sol_sock0.precision(8);
         sol_sock0 << "solution\n" << *mesh << *x0
                   << "window_title '" << oss.str() << "'"
                   << "window_geometry "
                   << Ws * (t0) << " " << Ws * (t0) << " "
                   << (int)(1.5 * Ww) << " " << (int)(1.5 * Wh)
                   << flush;
      }

      int t1 = types[i];
      FiniteElementCollection *fec1 = GetFECollection((SCA_TYPE)t1, orders[i]);
      ParFiniteElementSpace   *fes1 = GetFESpace((SCA_TYPE)t1, *mesh, *fec1);

      ParGridFunction *y1 = new ParGridFunction(fes1);

      mfem::out << GetTypeName((SCA_TYPE)t1) << "(" << orders[i] << ")"
                << ":" << endl;

      int c01 = conv_types[i];
      string cmnt = "";

      switch ((CONV_TYPE)c01)
      {
         case PROJECTION:
            Projection(*x0, *y1);
            break;
         case INTERPOLATION_OP:
            cmnt = (t0 == (int)H1_TYPE) || (t0 == t1) ?
                   "(should match projection)" : "(not expected to succeed)";
            InterpolationOp(*x0, *y1);
            break;
         case SOLVE:
            LeastSquares((SCA_TYPE)t0, *x0, (SCA_TYPE)t1, *y1);
            break;
         default:
            *y1 = 0.0;
      }

      {
         ostringstream oss;
         oss << field_names[i] << "_" << GetConvTypeShortName((CONV_TYPE)c01)
             << "_" << GetTypeShortName((SCA_TYPE)t1) << "_o" << orders[i];

         dc.RegisterField(oss.str(), y1);
      }

      if (visualization)
      {
         ostringstream oss;
         oss << GetConvTypeShortName((CONV_TYPE)c01) << "--> "
             << GetTypeName((SCA_TYPE)t1)<< "(" << orders[i] << ")";
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream sol_sock1(vishost, visport);
#ifdef MFEM_USE_MPI
         sol_sock1 << "parallel " << mesh->GetNRanks() << ' '
                   << mesh->GetMyRank() << '\n';
#endif
         sol_sock1.precision(8);
         sol_sock1 << "solution\n" << *mesh << y1
                   << "window_title '" << oss.str() << "'"
                   << "window_geometry "
                   << (int)((Ww + Fw) * (1.5 + c01) +
                            Ws * (t0))
                   << " " << (Wh + Fh) * (t1) + Ws * (t0)
                   << " " << Ww << " " << Wh
                   << flush;
      }
      mfem::out << endl;

      // delete fes1;
      // delete fec1;
   }
   dc.Save();

   return 0;
}

FiniteElementCollection * GetFECollection(SCA_TYPE type, int p)
{
   switch (type)
   {
      case H1_TYPE:
         return new H1_FECollection(p, dim);
      case L2_TYPE:
         return new L2_FECollection(p-1, dim);
      case L2I_TYPE:
         return new L2_FECollection(p-1, dim, BasisType::GaussLegendre,
                                    FiniteElement::INTEGRAL);
      default:
         return NULL;
   }
}

ParFiniteElementSpace * GetFESpace(SCA_TYPE type,
                                   ParMesh &pmesh,
                                   FiniteElementCollection &fec)
{
   return new ParFiniteElementSpace(&pmesh, &fec);
}

string GetTypeName(SCA_TYPE type)
{
   switch (type)
   {
      case H1_TYPE:
         return "    H1";
      case L2_TYPE:
         return "    L2";
      case L2I_TYPE:
         return "    L2I";
      default:
         return "--";
   }
}

string GetTypeShortName(SCA_TYPE type)
{
   switch (type)
   {
      case H1_TYPE:
         return "H1";
      case L2_TYPE:
         return "L2";
      case L2I_TYPE:
         return "L2I";
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

string GetConvTypeShortName(CONV_TYPE type)
{
   switch (type)
   {
      case PROJECTION:
         return "Proj";
      case INTERPOLATION_OP:
         return "Interp";
      case SOLVE:
         return "LS";
      case SOLVE_W_DBC:
         return "LSwBC";
      default:
         return "--";
   }
}

void parseFieldNames(const char * field_name_c_str, vector<string> &field_names)
{
   string field_name_str(field_name_c_str);
   string field_name;

   for (string::iterator it=field_name_str.begin();
        it!=field_name_str.end(); it++)
   {
      if (*it == '\\')
      {
         it++;
         field_name.push_back(*it);
      }
      else if (*it == ' ')
      {
         if (!field_name.empty())
         {
            field_names.push_back(field_name);
         }
         field_name.clear();
      }
      else if (it == field_name_str.end() - 1)
      {
         field_name.push_back(*it);
         field_names.push_back(field_name);
      }
      else
      {
         field_name.push_back(*it);
      }
   }
   if (field_names.size() == 0)
   {
      field_names.push_back("ALL");
   }
}

/** Perform a naive projection from one scalar field to another.

   This scheme simply evaluates v0 at the interpolation points of v1.

   If v0 has reduced continuity compared to v1 this can produce
   results that depend on the order in which the elements are
   traversed.

   Suitable conversions:
   H1 -> L2
   H1 -> DG (same as L2)
 */
void Projection(const ParGridFunction &v0, ParGridFunction &v1)
{
   GridFunctionCoefficient v0Coef(&v0);
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
void LeastSquares(SCA_TYPE t0, const ParGridFunction &v0,
                  SCA_TYPE t1, ParGridFunction &v1)
{
   ParFiniteElementSpace *fes0, *fes1;
   fes0 = v0.ParFESpace();
   fes1 = v1.ParFESpace();

   ParMixedBilinearForm op(fes0, fes1);
   op.AddDomainIntegrator(new MassIntegrator);
   op.Assemble();
   op.Finalize();

   ParLinearForm b(v1.ParFESpace());
   op.Mult(v0, b);

   ParBilinearForm m(v1.ParFESpace());
   m.AddDomainIntegrator(new MassIntegrator);
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
void LeastSquaresBC(SCA_TYPE t0, const ParGridFunction &v0,
                    SCA_TYPE t1, ParGridFunction &v1,
                    Coefficient &c)
{
   ParFiniteElementSpace *fes0, *fes1;
   fes0 = v0.ParFESpace();
   fes1 = v1.ParFESpace();

   ParMixedBilinearForm op(fes0, fes1);
   op.AddDomainIntegrator(new MassIntegrator);
   op.Assemble();
   op.Finalize();

   ParLinearForm b(v1.ParFESpace());
   op.Mult(v0, b);

   ParBilinearForm m(v1.ParFESpace());
   m.AddDomainIntegrator(new MassIntegrator);
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

   if (t1 == H1_TYPE)
   {
      v1.ProjectBdrCoefficient(c, ess_bdr);
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
