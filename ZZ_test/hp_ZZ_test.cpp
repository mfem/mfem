//                         Test of ZZ error estimator
//
// Compile with: make ZZ_test
//

#include "mfem.hpp"
#include <fstream>

// #include "exact.hpp"

using namespace std;
using namespace mfem;

double lshape_exsol(const Vector &p, double omega);
void   lshape_exgrad(const Vector &p, double omega, Vector &grad);
double lshape_laplace(const Vector &p);

int dim;
const char* keys = "Rjlmc*******";

bool ContainsVertex(Mesh *mesh, int elem, const Vertex& vert) // different name?
{
   IsoparametricTransformation Tr;
   mesh->GetElementTransformation(elem, &Tr);
   IntegrationPoint reference_pt;
   Vector physical_pt(3);
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         reference_pt.Set((float)i, (float)j, 0.0, 0.0);
         Tr.Transform(reference_pt, physical_pt);
         double dist = 0.0;
         for (int l = 0; l < 2; l++)
         {
            double d = physical_pt(l) - vert(l);
            dist += d*d;
         }
         if (dist == 0) { return true; }
      }
   }
   return false;
   // mesh->GetElementVertices(elem, v);
   // for (int j = 0; j < v.Size(); j++)
   // {
   //    double* vertex = mesh->GetVertex(v[j]);
   //    double dist = 0.0;
   //    for (int l = 0; l < 2; l++) // Euclidean distance in x-y plane
   //    {
   //       double d = vert(l) - vertex[l];
   //       dist += d*d;
   //    }
   //    if (dist == 0) { return true; }
   // }
   // return false;
}

GridFunction* ProlongToMaxOrder(const GridFunction *x)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   // find the max order in the space
   int max_order = 1;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      max_order = std::max(fespace->GetElementOrder(i), max_order);
   }

   // create a visualization space of max order for all elements
   FiniteElementCollection *l2fec =
      new L2_FECollection(max_order, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace *l2space = new FiniteElementSpace(mesh, l2fec);

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *prolonged_x = new GridFunction(l2space);

   // interpolate solution vector in the larger space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geom);

      Array<int> dofs;
      fespace->GetElementDofs(i, dofs);
      Vector elemvect, l2vect;
      x->GetSubVector(dofs, elemvect);

      const auto *fe = fec->GetFE(geom, fespace->GetElementOrder(i));
      const auto *l2fe = l2fec->GetFE(geom, max_order);

      l2fe->GetTransferMatrix(*fe, T, I);
      l2space->GetElementDofs(i, dofs);
      l2vect.SetSize(dofs.Size());

      I.Mult(elemvect, l2vect);
      prolonged_x->SetSubVector(dofs, l2vect);
   }

   prolonged_x->MakeOwner(l2fec);
   return prolonged_x;
}

int main(int argc, char *argv[])
{
   // Parse command-line options.
   int problem = 0;
   int order = 1;
   double ref_threshold = 0.8;
   int nc_limit = 1;
   const char *device_config = "cpu";
   bool visualization = false;
   int which_estimator = 0;
   double angle = 7.0*M_PI/4.0;
   // double angle = 3.0*M_PI/2.0;

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type: 0 = canonical L-shaped solution, 1 = sinusoid.");
   args.AddOption(&order, "-o", "--order",
                  "Initial mesh finite element order (polynomial degree).");
   args.AddOption(&ref_threshold, "-rt", "--ref-threshold",
                  "Refine elements with error larger than threshold * max_error.");
   args.AddOption(&angle, "-a", "--angle", "Angle of the reentrant corner.");
   args.AddOption(&nc_limit, "-nc", "--nc-limit",
                  "Set maximum difference of refinement levels of adjacent elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&which_estimator, "-est", "--estimator",
                  "Which estimator to use: "
                  "0 = ZZ, 1 = Kelly. Defaults to ZZ.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   const char *mesh_file;
   // if (problem == 0)
   // {
      mesh_file = "l-shape-benchmark.mesh";
   // }


   // 2. Read the (serial) mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   mesh.EnsureNodes();
   Vector nodes;
   mesh.GetNodes(nodes);
   int num_nodes = int(nodes.Size()/2);
   for (int i = 0; i < num_nodes; i++)
   {
      double x = nodes[2*i];
      double y = nodes[2*i+1];
      double theta = atan2(y, x);
      if (theta < 0) { theta += 2.0*M_PI; }
      double delta_theta = theta * (angle - 3.0*M_PI/2.0) / (3.0*M_PI/2.0);
      nodes[2*i] = x*cos(delta_theta) - y*sin(delta_theta);
      nodes[2*i+1] = x*sin(delta_theta) + y*cos(delta_theta);
   }
   mesh.SetNodes(nodes);

   dim = mesh.Dimension();
   mesh.EnsureNCMesh();
   mesh.UniformRefinement(); // ZZ doesn't work properly on the initial L-shaped mesh

   // Define a finite element space on the mesh.
   H1_FECollection fec(order, dim);
   L2_FECollection l2fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // Define the solution vector x as a finite element grid function
   // corresponding to fespace.
   GridFunction x(&fespace);

   // Define exact solutions
   FunctionCoefficient *exsol=nullptr;
   VectorFunctionCoefficient *exgrad=nullptr;
   FunctionCoefficient *rhs=nullptr;
   ConstantCoefficient one(1.0);

//    switch (problem)
//    {
//       case 1:
//       {
//          exsol = new FunctionCoefficient(sinsin_exsol);
//          exgrad = new VectorFunctionCoefficient(dim, sinsin_exgrad);
//          rhs = new FunctionCoefficient(sinsin_laplace);
//          break;
//       }
//       case 2:
//       {
//          exsol = new FunctionCoefficient(poly_exsol);
//          exgrad = new VectorFunctionCoefficient(dim, poly_exgrad);
//          rhs = new FunctionCoefficient(poly_laplace);
//          break;
//       }
//       default:
//       case 0:
//       {
         exsol = new FunctionCoefficient(lshape_exsol);
         exgrad = new VectorFunctionCoefficient(dim, lshape_exgrad);
         rhs = new FunctionCoefficient(lshape_laplace);
         exsol->SetTime(angle);
         exgrad->SetTime(angle);
//          break;
//       }
//    }

   // Set up the linear form b(.) and the bilinear form a(.,.).
   LinearForm b(&fespace);
   BilinearForm a(&fespace);

   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs));
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");   

   // Connect to GLVis.
   socketstream sol_sock, ord_sock, dbg_sock[3], err_sock;

   ostringstream file_name;
   file_name << "conv_order" << order << ".csv";
   ofstream conv(file_name.str().c_str());
//    std::ofstream elemerr(elemerr_file);

   if (visualization)
   {
      cout << "\n Press enter to advance... " << endl;
      cin.get();
   }

   cout << setw(4) << "\nRef." << setw(12) << "DOFs" << setw(21) << "H^1_0 error" << setw(21) << "error estimate" << setw(18) << "H^1_0 rate" << setw(18) << "estimator rate" <<  endl;
   conv << "DOFs " << ", " << "H^1_0 error" << ", " << "error estimate" << endl;

   double old_num_dofs = 0.0;
   double old_H10_error = 0.0;
   double old_ZZ_error = 0.0;
   const int max_dofs = 20000;
   for (int it = 0; ; it++)
   {
      int num_dofs = fespace.GetTrueVSize();

      // Set Dirichlet boundary values in the GridFunction x.
      // Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      x = 0.0;
      x.ProjectBdrCoefficient(*exsol, ess_bdr);
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   
      // Solve for the current mesh:
      b.Assemble();
      a.Assemble();
      a.Finalize();
      OperatorPtr A;
      Vector B, X;
   
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 2000, 1e-30, 0.0);

      a.RecoverFEMSolution(X, b, x);

      // Calculate the total error in the H^1_0 norm.
      double H10_error = x.ComputeGradError(exgrad);
      DiffusionIntegrator di;

      ErrorEstimator* estimator{nullptr};
      switch (which_estimator)
      {
         case 1:
         {
            auto flux_fes = new FiniteElementSpace(&mesh, &fec, dim);
            estimator = new NewZienkiewiczZhuEstimator(di, x, flux_fes);
            // int flux_order = 4;
            // estimator = new NewZienkiewiczZhuEstimator(di, x, flux_order);
            break;
         }

         case 2:
         {
            auto flux_fes = new FiniteElementSpace(&mesh, &l2fec, dim);
            estimator = new KellyErrorEstimator(di, x, flux_fes);
            break;
         }

         default:
            std::cout << "Unknown estimator. Falling back to ZZ." << std::endl;
         case 0:
         {
            auto flux_fes = new FiniteElementSpace(&mesh, &fec, dim);
            estimator = new ZienkiewiczZhuEstimator(di, x, flux_fes);
            break;
         }
      }
      StopWatch chrono;
      chrono.Clear();
      chrono.Start();
      const Vector &zzerr = estimator->GetLocalErrors();
      chrono.Stop();
      mfem::out << "get local errors time: " << chrono.RealTime() << endl;
      // double ZZ_error = zzerr.Norml2();
      double ZZ_error = estimator->GetTotalError();

      // estimate convergence rate
      double H10_rate = 0.0;
      double ZZ_rate = 0.0;
      if (old_H10_error > 0.0)
      {
          H10_rate = log(H10_error/old_H10_error) / log(old_num_dofs/num_dofs);
          ZZ_rate  = log(ZZ_error/old_ZZ_error)   / log(old_num_dofs/num_dofs);
      }

      cout << setw(4) << it << setw(12) << num_dofs << setw(21) << H10_error << setw(21) << ZZ_error << setw(18) << H10_rate << setw(18) << ZZ_rate << endl;

      // Send solution by socket to the GLVis server.
      if (visualization)
      {
         const char vishost[] = "localhost";
         const int  visport   = 19916;

         // Prolong the solution vector onto L2 space of max order (for GLVis)
         GridFunction *vis_x = ProlongToMaxOrder(&x);
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << *vis_x;
         sol_sock << "keys ARjlm\n";
         delete vis_x;

         // Visualize element orders
         if (true)
         {
            L2_FECollection l20fec(0, dim);
            FiniteElementSpace l20fes(&mesh, &l20fec);
            GridFunction orders(&l20fes);

            for (int i = 0; i < orders.Size(); i++)
            {
               orders(i) = fespace.GetElementOrder(i);
            }

            socketstream ord_sock(vishost, visport);
            ord_sock.precision(8);
            ord_sock << "solution\n" << mesh << orders;
            ord_sock << "keys ARjlmpc**]]]]]]]]]]\n";
         }
      }

      if (num_dofs > max_dofs)
      {
         cout << "\n Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // Save dofs and error for convergence plot
      conv << num_dofs << ", " << H10_error << ", " << ZZ_error << endl;

    //   for (int i = 0; i < mesh.GetNE(); i++)
    //   {
    //      elemerr << sqrt(elemError[i]) << ' ';
    //   }
    //   elemerr << endl;


    //   Array<Refinement> refinements;
    //   double err_max = zzerr.Max();
    //   for (int i = 0; i < mesh.GetNE(); i++)
    //   {
    //      if (zzerr[i] > ref_threshold * err_max)
    //      {
    //          refinements.Append(Refinement(i, 7));
    //      }
    //   }
    //   mesh.GeneralRefinement(refinements, -1, nc_limit);
      int h_refined = 0, p_refined = 0;

      Array<Refinement> h_refinements;
      Array<int> p_refinements;
      const Table& table = mesh.ElementToElementTable();
      const Vertex origin(0,0);

      double err_max = zzerr.Max();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (zzerr[i] > ref_threshold * err_max)
         {
            if (ContainsVertex(&mesh, i, origin))
            {
               h_refinements.Append(Refinement(i));
               h_refined++;
               cout << "=> h-refined" << endl;
            }
            else
            {
               p_refinements.Append(i);
            //    // Also refine face-neighbors up to this order
            //    int elem_p = fespace.GetElementOrder(i);
            //    const int* row = table.GetRow(i);
            //    int row_size = table.RowSize(i);
            //    for (int j = 0; j < row_size; j++)
            //    {
            //       int neig_p = fespace.GetElementOrder(row[j]);
            //       if (neig_p <= elem_p)
            //       {
            //          p_refinements.Append(row[j]);
            //       }
            //    }
            }
         }
      }
      // Filter for unique elements
      p_refinements.Sort();
      p_refinements.Unique();
      for (auto i: p_refinements)
      {
         int p = fespace.GetElementOrder(i);
         fespace.SetElementOrder(i, p+1);
         p_refined++;
         cout << "=> p-refined" << endl;
      }

      old_num_dofs = double(num_dofs);
      old_H10_error = H10_error;
      old_ZZ_error = ZZ_error;

      // Update the space, interpolate the solution.
      fespace.Update(false);
      mesh.GeneralRefinement(h_refinements, -1, nc_limit);
      fespace.Update(false);

      a.Update();
      b.Update();
      x.Update();

      if (visualization)
      {
         cin.get();
      }

      // Free the used memory.
      delete estimator;
   }

   // Free the used memory.
   delete exsol;
   delete exgrad;
   delete rhs;
   return 0;
}


// L-shape domain problem exact solution (2D)

double lshape_exsol(const Vector &p, double omega)
{
   double alpha = M_PI / omega;
   double x = p(0), y = p(1);
   double r = sqrt(x*x + y*y);
   double t = atan2(y, x);
   if (t < 0) { t += 2.0*M_PI; }
   return pow(r, alpha) * sin(t*alpha);
}

void lshape_exgrad(const Vector &p, double omega, Vector &grad)
{
   double alpha = M_PI / omega;
   double x = p(0), y = p(1);
   double t = atan2(y, x);
   if (t < 0) { t += 2*M_PI; }
   double talpha = t*alpha;
   double ralpha = pow(x*x + y*y, alpha);
   grad(0) = alpha*x*sin(talpha)/(ralpha) - alpha*y*cos(talpha)/(ralpha);
   grad(1) = alpha*y*sin(talpha)/(ralpha) + alpha*x*cos(talpha)/(ralpha);
}

double lshape_laplace(const Vector &p)
{
   return 0.0;
}