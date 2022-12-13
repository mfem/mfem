//                                MFEM Example 0
//
// Compile with: make ex-pref-estimator-conv
//
// Sample runs:  ex-pref-estimator-conv -o 2 -rs 0 -nrand 3 -prob 0.0 -type 1 -es 3
// ./ex-pref-estimator-conv -o 2 -rs 0 -type 1 -es 4 -refa 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int type = 0;

double sfun(const Vector & x)
{
    if (type == 0) { //Gaussian bump
        double xc = x(0) - 0.5;
        double yc = x(1) - 0.5;
        return std::exp(-100*(xc*xc+yc*yc));

    }
    else if (type == 1) { // sin(2 pi x)*sin(2 pi y) + cos(2 pi x)*cos(2 pi y)
        return std::sin(x(0)*2.0*M_PI)*std::sin(x(1)*2.0*M_PI) +
               std::cos(x(0)*2.0*M_PI)*std::cos(x(1)*2.0*M_PI);
    }
    else if (type == 2) { // sin(2 pi x)*sin(2 pi y)
        return std::sin(x(0)*2.0*M_PI)*std::sin(x(1)*2.0*M_PI);
    }
    else if (type == 3) { // sin(2 pi r) + sin(3 pi r)
        double xc = x(0) - 0.5;
        double yc = x(1) - 0.5;
        double rc =std::pow(xc*xc+yc*yc, 0.5);
        return std::sin(rc*2.0*M_PI) + std::sin(rc*3.0*M_PI);
    }
    else if (type == 4) { //Eq 3.3 from https://www.math.tamu.edu/~guermond/PUBLICATIONS/MS/non_stationnary_jlg_rp_bp.pdf
        double xc = x(0) - 1.0;
        double yc = x(1) - 1.0;
        double rc =std::pow(xc*xc+yc*yc, 0.5);
        if (std::fabs(2*rc-0.3) <= 0.25) {
            return std::exp(-100*(2*rc-0.3)*(2*rc-0.3)); //300
        }
        else if (std::fabs(2*rc-0.9) <= 0.25) {
            return std::exp(-100*(2*rc-0.3)*(2*rc-0.3)); //300
        }
        else if (std::fabs(2*rc-1.6) <= 0.2) {
            return std::pow(1.0 - std::pow((2*rc-1.6)/0.5, 2.0), 0.5); // /0.2
        }
        else if (std::fabs(rc-0.3) < 0.2) {
            return 0.0;
        }
        return 0.0;
    }
    else {
        MFEM_ABORT(" unknown function type. ");
    }
    return 0.0;
}

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int rs = 0;
   int nrand = 0;
   double probmin = 0.0;
   int estimator = 0;
   double theta = 0.0;
   int niters = 0;
   bool pref = false;
   bool vis = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&rs, "-rs", "--rs", "Number of refinements");
   args.AddOption(&type, "-type", "--type", "Type of function");
   args.AddOption(&nrand, "-nrand", "--nrand", "Number of random refinement");
   args.AddOption(&probmin, "-prob", "--prob", "Min probability of refinement when nrand > 0");
   args.AddOption(&estimator, "-es", "--estimator", "ZZ(1), Kelly(2), P-1(3), FaceJump(4), ZZ+SolJump(5)");
   args.AddOption(&theta, "-theta", "--theta", "AMR theta factor");
   args.AddOption(&niters, "-n", "--niter", "Number of AMR steps");
   args.AddOption(&pref, "-pref", "--pref", "-no-pref",
                  "--no-pref",
                  "Enable or disable p-refinement mode.");
   args.AddOption(&vis, "-vis", "--vis", "-no-vis",
                  "--no-vis",
                  "Enable or disable visualization.");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   for (int i = 0; i < rs; i++) {
       mesh.UniformRefinement();
   }
   mesh.EnsureNCMesh();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   int dim = mesh.Dimension();
   L2_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   //Do random p-refinement
   for (int i = 0; i < nrand; i++) {
       for (int e = 0; e < mesh.GetNE(); e++) {
           double probref = (double) rand() / RAND_MAX;
           double inc = probref > probmin ? 1.0 : 0.0;
           fespace.SetElementOrder(e,fespace.GetElementOrder(e)+inc);
       }
   }
   fespace.Update(false);

   GridFunction x(&fespace);
   x = 0.0;

   // Element order after p-refinement
   L2_FECollection visfec(0, mesh.Dimension());
   FiniteElementSpace visfespace(&mesh, &visfec);
   GridFunction ElOrder(&visfespace);
   for (int e = 0; e < mesh.GetNE(); e++) {
       ElOrder(e) = fespace.GetElementOrder(e);
   }
   int max_order = fespace.GetMaxElementOrder();

   // Function Coefficient
   FunctionCoefficient scoeff(sfun);
   x.ProjectCoefficient(scoeff);

   // Compute exact error
   Vector elem_errors_exact(mesh.GetNE());
   x.ComputeElementL2Errors(scoeff, elem_errors_exact, NULL);

   // Convergence
   ConvergenceStudy error_study;
   ConstantCoefficient one(1.0);
   DiffusionIntegrator integ(one);
   L2_FECollection flux_fec(max_order, mesh.Dimension());
   FiniteElementSpace flux_fespace(&mesh, &flux_fec);

   Array<double> estimates;
   Array<double> estimates_rate;
   Array<int> ndofs;
   Array<int> nels;
   estimates_rate.Append(0.0);

   ErrorEstimator *es = NULL;
   if (estimator == 0) {
       es = new ExactError(x, scoeff);
   }
   else if (estimator == 1) {
       es = new LSZienkiewiczZhuEstimator(integ, x);
   }
   else if (estimator == 2) {
       es = new KellyErrorEstimator(integ, x, &flux_fespace);
   }
   else if (estimator == 3) {
       es = new PRefDiffEstimator(x, -1);
   }
   else if (estimator == 4) {
       es = new PRefJumpEstimator(x);
   }
   else if (estimator == 5) {
       es = new LSZienkiewiczZhuEstimator(integ, x);
       dynamic_cast<LSZienkiewiczZhuEstimator *>(es)->EnableSolutionBasedFit();
   }
   else {
       MFEM_ABORT("invalid estimator type");
   }

   socketstream sol_out;
   socketstream mesh_out;
   char vishost[] = "localhost";
   int  visport   = 19916;

   BilinearForm m(&fespace);
   m.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));

   LinearForm l(&fespace);
   l.AddDomainIntegrator(new DomainLFIntegrator(scoeff));
   Vector element_estimates;
   Array<int> element_marker;

   // L2 Projection solution
   x = 0.0;
   m.Assemble();
   l.Assemble(false);
   m.Mult(l,x);

   double exact_err = x.ComputeL2Error(scoeff);
   Vector error_estimate = es->GetLocalErrors();
   double tot_es_err = es->GetTotalError();

   for (int it = 0; it < niters; it++)
   {

       for (int e = 0; e < mesh.GetNE(); e++)
       {
          ElOrder(e) = fespace.GetElementOrder(e);
       }

       // L2 projection or regular projection
       x = 0.0;
       m.Assemble();
       l.Assemble(false);
       m.Mult(l,x);
       // x.ProjectCoefficient(scoeff);

       //
       ndofs.Append(fespace.GetTrueVSize());
       nels.Append(mesh.GetNE());
       error_study.AddL2GridFunction(&x,&scoeff);
       element_estimates = es->GetLocalErrors();
       estimates.Append(element_estimates.Norml2());

       exact_err = x.ComputeL2Error(scoeff);
       tot_es_err = es->GetTotalError();

       if (it>0)
       {
          double num =  log(estimates[it-1]/estimates[it]);
          double den = log((double)ndofs[it]/ndofs[it-1]);
          estimates_rate.Append(dim*num/den);
       }

       const char * keys = (it == 0) ? "pppppjRmlk\n" : nullptr;
       const char * viskeys = (it == 0) ? "jRmlkc\n" : nullptr;
       if (vis) {
           if (pref)
           {
              GridFunction *xprolong = ProlongToMaxOrder(&x);
              common::VisualizeField(sol_out,vishost,visport,*xprolong,"",0,0,500,500,keys);
              common::VisualizeField(mesh_out,vishost,visport,ElOrder,"orders",550,0,500,500,viskeys);
              delete xprolong;
           }
           else
           {
              common::VisualizeField(sol_out,vishost,visport,x,"",0,0,800,800,keys);
           }
       }

       if (it == niters - 1)
       {
          break;
       }


       // Adaptive refinements;
       double max_estimate = element_estimates.Max();
       element_marker.SetSize(0);
       for (int iel = 0; iel<mesh.GetNE(); iel ++)
       {
          if (element_estimates[iel] >= theta * max_estimate)
          {
             element_marker.Append(iel);
          }
       }

       if (pref)
       {
          // Array<int> additional_elements;
          // const Table & e2e = mesh.ElementToElementTable();
          // for (int iel = 0; iel<element_marker.Size(); iel++)
          // {
          //    int el = element_marker[iel];
          //    int eorder = fespace.GetElementOrder(el);
          //    int size = e2e.RowSize(el);
          //    const int * row = e2e.GetRow(el);
          //    for (int j = 0; j<size; j++)
          //    {
          //       int norder = fespace.GetElementOrder(row[j]);
          //       if (norder <= eorder)
          //       {
          //          additional_elements.Append(row[j]);
          //       }
          //    }
          // }
          // element_marker.Append(additional_elements);
          // element_marker.Sort();
          // element_marker.Unique();

          for (int iel = 0; iel<element_marker.Size(); iel++)
          {
             int el = element_marker[iel];
             int eorder = fespace.GetElementOrder(el);
             fespace.SetElementOrder(el,eorder+1);
          }
       }
       else
       {
          mesh.GeneralRefinement(element_marker,-1,1);
       }

       fespace.Update(false);
       x.Update();
       l.Update();
       m.Update();
       visfespace.Update(false);
       ElOrder.Update();
   }

   // Info for python plots
   std::cout << "\n";
   std::cout << " ---------------------------------------------------------------------------------------" << "\n";
   std::cout <<  std::setw(82) << "                       Convergence Info                               " << "\n";
   std::cout << " ---------------------------------------------------------------------------------------"
      << "\n";
   std::cout << std::right<< std::setw(15)<< "Type "<<
                std::setw(8) << "Estimator "<<
                std::setw(8) << "Order "<<
                std::setw(8) << "NEL "<<
                std::setw(10) << "NDOFS "<<
                std::setw(10) << "ExactL2 "<<
                std::setw(10) << "EstimatorL2" <<
                std::setw(10) << "Ref Mode" << std::endl;
   std::cout << " ---------------------------------------------------------------------------------------"
      << "\n";
   std::cout << std::setprecision(4);
   for (int i = 0; i<niters; i++)
   {
      std::cout << std::right << std::setw(10)<< type
                << std::setw(10)<< estimator
                << std::setw(10)<< order
                << std::setw(10)<< nels[i]
                << std::setw(10)<< ndofs[i]
                << std::setw(10)<< error_study.GetL2Error(i)
                << std::setw(10)<< estimates[i]
                << std::setw(10)<< pref
                << " ConvInfo \n";
   }

   // Info for vis inspection
   error_study.Print();

   std::cout << "\n";
   std::cout << " -------------------------------------------" << "\n";
   std::cout <<  std::setw(31) << "   Estimates    " << "\n";
   std::cout << " -------------------------------------------"
      << "\n";
   std::cout << std::right<< std::setw(11)<< "DOFs "<< std::setw(15) << "Estimate ";
   std::cout <<  std::setw(13) << "Rate " << "\n";
   std::cout << " -------------------------------------------"
      << "\n";
   std::cout << std::setprecision(4);
   for (int i = 0; i<niters; i++)
   {
      std::cout << std::right << std::setw(10)<< ndofs[i] << std::setw(16)
            << std::scientific << estimates[i] << std::setw(13)
            << std::fixed << estimates_rate[i] << "\n";
   }

//   delete xprolong;
   delete es;
   return 0;
}
