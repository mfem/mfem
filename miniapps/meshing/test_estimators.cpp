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
            return std::exp(-300*(2*rc-0.3)*(2*rc-0.3));
        }
        else if (std::fabs(2*rc-0.9) <= 0.25) {
            return std::exp(-300*(2*rc-0.3)*(2*rc-0.3));
        }
        else if (std::fabs(2*rc-1.6) <= 0.2) {
            return std::pow(1.0 - std::pow((2*rc-1.6)/0.2, 2.0), 0.5);
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
   int ur = 0;
   int estimator = 0;
   double theta = 0.0;
   bool pref_mode = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&theta, "-theta", "--theta", "AMR theta factor");
   args.AddOption(&rs, "-rs", "--rs", "Number of AMR refinements");
   args.AddOption(&ur, "-ur", "--ur", "Number of uniform refinements");
   args.AddOption(&type, "-type", "--type", "Type of function");
   args.AddOption(&estimator, "-es", "--estimator", "Exact(0), LSZZ(1), LSZZmod(2), Kelly(3), P-1(4), FaceJump(5)");
   args.AddOption(&pref_mode, "-pref_mode", "--pref_mode", "-no-pref_mode",
                  "--no-pref_mode",
                  "Enable or disable p-refinement mode.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(mesh_file);
   mesh.EnsureNCMesh();

   for (int i = 0; i<ur; i++)
   {
      mesh.UniformRefinement();
   }

   int dim = mesh.Dimension();
   L2_FECollection fec(order-1, mesh.Dimension());
   L2_FECollection visfec(0, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   FiniteElementSpace visfespace(&mesh, &visfec);
   FunctionCoefficient scoeff(sfun);

   GridFunction x(&fespace);
   GridFunction orders_gf(&visfespace);

   ConvergenceStudy error_study;
   ErrorEstimator *es = NULL;
   ConstantCoefficient one(1.0);
   DiffusionIntegrator integ(one);
   L2_FECollection flux_fec(order, mesh.Dimension());
   FiniteElementSpace flux_fespace(&mesh, &flux_fec);
   Array<double> estimates; 
   Array<double> estimates_rate; 
   Array<int> ndofs; 
   estimates_rate.Append(0.0);
   if (estimator == 0) {
       es = new ExactError(x, scoeff);
   }
   else if (estimator == 1) {
       es = new LSZienkiewiczZhuEstimator(integ, x);
   }
   else if (estimator == 2) {
       es = new LSZienkiewiczZhuEstimator(integ, x);
       dynamic_cast<LSZienkiewiczZhuEstimator *>(es)->EnableSolutionBasedFit();
   }
   else if (estimator == 3) {
       es = new KellyErrorEstimator(integ, x, &flux_fespace);
   }
   else if (estimator == 4) {
       es = new PRefDiffEstimator(x, -1);
   }
   else if (estimator == 5) {
       es = new PRefJumpEstimator(x);
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
   for (int i = 0; i<rs; i++)
   {

      for (int e = 0; e < mesh.GetNE(); e++) 
      {
         orders_gf(e) = fespace.GetElementOrder(e);
      }   

      x = 0.0;
      m.Assemble();
      l.Assemble(false);
      m.Mult(l,x);
      // x.ProjectCoefficient(scoeff); // replace this with L2 projection;

      ndofs.Append(fespace.GetTrueVSize());
      error_study.AddL2GridFunction(&x,&scoeff);
      element_estimates = es->GetLocalErrors(); 
      estimates.Append(element_estimates.Norml2());
      if (i>0)
      {
         double num =  log(estimates[i-1]/estimates[i]);
         double den = log((double)ndofs[i]/ndofs[i-1]);
         estimates_rate.Append(dim*num/den);
      }
      const char * keys = (i == 0) ? "pppppjRmlk\n" : nullptr;
      const char * viskeys = (i == 0) ? "jRmlk\n" : nullptr;
      if (pref_mode)
      {
         GridFunction *xprolong = ProlongToMaxOrder(&x);
         common::VisualizeField(sol_out,vishost,visport,*xprolong,"",0,0,800,800,keys);
         common::VisualizeField(mesh_out,vishost,visport,orders_gf,"orders",0,0,800,800,viskeys);
      }
      else
      {
         common::VisualizeField(sol_out,vishost,visport,x,"",0,0,800,800,keys);
      }
      if (i == rs - 1)
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

      if (pref_mode)
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
      orders_gf.Update();
   }

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
   for (int i = 0; i<rs; i++)
   {
      std::cout << std::right << std::setw(10)<< ndofs[i] << std::setw(16)
            << std::scientific << estimates[i] << std::setw(13)
            << std::fixed << estimates_rate[i] << "\n";
   }

   return 0;

}