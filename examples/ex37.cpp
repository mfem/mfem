//                                MFEM Example 37
//
// Compile with: make ex37
//
// Sample runs:  ex37
//               ex37 -i surface
//               ex37 -i surface -o 0
//               ex37 -i surface -r 1
//               ex37 -i surface -o 4
//               ex37 -i surface -o 4 -r 5
//               ex37 -i volumetric
//               ex37 -i volumetric -o 0
//               ex37 -i volumetric -r 1
//               ex37 -i volumetric -o 4
//               ex37 -i volumetric -o 4 -r 5
//
// Description: This example code demonstrates the use of MFEM to integrate
//              functions over implicit interfaces and subdomains bounded by
//              implicit interfaces.
//
//              The quadrature rules are constructed by means of moment-fitting.
//              The interface is given by the zero iso line of a level-set
//              function ϕ and the subdomain is given as the domain where ϕ>0
//              holds. The algorithm for construction of the quadrature rules
//              was introduced by Mueller, Kummer and Oberlack [1].
//
//              There is an example for the integration of a quadratic function
//              over the sphere in 2 dimensions and an example computong the
//              arclength and area of an ellipse in 2 dimensions.
//
//              This example showcases how to set up integrators using the
//              integration rules on surfaces and subdomains.
//
// [1] Mueller, B., Kummer, F. and Obelrack, M. (2013) Highly accurate surface
//     and volume integration on implicit domains by means of moment-fitting.
//     Int. J. Numer. Meth. Engng. (96) 512-528. DOI:10.1002/nme.4569

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

/// @brief Integration rule the example should demonstrate
enum class IntegrationType { Surface, Volumetric };
IntegrationType itype;

/// @brief Level-set function defining the implicit interface
double lvlset(const Vector& X)
{
   switch (itype)
   {
      case IntegrationType::Surface:
         return 1. - (pow(X(0), 2.) + pow(X(1), 2.));
      case IntegrationType::Volumetric:
         return 1. - (pow(X(0) / 1.5, 2.) + pow(X(1) / .75, 2.));
      default:
         return 1.;
   }
}

/// @brief Function that should be integrated
double integrand(const Vector& X)
{
   switch (itype)
   {
      case IntegrationType::Surface:
         return 3. * pow(X(0), 2.) - pow(X(1), 2.);
      case IntegrationType::Volumetric:
         return 1.;
      default:
         return 0.;
   }
}

/// @brief Analytic surface integral
double Surface()
{
   switch (itype)
   {
      case IntegrationType::Surface:
         return 2. * M_PI;
      case IntegrationType::Volumetric:
         return 7.26633616541076;
      default:
         return 0.;
   }
}

/// @brief Analyitc volume integral over subdomain with positiv level-set
double Volume()
{
   switch (itype)
   {
      case IntegrationType::Surface:
         return NAN;
      case IntegrationType::Volumetric:
         return 9. / 8. * M_PI;
      default:
         return 0.;
   }
}

#ifdef MFEM_USE_LAPACK
/**
 @brief Class for surface linearform integrator

 Integrator to emonstrate the use of the surface integration rule on an
 implicit surface defined by a level-set.
 */
class SurfaceLFIntegrator : public LinearFormIntegrator
{
protected:
   /// @brief vector to evaluate the basis functions
   Vector shape;

   /// @brief surface integration rule
   SIntegrationRule* SIntRule;

   /// @brief coefficient representing the level-set defining the interface
   Coefficient &LevelSet;

   /// @brief coefficient representing the integrand
   Coefficient &Q;

public:
   /**
    @brief Constructor for the surface linear form integrator

    Constructor for the surface linear form integrator to demonstrate the use
    of the surface integration rule by means of moment-fitting.

    @param [in] q coefficient representing the inegrand
    @param [in] levelset level-set defining the implicit interfac
    @param [in] ir surface integrtion rule to be used
    */
   SurfaceLFIntegrator(Coefficient &q, Coefficient &levelset,
                       SIntegrationRule* ir)
      : LinearFormIntegrator(), Q(q), LevelSet(levelset), SIntRule(ir) {}

   /**
    @brief Constructor for the surface linear form integrator

    Constructor for the surface linear form integrator to demonstrate the use
    of the surface integration rule by means of moment-fitting.

    @param [in] q coefficient representing the inegrand
    @param [in] levelset level-set defining the implicit interfac
    */
   SurfaceLFIntegrator(Coefficient &q, Coefficient &levelset)
      : LinearFormIntegrator(), Q(q), LevelSet(levelset), SIntRule(NULL) {}

   /**
    @brief Assembly of the element vector

    Assemble the element vector of for the right hand side on the element given
    by the FiniteElement and ElementTransformation.

    @param [in] el finite Element the vector belongs to
    @param [in] Tr transformation of finite element
    @param [out] elvect vector containing the
   */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override
   {
      int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.;
      IsoparametricTransformation Trafo;
      Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

      // Update the surface integration rule for the current element
      SIntRule->Update(Trafo);

      for (int ip = 0; ip < SIntRule->GetNPoints(); ip++)
      {
         Trafo.SetIntPoint((&(SIntRule->IntPoint(ip))));
         double val = Trafo.Weight() * Q.Eval(Trafo, SIntRule->IntPoint(ip));
         el.CalcShape(SIntRule->IntPoint(ip), shape);
         add(elvect, SIntRule->IntPoint(ip).weight * val, shape, elvect);
      }
   }

   /// @brief Get the level-set defining the implicit interface
   void SetSurface(Coefficient &levelset) { LevelSet = levelset; }

   /// @brief Set the surface integration rule
   void SetSIntRule(SIntegrationRule *ir) { SIntRule = ir; }

   /// @brief Get the surface integration rule
   const SIntegrationRule* GetSIntRule() { return SIntRule; }
};

/**
 @brief Class for surface linearform integrator

 Integrator to emonstrate the use of the surface integration rule on an
 implicit surface defined by a level-set.
 */
class SubdomainLFIntegrator : public LinearFormIntegrator
{
protected:
   /// @brief vector to evaluate the basis functions
   Vector shape;

   /// @brief surface integration rule
   CutIntegrationRule* CutIntRule;

   /// @brief coefficient representing the level-set defining the interface
   Coefficient &LevelSet;

   /// @brief coefficient representing the integrand
   Coefficient &Q;

public:
   /**
    @brief Constructor for the volumetric subdomain linear form integrator

    Constructor for the subdomain linear form integrator to demonstrate the use
    of the volumeric subdomain integration rule by means of moment-fitting.

    @param [in] q coefficient representing the inegrand
    @param [in] levelset level-set defining the implicit interfac
    @param [in] ir subdomain integrtion rule to be used
    */
   SubdomainLFIntegrator(Coefficient &q, Coefficient &levelset,
                         CutIntegrationRule* ir)
      : LinearFormIntegrator(), Q(q), LevelSet(levelset), CutIntRule(ir) {}

   /**
    @brief Constructor for the volumetric subdomain linear form integrator

    Constructor for the subdomain linear form integrator to demonstrate the use
    of the volumeric subdomain integration rule by means of moment-fitting.

    @param [in] q coefficient representing the inegrand
    @param [in] levelset level-set defining the implicit interfac
    */
   SubdomainLFIntegrator(Coefficient &q, Coefficient &levelset)
      : LinearFormIntegrator(), Q(q), LevelSet(levelset), CutIntRule(NULL) {}

   /**
    @brief Assembly of the element vector

    Assemble the element vector of for the right hand side on the element given
    by the FiniteElement and ElementTransformation.

    @param [in] el finite Element the vector belongs to
    @param [in] Tr transformation of finite element
    @param [out] elvect vector containing the
   */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override
   {
      int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.;
      IsoparametricTransformation Trafo;
      Tr.mesh->GetElementTransformation(Tr.ElementNo, &Trafo);

      // Update the subdomain integration rule
      CutIntRule->Update(Trafo);

      for (int ip = 0; ip < CutIntRule->GetNPoints(); ip++)
      {
         Trafo.SetIntPoint((&(CutIntRule->IntPoint(ip))));
         double val = Trafo.Weight()
                      * Q.Eval(Trafo, CutIntRule->IntPoint(ip));
         el.CalcPhysShape(Trafo, shape);
         add(elvect, CutIntRule->IntPoint(ip).weight * val, shape, elvect);
      }
   }

   /// @brief Get the level-set defining the implicit interface
   void SetSurface(Coefficient &levelset) { LevelSet = levelset; }

   /// @brief Set the volumetric subdomain integration rule
   void SetCutIntRule(CutIntegrationRule *ir) { CutIntRule = ir; }

   /// @brief Get the volumetricsubdomain integration
   const CutIntegrationRule* GetCutIntRule() { return CutIntRule; }
};
#endif //MFEM_USE_LAPACK

int main(int argc, char *argv[])
{
#ifndef MFEM_USE_LAPACK
   cout << "MFEM must be build with LAPACK for this example." << endl;
   return EXIT_FAILURE;
#else
   // 1. Parse he command-line options.
   int ref_levels = 3;
   int order = 2;
   const char *inttype = "surface";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Order of quadrature rule");
   args.AddOption(&ref_levels, "-r", "--refine", "Number of meh refinements");
   args.AddOption(&inttype, "-i", "--integrationtype",
                  "IntegrationType to demonstrate");
   args.ParseCheck();

   if (strcmp(inttype, "surface") == 0 || strcmp(inttype, "Surface") == 0)
   {
      itype = IntegrationType::Surface;
   }
   else if (strcmp(inttype, "volumetric") == 0
            || strcmp(inttype, "Volumetric") == 0)
   {
      itype = IntegrationType::Volumetric;
   }

   // 2. Construct and refine the mesh.
   Mesh *mesh = new Mesh(2, 4, 1, 0, 2);
   mesh->AddVertex(-1.6,-1.6);
   mesh->AddVertex(1.6,-1.6);
   mesh->AddVertex(1.6,1.6);
   mesh->AddVertex(-1.6,1.6);
   mesh->AddQuad(0,1,2,3);
   mesh->FinalizeQuadMesh(1, 0, 1);

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 3. Define the necessary finite element space on the mesh.
   H1_FECollection fe_coll(1, mesh->Dimension());
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, &fe_coll);

   // 4.
   FunctionCoefficient levelset(lvlset);
   FunctionCoefficient u(integrand);

   // 5. Define the necessary Integration rules on element 0.
   IsoparametricTransformation Tr;
   mesh->GetElementTransformation(0, &Tr);
   SIntegrationRule* sir = new SIntegrationRule(order, Tr, levelset);
   CutIntegrationRule* cir = NULL;
   if (itype == IntegrationType::Volumetric)
   {
      cir = new CutIntegrationRule(order, Tr, levelset);
   }

   // 6. Define and assemble the linar forms on the finite element space.
   LinearForm surface(fespace);
   LinearForm volume(fespace);

   surface.AddDomainIntegrator(new SurfaceLFIntegrator(u, levelset, sir));
   surface.Assemble();

   if (itype == IntegrationType::Volumetric)
   {
      volume.AddDomainIntegrator(new SubdomainLFIntegrator(u, levelset, cir));
      volume.Assemble();
   }

   // 7. Print information, computed values and errors to the console.
   int qorder = 0;
   int nbasis = 2 * (order + 1) + (int)(order * (order + 1) / 2);
   IntegrationRules irs(0, Quadrature1D::GaussLegendre);
   IntegrationRule ir = irs.Get(Geometry::SQUARE, qorder);
   for (; ir.GetNPoints() <= nbasis; qorder++)
   {
      ir = irs.Get(Geometry::SQUARE, qorder);
   }
   cout << "============================================" << endl;
   cout << "Mesh size dx:                       ";
   cout << 3.2 / pow(2., (double)ref_levels) << endl;
   cout << "Number of div free basis functions: " << nbasis << endl;
   cout << "Number of quadrature points:        " << ir.GetNPoints() << endl;
   cout << scientific << setprecision(2);
   cout << "============================================" << endl;
   cout << "Computed value of surface integral: " << surface.Sum() << endl;
   cout << "True value of surface integral:     " << Surface() << endl;
   cout << "Absolut Error (Surface):            ";
   cout << abs(surface.Sum() - Surface()) << endl;
   cout << "Relative Error (Surface):           ";
   cout << abs(surface.Sum() - Surface()) / Surface() << endl;
   if (itype == IntegrationType::Volumetric)
   {
      cout << "--------------------------------------------" << endl;
      cout << "Computed value of volume integral:  " << volume.Sum() << endl;
      cout << "True value of volume integral:      " << Volume() << endl;
      cout << "Absolut Error (Volume):             ";
      cout << abs(volume.Sum() - Volume()) << endl;
      cout << "Relative Error (Volume):            ";
      cout << abs(volume.Sum() - Volume()) / Volume() << endl;
   }
   cout << "============================================" << endl;

   // 8. Plot the level-set function on a high order finite element space.
   H1_FECollection fe_coll2(5, 2);
   FiniteElementSpace fespace2(mesh, &fe_coll2);
   FunctionCoefficient levelset_coeff(levelset);
   GridFunction lgf(&fespace2);
   lgf.ProjectCoefficient(levelset_coeff);
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);
   sol_sock << "solution\n" << *mesh << lgf << flush;
   sol_sock << "keys pppppppppppppppppppppppppppcmmlRj\n";
   sol_sock << "levellines " << 0. << " " << 0. << " " << 1 << "\n" << flush;

   delete sir;
   delete fespace;
   delete mesh;
   return EXIT_SUCCESS;
#endif //MFEM_USE_LAPACK
}