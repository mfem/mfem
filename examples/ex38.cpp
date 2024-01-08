//                                MFEM Example 38
//
// Compile with: make ex38
//
// Sample runs:
// (since all sample runs require LAPACK, the * symbol is used to exclude them
// from the automatically generated internal MFEM tests).
//              * ex38
//              * ex38 -i volumetric1d
//              * ex38 -i surface2d
//              * ex38 -i surface2d -o 4 -r 5
//              * ex38 -i volumetric2d
//              * ex38 -i volumetric2d -o 4 -r 5
//              * ex38 -i surface3d
//              * ex38 -i surface3d -o 4 -r 5
//              * ex38 -i volumetric3d
//              * ex38 -i volumetric3d -o 4 -r 5
//
// Description: This example code demonstrates the use of MFEM to integrate
//              functions over implicit interfaces and subdomains bounded by
//              implicit interfaces.
//
//              The quadrature rules are constructed by means of moment-fitting.
//              The interface is given by the zero isoline of a level-set
//              function ϕ and the subdomain is given as the domain where ϕ>0
//              holds. The algorithm for construction of the quadrature rules
//              was introduced by Mueller, Kummer and Oberlack [1].
//
//              This example also showcases how to set up integrators using the
//              integration rules on implicit surfaces and subdomains.
//
// [1] Mueller, B., Kummer, F. and Oberlack, M. (2013) Highly accurate surface
//     and volume integration on implicit domains by means of moment-fitting.
//     Int. J. Numer. Meth. Engr. (96) 512-528. DOI:10.1002/nme.4569

#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

/// @brief Integration rule the example should demonstrate
enum class IntegrationType { Volumetric1D, Surface2D, Volumetric2D,
                             Surface3D, Volumetric3D
                           };
IntegrationType itype;

/// @brief Level-set function defining the implicit interface
double lvlset(const Vector& X)
{
   switch (itype)
   {
      case IntegrationType::Volumetric1D:
         return .55 - X(0);
      case IntegrationType::Surface2D:
         return 1. - (pow(X(0), 2.) + pow(X(1), 2.));
      case IntegrationType::Volumetric2D:
         return 1. - (pow(X(0) / 1.5, 2.) + pow(X(1) / .75, 2.));
      case IntegrationType::Surface3D:
         return 1. - (pow(X(0), 2.) + pow(X(1), 2.) + pow(X(2), 2.));
      case IntegrationType::Volumetric3D:
         return 1. - (pow(X(0) / 1.5, 2.) + pow(X(1) / .75, 2.) + pow(X(2) / .5, 2.));
      default:
         return 1.;
   }
}

/// @brief Function that should be integrated
double integrand(const Vector& X)
{
   switch (itype)
   {
      case IntegrationType::Volumetric1D:
         return 1.;
      case IntegrationType::Surface2D:
         return 3. * pow(X(0), 2.) - pow(X(1), 2.);
      case IntegrationType::Volumetric2D:
         return 1.;
      case IntegrationType::Surface3D:
         return 4. - 3. * pow(X(0), 2.) + 2. * pow(X(1), 2.) - pow(X(2), 2.);
      case IntegrationType::Volumetric3D:
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
      case IntegrationType::Volumetric1D:
         return 1.;
      case IntegrationType::Surface2D:
         return 2. * M_PI;
      case IntegrationType::Volumetric2D:
         return 7.26633616541076;
      case IntegrationType::Surface3D:
         return 40. / 3. * M_PI;
      case IntegrationType::Volumetric3D:
         return 9.90182151329315;
      default:
         return 0.;
   }
}

/// @brief Analytic volume integral over subdomain with positive level-set
double Volume()
{
   switch (itype)
   {
      case IntegrationType::Volumetric1D:
         return .55;
      case IntegrationType::Surface2D:
         return NAN;
      case IntegrationType::Volumetric2D:
         return 9. / 8. * M_PI;
      case IntegrationType::Surface3D:
         return NAN;
      case IntegrationType::Volumetric3D:
         return 3. / 4. * M_PI;
      default:
         return 0.;
   }
}

#ifdef MFEM_USE_LAPACK
/**
 @brief Class for surface IntegrationRule

 This class demonstrates how IntegrationRules computed as CutIntegrationRules
 can be saved to reduce the impact by computing them from scratch each time.
 */
class SIntegrationRule : public IntegrationRule
{
protected:
   /// @brief Space Dimension of the IntegrationRule
   int dim;
   /// @brief Column-wise matrix of the quadtrature weights
   DenseMatrix Weights;
   /// @brief Column-wise matrix of the transformation weights of the normal
   DenseMatrix SurfaceWeights;

public:
   /**
    @brief Constructor of SIntegrationRule

    The surface integrationRules are computed and saved in the constructor.

    @param [in] Order Order of the IntegrationRule
    @param [in] LvlSet Level-set defining the implicit interface
    @param [in] lsOrder Polynomial degree for approx of level-set function
    @param [in] mesh Pointer to the mesh that is used
   */
   SIntegrationRule(int Order, Coefficient& LvlSet, int lsOrder, Mesh* mesh)
   {
      dim = mesh->Dimension();

      IsoparametricTransformation Tr;
      MomentFittingIntRules MFIRs(Order, LvlSet, lsOrder);
      mesh->GetElementTransformation(0, &Tr);
      IntegrationRule ir;
      MFIRs.GetSurfaceIntegrationRule(Tr, ir);
      if (dim >1)
      {
         Weights.SetSize(ir.GetNPoints(), mesh->GetNE());
      }
      else
      {
         Weights.SetSize(2, mesh->GetNE());
      }
      SurfaceWeights.SetSize(ir.GetNPoints(), mesh->GetNE());
      Vector w;
      MFIRs.GetSurfaceWeights(Tr, ir, w);
      SurfaceWeights.SetCol(0, w);
      SetSize(ir.GetNPoints());

      for (int ip = 0; ip < GetNPoints(); ip++)
      {
         IntPoint(ip).index = ip;
         IntegrationPoint &intp = IntPoint(ip);
         intp.x = ir.IntPoint(ip).x;
         intp.y = ir.IntPoint(ip).y;
         intp.z = ir.IntPoint(ip).z;
         if (dim > 1)
         {
            Weights(ip, 0) = ir.IntPoint(ip).weight;
         }
         else
         {
            Weights(0, 0) = ir.IntPoint(ip).x;
            Weights(1, 0) = ir.IntPoint(ip).weight;
         }
      }


      for (int elem = 1; elem < mesh->GetNE(); elem++)
      {
         mesh->GetElementTransformation(elem, &Tr);
         MFIRs.GetSurfaceIntegrationRule(Tr, ir);
         Vector w;
         MFIRs.GetSurfaceWeights(Tr, ir, w);
         SurfaceWeights.SetCol(elem, w);

         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            if (dim > 1)
            {
               Weights(ip, elem) = ir.IntPoint(ip).weight;
            }
            else
            {
               Weights(0, elem) = ir.IntPoint(ip).x;
               Weights(1, elem) = ir.IntPoint(ip).weight;
            }
         }
      }
   }

   /**
    @brief Set the weights for the given element and multiply them with the
    transformation of the interface
    */
   void SetElementinclSurfaceWeight(int Element)
   {
      if (dim == 1)
      {
         IntegrationPoint &intp = IntPoint(0);
         intp.x = Weights(0, Element);
         intp.weight = Weights(1, Element);
         cout << intp.x << " " << Element << endl;
      }
      else
         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            IntegrationPoint &intp = IntPoint(ip);
            intp.weight = Weights(ip, Element) * SurfaceWeights(ip, Element);
         }
   }

   /// @brief Set the weights for the given element
   void SetElement(int Element)
   {
      if (dim == 1)
      {
         IntegrationPoint &intp = IntPoint(0);
         intp.x = Weights(0, Element);
         intp.weight = Weights(1, Element);
      }
      else
         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            IntegrationPoint &intp = IntPoint(ip);
            intp.weight = Weights(ip, Element);
         }
   }

   /// @brief Destructor of SIntegrationRule
   ~SIntegrationRule() {}
};

/**
 @brief Class for volume IntegrationRule

 This class demonstrates how IntegrationRules computed as CutIntegrationRules
 can be saved to reduce the impact by computing them from scratch each time.
 */
class CIntegrationRule : public IntegrationRule
{
protected:
   /// @brief Space Dimension of the IntegrationRule
   int dim;
   /// @brief Column-wise matrix of the quadtrature weights
   DenseMatrix Weights;

public:
   /**
    @brief Constructor of CIntegrationRule

    The volume integrationRules are computed and saved in the constructor.

    @param [in] Order Order of the IntegrationRule
    @param [in] LvlSet Level-set defining the implicit interface
    @param [in] lsOrder Polynomial degree for approx of level-set function
    @param [in] mesh Pointer to the mesh that is used
   */
   CIntegrationRule(int Order, Coefficient& LvlSet, int lsOrder, Mesh* mesh)
   {
      dim = mesh->Dimension();

      IsoparametricTransformation Tr;
      MomentFittingIntRules MFIRs(Order, LvlSet, lsOrder);
      mesh->GetElementTransformation(0, &Tr);
      IntegrationRule ir;
      MFIRs.GetVolumeIntegrationRule(Tr, ir);
      if (dim > 1)
      {
         Weights.SetSize(ir.GetNPoints(), mesh->GetNE());
      }
      else
      {
         Weights.SetSize(2 * ir.GetNPoints(), mesh->GetNE());
      }

      SetSize(ir.GetNPoints());
      for (int ip = 0; ip < GetNPoints(); ip++)
      {
         IntPoint(ip).index = ip;
         IntegrationPoint &intp = IntPoint(ip);
         intp.x = ir.IntPoint(ip).x;
         intp.y = ir.IntPoint(ip).y;
         intp.z = ir.IntPoint(ip).z;
         if (dim > 1)
         {
            Weights(ip, 0) = ir.IntPoint(ip).weight;
         }
         else
         {
            Weights(2 * ip, 0) = ir.IntPoint(ip).x;
            Weights(2 * ip + 1, 0) = ir.IntPoint(ip).weight;
         }
      }

      for (int elem = 1; elem < mesh->GetNE(); elem++)
      {
         mesh->GetElementTransformation(elem, &Tr);
         MFIRs.GetVolumeIntegrationRule(Tr, ir);

         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            if (dim > 1)
            {
               Weights(ip, elem) = ir.IntPoint(ip).weight;
            }
            else
            {
               Weights(2 * ip, elem) = ir.IntPoint(ip).x;
               Weights(2 * ip + 1, elem) = ir.IntPoint(ip).weight;
            }
         }
      }
   }

   /// @brief Set the weights for the given element
   void SetElement(int Element)
   {
      if (dim == 1)
         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            IntegrationPoint &intp = IntPoint(ip);
            intp.x = Weights(2 * ip, Element);
            intp.weight = Weights(2 * ip + 1, Element);
         }
      else
         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            IntegrationPoint &intp = IntPoint(ip);
            intp.weight = Weights(ip, Element);
         }
   }

   /// @brief Destructor of CIntegrationRule
   ~CIntegrationRule() {}
};
/**
 @brief Class for surface linearform integrator

 Integrator to demonstrate the use of the surface integration rule on an
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
      : LinearFormIntegrator(), SIntRule(ir), LevelSet(levelset), Q(q) { }

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

      // Update the surface integration rule for the current element
      SIntRule->SetElementinclSurfaceWeight(Tr.ElementNo);

      for (int ip = 0; ip < SIntRule->GetNPoints(); ip++)
      {
         Tr.SetIntPoint((&(SIntRule->IntPoint(ip))));
         double val = Tr.Weight() * Q.Eval(Tr, SIntRule->IntPoint(ip));
         el.CalcShape(SIntRule->IntPoint(ip), shape);
         add(elvect, SIntRule->IntPoint(ip).weight * val, shape, elvect);
      }
   }
};

/**
 @brief Class for subdomain linearform integrator

 Integrator to demonstrate the use of the subdomain integration rule within
 an area defined by an implicit surface defined by a level-set.
 */
class SubdomainLFIntegrator : public LinearFormIntegrator
{
protected:
   /// @brief vector to evaluate the basis functions
   Vector shape;

   /// @brief surface integration rule
   CIntegrationRule* CIntRule;

   /// @brief coefficient representing the level-set defining the interface
   Coefficient &LevelSet;

   /// @brief coefficient representing the integrand
   Coefficient &Q;

public:
   /**
    @brief Constructor for the volumetric subdomain linear form integrator

    Constructor for the subdomain linear form integrator to demonstrate the use
    of the volumetric subdomain integration rule by means of moment-fitting.

    @param [in] q coefficient representing the inegrand
    @param [in] levelset level-set defining the implicit interfac
    @param [in] ir subdomain integrtion rule to be used
    */
   SubdomainLFIntegrator(Coefficient &q, Coefficient &levelset,
                         CIntegrationRule* ir)
      : LinearFormIntegrator(), CIntRule(ir), LevelSet(levelset), Q(q) { }

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

      // Update the subdomain integration rule
      CIntRule->SetElement(Tr.ElementNo);

      for (int ip = 0; ip < CIntRule->GetNPoints(); ip++)
      {
         Tr.SetIntPoint((&(CIntRule->IntPoint(ip))));
         double val = Tr.Weight()
                      * Q.Eval(Tr, CIntRule->IntPoint(ip));
         el.CalcPhysShape(Tr, shape);
         add(elvect, CIntRule->IntPoint(ip).weight * val, shape, elvect);
      }
   }
};
#endif // MFEM_USE_LAPACK

int main(int argc, char *argv[])
{
#ifndef MFEM_USE_LAPACK
   cout << "MFEM must be build with LAPACK for this example." << endl;
   return EXIT_FAILURE;
#else
   // 1. Parse he command-line options.
   int ref_levels = 3;
   int order = 2;
   const char *inttype = "surface2d";
   bool visualization = true;
   itype = IntegrationType::Surface2D;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Order of quadrature rule");
   args.AddOption(&ref_levels, "-r", "--refine", "Number of meh refinements");
   args.AddOption(&inttype, "-i", "--integrationtype",
                  "IntegrationType to demonstrate");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   if (strcmp(inttype, "volumetric1d") == 0
       || strcmp(inttype, "Volumetric1D") == 0)
   {
      itype = IntegrationType::Volumetric1D;
   }
   else if (strcmp(inttype, "surface2d") == 0
            || strcmp(inttype, "Surface2D") == 0)
   {
      itype = IntegrationType::Surface2D;
   }
   else if (strcmp(inttype, "volumetric2d") == 0
            || strcmp(inttype, "Volumetric2D") == 0)
   {
      itype = IntegrationType::Volumetric2D;
   }
   else if (strcmp(inttype, "surface3d") == 0
            || strcmp(inttype, "Surface3d") == 0)
   {
      itype = IntegrationType::Surface3D;
   }
   else if (strcmp(inttype, "volumetric3d") == 0
            || strcmp(inttype, "Volumetric3d") == 0)
   {
      itype = IntegrationType::Volumetric3D;
   }

   // 2. Construct and refine the mesh.
   Mesh *mesh;
   if (itype == IntegrationType::Volumetric1D)
   {
      mesh = new Mesh("../data/inline-segment.mesh");
   }
   if (itype == IntegrationType::Surface2D
       || itype == IntegrationType::Volumetric2D)
   {
      mesh = new Mesh(2, 4, 1, 0, 2);
      mesh->AddVertex(-1.6,-1.6);
      mesh->AddVertex(1.6,-1.6);
      mesh->AddVertex(1.6,1.6);
      mesh->AddVertex(-1.6,1.6);
      mesh->AddQuad(0,1,2,3);
      mesh->FinalizeQuadMesh(1, 0, 1);
   }
   else if (itype == IntegrationType::Surface3D
            || itype == IntegrationType::Volumetric3D)
   {
      mesh = new Mesh(3, 8, 1, 0, 3);
      mesh->AddVertex(-1.6,-1.6,-1.6);
      mesh->AddVertex(1.6,-1.6,-1.6);
      mesh->AddVertex(1.6,1.6,-1.6);
      mesh->AddVertex(-1.6,1.6,-1.6);
      mesh->AddVertex(-1.6,-1.6,1.6);
      mesh->AddVertex(1.6,-1.6,1.6);
      mesh->AddVertex(1.6,1.6,1.6);
      mesh->AddVertex(-1.6,1.6,1.6);
      mesh->AddHex(0,1,2,3,4,5,6,7);
      mesh->FinalizeHexMesh(1, 0, 1);
   }

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 3. Define the necessary finite element space on the mesh.
   H1_FECollection fe_coll(1, mesh->Dimension());
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, &fe_coll);

   // 4. Construction Coefficients for the level set and the integrand.
   FunctionCoefficient levelset(lvlset);
   FunctionCoefficient u(integrand);

   // 5. Define the necessary Integration rules on element 0.
   IsoparametricTransformation Tr;
   mesh->GetElementTransformation(0, &Tr);
   SIntegrationRule* sir = new SIntegrationRule(order, levelset, 2, mesh);
   CIntegrationRule* cir = NULL;
   if (itype == IntegrationType::Volumetric1D
       || itype == IntegrationType::Volumetric2D
       || itype == IntegrationType::Volumetric3D)
   {
      cir = new CIntegrationRule(order, levelset, 2, mesh);
   }

   // 6. Define and assemble the linear forms on the finite element space.
   LinearForm surface(fespace);
   LinearForm volume(fespace);

   surface.AddDomainIntegrator(new SurfaceLFIntegrator(u, levelset, sir));
   surface.Assemble();

   if (itype == IntegrationType::Volumetric1D
       || itype == IntegrationType::Volumetric2D
       || itype == IntegrationType::Volumetric3D)
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
   if (itype != IntegrationType::Volumetric1D)
   {
      cout << 3.2 / pow(2., (double)ref_levels) << endl;
   }
   else
   {
      cout << .25 / pow(2., (double)ref_levels) << endl;
   }
   if (itype == IntegrationType::Surface2D
       || itype == IntegrationType::Volumetric2D)
   {
      cout << "Number of div free basis functions: " << nbasis << endl;
      cout << "Number of quadrature points:        " << ir.GetNPoints() << endl;
   }
   cout << scientific << setprecision(2);
   cout << "============================================" << endl;
   cout << "Computed value of surface integral: " << surface.Sum() << endl;
   cout << "True value of surface integral:     " << Surface() << endl;
   cout << "Absolut Error (Surface):            ";
   cout << abs(surface.Sum() - Surface()) << endl;
   cout << "Relative Error (Surface):           ";
   cout << abs(surface.Sum() - Surface()) / Surface() << endl;
   if (itype == IntegrationType::Volumetric1D
       || itype == IntegrationType::Volumetric2D
       || itype == IntegrationType::Volumetric3D)
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
   if (visualization)
   {
      H1_FECollection fe_coll2(5, mesh->Dimension());
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
   }

   delete sir;
   delete cir;
   delete fespace;
   delete mesh;
   return EXIT_SUCCESS;
#endif //MFEM_USE_LAPACK
}
