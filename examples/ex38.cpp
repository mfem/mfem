//                                MFEM Example 38
//
// Compile with: make ex38
//
// Sample runs:
// (since all sample runs require LAPACK or ALGOIM, the * symbol is used to
//  exclude them from the automatically generated internal MFEM tests).
//              * ex38
//              * ex38 -i volumetric1d
//              * ex38 -i surface2d
//              * ex38 -i surface2d -o 4 -r 5 -m 1
//              * ex38 -i volumetric2d
//              * ex38 -i volumetric2d -o 4 -r 5 -m 1
//              * ex38 -i surface3d
//              * ex38 -i surface3d -o 3 -r 4 -m 1
//              * ex38 -i volumetric3d
//              * ex38 -i volumetric3d -o 3 -r 4 -m 1
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
real_t lvlset(const Vector& X)
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
real_t integrand(const Vector& X)
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
real_t Surface()
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
real_t Volume()
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

/**
 @brief Class for surface IntegrationRule

 This class demonstrates how IntegrationRules computed as CutIntegrationRules
 can be saved to reduce the impact by computing them from scratch each time.
 */
class SIntegrationRule : public IntegrationRule
{
protected:
   /// method 0 is moments-based, 1 is Algoim.
   int method, ir_order, ls_order;
   Coefficient &level_set;
   /// Space Dimension of the IntegrationRule
   int dim;
   /// Column-wise matrix of the quadtrature weights
   DenseMatrix Weights;
   /// Column-wise matrix of the transformation weights of the normal
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
   SIntegrationRule(int method_, int Order,
                    Coefficient& LvlSet, int lsOrder, Mesh* mesh)
      : method(method_), ir_order(Order), ls_order(lsOrder),
        level_set(LvlSet), dim(mesh->Dimension())
   {
      // Nothing gets pre-computed for Algoim.
      if (method == 1) { return; }

#ifdef MFEM_USE_LAPACK
      MomentFittingIntRules mf_ir(ir_order, level_set, ls_order);

      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(0, &Tr);
      IntegrationRule ir;
      mf_ir.GetSurfaceIntegrationRule(Tr, ir);
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
      mf_ir.GetSurfaceWeights(Tr, ir, w);
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
         mf_ir.GetSurfaceIntegrationRule(Tr, ir);
         mf_ir.GetSurfaceWeights(Tr, ir, w);
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
#else
      MFEM_ABORT("Moment-fitting requires MFEM to be built with LAPACK!");
#endif
   }

   /**
    @brief Set the weights for the given element and multiply them with the
    transformation of the interface
    */
   void SetElementAndSurfaceWeight(ElementTransformation &Tr)
   {
      if (method == 1)
      {
#ifdef MFEM_USE_ALGOIM
         AlgoimIntegrationRules a_ir(ir_order, level_set, ls_order);
         a_ir.GetSurfaceIntegrationRule(Tr, *this);
         Vector w;
         a_ir.GetSurfaceWeights(Tr, *this, w);
         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            IntPoint(ip).weight *= w(ip);
         }
         return;
#else
         MFEM_ABORT("MFEM is not built with Algoim support!");
#endif
      }

      if (dim == 1)
      {
         IntPoint(0).x = Weights(0, Tr.ElementNo);
         IntPoint(0).weight = Weights(1, Tr.ElementNo);
      }
      else
      {
         for (int ip = 0; ip < GetNPoints(); ip++)
         {
            IntPoint(ip).weight = Weights(ip, Tr.ElementNo) *
                                  SurfaceWeights(ip, Tr.ElementNo);
         }
      }
   }
};

/**
 @brief Class for volume IntegrationRule

 This class demonstrates how IntegrationRules computed as CutIntegrationRules
 can be saved to reduce the impact by computing them from scratch each time.
 */
class CIntegrationRule : public IntegrationRule
{
protected:
   /// method 0 is moments-based, 1 is Algoim.
   int method, ir_order, ls_order;
   Coefficient &level_set;
   /// Space Dimension of the IntegrationRule
   int dim;
   /// Column-wise matrix of the quadtrature positions and weights.
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
   CIntegrationRule(int method_, int Order,
                    Coefficient &LvlSet, int lsOrder, Mesh *mesh)
      : method(method_), ir_order(Order), ls_order(lsOrder),
        level_set(LvlSet), dim(mesh->Dimension())
   {
      // Nothing gets pre-computed for Algoim.
      if (method == 1) { return; }

#ifdef MFEM_USE_LAPACK
      MomentFittingIntRules mf_ir(ir_order, level_set, ls_order);

      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(0, &Tr);
      IntegrationRule ir;
      mf_ir.GetVolumeIntegrationRule(Tr, ir);
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
         mf_ir.GetVolumeIntegrationRule(Tr, ir);

         for (int ip = 0; ip < ir.GetNPoints(); ip++)
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
#else
      MFEM_ABORT("Moment-fitting requires MFEM to be built with LAPACK!");
#endif
   }

   /// @brief Set the weights for the given element
   void SetElement(ElementTransformation &Tr)
   {
      if (method == 1)
      {
#ifdef MFEM_USE_ALGOIM
         AlgoimIntegrationRules a_ir(ir_order, level_set, ls_order);
         a_ir.GetVolumeIntegrationRule(Tr, *this);
         return;
#else
         MFEM_ABORT("MFEM is not built with Algoim support!");
#endif
      }

      for (int ip = 0; ip < GetNPoints(); ip++)
      {
         IntegrationPoint &intp = IntPoint(ip);
         if (dim == 1)
         {
            intp.x = Weights(2 * ip, Tr.ElementNo);
            intp.weight = Weights(2 * ip + 1, Tr.ElementNo);
         }
         else { intp.weight = Weights(ip, Tr.ElementNo); }
      }
   }
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
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.;

      // Update the surface integration rule for the current element
      SIntRule->SetElementAndSurfaceWeight(Tr);

      for (int ip = 0; ip < SIntRule->GetNPoints(); ip++)
      {
         Tr.SetIntPoint((&(SIntRule->IntPoint(ip))));
         real_t val = Tr.Weight() * Q.Eval(Tr, SIntRule->IntPoint(ip));
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
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.;

      // Update the subdomain integration rule
      CIntRule->SetElement(Tr);

      for (int ip = 0; ip < CIntRule->GetNPoints(); ip++)
      {
         Tr.SetIntPoint((&(CIntRule->IntPoint(ip))));
         real_t val = Tr.Weight()
                      * Q.Eval(Tr, CIntRule->IntPoint(ip));
         el.CalcPhysShape(Tr, shape);
         add(elvect, CIntRule->IntPoint(ip).weight * val, shape, elvect);
      }
   }
};

int main(int argc, char *argv[])
{
#if defined(MFEM_USE_LAPACK) || defined(MFEM_USE_ALGOIM)
   // 1. Parse he command-line options.
   int ref_levels = 3;
   int order = 2;
   int method = 0;
   const char *inttype = "surface2d";
   bool visualization = true;
   itype = IntegrationType::Surface2D;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Order of quadrature rule");
   args.AddOption(&ref_levels, "-r", "--refine", "Number of meh refinements");
   args.AddOption(&method, "-m", "--method",
                  "Cut integration method: 0 for moments-based, 1 for Algoim.");
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
   Mesh *mesh = nullptr;
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
   SIntegrationRule* sir = new SIntegrationRule(method, order,
                                                levelset, 2, mesh);
   CIntegrationRule* cir = NULL;
   if (itype == IntegrationType::Volumetric1D
       || itype == IntegrationType::Volumetric2D
       || itype == IntegrationType::Volumetric3D)
   {
      cir = new CIntegrationRule(method, order, levelset, 2, mesh);
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
      cout << 3.2 / pow(2., (real_t)ref_levels) << endl;
   }
   else
   {
      cout << .25 / pow(2., (real_t)ref_levels) << endl;
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
   cout << "Absolute Error (Surface):           ";
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
      cout << "Absolute Error (Volume):            ";
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
#else
   cout << "MFEM must be built with LAPACK or ALGOIM for this example." << endl;
   return MFEM_SKIP_RETURN_VALUE;
#endif // MFEM_USE_LAPACK
}
