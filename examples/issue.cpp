
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


/** Mass integrator (u⋅d, v⋅d) restricted to the boundary of a domain */
class VectorBoundaryDirectionalMassIntegrator: public BilinearFormIntegrator
{
private:
   VectorCoefficient &direction;
   int vdim;
   int oa, ob;
   const double k;

public:
   /// Construct an integrator with coefficient 1.0
   VectorBoundaryDirectionalMassIntegrator(const double k,
                                           VectorCoefficient &direction,
                                           const int oa=1, const int ob=1)
      : k(k), vdim(direction.GetVDim()), direction(direction),
        oa(oa), ob(ob) { }

   using BilinearFormIntegrator::AssembleElementMatrix;
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat)
   {
      int dof = el.GetDof();
      Vector shape(dof), vec(vdim);

      out << Tr.Attribute - 1 << " " << dof << " LHSElement" << std::endl;
      elmat.SetSize(dof*vdim);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.GetGeometryType(), intorder); // of integration order
      }

      DenseMatrix elmat_scalar(dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetIntPoint(&ip);

         // Access the neighboring element's integration point
         direction.Eval(vec, Tr, ip);
         double val = k*Tr.Weight() * ip.weight;

         el.CalcShape(ip, shape);
         MultVVt(shape, elmat_scalar);
         for (int row = 0; row < vdim; row++)
         {
            for (int col = 0; col < vdim; col++)
            {
               elmat.AddMatrix(val*vec(row)*vec(col), elmat_scalar, dof*row, dof*col);
            }
         }
      }
   }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el,
                                   const FiniteElement &dummy,
                                   FaceElementTransformations &Tr,
                                   DenseMatrix &elmat)
   {
      int dof = el.GetDof();
      Vector shape(dof), vec(vdim);

      out << Tr.Attribute - 1 << " " << dof << " LHSFace" << std::endl;

      elmat.SetSize(dof*vdim);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
      }

      DenseMatrix elmat_scalar(dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetAllIntPoints(&ip);

         // Access the neighboring element's integration point
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         direction.Eval(vec, *Tr.Face, ip);
         double val = k*Tr.Face->Weight() * ip.weight;

         el.CalcShape(eip, shape);

         for (int row = 0; row < vdim; row++)
         {
            for (int col = 0; col < vdim; col++)
            {
               elmat.AddMatrix(val*vec(row)*vec(col), elmat_scalar, dof*row, dof*col);
            }
         }
      }
   }
};
/** Mass integrator (u⋅n, v⋅n) restricted to the boundary of a domain */
class VectorBoundaryDirectionalLFIntegrator : public LinearFormIntegrator
{
   VectorCoefficient &direction, &force;
   int oa, ob, vdim;
public:
   /** @brief Constructs a boundary integrator with a given Coefficient @a QG.
       Integration order will be @a a * basis_order + @a b. */
   VectorBoundaryDirectionalLFIntegrator(VectorCoefficient &direction,
                                         VectorCoefficient &force,
                                         int a = 1, int b = 1)
      : direction(direction), force(force), oa(a), ob(b), vdim(direction.GetVDim()) { }

   /** Given a particular boundary Finite Element and a transformation (Tr)
       computes the element boundary vector, elvect. */
   using LinearFormIntegrator::AssembleRHSElementVect;
   virtual void AssembleRHSElementVect(
      const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
   {
      int dof = el.GetDof();

      out << Tr.Attribute - 1 << " " << dof << " RHSElement" << std::endl;

      Vector shape(dof), vec(vdim), vecF(vdim);
      elvect.SetSize(dof*vdim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.GetGeometryType(), intorder); // of integration order
      }
      double * data = elvect.GetData();
      Vector elvect_loc(data, dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         direction.Eval(vec, Tr, ip);
         force.Eval(vecF, Tr, ip);
         double val = Tr.Weight() * ip.weight * (vec * vecF);

         el.CalcShape(ip, shape);
         for (int row = 0; row < vdim; row++)
         {
            elvect_loc.SetData(data + dof*row);
            elvect_loc.Add(val*vec(row), shape);
         }
      }
   }
   virtual void AssembleRHSElementVect(
      const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
   {
      int dof = el.GetDof();

      out << Tr.Attribute - 1 << " " << dof << " RHSFace" << std::endl;

      Vector shape(dof), vec(vdim), vecF(vdim);
      elvect.SetSize(dof*vdim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
      }
      double * data = elvect.GetData();
      Vector elvect_loc(data, dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetAllIntPoints(&ip);

         // Access the neighboring element's integration point
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         direction.Eval(vec, Tr, ip);
         force.Eval(vecF, Tr, ip);
         double val = Tr.Face->Weight() * ip.weight * (vec * vecF);

         el.CalcShape(eip, shape);
         for (int row = 0; row < vdim; row++)
         {
            elvect_loc.SetData(data + dof*row);
            elvect_loc.Add(val*vec(row), shape);
         }
      }
   }
};

enum BdrType
{
   Fixed,
   XRoller,
   YRoller,
   ZRoller,
   Input,
   Output,
   Free,
   NumBdr
};

int main(int argc, char *argv[])
{
   int p=1;
   int nel = 40;
   int numelx = nel*2;
   int numely = nel;
   const double len = nel*0.025; // fixed, input, output boundary length
   // Setup spring
   double input_spring = 1;
   double output_spring = 0.0001;
   Vector input_direction(2), output_direction(2);
   input_direction = 0.0; output_direction = 0.0;
   input_direction[0] = 1.0;
   output_direction[0] = -1.0;


   // Mesh
   Mesh mesh = mesh.MakeCartesian2D(numelx, numely,
                                    mfem::Element::Type::QUADRILATERAL,
                                    true,
                                    (double)numelx, (double)numely);
   // Setup boundary
   //
   //          ooooooooooooooooooooooo <- x roller (Y fixed)
   // Input -> II                   II <- Output
   //          |                     |
   //          |                     |
   // Fixed -> II--------------------|
   // 
   // Otherwise, free.
   Array2D<int> ess_bdr(mesh.SpaceDimension(), BdrType::NumBdr); // [X-fixed; Y-fixed; All-fixed]
   ess_bdr = 0;
   ess_bdr(0, BdrType::YRoller) = 1; // y-roller - x direction fixed
   ess_bdr(1, BdrType::XRoller) = 1; // x-roller - y direction fixed
   ess_bdr(2, BdrType::Fixed) = 1; // all direction fixed
   Array<int> input_bdr(BdrType::NumBdr), output_bdr(BdrType::NumBdr);
   input_bdr = 0; output_bdr = 0;
   input_bdr[BdrType::Input] = 1; output_bdr[BdrType::Output] = 1;
   // To ensure that there are input/output boundaries
   int nrInputBdrFace = 0;
   int nrOutputBdrFace = 0;
   // Set boundary attributes   
   for (int i = 0; i<mesh.GetNBE(); i++)
   {
      Element * be = mesh.GetBdrElement(i);
      Array<int> vertices;
      be->GetVertices(vertices);

      double * coords1 = mesh.GetVertex(vertices[0]);
      double * coords2 = mesh.GetVertex(vertices[1]);

      Vector fc(2);
      fc(0) = 0.5*(coords1[0] + coords2[0]);
      fc(1) = 0.5*(coords1[1] + coords2[1]);

      switch (be->GetAttribute())
      {
         case 1: // bottom
            be->SetAttribute(BdrType::Free + 1);
            break;
         case 2: // right
            if (fc(1) > numely - len)
            {
               be->SetAttribute(BdrType::Output + 1);
               nrOutputBdrFace++;
               break;
            }
            be->SetAttribute(BdrType::Free + 1);
            break;
         case 3: // top
            be->SetAttribute(BdrType::XRoller + 1);
            break;
         case 4: // left
            if (fc(1) > numely - len)
            {
               be->SetAttribute(BdrType::Input + 1);
               nrInputBdrFace++;
               break;
            }
            else if (fc(1) < len)
            {
               be->SetAttribute(BdrType::Fixed + 1);
               break;
            }
            be->SetAttribute(BdrType::Free + 1);
            break;
         default:
            mfem_error("Something went wrong");
      }
   }

   out << "(# Input, # Output) = (" << nrInputBdrFace << ", " << nrOutputBdrFace << ")" << std::endl;

   

   H1_FECollection fec(p);
   FiniteElementSpace fes(&mesh, &fec, mesh.SpaceDimension(), Ordering::byNODES);
   VectorConstantCoefficient output_d_cf(output_direction), input_d_cf(input_direction);
   for(int i=0; i<10; i++)
   {
      LinearForm b(&fes);
      // Expected output for each iteration:
      // BdrType::Input (p+1)*dim RHSFace
      // BdrType::Output (p+1)*dim RHSFace
      // 
      // When p = 1 and dim = 2,
      // 4 4 RHSFace
      // 5 4 RHSFace
      // 
      out << i << std::endl;
      b.AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(output_d_cf, output_d_cf), output_bdr);
      b.AddBdrFaceIntegrator(new VectorBoundaryDirectionalLFIntegrator(input_d_cf, input_d_cf), input_bdr);
      b.Assemble();
      out << std::endl;
   }
}
