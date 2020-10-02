#include "mfem.hpp"

using namespace std;
using namespace mfem;

bool maxtau=true;   //take a maximum on each element
double vA = 1.0;    
double ALPHA = 0.1; //the parameter in stabilization B terms
bool dtfloor=false;
double dtmin = 0.025;
double dtfactor=10.;    //a scale factor in tau of dt/dtfactor

// Integrator for the boundary gradient integral from the Laplacian operator
// this is used in the auxiliary variable where the boundary condition is not needed
class BoundaryGradIntegrator: public BilinearFormIntegrator
{
private:
   Vector shape1, dshape_dn, nor;
   DenseMatrix dshape, dshapedxt, invdfdx;

public:
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

void BoundaryGradIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int i, j, ndof1;
   int dim, order;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   // set to this for now, integration includes rational terms
   order = 2*el1.GetOrder() + 1;

   nor.SetSize(dim);
   shape1.SetSize(ndof1);
   dshape_dn.SetSize(ndof1);
   dshape.SetSize(ndof1,dim);
   dshapedxt.SetSize(ndof1,dim);
   invdfdx.SetSize(dim);

   elmat.SetSize(ndof1);
   elmat = 0.0;

   const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1;
      Trans.Loc1.Transform(ip, eip1);
      el1.CalcShape(eip1, shape1);
      //d of shape function, evaluated at eip1
      el1.CalcDShape(eip1, dshape);

      Trans.Elem1->SetIntPoint(&eip1);

      CalcInverse(Trans.Elem1->Jacobian(), invdfdx); //inverse Jacobian
      //invdfdx.Transpose();
      Mult(dshape, invdfdx, dshapedxt); // dshapedxt = grad phi* J^-1

      //get normal vector
      Trans.Face->SetIntPoint(&ip);
      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      /* this is probably the old way
      const DenseMatrix &J = Trans.Face->Jacobian(); //is this J^{-1} or J^{T}?
      else if (dim == 2)
      {
         nor(0) =  J(1,0);
         nor(1) = -J(0,0);
      }
      else if (dim == 3)
      {
         nor(0) = J(1,0)*J(2,1) - J(2,0)*J(1,1);
         nor(1) = J(2,0)*J(0,1) - J(0,0)*J(2,1);
         nor(2) = J(0,0)*J(1,1) - J(1,0)*J(0,1);
      }
      */

      // multiply weight into normal, make answer negative
      // (boundary integral is subtracted)
      nor *= -ip.weight;

      dshapedxt.Mult(nor, dshape_dn);

      for (i = 0; i < ndof1; i++)
         for (j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i)*dshape_dn(j);
         }
   }
}

//this is a test integrator (from diffusion integrator)
class TestIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, gshape, Jinv;
   Coefficient *nuCoef;

public:
   TestIntegrator (double visc)  
   { nuCoef = new ConstantCoefficient(visc); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat);

   virtual ~TestIntegrator()
   { delete nuCoef; }
};

void TestIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        DenseMatrix &elmat)
{
    int dim = 2;
    int nd = el.GetDof();

    dshape.SetSize(nd, dim);
    gshape.SetSize(nd, dim);
    Jinv  .SetSize(dim);


    elmat.SetSize(nd);
    elmat=0.;
    
    double w;
    const IntegrationRule *ir = IntRule ? IntRule : &DiffusionIntegrator::GetRule(el, el);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Tr.SetIntPoint(&ip);
      w = Tr.Weight();
      w = ip.weight / w;
      Mult(dshape, Tr.AdjugateJacobian(), gshape);
            
      w *= nuCoef->Eval(Tr, ip);;
      AddMult_a_AAt(w, gshape, elmat);
    }
}

// Integrator for (tau * Q.grad func , grad v) where Q=[v1**2, 0; 0, v2**2]
class SpecialConvectionIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, gshape, Jinv, V_ir;
   Coefficient *nuCoef;
   MyCoefficient *V; 
   double dt;
   int itau;

public:
   SpecialConvectionIntegrator (double dt_, double visc, MyCoefficient &q, int itau_=2) : 
       V(&q), dt(dt_), itau(itau_)
   { nuCoef = new ConstantCoefficient(visc); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat);

   virtual ~SpecialConvectionIntegrator()
   { delete nuCoef; }
};

void SpecialConvectionIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        DenseMatrix &elmat)
{
    double norm, tau;
    double Unorm, invtau;
    int dim = 2;
    int nd = el.GetDof();
    Vector advGrad(nd), advGrad2(nd), vec1(dim), vec2(dim);
    DenseMatrix invdfdx(dim,dim);

    dshape.SetSize(nd, dim);
    gshape.SetSize(nd, dim);
    Jinv  .SetSize(dim);

    elmat.SetSize(nd);
    elmat=0.;
  
    //here we assume 2d quad
    double eleLength = sqrt( Geometry::Volume[el.GetGeomType()] * Tr.Weight() );   
    //integration order is el.order + grad.order-1 (-1 due to another derivative taken in V)
    int intorder = 2 * (el.GetOrder() + Tr.OrderGrad(&el)-1);
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

    V->Eval(V_ir, Tr, ir);

    //compare maximum tau
    double tauMax=0.0;

    /*
    if (maxtau)
    {
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            double nu = nuCoef->Eval (Tr, ip);

            V_ir.GetColumnReference(i, vec1);
            Unorm = vec1.Norml2();
            if (itau==1)
            {
                invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==2)
            {
                invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==3)
            {
                invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==4)
            {
                invtau = 2./dt;
            }
            else
                invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            tau = 1.0/invtau;

            tauMax = max(tauMax, tau);
        }
    }
    invdfdx=0.;
    */

    tauMax = dt/2.;

    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        el.CalcDShape(ip, dshape);
        
        Tr.SetIntPoint(&ip);
        norm = ip.weight / Tr.Weight();
        Mult(dshape, Tr.AdjugateJacobian(), gshape);
        
        V_ir.GetColumnReference(i, vec1);
        Unorm = vec1.Norml2();

        norm *= tauMax*Unorm*Unorm;

        AddMult_a_AAt(norm, gshape, elmat);

        /*
        invdfdx(0,0)=vec1(0)*vec1(0)+vec1(1)*vec1(1);
        invdfdx(1,1)=invdfdx(0,0);
        //invdfdx(1,1)=vec1(1)*vec1(1);
        invdfdx *= norm;

        Mult(gshape, invdfdx, dshape);
        AddMultABt(dshape, gshape, elmat);
        */
    }
}

// Integrator for (tau * Q.grad func , V.grad v)
// Here we always assume V is the advection speed
// It also supports V as the magnetic field
class StabConvectionIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, gshape, Jinv, V_ir, Q_ir;
   Coefficient *nuCoef;
   MyCoefficient *V, *Q; 
   double dt;
   bool FieldDiff; //field line diffusion along magnetic field
   int itau;
   double alpha;

public:
   StabConvectionIntegrator (double dt_, double visc, MyCoefficient &q, int itau_=2, double alpha_=0.) : 
       V(&q), Q(NULL), dt(dt_), FieldDiff(false), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc); }

   StabConvectionIntegrator (double dt_, double visc, MyCoefficient &q, MyCoefficient &v, int itau_=2, double alpha_=0.) : 
       V(&v), Q(&q), dt(dt_), FieldDiff(false), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc); }

   //Field line diffusion
   StabConvectionIntegrator (double dt_, double visc, MyCoefficient &q, bool FieldDiff_, int itau_=2, double alpha_=0.) : 
       V(&q), Q(NULL), dt(dt_), FieldDiff(FieldDiff_), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc); 
     if ((!maxtau) && FieldDiff) MFEM_ABORT("Invalid choice in FDFEM."); }

   //Field line diffusion
   StabConvectionIntegrator (double dt_, double visc, MyCoefficient &q, MyCoefficient &v, bool FieldDiff_, int itau_=2, double alpha_=0.) : 
       V(&v), Q(&q), dt(dt_), FieldDiff(FieldDiff_), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc);
     if ((!maxtau) && FieldDiff) MFEM_ABORT("Invalid choice in FDFEM."); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat);

   virtual ~StabConvectionIntegrator()
   { delete nuCoef; }
};

void StabConvectionIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        DenseMatrix &elmat)
{
    double norm, tau;
    double Unorm, invtau;
    int dim = 2;
    int nd = el.GetDof();
    Vector advGrad(nd), advGrad2(nd), vec1(dim), vec2(dim);

    dshape.SetSize(nd, dim);
    gshape.SetSize(nd, dim);
    Jinv  .SetSize(dim);

    elmat.SetSize(nd);
    elmat=0.;
  
    //here we assume 2d quad
    double eleLength = sqrt( Geometry::Volume[el.GetGeomType()] * Tr.Weight() );   
    //integration order is el.order + grad.order-1 (-1 due to another derivative taken in V)
    int intorder = 2 * (el.GetOrder() + Tr.OrderGrad(&el)-1);
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

    V->Eval(V_ir, Tr, ir);
    if(Q!=NULL) Q->Eval(Q_ir, Tr, ir);

    if (dtfloor && dt<dtmin)
        dt=dtmin;

    //compare maximum tau
    double tauMax=0.0;
    if (maxtau)
    {
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            double nu = nuCoef->Eval (Tr, ip);

            if (FieldDiff){
               invtau = sqrt( pow(2./dt,2) + pow(2.0*vA/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2) );
               tau = ALPHA*eleLength*eleLength/invtau;
               //tau = 1e-4/invtau;
            }
            else{
               V_ir.GetColumnReference(i, vec1);
               Unorm = vec1.Norml2();
               if (itau==1)
               {
                   invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
               }
               else if (itau==2)
               {
                   invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
               }
               else if (itau==3)
               {
                   if (alpha>0.)
                      invtau = sqrt( pow(alpha/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                   else
                      invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
               }
               else if (itau==4)
               {
                   invtau = 2./dt;
               }
               else
                   invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
               tau = 1.0/invtau;
            }
            tauMax = max(tauMax, tau);
        }
    }
    //if (!FieldDiff)
    //    cout <<"tauMax="<<tauMax<<" length="<<eleLength<<" ";

    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        el.CalcDShape(ip, dshape);
        
        Tr.SetIntPoint(&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        Mult(dshape, Jinv, gshape); 
        
        V_ir.GetColumnReference(i, vec1);

        if (maxtau)
            norm *= tauMax;
        else
        {
            //compute tau
            double nu = nuCoef->Eval (Tr, ip);
            Unorm = vec1.Norml2();
            if (itau==1)
            {
                invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                //invtau = 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            }
            else if (itau==2)
            {
                //fix dt=0.1 so that it is comparable to an implicit scheme
                //invtau = sqrt( pow(2./.1,2) +  pow(2.0 * Unorm / eleLength, 2) );
                invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==3)
            {
                if (alpha>0.)
                    invtau = sqrt( pow(alpha/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                else
                    invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==4)
            {
                   invtau = 2./dt;
            }
            else
                invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            //cout <<"invtau ="<<invtau<<" length="<<eleLength<<" ";
            tau = 1.0/invtau;

            norm *= tau;
        }

        gshape.Mult(vec1, advGrad);
        if (Q==NULL) 
        {
            AddMult_a_VVt(norm, advGrad, elmat);
        }
        else{
            Q_ir.GetColumnReference(i, vec2);
            gshape.Mult(vec2, advGrad2);
            AddMult_a_VWt(norm, advGrad, advGrad2, elmat);
        }
    }
}

// Integrator for (tau * func , V.grad v)
class StabMassIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape;
   DenseMatrix dshape, gshape, Jinv, V_ir;
   Coefficient *nuCoef;
   MyCoefficient *V; 
   double dt;
   bool FieldDiff; //field line diffusion along magnetic field
   int itau;
   double alpha;

public:
   StabMassIntegrator (double dt_, double visc, MyCoefficient &q, int itau_=2, double alpha_=0.) : 
       V(&q), dt(dt_), FieldDiff(false), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc); }

   StabMassIntegrator (double dt_, double visc, MyCoefficient &q, bool FieldDiff_, int itau_=2, double alpha_=0.) : 
       V(&q), dt(dt_), FieldDiff(FieldDiff_), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      DenseMatrix &elmat);

   virtual ~StabMassIntegrator()
   { delete nuCoef; }
};

void StabMassIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        DenseMatrix &elmat)
{
    double norm, tau;
    double Unorm, invtau;
    int dim = 2;
    int nd = el.GetDof();
    Vector advGrad(nd), vec1(dim);

    shape.SetSize(nd);
    dshape.SetSize(nd, dim);
    gshape.SetSize(nd, dim);
    Jinv  .SetSize(dim);

    if (dtfloor && dt<dtmin)
        dt=dtmin;

    //here we assume 2d quad
    double eleLength = sqrt( Geometry::Volume[el.GetGeomType()] * Tr.Weight() );   

    elmat.SetSize(nd);
    elmat=0.;
    
    //this is from ConvectionIntegrator, maybe too high?
    int intorder = el.GetOrder() + Tr.OrderGrad(&el) + Tr.Order();
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

    V->Eval(V_ir, Tr, ir);
    //compare maximum tau
    double tauMax=0.0;
    double nu=0.;
    if (maxtau)
    {
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            nu = nuCoef->Eval (Tr, ip);

            if (FieldDiff){
               invtau = sqrt( pow(2./dt,2) + pow(2.0*vA/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2) );
               tau = ALPHA*eleLength*eleLength/invtau;
               //tau = 1e-4/invtau;
            }
            else{
               V_ir.GetColumnReference(i, vec1);
               Unorm = vec1.Norml2();
               if (itau==1)
               {
                   invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                   //invtau = 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
               }
               else if (itau==2)
                   invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
               else if (itau==3)
               {
                   if (alpha>0.)
                       invtau = sqrt( pow(alpha/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                   else
                       invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
               }
               else if (itau==4)
               {
                   invtau = 2./dt;
               }
               else
                   invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
               tau = 1.0/invtau;
            }
            tauMax = max(tauMax, tau);
        }
    }
    //cout <<"tauMax ="<<tauMax<<" length="<<eleLength<<" h^2/nu="<<(eleLength*eleLength)/nu<<" dt="<<dt<<" ";
  
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcDShape(ip, dshape);
        el.CalcShape(ip, shape);
        
        Tr.SetIntPoint(&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        Mult(dshape, Jinv, gshape); 
        
        V_ir.GetColumnReference(i, vec1);

        if (maxtau)
            norm*=tauMax;
        else
        {
            //compute tau
            double nu = nuCoef->Eval (Tr, ip);
            Unorm = vec1.Norml2();
            if (itau==1)
            {
                invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                //invtau = 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            }
            else if (itau==2)
                invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            else if (itau==3)
            {
                if (alpha>0.)
                    invtau = sqrt( pow(alpha/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                else
                    invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==4)
            {
                invtau = 2./dt;
            }
            else
                invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            tau = 1.0/invtau;
            norm *= tau;
            //cout <<"tau="<<tau<<" Unorm="<<Unorm<<" "<<" h="<<eleLength<<" ";
        }
 
        gshape.Mult(vec1, advGrad);
        
        AddMult_a_VWt(norm, advGrad, shape, elmat);
    }
}

// Integrator to check Tau values
class CheckTauIntegrator : public LinearFormIntegrator
{
private:
   DenseMatrix V_ir;
   MyCoefficient *V; 
   Coefficient *nuCoef;
   double dt;
   int itau;

public:
   CheckTauIntegrator (double dt_, double visc, MyCoefficient &q, int itau_=2) : 
       V(&q),  dt(dt_), itau(itau_)
   { nuCoef = new ConstantCoefficient(visc); }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

   virtual ~CheckTauIntegrator()
   { delete nuCoef; }
};

void CheckTauIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
{
    double norm, tau;
    double Unorm, invtau;
    int dim = 2;
    int nd = el.GetDof();
    Vector vec1(dim);

    //here we assume 2d quad
    double eleLength = sqrt( Geometry::Volume[el.GetGeomType()] * Tr.Weight() );   

    //here we only supports the max tau, so it needs to be DG of order 0
    if(nd>1) 
    {
        cout <<"nd = "<<nd<<endl;
        MFEM_ABORT("Error in checktauintegrator: only support DG of order=0"); 
    }

    elvect.SetSize(nd);
    elvect=0.;
    
    int intorder = 2*el.GetOrder() + Tr.OrderGrad(&el)-1;
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

    V->Eval(V_ir, Tr, ir);

    if (dtfloor && dt<dtmin)
        dt=dtmin;

    //compare maximum tau
    double tauMax=0.0;
    if (maxtau)
    {
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            double nu = nuCoef->Eval (Tr, ip);

            V_ir.GetColumnReference(i, vec1);
            Unorm = vec1.Norml2();
            if (itau==1)
            {
                invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                //invtau = 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            }
            else if (itau==2)
                invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            else if (itau==3)
                invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            else if (itau==4)
                invtau = 2./dt;
            else
                invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            tau = 1.0/invtau;
            tauMax = max(tauMax, tau);
        }
    }
    else
        MFEM_ABORT("Error in checktauintegrator: only support maxtau"); 

    elvect=tauMax;
}

// Integrator for (tau * rhs , V.grad v)
class StabDomainLFIntegrator : public LinearFormIntegrator
{
private:
   DenseMatrix dshape, gshape, Jinv, V_ir;
   MyCoefficient *V; 
   Coefficient *nuCoef, &Q;
   double dt;
   int itau;
   double alpha;

public:
   StabDomainLFIntegrator (double dt_, double visc, MyCoefficient &q, Coefficient &QF, int itau_=2, double alpha_=0.) : 
       V(&q),  Q(QF), dt(dt_), itau(itau_), alpha(alpha_)
   { nuCoef = new ConstantCoefficient(visc); }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

   virtual ~StabDomainLFIntegrator()
   { delete nuCoef; }
};

void StabDomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
{
    double norm, tau;
    double Unorm, invtau;
    int dim = 2;
    int nd = el.GetDof();
    Vector advGrad(nd), vec1(dim);

    dshape.SetSize(nd, dim);
    gshape.SetSize(nd, dim);
    Jinv  .SetSize(dim);

    //here we assume 2d quad
    double eleLength = sqrt( Geometry::Volume[el.GetGeomType()] * Tr.Weight() );   

    elvect.SetSize(nd);
    elvect=0.;
    
    int intorder = 2*el.GetOrder() + Tr.OrderGrad(&el)-1;
    const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

    V->Eval(V_ir, Tr, ir);

    if (dtfloor && dt<dtmin)
        dt=dtmin;

    //compare maximum tau
    double tauMax=0.0;
    if (maxtau)
    {
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            double nu = nuCoef->Eval (Tr, ip);

            V_ir.GetColumnReference(i, vec1);
            Unorm = vec1.Norml2();
            if (itau==1)
            {
                invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                //invtau = 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            }
            else if (itau==2)
                invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            else if (itau==3)
            {
                if (alpha>0.)
                    invtau = sqrt( pow(alpha/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                else
                    invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==4)
                invtau = 2./dt;
            else
                invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            tau = 1.0/invtau;
            tauMax = max(tauMax, tau);
        }
    }
    //cout <<"tauMax="<<tauMax<<" length="<<eleLength<<" "<<"dt="<<dt<<" U="<<Unorm<<" ";
  
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir.IntPoint(i);
        
        el.CalcDShape(ip, dshape);
        
        Tr.SetIntPoint(&ip);
        norm = ip.weight * Tr.Weight();
        CalcInverse (Tr.Jacobian(), Jinv);
        Mult(dshape, Jinv, gshape); 
        
        V_ir.GetColumnReference(i, vec1);

        if(maxtau)
            norm*=tauMax*Q.Eval(Tr,ip);
        else
        {
            //compute tau
            double nu = nuCoef->Eval (Tr, ip);
            Unorm = vec1.Norml2();
            if (itau==1)
            {
                invtau = sqrt( pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                //invtau = 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            }
            else if (itau==2)
                invtau = sqrt( pow(2./dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            else if (itau==3)
            {
                if (alpha>0.)
                    invtau = sqrt( pow(alpha/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
                else
                    invtau = sqrt( pow(dtfactor/dt,2) + pow(2.0*Unorm/eleLength,2) + pow(4.0*nu/(eleLength*eleLength),2));
            }
            else if (itau==4)
                invtau = 2./dt;
            else
                invtau = 2.0/dt + 2.0 * Unorm / eleLength + 4.0 * nu / (eleLength * eleLength);
            tau = 1.0/invtau;

            norm *= (tau * Q.Eval (Tr, ip));
        }
 
        gshape.Mult(vec1, advGrad);

        add(elvect, norm, advGrad, elvect);
    }
}
