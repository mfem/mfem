#include "integ_algoim.hpp"


#ifdef MFEM_USE_ALGOIM


namespace mfem
{

AlgoimIntegrationRule::AlgoimIntegrationRule(int o, const FiniteElement &el,
                                             ElementTransformation &trans,
                                             const Vector &lsfun)
{
   int_order=o;
   vir=nullptr;
   sir=nullptr;

   if (el.GetGeomType()==Geometry::Type::SQUARE)
   {
      pe=new H1Pos_QuadrilateralElement(el.GetOrder());
   }
   else if (el.GetGeomType()==Geometry::Type::CUBE)
   {
      pe=new H1Pos_HexahedronElement(el.GetOrder());
   }
   else
   {
      MFEM_ABORT("Currently MFEM + Algoim supports only quads and hexes.");
   }

   //change the basis of the level-set function
   //from Lagrangian to Bernstein (positive)
   lsvec.SetSize(pe->GetDof());
   DenseMatrix T(pe->GetDof());
   pe->Project(el,trans,T);
   T.Mult(lsfun,lsvec);
}

AlgoimIntegrationRule::AlgoimIntegrationRule(int o,ElementTransformation &trans,
                                             Coefficient& lsfun, int lso)
{
    int_order=o;
    vir=nullptr;
    sir=nullptr;

    if(trans.GetGeometryType()==Geometry::Type::SQUARE)
    {
        if(lso==-1){
            pe=new H1Pos_QuadrilateralElement(trans.Order());
        }else{
            pe=new H1Pos_QuadrilateralElement(lso);
        }
    }
    else if (trans.GetGeometryType()==Geometry::Type::CUBE)
    {
        if(lso==-1){
            pe=new H1Pos_HexahedronElement(trans.Order());
        }else{
            pe=new H1Pos_HexahedronElement(lso);
        }
    }
    else
    {
        MFEM_ABORT("Currently MFEM + Algoim supports only quads and hexes.");
    }

    //evaluate the level-set function
    DenseMatrix M; M.SetSize(pe->GetDof()); M=0.0;
    Vector rhs; rhs.SetSize(pe->GetDof()); rhs=0.0;
    Vector shape; shape.SetSize(pe->GetDof()); shape=0.0;

    const IntegrationRule* ir= nullptr;
    int io=2*pe->GetOrder()+trans.OrderJ();
    ir=&IntRules.Get(trans.GetGeometryType(),io);

    lsvec.SetSize(pe->GetDof());

    // form the mass matrix and the rhs
    double w;
    for(int i=0;i<ir->GetNPoints();i++){
        const IntegrationPoint &ip = ir->IntPoint(i);
        trans.SetIntPoint(&ip);

        w=trans.Weight();
        w = ip.weight * w;

        pe->CalcPhysShape(trans,shape);

        for(int ii=0;ii<pe->GetDof();ii++){
                rhs[ii]=rhs[ii]+w*shape[ii]*lsfun.Eval(trans,ip);
                M(ii,ii)=M(ii,ii)+w*shape[ii]*shape[ii];
            for(int jj=0;jj<ii;jj++){
                M(ii,jj)=M(ii,jj)+w*shape[ii]*shape[jj];
        }}
    }

    for(int ii=0;ii<pe->GetDof();ii++){
        for(int jj=0;jj<ii;jj++){
            M(jj,ii)=M(ii,jj);
        }
    }

    DenseMatrixInverse im(&M,true);
    im.Factor();
    im.Mult(rhs,lsvec);
}

AlgoimIntegrationRule::   AlgoimIntegrationRule(int o, ElementTransformation &trans,
                                                GridFunction& lsfun)
{

    int lso= lsfun.FESpace()->GetElementOrder(trans.ElementNo);

    int_order=o;
    vir=nullptr;
    sir=nullptr;

    if(trans.GetGeometryType()==Geometry::Type::SQUARE)
    {
        pe=new H1Pos_QuadrilateralElement(lso);
    }
    else if (trans.GetGeometryType()==Geometry::Type::CUBE)
    {
        pe=new H1Pos_HexahedronElement(lso);
    }
    else
    {
        MFEM_ABORT("Currently MFEM + Algoim supports only quads and hexes.");
    }

    //evaluate the level-set function
    DenseMatrix M; M.SetSize(pe->GetDof()); M=0.0;
    Vector rhs; rhs.SetSize(pe->GetDof()); rhs=0.0;
    Vector shape; shape.SetSize(pe->GetDof()); shape=0.0;

    const IntegrationRule* ir= nullptr;
    int io=2*pe->GetOrder()+trans.OrderJ();
    ir=&IntRules.Get(trans.GetGeometryType(),io);

    lsvec.SetSize(pe->GetDof());

    // form the mass matrix and the rhs
    double w;
    for(int i=0;i<ir->GetNPoints();i++){
        const IntegrationPoint &ip = ir->IntPoint(i);
        trans.SetIntPoint(&ip);

        w=trans.Weight();
        w = ip.weight * w;

        pe->CalcPhysShape(trans,shape);

        for(int ii=0;ii<pe->GetDof();ii++){
                rhs[ii]=rhs[ii]+w*shape[ii]*lsfun.GetValue(trans,ip);
                M(ii,ii)=M(ii,ii)+w*shape[ii]*shape[ii];
            for(int jj=0;jj<ii;jj++){
                M(ii,jj)=M(ii,jj)+w*shape[ii]*shape[jj];
        }}
    }

    for(int ii=0;ii<pe->GetDof();ii++){
        for(int jj=0;jj<ii;jj++){
            M(jj,ii)=M(ii,jj);
        }
    }

    DenseMatrixInverse im(&M,true);
    im.Factor();
    im.Mult(rhs,lsvec);
}


const IntegrationRule* AlgoimIntegrationRule::GetVolumeIntegrationRule()
{
   if (vir!=nullptr) {return vir;}

   const int dim=pe->GetDim();
   int np1d=int_order/2+1;
   if (dim==2)
   {
      LevelSet2D ls(pe,lsvec);
      auto q = Algoim::quadGen<2>(ls,Algoim::BoundingBox<double,2>(0.0,1.0),
                                  -1, -1, np1d);

      vir=new IntegrationRule(q.nodes.size());
      vir->SetOrder(int_order);
      for (size_t i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=vir->IntPoint(i);
         ip.Set2w(q.nodes[i].x(0),q.nodes[i].x(1),q.nodes[i].w);
      }
   }
   else
   {
      LevelSet3D ls(pe,lsvec);
      auto q = Algoim::quadGen<3>(ls,Algoim::BoundingBox<double,3>(0.0,1.0),
                                  -1, -1, np1d);

      vir=new IntegrationRule(q.nodes.size());
      vir->SetOrder(int_order);
      for (size_t i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=vir->IntPoint(i);
         ip.Set(q.nodes[i].x(0),q.nodes[i].x(1),q.nodes[i].x(2),q.nodes[i].w);
      }
   }

   return vir;
}

const IntegrationRule* AlgoimIntegrationRule::GetSurfaceIntegrationRule()
{
   if (sir!=nullptr) {return sir;}

   int np1d=int_order/2+1;
   const int dim=pe->GetDim();
   if (dim==2)
   {
      LevelSet2D ls(pe,lsvec);
      auto q = Algoim::quadGen<2>(ls,Algoim::BoundingBox<double,2>(0.0,1.0),
                                  2, -1, np1d);

      sir=new IntegrationRule(q.nodes.size());
      sir->SetOrder(int_order);
      for (size_t i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=sir->IntPoint(i);
         ip.Set2w(q.nodes[i].x(0),q.nodes[i].x(1),q.nodes[i].w);
      }
   }
   else
   {
      LevelSet3D ls(pe,lsvec);
      auto q = Algoim::quadGen<3>(ls,Algoim::BoundingBox<double,3>(0.0,1.0),
                                  3, -1, np1d);

      sir=new IntegrationRule(q.nodes.size());
      sir->SetOrder(int_order);
      for (size_t i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=sir->IntPoint(i);
         ip.Set(q.nodes[i].x(0),q.nodes[i].x(1),q.nodes[i].x(2),q.nodes[i].w);
      }
   }

   return sir;
}


double AlgoimIntegrationRule::SurfaceWeight(ElementTransformation &trans,
                                            const IntegrationPoint& ip)
{
    DenseMatrix bmat; //gradients of the shape functions in isoparametric space
    DenseMatrix pmat; //gradients of the shape functions in physical space
    Vector inormal; //normal to the level set in isoparametric space
    Vector tnormal; //normal to the level set in physical space

    bmat.SetSize(pe->GetDof(),pe->GetDim());
    pmat.SetSize(pe->GetDof(),pe->GetDim());
    inormal.SetSize(pe->GetDim());
    tnormal.SetSize(pe->GetDim());

    pe->CalcDShape(ip,bmat);
    Mult(bmat, trans.AdjugateJacobian(), pmat);
    //compute the normal to the LS in isoparametric space
    bmat.MultTranspose(lsvec,inormal);
    //compute the normal to the LS in physical space
    pmat.MultTranspose(lsvec,tnormal);

    return tnormal.Norml2() / inormal.Norml2();
}

const Array<double>* AlgoimIntegrationRule::GetSurfaceWeights(
                                            ElementTransformation &trans)
{
    const IntegrationRule* lsir=GetSurfaceIntegrationRule();

    sweight.SetSize(lsir->GetNPoints());

    DenseMatrix bmat; //gradients of the shape functions in isoparametric space
    DenseMatrix pmat; //gradients of the shape functions in physical space
    Vector inormal; //normal to the level set in isoparametric space
    Vector tnormal; //normal to the level set in physical space

    bmat.SetSize(pe->GetDof(),pe->GetDim());
    pmat.SetSize(pe->GetDof(),pe->GetDim());
    inormal.SetSize(pe->GetDim());
    tnormal.SetSize(pe->GetDim());

    for(int i=0;i<lsir->GetNPoints();i++){
        const IntegrationPoint &ip = lsir->IntPoint(i);
        trans.SetIntPoint(&ip);
        pe->CalcDShape(ip,bmat);
        Mult(bmat, trans.AdjugateJacobian(), pmat);
        //compute the normal to the LS in isoparametric space
        bmat.MultTranspose(lsvec,inormal);
        //compute the normal to the LS in physical space
        pmat.MultTranspose(lsvec,tnormal);
        sweight[i]=tnormal.Norml2() / inormal.Norml2();
    }

    return &sweight;
}


}

#endif


