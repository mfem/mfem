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

}

#endif


