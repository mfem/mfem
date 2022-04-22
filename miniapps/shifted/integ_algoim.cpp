#include "integ_algoim.hpp"


#ifdef MFEM_USE_ALGOIM


namespace mfem
{

const IntegrationRule* AlgoimIntegrationRule::GetVolumeIntegrationRule()
{
   if (vir!=nullptr) {return vir;}

   const int dim=pe->GetDim();
   if (dim==2)
   {
      LevelSet2D ls(pe,lsvec);
      auto q = Algoim::quadGen<2>(ls,Algoim::BoundingBox<double,2>(0.0,1.0),-1,-1,
                                  int_order);

      vir=new IntegrationRule(q.nodes.size());
      vir->SetOrder(int_order);
      for (int i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=vir->IntPoint(i);
         ip.Set2w(q.nodes[i].x[0],q.nodes[i].x[1],q.nodes[i].w);
      }
   }
   else
   {
      LevelSet3D ls(pe,lsvec);
      auto q = Algoim::quadGen<3>(ls,Algoim::BoundingBox<double,3>(0.0,1.0),-1,-1,
                                  int_order);

      vir=new IntegrationRule(q.nodes.size());
      vir->SetOrder(int_order);
      for (int i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=vir->IntPoint(i);
         ip.Set(q.nodes[i].x[0],q.nodes[i].x[1],q.nodes[i].x[2],q.nodes[i].w);
      }
   }

   return vir;
}

const IntegrationRule* AlgoimIntegrationRule::GetSurfaceIntegrationRule()
{
   if (sir!=nullptr) {return sir;}

   const int dim=pe->GetDim();
   if (dim==2)
   {
      LevelSet2D ls(pe,lsvec);
      auto q = Algoim::quadGen<2>(ls,Algoim::BoundingBox<double,2>(0.0,1.0),2,-1,
                                  int_order);

      sir=new IntegrationRule(q.nodes.size());
      sir->SetOrder(int_order);
      for (int i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=sir->IntPoint(i);
         ip.Set2w(q.nodes[i].x[0],q.nodes[i].x[1],q.nodes[i].w);
      }
   }
   else
   {
      LevelSet3D ls(pe,lsvec);
      auto q = Algoim::quadGen<3>(ls,Algoim::BoundingBox<double,3>(0.0,1.0),3,-1,
                                  int_order);

      sir=new IntegrationRule(q.nodes.size());
      sir->SetOrder(int_order);
      for (int i=0; i<q.nodes.size(); i++)
      {
         IntegrationPoint& ip=sir->IntPoint(i);
         ip.Set(q.nodes[i].x[0],q.nodes[i].x[1],q.nodes[i].x[2],q.nodes[i].w);
      }
   }

   return sir;
}

}

#endif


