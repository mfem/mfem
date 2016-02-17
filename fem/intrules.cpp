// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of IntegrationRule(s) classes

// Acknowledgment: Some of the high-precision triangular and tetrahedral
// quadrature rules below were obtained from the Encyclopaedia of Cubature
// Formulas at http://nines.cs.kuleuven.be/research/ecf/ecf.html

#include "fem.hpp"
#include <cmath>

using namespace std;

namespace mfem
{

IntegrationRule::IntegrationRule(IntegrationRule &irx, IntegrationRule &iry)
{
   int i, j, nx, ny;

   nx = irx.GetNPoints();
   ny = iry.GetNPoints();
   SetSize(nx * ny);

   for (j = 0; j < ny; j++)
   {
      IntegrationPoint &ipy = iry.IntPoint(j);
      for (i = 0; i < nx; i++)
      {
         IntegrationPoint &ipx = irx.IntPoint(i);
         IntegrationPoint &ip  = IntPoint(j*nx+i);

         ip.x = ipx.x;
         ip.y = ipy.x;
         ip.weight = ipx.weight * ipy.weight;
      }
   }
}

IntegrationRule::IntegrationRule(IntegrationRule &irx, IntegrationRule &iry,
                                 IntegrationRule &irz)
{
   const int nx = irx.GetNPoints();
   const int ny = iry.GetNPoints();
   const int nz = irz.GetNPoints();
   SetSize(nx*ny*nz);

   for (int iz = 0; iz < nz; ++iz)
   {
      IntegrationPoint &ipz = irz.IntPoint(iz);
      for (int iy = 0; iy < ny; ++iy)
      {
         IntegrationPoint &ipy = iry.IntPoint(iy);
         for (int ix = 0; ix < nx; ++ix)
         {
            IntegrationPoint &ipx = irx.IntPoint(ix);
            IntegrationPoint &ip  = IntPoint(iz*nx*ny + iy*nx + ix);

            ip.x = ipx.x;
            ip.y = ipy.x;
            ip.z = ipz.x;
            ip.weight = ipx.weight*ipy.weight*ipz.weight;
         }
      }
   }
}

void IntegrationRule::GaussianRule()
{
   int n = Size();
   int m = (n+1)/2;
   int i, j;
   double p1, p2, p3;
   double pp, z;
   for (i = 1; i <= m; i++)
   {
      z = cos(M_PI * (i - 0.25) / (n + 0.5));

      while (1)
      {
         p1 = 1;
         p2 = 0;
         for (j = 1; j <= n; j++)
         {
            p3 = p2;
            p2 = p1;
            p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
         }
         // p1 is Legendre polynomial

         pp = n * (z*p1-p2) / (z*z - 1);

         if (fabs(p1/pp) < 2e-16) { break; }

         z = z - p1/pp;
      }

      z = ((1 - z) + p1/pp)/2;

      IntPoint(i-1).x  = z;
      IntPoint(n-i).x  = 1 - z;
      IntPoint(i-1).weight =
         IntPoint(n-i).weight = 1./(4*z*(1 - z)*pp*pp);
   }
}

void IntegrationRule::UniformRule()
{
   int i;
   double h;

   h = 1.0 / (Size() - 1);
   for (i = 0; i < Size(); i++)
   {
      IntPoint(i).x = double(i) / (Size() - 1);
      IntPoint(i).weight = h;
   }
   IntPoint(0).weight = 0.5 * h;
   IntPoint(Size()-1).weight = 0.5 * h;
}

void IntegrationRule::GrundmannMollerSimplexRule(int s, int n)
{
   // for pow on older compilers
   using std::pow;
   const int d = 2*s + 1;
   Vector fact(d + n + 1);
   Array<int> beta(n), sums(n);

   fact(0) = 1.;
   for (int i = 1; i < fact.Size(); i++)
   {
      fact(i) = fact(i - 1)*i;
   }

   // number of points is \binom{n + s + 1}{n + 1}
   int np = 1, f = 1;
   for (int i = 0; i <= n; i++)
   {
      np *= (s + i + 1), f *= (i + 1);
   }
   np /= f;
   SetSize(np);

   int pt = 0;
   for (int i = 0; i <= s; i++)
   {
      double weight;

      weight = pow(2., -2*s)*pow(static_cast<double>(d + n - 2*i),
                                 d)/fact(i)/fact(d + n - i);
      if (i%2)
      {
         weight = -weight;
      }

      // loop over all beta : beta_0 + ... + beta_{n-1} <= s - i
      int k = s - i;
      beta = 0;
      sums = 0;
      while (true)
      {
         IntegrationPoint &ip = IntPoint(pt++);
         ip.weight = weight;
         ip.x = double(2*beta[0] + 1)/(d + n - 2*i);
         ip.y = double(2*beta[1] + 1)/(d + n - 2*i);
         if (n == 3)
         {
            ip.z = double(2*beta[2] + 1)/(d + n - 2*i);
         }

         int j = 0;
         while (sums[j] == k)
         {
            beta[j++] = 0;
            if (j == n)
            {
               goto done_beta;
            }
         }
         beta[j]++;
         sums[j]++;
         for (j--; j >= 0; j--)
         {
            sums[j] = sums[j+1];
         }
      }
   done_beta:
      ;
   }
}


IntegrationRules IntRules(0);

IntegrationRules RefinedIntRules(1);

IntegrationRules::IntegrationRules(int Ref)
{
   refined = Ref;
   if (refined < 0) { own_rules = 0; return; }

   own_rules = 1;

   PointIntRules.SetSize(2);
   PointIntRules = NULL;

   SegmentIntRules.SetSize(32);
   SegmentIntRules = NULL;

   // TriangleIntegrationRule() assumes that this size is >= 26
   TriangleIntRules.SetSize(32);
   TriangleIntRules = NULL;

   SquareIntRules.SetSize(32);
   SquareIntRules = NULL;

   // TetrahedronIntegrationRule() assumes that this size is >= 10
   TetrahedronIntRules.SetSize(32);
   TetrahedronIntRules = NULL;

   CubeIntRules.SetSize(32);
   CubeIntRules = NULL;
}

const IntegrationRule &IntegrationRules::Get(int GeomType, int Order)
{
   Array<IntegrationRule *> *ir_array;

   switch (GeomType)
   {
      case Geometry::POINT:       ir_array = &PointIntRules; Order = 0; break;
      case Geometry::SEGMENT:     ir_array = &SegmentIntRules; break;
      case Geometry::TRIANGLE:    ir_array = &TriangleIntRules; break;
      case Geometry::SQUARE:      ir_array = &SquareIntRules; break;
      case Geometry::TETRAHEDRON: ir_array = &TetrahedronIntRules; break;
      case Geometry::CUBE:        ir_array = &CubeIntRules; break;
      default:
         mfem_error("IntegrationRules::Get(...) : Unknown geometry type!");
         ir_array = NULL;
   }

   if (Order < 0)
   {
      Order = 0;
   }

   if (!HaveIntRule(*ir_array, Order))
   {
      GenerateIntegrationRule(GeomType, Order);
   }

   return *(*ir_array)[Order];
}

void IntegrationRules::Set(int GeomType, int Order, IntegrationRule &IntRule)
{
   Array<IntegrationRule *> *ir_array;

   switch (GeomType)
   {
      case Geometry::POINT:       ir_array = &PointIntRules; break;
      case Geometry::SEGMENT:     ir_array = &SegmentIntRules; break;
      case Geometry::TRIANGLE:    ir_array = &TriangleIntRules; break;
      case Geometry::SQUARE:      ir_array = &SquareIntRules; break;
      case Geometry::TETRAHEDRON: ir_array = &TetrahedronIntRules; break;
      case Geometry::CUBE:        ir_array = &CubeIntRules; break;
      default:
         mfem_error("IntegrationRules::Set(...) : Unknown geometry type!");
         ir_array = NULL;
   }

   if (HaveIntRule(*ir_array, Order))
   {
      MFEM_ABORT("Overwriting set rules is not supported!");
   }

   AllocIntRule(*ir_array, Order);

   (*ir_array)[Order] = &IntRule;
}

void IntegrationRules::DeleteIntRuleArray(Array<IntegrationRule *> &ir_array)
{
   int i;
   IntegrationRule *ir = NULL;

   // Many of the intrules have multiple contiguous copies in the ir_array
   // so we have to be careful to not delete them twice.
   for (i = 0; i < ir_array.Size(); i++)
   {
      if (ir_array[i] != NULL && ir_array[i] != ir)
      {
         ir = ir_array[i];
         delete (ir_array[i]);
      }
   }
}

IntegrationRules::~IntegrationRules()
{
   if (!own_rules) { return; }

   DeleteIntRuleArray(PointIntRules);
   DeleteIntRuleArray(SegmentIntRules);
   DeleteIntRuleArray(TriangleIntRules);
   DeleteIntRuleArray(SquareIntRules);
   DeleteIntRuleArray(TetrahedronIntRules);
   DeleteIntRuleArray(CubeIntRules);
}


IntegrationRule *IntegrationRules::GenerateIntegrationRule(int GeomType,
                                                           int Order)
{
   switch (GeomType)
   {
      case Geometry::POINT:
         return PointIntegrationRule(Order);
      case Geometry::SEGMENT:
         return SegmentIntegrationRule(Order);
      case Geometry::TRIANGLE:
         return TriangleIntegrationRule(Order);
      case Geometry::SQUARE:
         return SquareIntegrationRule(Order);
      case Geometry::TETRAHEDRON:
         return TetrahedronIntegrationRule(Order);
      case Geometry::CUBE:
         return CubeIntegrationRule(Order);
      default:
         mfem_error("IntegrationRules::Set(...) : Unknown geometry type!");
         return NULL;
   }
}


// Integration rules for a point
IntegrationRule *IntegrationRules::PointIntegrationRule(int Order)
{
   if (Order > 1)
   {
      mfem_error("Point Integration Rule of Order > 1 not defined");
      return NULL;
   }

   PointIntRules[0] = new IntegrationRule(1);
   PointIntRules[0] -> IntPoint(0).x = .0;
   PointIntRules[0] -> IntPoint(0).weight = 1.;

   PointIntRules[1] = PointIntRules[0];

   return PointIntRules[0];
}

// Integration rules for line segment [0,1]
IntegrationRule *IntegrationRules::SegmentIntegrationRule(int Order)
{
   int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order

   AllocIntRule(SegmentIntRules, i);

   if (refined)
   {
      int n = i/2 + 1;

      IntegrationRule *tmp = new IntegrationRule(n);
      tmp->GaussianRule();

      IntegrationRule *ir = new IntegrationRule(2*n);
      SegmentIntRules[i-1] = SegmentIntRules[i] = ir;
      for (int j = 0; j < n; j++)
      {
         ir->IntPoint(j).x = tmp->IntPoint(j).x/2.0;
         ir->IntPoint(j).weight = tmp->IntPoint(j).weight/2.0;
         ir->IntPoint(j+n).x = 0.5 + tmp->IntPoint(j).x/2.0;
         ir->IntPoint(j+n).weight = tmp->IntPoint(j).weight/2.0;
      }
      delete tmp;

      return ir;
   }

   switch (Order / 2)
   {
      case 0:  // 1 point - 1 degree
         SegmentIntRules[0] = SegmentIntRules[1] = new IntegrationRule(1);
         SegmentIntRules[0] -> IntPoint(0).x = .5;
         SegmentIntRules[0] -> IntPoint(0).weight = 1.;
         return SegmentIntRules[0];
      case 1:  // 2 point - 3 degree
         SegmentIntRules[2] = SegmentIntRules[3] = new IntegrationRule(2);
         SegmentIntRules[2] -> IntPoint(0).x = 0.21132486540518711775;
         SegmentIntRules[2] -> IntPoint(0).weight = .5;
         SegmentIntRules[2] -> IntPoint(1).x = 0.78867513459481288225;
         SegmentIntRules[2] -> IntPoint(1).weight = .5;
         return SegmentIntRules[2];
      case 2:  // 3 point - 5 degree
         SegmentIntRules[4] = SegmentIntRules[5] = new IntegrationRule(3);
         SegmentIntRules[4] -> IntPoint(0).x = 0.11270166537925831148;
         SegmentIntRules[4] -> IntPoint(0).weight = 5./18.;
         SegmentIntRules[4] -> IntPoint(1).x = 0.5;
         SegmentIntRules[4] -> IntPoint(1).weight = 4./9.;
         SegmentIntRules[4] -> IntPoint(2).x = 0.88729833462074168852;
         SegmentIntRules[4] -> IntPoint(2).weight = 5./18.;
         return SegmentIntRules[4];
      default:
         SegmentIntRules[i-1] = SegmentIntRules[i] = new IntegrationRule(i/2+1);
         SegmentIntRules[i]->GaussianRule();
         return SegmentIntRules[i];
   }
}

// Integration rules for reference triangle {[0,0],[1,0],[0,1]}
IntegrationRule *IntegrationRules::TriangleIntegrationRule(int Order)
{
   IntegrationRule *ir = NULL;

   // assuming that orders <= 25 are pre-allocated
   switch (Order)
   {
      case 0:  // 1 point - 0 degree
      case 1:
         TriangleIntRules[0] =
            TriangleIntRules[1] = ir = new IntegrationRule(1);
         TriangleIntRules[0]->AddTriMidPoint(0, 0.5);
         return ir;

      case 2:  // 3 point - 2 degree
         TriangleIntRules[2] = ir = new IntegrationRule(3);
         //   interior points
         TriangleIntRules[2]->AddTriPoints3(0, 1./6., 1./6.);
         return ir;

      case 3:  // 4 point - 3 degree (has one negative weight)
         TriangleIntRules[3] = ir = new IntegrationRule(4);
         ir->AddTriMidPoint(0, -0.28125); // -9./32.
         ir->AddTriPoints3(1, 0.2, 25./96.);
         return ir;

      case 4:  // 6 point - 4 degree
         TriangleIntRules[4] = ir = new IntegrationRule(6);
         ir->AddTriPoints3(0, 0.091576213509770743460, 0.054975871827660933819);
         ir->AddTriPoints3(3, 0.44594849091596488632, 0.11169079483900573285);
         return ir;

      case 5:  // 7 point - 5 degree
         TriangleIntRules[5] = ir = new IntegrationRule(7);
         ir->AddTriMidPoint(0, 0.1125);
         ir->AddTriPoints3(1, 0.10128650732345633880, 0.062969590272413576298);
         ir->AddTriPoints3(4, 0.47014206410511508977, 0.066197076394253090369);
         return ir;

      case 6:  // 12 point - 6 degree
         TriangleIntRules[6] = ir = new IntegrationRule(12);
         ir->AddTriPoints3(0, 0.063089014491502228340, 0.025422453185103408460);
         ir->AddTriPoints3(3, 0.24928674517091042129, 0.058393137863189683013);
         ir->AddTriPoints6(6, 0.053145049844816947353, 0.31035245103378440542,
                           0.041425537809186787597);
         return ir;

      case 7:  // 12 point - degree 7
         TriangleIntRules[7] = ir = new IntegrationRule(12);
         ir->AddTriPoints3R(0, 0.062382265094402118174, 0.067517867073916085443,
                            0.026517028157436251429);
         ir->AddTriPoints3R(3, 0.055225456656926611737, 0.32150249385198182267,
                            0.043881408714446055037);
         //  slightly better with explicit 3rd area coordinate
         ir->AddTriPoints3R(6, 0.034324302945097146470, 0.66094919618673565761,
                            0.30472650086816719592, 0.028775042784981585738);
         ir->AddTriPoints3R(9, 0.51584233435359177926, 0.27771616697639178257,
                            0.20644149867001643817, 0.067493187009802774463);
         return ir;

      case 8:  // 16 point - 8 degree
         TriangleIntRules[8] = ir = new IntegrationRule(16);
         ir->AddTriMidPoint(0, 0.0721578038388935841255455552445323);
         ir->AddTriPoints3(1, 0.170569307751760206622293501491464,
                           0.0516086852673591251408957751460645);
         ir->AddTriPoints3(4, 0.0505472283170309754584235505965989,
                           0.0162292488115990401554629641708902);
         ir->AddTriPoints3(7, 0.459292588292723156028815514494169,
                           0.0475458171336423123969480521942921);
         ir->AddTriPoints6(10, 0.008394777409957605337213834539296,
                           0.263112829634638113421785786284643,
                           0.0136151570872174971324223450369544);
         return ir;

      case 9:  // 19 point - 9 degree
         TriangleIntRules[9] = ir = new IntegrationRule(19);
         ir->AddTriMidPoint(0, 0.0485678981413994169096209912536443);
         ir->AddTriPoints3b(1, 0.020634961602524744433,
                            0.0156673501135695352684274156436046);
         ir->AddTriPoints3b(4, 0.12582081701412672546,
                            0.0389137705023871396583696781497019);
         ir->AddTriPoints3(7, 0.188203535619032730240961280467335,
                           0.0398238694636051265164458871320226);
         ir->AddTriPoints3(10, 0.0447295133944527098651065899662763,
                           0.0127888378293490156308393992794999);
         ir->AddTriPoints6(13, 0.0368384120547362836348175987833851,
                           0.2219629891607656956751025276931919,
                           0.0216417696886446886446886446886446);
         return ir;

      case 10:  // 25 point - 10 degree
         TriangleIntRules[10] = ir = new IntegrationRule(25);
         ir->AddTriMidPoint(0, 0.0454089951913767900476432975500142);
         ir->AddTriPoints3b(1, 0.028844733232685245264984935583748,
                            0.0183629788782333523585030359456832);
         ir->AddTriPoints3(4, 0.109481575485037054795458631340522,
                           0.0226605297177639673913028223692986);
         ir->AddTriPoints6(7, 0.141707219414879954756683250476361,
                           0.307939838764120950165155022930631,
                           0.0363789584227100543021575883096803);
         ir->AddTriPoints6(13, 0.025003534762686386073988481007746,
                           0.246672560639902693917276465411176,
                           0.0141636212655287424183685307910495);
         ir->AddTriPoints6(19, 0.0095408154002994575801528096228873,
                           0.0668032510122002657735402127620247,
                           4.71083348186641172996373548344341E-03);
         return ir;

      case 11: // 28 point -- 11 degree
         TriangleIntRules[11] = ir = new IntegrationRule(28);
         ir->AddTriPoints6(0, 0.0,
                           0.141129718717363295960826061941652,
                           3.68119189165027713212944752369032E-03);
         ir->AddTriMidPoint(6, 0.0439886505811161193990465846607278);
         ir->AddTriPoints3(7, 0.0259891409282873952600324854988407,
                           4.37215577686801152475821439991262E-03);
         ir->AddTriPoints3(10, 0.0942875026479224956305697762754049,
                           0.0190407859969674687575121697178070);
         ir->AddTriPoints3b(13, 0.010726449965572372516734795387128,
                            9.42772402806564602923839129555767E-03);
         ir->AddTriPoints3(16, 0.207343382614511333452934024112966,
                           0.0360798487723697630620149942932315);
         ir->AddTriPoints3b(19, 0.122184388599015809877869236727746,
                            0.0346645693527679499208828254519072);
         ir->AddTriPoints6(22, 0.0448416775891304433090523914688007,
                           0.2772206675282791551488214673424523,
                           0.0205281577146442833208261574536469);
         return ir;

      case 12: // 33 point - 12 degree
         TriangleIntRules[12] = ir = new IntegrationRule(33);
         ir->AddTriPoints3b(0, 2.35652204523900E-02, 1.28655332202275E-02);
         ir->AddTriPoints3b(3, 1.20551215411079E-01, 2.18462722690190E-02);
         ir->AddTriPoints3(6, 2.71210385012116E-01, 3.14291121089425E-02);
         ir->AddTriPoints3(9, 1.27576145541586E-01, 1.73980564653545E-02);
         ir->AddTriPoints3(12, 2.13173504532100E-02, 3.08313052577950E-03);
         ir->AddTriPoints6(15, 1.15343494534698E-01, 2.75713269685514E-01,
                           2.01857788831905E-02);
         ir->AddTriPoints6(21, 2.28383322222570E-02, 2.81325580989940E-01,
                           1.11783866011515E-02);
         ir->AddTriPoints6(27, 2.57340505483300E-02, 1.16251915907597E-01,
                           8.65811555432950E-03);
         return ir;

      case 13: // 37 point - 13 degree
         TriangleIntRules[13] = ir = new IntegrationRule(37);
         ir->AddTriPoints3b(0, 0.0,
                            2.67845189554543044455908674650066E-03);
         ir->AddTriMidPoint(3, 0.0293480398063595158995969648597808);
         ir->AddTriPoints3(4, 0.0246071886432302181878499494124643,
                           3.92538414805004016372590903990464E-03);
         ir->AddTriPoints3b(7, 0.159382493797610632566158925635800,
                            0.0253344765879434817105476355306468);
         ir->AddTriPoints3(10, 0.227900255506160619646298948153592,
                           0.0250401630452545330803738542916538);
         ir->AddTriPoints3(13, 0.116213058883517905247155321839271,
                           0.0158235572961491595176634480481793);
         ir->AddTriPoints3b(16, 0.046794039901841694097491569577008,
                            0.0157462815379843978450278590138683);
         ir->AddTriPoints6(19, 0.0227978945382486125477207592747430,
                           0.1254265183163409177176192369310890,
                           7.90126610763037567956187298486575E-03);
         ir->AddTriPoints6(25, 0.0162757709910885409437036075960413,
                           0.2909269114422506044621801030055257,
                           7.99081889046420266145965132482933E-03);
         ir->AddTriPoints6(31, 0.0897330604516053590796290561145196,
                           0.2723110556841851025078181617634414,
                           0.0182757511120486476280967518782978);
         return ir;

      case 14: // 42 point - 14 degree
         TriangleIntRules[14] = ir = new IntegrationRule(42);
         ir->AddTriPoints3b(0, 2.20721792756430E-02, 1.09417906847145E-02);
         ir->AddTriPoints3b(3, 1.64710561319092E-01, 1.63941767720625E-02);
         ir->AddTriPoints3(6, 2.73477528308839E-01, 2.58870522536460E-02);
         ir->AddTriPoints3(9, 1.77205532412543E-01, 2.10812943684965E-02);
         ir->AddTriPoints3(12, 6.17998830908730E-02, 7.21684983488850E-03);
         ir->AddTriPoints3(15, 1.93909612487010E-02, 2.46170180120000E-03);
         ir->AddTriPoints6(18, 5.71247574036480E-02, 1.72266687821356E-01,
                           1.23328766062820E-02);
         ir->AddTriPoints6(24, 9.29162493569720E-02, 3.36861459796345E-01,
                           1.92857553935305E-02);
         ir->AddTriPoints6(30, 1.46469500556540E-02, 2.98372882136258E-01,
                           7.21815405676700E-03);
         ir->AddTriPoints6(36, 1.26833093287200E-03, 1.18974497696957E-01,
                           2.50511441925050E-03);
         return ir;

      case 15: // 54 point - 15 degree
         TriangleIntRules[15] = ir = new IntegrationRule(54);
         ir->AddTriPoints3b(0, 0.0834384072617499333, 0.016330909424402645);
         ir->AddTriPoints3b(3, 0.192779070841738867, 0.01370640901568218);
         ir->AddTriPoints3(6, 0.293197167913025367, 0.01325501829935165);
         ir->AddTriPoints3(9, 0.146467786942772933, 0.014607981068243055);
         ir->AddTriPoints3(12, 0.0563628676656034333, 0.005292304033121995);
         ir->AddTriPoints3(15, 0.0165751268583703333, 0.0018073215320460175);
         ir->AddTriPoints6(18, 0.0099122033092248, 0.239534554154794445,
                           0.004263874050854718);
         ir->AddTriPoints6(24, 0.015803770630228, 0.404878807318339958,
                           0.006958088258345965);
         ir->AddTriPoints6(30, 0.00514360881697066667, 0.0950021131130448885,
                           0.0021459664703674175);
         ir->AddTriPoints6(36, 0.0489223257529888, 0.149753107322273969,
                           0.008117664640887445);
         ir->AddTriPoints6(42, 0.0687687486325192, 0.286919612441334979,
                           0.012803670460631195);
         ir->AddTriPoints6(48, 0.1684044181246992, 0.281835668099084562,
                           0.016544097765822835);
         return ir;

      case 16:  // 61 point - 17 degree (used for 16 as well)
      case 17:
         TriangleIntRules[16] =
            TriangleIntRules[17] = ir = new IntegrationRule(61);
         ir->AddTriMidPoint(0,  1.67185996454015E-02);
         ir->AddTriPoints3b(1,  5.65891888645200E-03, 2.54670772025350E-03);
         ir->AddTriPoints3b(4,  3.56473547507510E-02, 7.33543226381900E-03);
         ir->AddTriPoints3b(7,  9.95200619584370E-02, 1.21754391768360E-02);
         ir->AddTriPoints3b(10, 1.99467521245206E-01, 1.55537754344845E-02);
         ir->AddTriPoints3 (13, 2.52141267970953E-01, 1.56285556093100E-02);
         ir->AddTriPoints3 (16, 1.62047004658461E-01, 1.24078271698325E-02);
         ir->AddTriPoints3 (19, 7.58758822607460E-02, 7.02803653527850E-03);
         ir->AddTriPoints3 (22, 1.56547269678220E-02, 1.59733808688950E-03);
         ir->AddTriPoints6 (25, 1.01869288269190E-02, 3.34319867363658E-01,
                            4.05982765949650E-03);
         ir->AddTriPoints6 (31, 1.35440871671036E-01, 2.92221537796944E-01,
                            1.34028711415815E-02);
         ir->AddTriPoints6 (37, 5.44239242905830E-02, 3.19574885423190E-01,
                            9.22999660541100E-03);
         ir->AddTriPoints6 (43, 1.28685608336370E-02, 1.90704224192292E-01,
                            4.23843426716400E-03);
         ir->AddTriPoints6 (49, 6.71657824135240E-02, 1.80483211648746E-01,
                            9.14639838501250E-03);
         ir->AddTriPoints6 (55, 1.46631822248280E-02, 8.07113136795640E-02,
                            3.33281600208250E-03);
         return ir;

      case 18: // 73 point - 19 degree (used for 18 as well)
      case 19:
         TriangleIntRules[18] =
            TriangleIntRules[19] = ir = new IntegrationRule(73);
         ir->AddTriMidPoint(0,  0.0164531656944595);
         ir->AddTriPoints3b(1,  0.020780025853987, 0.005165365945636);
         ir->AddTriPoints3b(4,  0.090926214604215, 0.011193623631508);
         ir->AddTriPoints3b(7,  0.197166638701138, 0.015133062934734);
         ir->AddTriPoints3 (10, 0.255551654403098, 0.015245483901099);
         ir->AddTriPoints3 (13, 0.17707794215213,  0.0120796063708205);
         ir->AddTriPoints3 (16, 0.110061053227952, 0.0080254017934005);
         ir->AddTriPoints3 (19, 0.05552862425184,  0.004042290130892);
         ir->AddTriPoints3 (22, 0.012621863777229, 0.0010396810137425);
         ir->AddTriPoints6 (25, 0.003611417848412, 0.395754787356943,
                            0.0019424384524905);
         ir->AddTriPoints6 (31, 0.13446675453078, 0.307929983880436,
                            0.012787080306011);
         ir->AddTriPoints6 (37, 0.014446025776115, 0.26456694840652,
                            0.004440451786669);
         ir->AddTriPoints6 (43, 0.046933578838178, 0.358539352205951,
                            0.0080622733808655);
         ir->AddTriPoints6 (49, 0.002861120350567, 0.157807405968595,
                            0.0012459709087455);
         ir->AddTriPoints6 (55, 0.075050596975911, 0.223861424097916,
                            0.0091214200594755);
         ir->AddTriPoints6 (61, 0.03464707481676, 0.142421601113383,
                            0.0051292818680995);
         ir->AddTriPoints6 (67, 0.065494628082938, 0.010161119296278,
                            0.001899964427651);
         return ir;

      case 20: // 85 point - 20 degree
         TriangleIntRules[20] = ir = new IntegrationRule(85);
         ir->AddTriMidPoint(0, 0.01380521349884976);
         ir->AddTriPoints3b(1, 0.001500649324429,     0.00088951477366337);
         ir->AddTriPoints3b(4, 0.0941397519389508667, 0.010056199056980585);
         ir->AddTriPoints3b(7, 0.2044721240895264,    0.013408923629665785);
         ir->AddTriPoints3(10, 0.264500202532787333,  0.012261566900751005);
         ir->AddTriPoints3(13, 0.211018964092076767,  0.008197289205347695);
         ir->AddTriPoints3(16, 0.107735607171271333,  0.0073979536993248);
         ir->AddTriPoints3(19, 0.0390690878378026667, 0.0022896411388521255);
         ir->AddTriPoints3(22, 0.0111743797293296333, 0.0008259132577881085);
         ir->AddTriPoints6(25, 0.00534961818733726667, 0.0635496659083522206,
                           0.001174585454287792);
         ir->AddTriPoints6(31, 0.00795481706619893333, 0.157106918940706982,
                           0.0022329628770908965);
         ir->AddTriPoints6(37, 0.0104223982812638,     0.395642114364374018,
                           0.003049783403953986);
         ir->AddTriPoints6(43, 0.0109644147961233333,  0.273167570712910522,
                           0.0034455406635941015);
         ir->AddTriPoints6(49, 0.0385667120854623333,  0.101785382485017108,
                           0.0039987375362390815);
         ir->AddTriPoints6(55, 0.0355805078172182,     0.446658549176413815,
                           0.003693067142668012);
         ir->AddTriPoints6(61, 0.0496708163627641333,  0.199010794149503095,
                           0.00639966593932413);
         ir->AddTriPoints6(67, 0.0585197250843317333,  0.3242611836922827,
                           0.008629035587848275);
         ir->AddTriPoints6(73, 0.121497787004394267,   0.208531363210132855,
                           0.009336472951467735);
         ir->AddTriPoints6(79, 0.140710844943938733,   0.323170566536257485,
                           0.01140911202919763);
         return ir;

      case 21: // 126 point - 25 degree (used also for degrees from 21 to 24)
      case 22:
      case 23:
      case 24:
      case 25:
         TriangleIntRules[21] =
            TriangleIntRules[22] =
               TriangleIntRules[23] =
                  TriangleIntRules[24] =
                     TriangleIntRules[25] = ir = new IntegrationRule(126);
         ir->AddTriPoints3b(0, 0.0279464830731742,   0.0040027909400102085);
         ir->AddTriPoints3b(3, 0.131178601327651467, 0.00797353841619525);
         ir->AddTriPoints3b(6, 0.220221729512072267, 0.006554570615397765);
         ir->AddTriPoints3 (9, 0.298443234019804467,   0.00979150048281781);
         ir->AddTriPoints3(12, 0.2340441723373718,     0.008235442720768635);
         ir->AddTriPoints3(15, 0.151468334609017567,   0.00427363953704605);
         ir->AddTriPoints3(18, 0.112733893545993667,   0.004080942928613246);
         ir->AddTriPoints3(21, 0.0777156920915263,     0.0030605732699918895);
         ir->AddTriPoints3(24, 0.034893093614297,      0.0014542491324683325);
         ir->AddTriPoints3(27, 0.00725818462093236667, 0.00034613762283099815);
         ir->AddTriPoints6(30,  0.0012923527044422,     0.227214452153364077,
                           0.0006241445996386985);
         ir->AddTriPoints6(36,  0.0053997012721162,     0.435010554853571706,
                           0.001702376454401511);
         ir->AddTriPoints6(42,  0.006384003033975,      0.320309599272204437,
                           0.0016798271630320255);
         ir->AddTriPoints6(48,  0.00502821150199306667, 0.0917503222800051889,
                           0.000858078269748377);
         ir->AddTriPoints6(54,  0.00682675862178186667, 0.0380108358587243835,
                           0.000740428158357803);
         ir->AddTriPoints6(60,  0.0100161996399295333,  0.157425218485311668,
                           0.0017556563053643425);
         ir->AddTriPoints6(66,  0.02575781317339,       0.239889659778533193,
                           0.003696775074853242);
         ir->AddTriPoints6(72,  0.0302278981199158,     0.361943118126060531,
                           0.003991543738688279);
         ir->AddTriPoints6(78,  0.0305049901071620667,  0.0835519609548285602,
                           0.0021779813065790205);
         ir->AddTriPoints6(84,  0.0459565473625693333,  0.148443220732418205,
                           0.003682528350708916);
         ir->AddTriPoints6(90,  0.0674428005402775333,  0.283739708727534955,
                           0.005481786423209775);
         ir->AddTriPoints6(96,  0.0700450914159106,     0.406899375118787573,
                           0.00587498087177056);
         ir->AddTriPoints6(102, 0.0839115246401166,     0.194113987024892542,
                           0.005007800356899285);
         ir->AddTriPoints6(108, 0.120375535677152667,   0.32413434700070316,
                           0.00665482039381434);
         ir->AddTriPoints6(114, 0.148066899157366667,   0.229277483555980969,
                           0.00707722325261307);
         ir->AddTriPoints6(120, 0.191771865867325067,   0.325618122595983752,
                           0.007440689780584005);
         return ir;

      default:
         // Grundmann-Moller rules
         int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order
         AllocIntRule(TriangleIntRules, i);
         TriangleIntRules[i-1] = TriangleIntRules[i] = ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(i/2,2);
         return ir;
   }
}

// Integration rules for unit square
IntegrationRule *IntegrationRules::SquareIntegrationRule(int Order)
{
   int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order

   if (!HaveIntRule(SegmentIntRules, i))
   {
      SegmentIntegrationRule(i);
   }
   AllocIntRule(SquareIntRules, i);
   SquareIntRules[i-1] =
      SquareIntRules[i] =
         new IntegrationRule(*SegmentIntRules[i], *SegmentIntRules[i]);
   return SquareIntRules[i];
}

/** Integration rules for reference tetrahedron
    {[0,0,0],[1,0,0],[0,1,0],[0,0,1]}          */
IntegrationRule *IntegrationRules::TetrahedronIntegrationRule(int Order)
{
   IntegrationRule *ir;

   // assuming that orders <= 9 are pre-allocated
   switch (Order)
   {
      case 0:  // 1 point - degree 1
      case 1:
         TetrahedronIntRules[0] =
            TetrahedronIntRules[1] = ir = new IntegrationRule(1);
         ir->AddTetMidPoint(0, 1./6.);
         return ir;

      case 2:  // 4 points - degree 2
         TetrahedronIntRules[2] = ir = new IntegrationRule(4);
         // ir->AddTetPoints4(0, 0.13819660112501051518, 1./24.);
         ir->AddTetPoints4b(0, 0.58541019662496845446, 1./24.);
         return ir;

      case 3:  // 5 points - degree 3 (negative weight)
         TetrahedronIntRules[3] = ir = new IntegrationRule(5);
         ir->AddTetMidPoint(0, -2./15.);
         ir->AddTetPoints4b(1, 0.5, 0.075);
         return ir;

      case 4:  // 11 points - degree 4 (negative weight)
         TetrahedronIntRules[4] = ir = new IntegrationRule(11);
         ir->AddTetPoints4(0, 1./14., 343./45000.);
         ir->AddTetMidPoint(4, -74./5625.);
         ir->AddTetPoints6(5, 0.10059642383320079500, 28./1125.);
         return ir;

      case 5:  // 14 points - degree 5
         TetrahedronIntRules[5] = ir = new IntegrationRule(14);
         ir->AddTetPoints6(0, 0.045503704125649649492,
                           7.0910034628469110730E-03);
         ir->AddTetPoints4(6, 0.092735250310891226402, 0.012248840519393658257);
         ir->AddTetPoints4b(10, 0.067342242210098170608,
                            0.018781320953002641800);
         return ir;

      case 6:  // 24 points - degree 6
         TetrahedronIntRules[6] = ir = new IntegrationRule(24);
         ir->AddTetPoints4(0, 0.21460287125915202929,
                           6.6537917096945820166E-03);
         ir->AddTetPoints4(4, 0.040673958534611353116,
                           1.6795351758867738247E-03);
         ir->AddTetPoints4b(8, 0.032986329573173468968,
                            9.2261969239424536825E-03);
         ir->AddTetPoints12(12, 0.063661001875017525299, 0.26967233145831580803,
                            8.0357142857142857143E-03);
         return ir;

      case 7:  // 31 points - degree 7 (negative weight)
         TetrahedronIntRules[7] = ir = new IntegrationRule(31);
         ir->AddTetPoints6(0, 0.0, 9.7001763668430335097E-04);
         ir->AddTetMidPoint(6, 0.018264223466108820291);
         ir->AddTetPoints4(7, 0.078213192330318064374, 0.010599941524413686916);
         ir->AddTetPoints4(11, 0.12184321666390517465,
                           -0.062517740114331851691);
         ir->AddTetPoints4b(15, 2.3825066607381275412E-03,
                            4.8914252630734993858E-03);
         ir->AddTetPoints12(19, 0.1, 0.2, 0.027557319223985890653);
         return ir;

      case 8:  // 43 points - degree 8 (negative weight)
         TetrahedronIntRules[8] = ir = new IntegrationRule(43);
         ir->AddTetPoints4(0, 5.7819505051979972532E-03,
                           1.6983410909288737984E-04);
         ir->AddTetPoints4(4, 0.082103588310546723091,
                           1.9670333131339009876E-03);
         ir->AddTetPoints12(8, 0.036607749553197423679, 0.19048604193463345570,
                            2.1405191411620925965E-03);
         ir->AddTetPoints6(20, 0.050532740018894224426,
                           4.5796838244672818007E-03);
         ir->AddTetPoints12(26, 0.22906653611681113960, 0.035639582788534043717,
                            5.7044858086819185068E-03);
         ir->AddTetPoints4(38, 0.20682993161067320408, 0.014250305822866901248);
         ir->AddTetMidPoint(42, -0.020500188658639915841);
         return ir;

      case 9: // orders 9 and higher -- Grundmann-Moller rules
         TetrahedronIntRules[9] = ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(4,3);
         return ir;

      default: // Grundmann-Moller rules
         int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order
         AllocIntRule(TetrahedronIntRules, i);
         TetrahedronIntRules[i-1] =
            TetrahedronIntRules[i] = ir = new IntegrationRule;
         ir->GrundmannMollerSimplexRule(i/2,3);
         return ir;
   }
}

// Integration rules for reference cube
IntegrationRule *IntegrationRules::CubeIntegrationRule(int Order)
{
   int i = (Order / 2) * 2 + 1;   // Get closest odd # >= Order

   if (!HaveIntRule(SegmentIntRules, i))
   {
      SegmentIntegrationRule(i);
   }
   AllocIntRule(CubeIntRules, i);
   CubeIntRules[i-1] =
      CubeIntRules[i] =
         new IntegrationRule(*SegmentIntRules[i], *SegmentIntRules[i],
                             *SegmentIntRules[i]);
   return CubeIntRules[i];
}

}
