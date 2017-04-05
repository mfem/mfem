#include "mfem.hpp"

class MaglevProblemGeometry
{
   public:
   MaglevProblemGeometry(double conductor_bottom, double conductor_top,
                         double conductor_vx, double conductor_mu, double conductor_sigma,
                         double ha_left, double ha_right, double ha_bottom, double ha_top,
                         int ha_num_magnets, double ha_vx, double ha_mu, double ha_sigma, double ha_mag,
                         double air_mu,  double air_sigma) :
      m_conductor_bottom(conductor_bottom),
      m_conductor_top(conductor_top),
      m_conductor_vx(conductor_vx),
      m_conductor_mu(conductor_mu),
      m_conductor_sigma(conductor_sigma),
      m_ha_left(ha_left),
      m_ha_right(ha_right),
      m_ha_bottom(ha_bottom),
      m_ha_top(ha_top),
      m_ha_num_magnets(ha_num_magnets),
      m_ha_vx(ha_vx),
      m_ha_mu(ha_mu),
      m_ha_sigma(ha_sigma),
      m_ha_mag(ha_mag),
      m_air_mu(air_mu),
      m_air_sigma(air_sigma) {m_magnet_size = (m_ha_right - m_ha_left) / double(m_ha_num_magnets);}

   inline int getMagnetNumber(const mfem::Vector &x);
   inline void getMuSigmaV(const mfem::Vector &x, mfem::Vector &out);
   inline void getMPerp(const mfem::Vector &x, mfem::Vector &out);

   private:
   double m_conductor_bottom;
   double m_conductor_top;
   double m_conductor_vx;
   double m_conductor_mu;   
   double m_conductor_sigma;
   double m_ha_left;
   double m_ha_right;
   double m_ha_bottom;
   double m_ha_top;
   int m_ha_num_magnets;
   double m_ha_vx;
   double m_ha_mu;
   double m_ha_sigma;
   double m_ha_mag;
   double m_air_mu;
   double m_air_sigma;

   double m_magnet_size;
};


class ConvectionCoeff : public mfem::VectorCoefficient
{
   public:
   ConvectionCoeff(MaglevProblemGeometry *problem) :
      VectorCoefficient(2),
      m_problem(problem) {}

   virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                     const mfem::IntegrationPoint &ip);

   private:
   MaglevProblemGeometry *m_problem;

};


class MagnetizationCoeff : public mfem::VectorCoefficient
{
   public:
   MagnetizationCoeff(MaglevProblemGeometry *problem) :
      VectorCoefficient(2),
      m_problem(problem) {}

   virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                     const mfem::IntegrationPoint &ip);

   private:
   MaglevProblemGeometry *m_problem;

};



inline int MaglevProblemGeometry::getMagnetNumber(const mfem::Vector &x)
{
   return int((x[0] - m_ha_left) / m_magnet_size);
}


inline void MaglevProblemGeometry::getMuSigmaV(const mfem::Vector &x, mfem::Vector &out)
{
   out = 0.0;
   if (x[1] >= m_conductor_bottom && x[1] <= m_conductor_top)
   {
      out[0] = m_conductor_mu*m_conductor_sigma*m_conductor_vx;
   }
}


inline void MaglevProblemGeometry::getMPerp(const mfem::Vector &x, mfem::Vector &out)
{
   out = 0.0;
   if (x[0] >= m_ha_left && x[0] <= m_ha_right && x[1] >= m_ha_bottom && x[1] <= m_ha_top)
   {
      int mag = getMagnetNumber(x) % 4;
      double a = 0.0;
      double b = 0.0;
      if (mag == 0)
      {
         a = -m_ha_mag;
      }
      else if (mag == 1)
      {
         b = m_ha_mag;
      }
      else if (mag == 2)
      {
         a = m_ha_mag;
      }
      else
      {
         b = -m_ha_mag;
      }

      //perpendicular to (a,b)
      out[0] = -b;
      out[1] = a;
   }
}


void ConvectionCoeff::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                           const mfem::IntegrationPoint &ip)
{
   double x[3];
   mfem::Vector transip(x, 3);

   T.Transform(ip, transip);

   V.SetSize(2);
   m_problem->getMuSigmaV(transip, V);
}


void MagnetizationCoeff::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                              const mfem::IntegrationPoint &ip)
{
   double x[3];
   mfem::Vector transip(x, 3);

   T.Transform(ip, transip);

   V.SetSize(2);
   m_problem->getMPerp(transip, V);
}