#ifndef MTOP_COEFFICIENTS_HPP
#define MTOP_COEFFICIENTS_HPP

#include "mfem.hpp"
#include <random>

namespace mfem {


/// Generates a coefficient with value zero or one.
/// The value is zero if the point is within a ball
/// with radius r and randomly generated center within
/// the boundig box of the mesh. The value is one otherwise.
/// If the mesh is not rectangular one should check
/// if the generated sample is within the mesh.
class RandShootingCoefficient:public Coefficient
{
public:
    RandShootingCoefficient(Mesh* mesh_, double r):udist(0,1)
    {
        mesh=mesh_;
        mesh->GetBoundingBox(cmin,cmax);
        radius=r;

        center.SetSize(cmin.Size());
        center=0.0;

        Sample();
    }

    virtual
    ~RandShootingCoefficient()
    {

    }

    /// Evaluates the coefficient
    virtual
    double Eval(ElementTransformation& T, const IntegrationPoint& ip);

    /// Generates a new random center for the ball.
    void Sample();

private:

    double radius;
    Vector center;
    Vector cmin;
    Vector cmax;

    Vector tmpv;

    Mesh* mesh;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> udist;

};



}

#endif
