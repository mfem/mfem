#ifndef __MESH_MORPHING_OPT_SOLVER_HPP__
#define __MESH_MORPHING_OPT_SOLVER_HPP__

#include <mfem.hpp>

//#include <memory>

namespace mfem
{

class MeshMorphingSolver
{
protected:
    mfem::ParMesh* pmesh_;
    int quadOrder_;         // Order of the quadrature rule
    int targetId_;          // Target type
    int metricId_;          // Metric type
    int newtonIter_;        // Number of Newton iterations
    bool moveBnd_;          // Move Boundary nodes or not
    int verbosityLevel_;    // Verbose output
    double surfaceFit_;     // Weight of the surface fitting term
    mutable mfem::Array<bool> marker_; // identify elements to be morphed
    mutable mfem::ParGridFunction markerGF;
    mutable double initTMOPEnergy;
    mutable double finalTMOPEnergy;
    mutable double interfaceFittingErrorAvg;
    mutable double interfaceFittingErrorMax;
    //Metric ID and Newton Iterations for Relaxation sweep
    mutable int metricIDRelax_ = -1; // will default to 1 in 2D, 301 in 3D
    mutable int newtonIterRelax_ = 3;
    mutable bool relax = false;

public:
    // Constructor that specifies a level-set function that must be used to
    // setup the marker
    MeshMorphingSolver(
        mfem::ParMesh* pmesh,
        const int quadOrder,
        const int targetId,
        const int metricId,
        const int newtonIter,
        const bool moveBnd,
        const int verbosityLevel,
        const double surfaceFit);

    ~MeshMorphingSolver() {}

    // Get marker grid function
    void GetMarkerGF(mfem::ParGridFunction &morphMarker)
    {
        morphMarker = markerGF;
    }

    /// get the solution
    /// When relax = true, a few Newton iterations are done without
    /// interface fitting. This is important for cases where we want
    /// to fit the same mesh to different level-sets.
    int Solve(
        mfem::ParGridFunction & LSF_OptStep,
        bool relax = false);

    int Solve(
        mfem::ParGridFunction &ls_0,
        mfem::ParGridFunction & nodeDisp,
        bool relax = false);

    double GetInitTMOPEnergy() { return initTMOPEnergy; }
    double GetFinalTMOPEnergy() { return finalTMOPEnergy; }
    double GetInterfaceFittingErrorAvg() { return interfaceFittingErrorAvg; }
    double GetInterfaceFittingErrorMax() { return interfaceFittingErrorMax; }

    /// Set options for relaxation sweep using TMOP integrator.
    void SetRelaxationParameters(int metricID, int newtonIter)
    {
        metricIDRelax_ = metricID;
        newtonIterRelax_ = newtonIter;
    }
};

}

#endif
