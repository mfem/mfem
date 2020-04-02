#ifndef HYPSYS_TOOLS
#define HYPSYS_TOOLS

#include <fstream>
#include <iostream>
#include "../../../mfem.hpp"

using namespace std;
using namespace mfem;

const IntegrationRule* GetElementIntegrationRule(FiniteElementSpace *fes, bool NodalQuadRule = false);

// Appropriate quadrature rule for faces, conforming with DGTraceIntegrator.
const IntegrationRule *GetFaceIntegrationRule(FiniteElementSpace *fes, bool NodalQuadRule = false);

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    string ProblemName, GridFunction &gf, string valuerange, bool vec = false);

#endif
