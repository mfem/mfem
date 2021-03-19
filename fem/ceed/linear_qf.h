
#ifndef QCOARSEN_LINEAR_H
#define QCOARSEN_LINEAR_H

struct LinearQFunctionContext { 
  CeedInt dim;
  CeedInt ncomp; // beginnning to think this is only thing that matters
  CeedInt layout[3];
};

// the idea here is the context has whatever comes out
// of CeedOperatorLinearAssembleQFunction, and then we can
// automatically apply it here
CEED_QFUNCTION(qcoarsen_linearfunc)(void *ctx, const CeedInt Q,
                                    const CeedScalar *const *in,
                                    CeedScalar *const *out) {
  const CeedScalar *ug = in[0];
  const CeedScalar *linear_data = in[1];
  CeedScalar *vg = out[0];

  struct LinearQFunctionContext *context = (struct LinearQFunctionContext *) ctx;

  const int ncomp = context->ncomp;
  // const int elemsize = context->layout[1];
  // const int lsize = context->layout[2];

  switch (ncomp) {
  case 1:
    for (CeedInt i=0; i<Q; i++) {
      vg[i] = ug[i] * linear_data[i];
    }
    break;
  case 4:
    for (CeedInt i=0; i<Q; i++) {
      const CeedScalar ug0 = ug[i+Q*0];
      const CeedScalar ug1 = ug[i+Q*1];
      vg[i+Q*0] = linear_data[i+Q*0]*ug0 + linear_data[i+Q*2]*ug1;
      vg[i+Q*1] = linear_data[i+Q*1]*ug0 + linear_data[i+Q*3]*ug1;
    }
    break;
  case 9:
    for (CeedInt i=0; i<Q; i++) {
      const CeedScalar ug0 = ug[i+Q*0];
      const CeedScalar ug1 = ug[i+Q*1];
      const CeedScalar ug2 = ug[i+Q*2];
      vg[i+Q*0] = linear_data[i+Q*0]*ug0 + linear_data[i+Q*3]*ug1 + linear_data[i+Q*6]*ug2;
      vg[i+Q*1] = linear_data[i+Q*1]*ug0 + linear_data[i+Q*4]*ug1 + linear_data[i+Q*7]*ug2;
      vg[i+Q*2] = linear_data[i+Q*2]*ug0 + linear_data[i+Q*5]*ug1 + linear_data[i+Q*8]*ug2;
    }
    break;
  }

  return 0;
}

#endif
