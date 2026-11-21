#ifndef COST_OPTIONS_H
#define COST_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

C     upstream pkg/cost/COST_OPTIONS.h with one
C     change, ALLOW_COST_TEST defined, so that objf_test (accumulated by
C     the COST_TILE override) enters COST_FINAL (cost_final.F:163-191).
#ifdef ALLOW_COST
#undef ALLOW_COST_STATE_FINAL
#undef ALLOW_COST_VECTOR
#undef ALLOW_COST_ATLANTIC_HEAT
#undef ALLOW_COST_ATLANTIC_HEAT_DOMASS
#define ALLOW_COST_TEST
#undef ALLOW_COST_TSQUARED
#undef ALLOW_COST_TRACER
#undef ALLOW_COST_TEMP
#undef ALLOW_COST_HFLUXM
#undef ALLOW_DIC_COST
#undef ALLOW_THSICE_COST_TEST
#undef ALLOW_COST_SHELFICE
#undef ALLOW_COST_STREAMICE
#else
#error rpair section cost requires the cost package
#endif /* ALLOW_COST */
#endif /* COST_OPTIONS_H */
