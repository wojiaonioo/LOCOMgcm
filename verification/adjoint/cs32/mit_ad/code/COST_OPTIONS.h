#ifndef COST_OPTIONS_H
#define COST_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

C     Use objf_test and the existing COST_FINAL global scalar reduction.
#define ALLOW_COST_TEST
#undef ALLOW_COST_TSQUARED
#undef ALLOW_COST_STATE_FINAL
#undef ALLOW_COST_VECTOR
#undef ALLOW_COST_ATLANTIC_HEAT
#undef ALLOW_COST_ATLANTIC_HEAT_DOMASS
#undef ALLOW_COST_DEPTH
#undef ALLOW_COST_TRACER
#undef ALLOW_COST_TEMP
#undef ALLOW_COST_HFLUXM
#undef ALLOW_DIC_COST
#undef ALLOW_THSICE_COST_TEST
#undef ALLOW_COST_SHELFICE

#if !defined(ALLOW_COST) || !defined(ALLOW_EXCH2)
#error CS32 section cost requires cost and exch2 packages
#endif
#endif
