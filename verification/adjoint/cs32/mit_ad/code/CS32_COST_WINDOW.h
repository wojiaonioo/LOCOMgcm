#ifndef CS32_COST_WINDOW_H
#define CS32_COST_WINDOW_H

C     Inclusive completed-step iteration numbers; restart nIter0=72.
#ifndef CS32_COST_FIRST_ITER
#define CS32_COST_FIRST_ITER 1
#endif
#ifndef CS32_COST_LAST_ITER
#define CS32_COST_LAST_ITER 3600
#endif
#if CS32_COST_FIRST_ITER < 1
#error CS32_COST_FIRST_ITER must be positive
#endif
#if CS32_COST_LAST_ITER < CS32_COST_FIRST_ITER
#error CS32 cost interval is empty
#endif

#endif
