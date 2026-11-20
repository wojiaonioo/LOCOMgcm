#include "COST_OPTIONS.h"
#include "CS32_COST_WINDOW.h"

      SUBROUTINE COST_TILE( myTime, myIter, myThid )
C     Completed-step face-1 U-section transport, averaged in Sv.
C     TO face=0, i_face=16, j=0:31 maps to MIT face=1, i=17, j=1:32.
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "GRID.h"
#include "W2_EXCH2_SIZE.h"
#include "W2_EXCH2_TOPOLOGY.h"
#include "cost.h"

      _RL myTime
      INTEGER myIter, myThid
      INTEGER bi, bj, tile, i, j, k, jFace
      _RL sampleSv, inverseCount

      IF ( myIter.GE.CS32_COST_FIRST_ITER .AND.
     &     myIter.LE.CS32_COST_LAST_ITER ) THEN
        inverseCount = 1. _d 0 /
     &       ( CS32_COST_LAST_ITER - CS32_COST_FIRST_ITER + 1 )
        DO bj=myByLo(myThid),myByHi(myThid)
          DO bi=myBxLo(myThid),myBxHi(myThid)
            tile = W2_myTileList(bi,bj)
            i = 17 - exch2_tBasex(tile)
            IF ( exch2_myFace(tile).EQ.1 .AND.
     &           i.GE.1 .AND. i.LE.sNx ) THEN
              sampleSv = 0. _d 0
              DO k=1,Nr
                DO j=1,sNy
                  jFace = exch2_tBasey(tile) + j
                  IF ( jFace.GE.1 .AND. jFace.LE.32 ) THEN
                    sampleSv = sampleSv + uVel(i,j,k,bi,bj)
     &                   * hFacW(i,j,k,bi,bj) * dyG(i,j,bi,bj)
     &                   * drF(k)
                  ENDIF
                ENDDO
              ENDDO
              objf_test(bi,bj) = objf_test(bi,bj)
     &             + sampleSv * 1. _d -6 * inverseCount
            ENDIF
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
