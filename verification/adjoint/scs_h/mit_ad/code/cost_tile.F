#include "COST_OPTIONS.h"
#include "RPAIR_COST_SECTION.h"

      SUBROUTINE COST_TILE( myTime, myIter, myThid )
C     completed-step U-section transport, window
C     arithmetic mean, in Sv:
C     J = (1/nTimeSteps) SUM_n SUM_{j,k} uVel(iFace,j,k)*hFacW*dyG*drF*1e-6
C     COST_TILE is called from model/src/forward_step.F:1163, after
C     :807-808 advanced myIter/myTime to the completed step and after the
C     field exchanges of :1093, and just before DO_THE_MODEL_IO (:1182), so
C     uVel is the field written to the U snapshot of step myIter, i.e.
C     the u that LOCOMgcm observes after each step() returns.
C     Lat-lon grid without exch2: a global (i,j) belongs to exactly one tile
C     interior, so the section is neither double counted nor missed.
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "GRID.h"
#include "cost.h"

      _RL myTime
      INTEGER myIter, myThid
      INTEGER bi, bj, i, j, k, iGlobal, jGlobal
      _RL sampleSv, inverseCount

      IF ( myIter.GT.nIter0 .AND. myIter.LE.nEndIter .AND.
     &     nTimeSteps.GT.0 ) THEN
        inverseCount = 1. _d 0 / DBLE(nTimeSteps)
        DO bj=myByLo(myThid),myByHi(myThid)
          DO bi=myBxLo(myThid),myBxHi(myThid)
            sampleSv = 0. _d 0
            DO k=1,Nr
              DO j=1,sNy
                jGlobal = myYGlobalLo-1+(bj-1)*sNy+j
                DO i=1,sNx
                  iGlobal = myXGlobalLo-1+(bi-1)*sNx+i
                  IF ( iGlobal.EQ.RPAIR_COST_IFACE .AND.
     &                 jGlobal.GE.RPAIR_COST_J0 .AND.
     &                 jGlobal.LE.RPAIR_COST_J1 ) THEN
                    sampleSv = sampleSv + uVel(i,j,k,bi,bj)
     &                   * hFacW(i,j,k,bi,bj) * dyG(i,j,bi,bj)
     &                   * drF(k)
                  ENDIF
                ENDDO
              ENDDO
            ENDDO
            objf_test(bi,bj) = objf_test(bi,bj)
     &           + sampleSv * 1. _d -6 * inverseCount
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
