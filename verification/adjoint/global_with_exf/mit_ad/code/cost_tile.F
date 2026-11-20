#include "COST_OPTIONS.h"

      SUBROUTINE COST_TILE( myTime, myIter, myThid )
C     Terminal native regional heat content in 1e22 J, zero Celsius datum.
C     Python j=[20,30), i=[67,90); all levels, rA*drF*hFacC volume.
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "GRID.h"
#include "cost.h"
      _RL myTime, heat
      INTEGER myIter, myThid, bi, bj, i, j, k, iGlobal, jGlobal

      IF ( myIter.EQ.nIter0+nTimeSteps ) THEN
        DO bj=myByLo(myThid),myByHi(myThid)
          DO bi=myBxLo(myThid),myBxHi(myThid)
            heat = 0. _d 0
            DO k=1,Nr
              DO j=1,sNy
                jGlobal = myYGlobalLo-1+(bj-1)*sNy+j
                DO i=1,sNx
                  iGlobal = myXGlobalLo-1+(bi-1)*sNx+i
                  IF ( iGlobal.GE.68 .AND. iGlobal.LE.90 .AND.
     &                 jGlobal.GE.21 .AND. jGlobal.LE.30 ) THEN
                    heat = heat + theta(i,j,k,bi,bj)
     &                   * rA(i,j,bi,bj)*drF(k)*hFacC(i,j,k,bi,bj)
                  ENDIF
                ENDDO
              ENDDO
            ENDDO
            objf_test(bi,bj) = objf_test(bi,bj)
     &           + heat * (1035. _d 0 * 3994. _d 0 / 1. _d 22)
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
