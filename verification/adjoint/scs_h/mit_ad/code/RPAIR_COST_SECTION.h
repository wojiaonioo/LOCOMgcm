#ifndef RPAIR_COST_SECTION_H
#define RPAIR_COST_SECTION_H

C     Section of verification/adjoint/scs_h/configs/pair_expl.yaml,
C     observations.section_transport_u {i_face: 82, j0: 99, j1: 112}
C     (LOCOMgcm, 0-based, both ends inclusive).  LOCOMgcm u face i is
C     MITgcm uVel(i+1), cell row j is MITgcm j+1, so the section is
C     uVel(83, 100..113, 1..Nr) in MITgcm 1-based global indices.
#define RPAIR_COST_IFACE 83
#define RPAIR_COST_J0 100
#define RPAIR_COST_J1 113

#if RPAIR_COST_J1 < RPAIR_COST_J0
#error rpair cost section is empty
#endif

#endif /* RPAIR_COST_SECTION_H */
