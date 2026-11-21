// Distinct-basename translation-unit shim.  torch cpp_extension derives the
// object file name from the source basename, so compiling several files all
// named kernel.cu into one build directory makes ninja refuse ("multiple
// rules generate" the same object file).  Including each package's
// kernel in a uniquely named TU fixes that without touching the legacy
// package (read-only).  Nested quoted includes still resolve against the
// included file's own directory.
#include "../_kpp_statekpp_fused/kernel.cu"
