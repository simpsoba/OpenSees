/* ****************************************************************** **
** Env-gated per-step wall timing for CudaExplicitAlpha / DistCuDSS. **
** OPS_CUDA_STEP_TIMING=1 enables; OPS_CUDA_STEP_TIMING_FILE sets CSV. **
** ****************************************************************** */

#ifndef CudaStepTiming_h
#define CudaStepTiming_h

namespace OpsCudaStepTiming {

/** True when OPS_CUDA_STEP_TIMING is set to a non-empty, non-"0" value. */
bool enabled();

/** Start a new step (rank 0 only should call; resets phase buckets). */
void beginStep();

/** Accumulate wall seconds into a named phase bucket. */
void add(const char *phase, double seconds);

/** Append one CSV line for the current step (rank 0). No-op if disabled. */
void endStep();

} // namespace OpsCudaStepTiming

#endif
