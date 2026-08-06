/*
 * SuperLU_DIST / DistributedSuperLU reference __saxpy (GNU f2c-era BLAS name).
 * Intel MKL and modern libblas expose saxpy_ / cblas_saxpy only — forward to Fortran saxpy_.
 */
#ifdef __cplusplus
extern "C" {
#endif

extern void saxpy_(const int *n, const float *sa, const float *sx, const int *incx, float *sy,
                   const int *incy);

void __saxpy(const int *n, const float *sa, const float *sx, const int *incx, float *sy,
             const int *incy)
{
    saxpy_(n, sa, sx, incx, sy, incy);
}

#ifdef __cplusplus
}
#endif
