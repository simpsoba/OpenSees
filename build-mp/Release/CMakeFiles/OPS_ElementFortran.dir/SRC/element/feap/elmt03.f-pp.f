# 1 "/home/garaujor/OpenSees-CUDA/SRC/element/feap/elmt03.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/garaujor/OpenSees-CUDA/SRC/element/feap/elmt03.f"
      subroutine ELMT03(d,ul,xl,ix,tl,s,p,ndf,ndm,nst,isw)
      implicit none

      real*8  d(*), ul(*), xl(*), tl(*), s(*), p(*)
      integer ix(*), ndf, ndm, nst, isw

      if(isw.gt.0) write(*,1000)
1000  format('WARNING: elmt03()-dummy subroutine, no elmt03() linked')
      end

