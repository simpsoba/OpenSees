# 1 "/home/garaujor/OpenSees-CUDA/OTHER/ARPACK/iset.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/garaujor/OpenSees-CUDA/OTHER/ARPACK/iset.f"
c
c-----------------------------------------------------------------------
c
c     Only work with increment equal to 1 right now.
c
      subroutine iset (n, value, array, inc)
c
      integer    n, value, inc
      integer    array(*)
c
      do 10 i = 1, n
         array(i) = value
   10 continue
c
      return
      end
