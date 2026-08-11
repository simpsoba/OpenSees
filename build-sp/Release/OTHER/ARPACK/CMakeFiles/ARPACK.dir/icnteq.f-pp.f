# 1 "/home/garaujor/OpenSees-CUDA/OTHER/ARPACK/icnteq.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/garaujor/OpenSees-CUDA/OTHER/ARPACK/icnteq.f"
c
c-----------------------------------------------------------------------
c
c     Count the number of elements equal to a specified integer value.
c
      integer function icnteq (n, array, value)
c
      integer    n, value
      integer    array(*)
c
      k = 0
      do 10 i = 1, n
         if (array(i) .eq. value) k = k + 1
   10 continue
      icnteq = k
c
      return
      end
