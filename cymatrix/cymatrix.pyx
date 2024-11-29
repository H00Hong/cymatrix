# distutils: language=c++
# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=True
from libcpp.vector cimport vector as cvector
from libcpp.algorithm cimport sort as _sort
from libc cimport math
from cpython cimport array as parray
import array as parray

ctypedef parray.array Parray
ctypedef fused numberType:
    double # float64
    long # int32
ctypedef fused numberType2:
    double # float64
    long # int32

cdef double dInf = math.INFINITY
cdef double dNan = math.NAN
cdef double dE = 2.71828182845904523536
cdef double dPI = 3.14159265358979323846
cdef double d2PI = 6.28318530717958647692
cdef double dPI_2 = 1.57079632679489661923
cdef double d1_PI = 0.31830988618379067154
cdef double d2_PI = 0.63661977236758134308
cdef double d1_2PI = 0.15915494309189533577
cdef double dSqrt2 = 1.41421356237309504880
cdef double dSqrt3 = 1.73205080756887729352
cdef double dLn2 = 0.69314718055994530942
cdef double dLn10 = 2.30258509299404568402
cdef double dLog2e = 1.44269504088896340736
cdef double dLog10e = 0.43429448190325182765
cdef double dSqrt1_2 = 0.70710678118654752440


cdef class cyMatrix:
    cdef:
        Py_ssize_t naxis0, naxis1, size, ndim
        cvector[double] dDat
        # cvector[long] lDat
        # bint bDType

    def __cinit__(self, Py_ssize_t nrows, Py_ssize_t ncols, Py_ssize_t ndim):
        self.ndim = ndim
        if ndim == 0:
            self.naxis0 = 1
            self.naxis1 = 1
        elif ndim == 1:
            self.naxis0 = 1
            self.naxis1 = ncols if nrows == 1 else nrows
        else:
            self.naxis0 = nrows
            self.naxis1 = ncols
        self.size = self.naxis0 * self.naxis1
        self.dDat.resize(self.size)
    # cdef void astype(self, bint dtype_s):
    #     if self.bDType ^ dtype_s:
    #         self.bDType = dtype_s
    #         if dtype_s:
    #             self.dDat.reserve(self.size)
    #             for i in range(self.size):
    #                 self.dDat.push_back(<double>self.lDat[i])
    #             self.lDat = cvector[long]()
    #         else:
    #             self.lDat.reserve(self.size)
    #             for i in range(self.size):
    #                 self.lDat.push_back(<long>self.dDat[i])
    #             self.dDat = cvector[double]()
    cdef void reshape(self, Py_ssize_t ndim, Py_ssize_t nrows, Py_ssize_t ncols):
        if ndim == 1:
            self.naxis0 = 1
            self.naxis1 = self.size
        else:
            assert nrows * ncols == self.size
            self.naxis0 = nrows
            self.naxis1 = ncols
        self.ndim = ndim

    cdef void fill(self, numberType value):
        cdef Py_ssize_t i
        cdef double dNum = <double>value
        for i in range(self.size):
            self.dDat[i] = dNum
    
    cdef double get_item(self, Py_ssize_t nrow, Py_ssize_t ncol):
        return self.dDat[nrow * self.naxis1 + ncol]


cpdef cyMatrix copy(cyMatrix m):
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    res.dDat.assign(m.dDat.begin(), m.dDat.end())
    return res

cdef void checkNumber(cyMatrix m):
    if m.ndim == 0:
        raise ValueError('检测为数字 checkNumber: m.ndim == 0')

cdef Py_ssize_t check_get(Py_ssize_t n, Py_ssize_t m):
    if n < 0:
        n += m
    return n

def setitem(cyMatrix m, numberType value, Py_ssize_t nrow, Py_ssize_t ncol):
    m.dDat[nrow * m.naxis1 + ncol] = <double>value

def setitems(cyMatrix m, cyMatrix o, Py_ssize_t nrow, Py_ssize_t ncol, Py_ssize_t rstep, Py_ssize_t cstep):
    # 将o的数据复制到m里面, 起始位置为nrow,ncol, 步长为rstep,cstep
    cdef bint m_nd = m.ndim == 1, o_nd = o.ndim < 2
    cdef Py_ssize_t i, j, n1, n2 = 0, IR, IC, m_naxis0 = m.naxis0, m_naxis1 = m.naxis1
    cdef Py_ssize_t o_naxis0 = o.naxis0, o_naxis1 = o.naxis1, m_next_row, o_next_row
    if o.ndim > m.ndim:
        raise ValueError('超出索引 setitems: o.ndim > m.ndim')
    nrow = check_get(nrow, m_naxis0)
    ncol = check_get(ncol, m_naxis1)
    if o_nd:
        if m_nd:
            IR = o.size if o.size < m.size else m.size
            j = ncol
            for i in range(IR):
                m.dDat[j] = o.dDat[i]
                j += cstep
        else:
            IR = o_naxis1 if o_naxis1 < m_naxis1 else m_naxis1
            n1 = nrow * m_naxis0 + ncol
            for i in range(IR):
                m.dDat[n1] = o.dDat[i]
                n1 += cstep
    else:
        if not m_nd:
            if o_naxis1 * cstep + ncol <= m_naxis1:
                IR = o_naxis1
            else:
                IR = <Py_ssize_t>math.ceil(<double>(m_naxis1 - ncol) / cstep)
            m_next_row = rstep * m_naxis1 - cstep * IR  # (rstep-1)*m_naxis1 + ncol + m_naxis1 - cstep * (IR-1) - ncol - cstep 
            o_next_row = o_naxis1 - IR
            IC = o_naxis0 if o_naxis0 * rstep + nrow <= m_naxis0 else <Py_ssize_t>math.ceil(<double>(m_naxis0 - nrow) / rstep)
            n1 = nrow * m_naxis1 + ncol
            for i in range(IC):
                for j in range(IR):
                    m.dDat[n1] = o.dDat[n2]
                    n2 += 1
                    n1 += cstep
                n1 += m_next_row
                n2 += o_next_row
        else:
            raise ValueError('超出索引 setitems: 将二维赋值进一维')


def getitem(cyMatrix m, Py_ssize_t nrow, Py_ssize_t ncol):
    # 从矩阵m中获取数据
    cdef cyMatrix res = cyMatrix(1, 1, 0)
    res.dDat[0] = m.dDat[nrow * m.naxis1 + ncol]
    return res

def getitems(cyMatrix m, Py_ssize_t ndim, slice n1, slice n2):
    # 提取矩阵m中的块
    cdef Py_ssize_t n1_start, n1_stop, n1_step, n2_start, n2_stop, n2_step
    cdef Py_ssize_t n1lim, n2lim, i1, i2, k, j = 0, ste1
    n1_start = 0 if n1.start is None else n1.start
    n1_step = 1 if n1.step is None else n1.step
    n1_stop = m.naxis0 if n1.stop is None else n1.stop
    n1_start = check_get(n1_start, m.naxis0)
    n1_stop = check_get(n1_stop, m.naxis0)
    if n1_stop <= n1_start + n1_step:
        n1_stop = n1_start + n1_step
    n1lim = (n1_stop - n1_start) / n1_step
    if n1_start + n1_step * n1lim < n1_stop:
        n1lim += 1

    n2_start = 0 if n2.start is None else n2.start
    n2_step = 1 if n2.step is None else n2.step
    n2_stop = m.naxis1 if n2.stop is None else n2.stop
    n2_start = check_get(n2_start, m.naxis1)
    n2_stop = check_get(n2_stop, m.naxis1)
    if n2_stop <= n2_start + n2_step:
        n2_stop = n2_start + n2_step
    n2lim = (n2_stop - n2_start) / n2_step
    if n2_start + n2_step * n2lim < n2_stop:
        n2lim += 1
    
    cdef cyMatrix res = cyMatrix(n1lim, n2lim, ndim)
    k = n1_start * m.naxis1 + n2_start
    if n1lim == 1: # 行向量 or number
        for i1 in range(res.size):
            res.dDat[i1] = m.dDat[k]
            k += n2_step
    elif n2lim == 1: # 列向量
        ste1 = m.naxis1 * n1_step
        for i1 in range(res.size):
            res.dDat[i1] = m.dDat[k]
            k += ste1
    else: # 矩阵
        for i1 in range(res.naxis0):
            k = (n1_start + i1) * m.naxis1 * n1_step + n2_start
            for i2 in range(res.naxis1):
                res.dDat[j] = m.dDat[k]
                j += n2_step
                k += n2_step
    return res

def transpos(cyMatrix m):
    cdef cyMatrix res = cyMatrix(m.naxis1, m.naxis0, m.ndim)
    cdef Py_ssize_t i1, i2, j, k = 0
    for i1 in range(m.naxis0):
        j = i1
        for i2 in range(m.naxis1):
            res.dDat[j] = m.dDat[k]
            k += 1
            j += m.naxis0
    return res


cpdef cyMatrix matmul(cyMatrix m1, cyMatrix m2): # inplace
    cdef Py_ssize_t i1, i2, k1, k2, k3 = 0
    cdef double dNum
    cdef bint m1_nd = m1.ndim == 1, m2_nd = m2.ndim == 1
    cdef cyMatrix res
    checkNumber(m1)
    checkNumber(m2)
    if m1_nd and m2_nd: # 向量
        assert m1.size == m2.size
        res = cyMatrix(1, 1, 0)
        dNum = 0.
        for i1 in range(m1.size):
            dNum += m1.dDat[i1] * m2.dDat[i1]
        res.dDat[0] = dNum
    elif not m1_nd and not m2_nd: # 矩阵
        assert m1.naxis1 == m2.naxis0
        res = cyMatrix(m1.naxis0, m2.naxis1, 2)
        for i1 in range(res.naxis0):
            for i2 in range(res.naxis1):
                dNum = 0.
                k1 = i1 * m1.naxis1
                k2 = i2
                for i3 in range(m1.naxis1):
                    dNum += m1.dDat[k1] * m2.dDat[k2]
                    k1 += 1
                    k2 += res.naxis1
                res.dDat[k3] = dNum
                k3 += 1
    elif m1_nd: # 向量 @ 矩阵
        assert m1.size == m2.naxis0
        res = cyMatrix(m2.naxis1, 1, 1)
        for i1 in range(res.size):
            dNum = 0.
            k3 = i1
            for i2 in range(m2.naxis0):
                dNum += m1.dDat[i2] * m2.dDat[k3]
                k3 += res.size
            res.dDat[i1] = dNum
    else: # 矩阵 @ 向量
        assert m1.naxis1 == m2.size
        res = cyMatrix(m1.naxis0, 1, 1)
        for i1 in range(res.size):
            dNum = 0.
            for i2 in range(m1.naxis1):
                dNum += m1.dDat[k3] * m2.dDat[i2]
                k3 += 1
            res.dDat[i1] = dNum
    return res


cdef bint is_int(double num):
    cdef double fractpart, intpart
    fractpart = math.modf(num, &intpart)
    return fractpart == 0
cdef bint is_fract(double num):
    cdef double fractpart, intpart
    fractpart = math.modf(num, &intpart)
    return fractpart != 0

cdef cyMatrix iadd_number(cyMatrix m1, double num): # 数字加到矩阵内
    cdef Py_ssize_t i
    for i in range(m1.size):
        m1.dDat[i] += num
    return m1
cdef cyMatrix isub_number(cyMatrix m1, double num, bint isAfter): # 数字减到矩阵内
    cdef Py_ssize_t i
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] -= num
    else:
        for i in range(m1.size):
            m1.dDat[i] = num - m1.dDat[i]
    return m1
cdef cyMatrix imul_number(cyMatrix m1, double num): # 数字乘到矩阵内
    cdef Py_ssize_t i
    for i in range(m1.size):
        m1.dDat[i] *= num
    return m1
cdef cyMatrix itruediv_number(cyMatrix m1, double num, bint isAfter): # 数字除到矩阵内
    cdef Py_ssize_t i
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] /= num
    else:
        for i in range(m1.size):
            m1.dDat[i] = num / m1.dDat[i]
    return m1
cdef cyMatrix imod_number(cyMatrix m1, double num, bint isAfter): # 矩阵取模
    cdef Py_ssize_t i
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] %= num
    else:
        for i in range(m1.size):
            m1.dDat[i] = num % m1.dDat[i]
    return m1
cdef cyMatrix ifloordiv_number(cyMatrix m1, double num, bint isAfter): # 矩阵整除
    cdef Py_ssize_t i
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] //= num
    else:
        for i in range(m1.size):
            m1.dDat[i] = num // m1.dDat[i]
    return m1
cdef cyMatrix ipow_number(cyMatrix m1, double num): # 矩阵取幂
    cdef Py_ssize_t i
    cdef bint is_fract_ = is_fract(num)
    if is_fract_:
        for i in range(m1.size):
            if m1.dDat[i] < 0:
                print('产生虚数')
                m1.dDat[i] = dNan
            else:
                m1.dDat[i] **= num
    else:
        for i in range(m1.size):
            m1.dDat[i] **= num
    return m1
cdef cyMatrix _ipow_number(cyMatrix m1, double num): # 矩阵取幂 无检测
    cdef Py_ssize_t i
    for i in range(m1.size):
        m1.dDat[i] **= num
    return m1

cdef cyMatrix broadcast(cyMatrix ma, cyMatrix mb): # 广播 把ma扩展到mb的形状
    cdef:
        Py_ssize_t i, j, k = 0
        cvector[double] dDat
        double dNum
    if ma.naxis0 == mb.naxis0 and ma.naxis1 == mb.naxis1:
        return ma
    if ma.naxis0 < mb.naxis0:
        if ma.naxis1 != mb.naxis1:
            print(ma.naxis1)
            print(mb.naxis1)
            raise ValueError('广播: 两个矩阵的列数不一致')
        
        dDat.reserve(mb.naxis0 * ma.naxis1)
        for i in range(mb.naxis0):
            if k == ma.size:
                k = 0
            for j in range(ma.naxis1):
                dDat.push_back(ma.dDat[k])
                k += 1
        
        ma.dDat.reserve(ma.naxis0 * mb.naxis1)
        ma.dDat.assign(dDat.begin(), dDat.end())
        ma.naxis0 = mb.naxis0
        ma.size = mb.naxis0 * ma.naxis1

    if ma.naxis1 < mb.naxis1:
        if ma.naxis0 != mb.naxis0:
            raise ValueError('广播: 两个矩阵的行数不一致')
        dDat.reserve(ma.naxis0 * mb.naxis1)
        for i in range(ma.naxis0):
            dNum = ma.dDat[i]
            for j in range(mb.naxis1):
                dDat.push_back(dNum)
        ma.dDat.reserve(ma.naxis0 * mb.naxis1)
        ma.dDat.assign(dDat.begin(), dDat.end())
        ma.naxis1 = mb.naxis1
        ma.size = mb.naxis0 * ma.naxis1
    return ma
cdef void _check_broadcast(cyMatrix m1, cyMatrix m2):
    if m1.ndim == 2 or m2.ndim == 2:
        m1.ndim = 2
        m2.ndim = 2
    broadcast(m1, m2)
    broadcast(m2, m1)

def iadd(cyMatrix m1, cyMatrix m_other): # 矩阵相加
    cdef cyMatrix m2 = copy(m_other)
    if m_other.ndim == 0:
        return iadd_number(m1, m2.dDat[0])
    if m1.ndim == 0:
        return iadd_number(m2, m1.dDat[0])
    cdef Py_ssize_t i
    _check_broadcast(m1, m2)
    for i in range(m1.size):
        m1.dDat[i] += m2.dDat[i]
    return m1
def isub(cyMatrix m1, cyMatrix m_other, bint isAfter): # 矩阵相减
    cdef cyMatrix m2 = copy(m_other)
    if m_other.ndim == 0:
        return isub_number(m1, m2.dDat[0], isAfter)
    if m1.ndim == 0:
        return isub_number(m2, m1.dDat[0], not isAfter)
    cdef Py_ssize_t i
    _check_broadcast(m1, m2)
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] -= m2.dDat[i]
    else:
        for i in range(m1.size):
            m1.dDat[i] = m2.dDat[i] - m1.dDat[i]
    return m1
def imul(cyMatrix m1, cyMatrix m_other): # 矩阵相乘
    cdef cyMatrix m2 = copy(m_other)
    if m_other.ndim == 0:
        return imul_number(m1, m2.dDat[0])
    if m1.ndim == 0:
        return imul_number(m2, m1.dDat[0])
    cdef Py_ssize_t i
    _check_broadcast(m1, m2)
    for i in range(m1.size):
        m1.dDat[i] *= m2.dDat[i]
    return m1
cpdef cyMatrix itruediv(cyMatrix m1, cyMatrix m_other, bint isAfter): # 矩阵相除
    cdef cyMatrix m2 = copy(m_other)
    if m_other.ndim == 0:
        return itruediv_number(m1, m2.dDat[0], isAfter)
    if m1.ndim == 0:
        return itruediv_number(m2, m1.dDat[0], not isAfter)
    cdef Py_ssize_t i
    _check_broadcast(m1, m2)
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] /= m2.dDat[i]
    else:
        for i in range(m1.size):
            m1.dDat[i] = m2.dDat[i] / m1.dDat[i]
    return m1
def imod(cyMatrix m1, cyMatrix m_other, bint isAfter): # 矩阵取模
    cdef cyMatrix m2 = copy(m_other)
    if m_other.ndim == 0:
        return imod_number(m1, m2.dDat[0], isAfter)
    if m1.ndim == 0:
        return imod_number(m2, m1.dDat[0], not isAfter)
    cdef Py_ssize_t i
    _check_broadcast(m1, m2)
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] %= m2.dDat[i]
    else:
        for i in range(m1.size):
            m1.dDat[i] = m2.dDat[i] % m1.dDat[i]
    return m1
def ifloordiv(cyMatrix m1, cyMatrix m_other, bint isAfter): # 矩阵整除
    cdef cyMatrix m2 = copy(m_other)
    if m_other.ndim == 0:
        return ifloordiv_number(m1, m2.dDat[0], isAfter)
    if m1.ndim == 0:
        return ifloordiv_number(m2, m1.dDat[0], not isAfter)
    cdef Py_ssize_t i
    _check_broadcast(m1, m2)
    if isAfter:
        for i in range(m1.size):
            m1.dDat[i] //= m2.dDat[i]
    else:
        for i in range(m1.size):
            m1.dDat[i] = m2.dDat[i] // m1.dDat[i]
    return m1
cpdef cyMatrix ipow(cyMatrix m1, cyMatrix m_other): # 矩阵取幂
    if m_other.ndim == 0:
        return ipow_number(m1, m_other.dDat[0])
    cdef Py_ssize_t i
    cdef cyMatrix m2 = copy(m_other)
    cdef double m1num, m2num
    _check_broadcast(m1, m2)
    for i in range(m1.size):
        m2num = m2.dDat[i]
        if m1.dDat[i] < 0 and is_fract(m2num):
            m1.dDat[i] = dNan
        else:
            m1.dDat[i] **= m2num
    return m1

# cdef cvector[cvector[bint]] _compare(cyMatrix m1, cyMatrix m2, bint* cmp_func):
#     cdef Py_ssize_t i, j, k=0
#     cdef cvector[bint] bres
#     cdef cvector[cvector[bint]] bres_
#     bres.reserve(m1.size)
    
#     if m2.ndim == 0:
#         bres_.reserve(m1.naxis0)
#         for i in range(m1.naxis0):
#             bres.resize(m1.naxis1)
#             for j in range(m1.naxis1):
#                 bres[j] = cmp_func(m1.dDat[k], m2.dDat[0])
#                 k += 1
#             bres_.push_back(bres)
#     elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
#         bres_.reserve(m1.naxis0)
#         for i in range(m1.naxis0):
#             bres.resize(m1.naxis1)
#             for j in range(m1.naxis1):
#                 bres[j] = cmp_func(m1.dDat[k], m2.dDat[k])
#                 k += 1
#             bres_.push_back(bres)
#     else:
#         raise ValueError(f"{cmp_func.__name__} : shape不匹配")
    
#     # if m1.ndim == 0:
#     #     return bres[0]
#     # if m1.ndim == 1:
#     #     return bres_[0]
#     return bres_


def lt(cyMatrix m1, cyMatrix m2): # <  retern list
    cdef Py_ssize_t i, j, k=0
    cdef cvector[bint] bres
    cdef cvector[cvector[bint]] bres_
    bres.reserve(m1.size)
    if m2.ndim == 0:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] < m2.dDat[0]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] < m2.dDat[k]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    else:
        raise ValueError('< : shape不匹配')
    
def le(cyMatrix m1, cyMatrix m2): # <=
    cdef Py_ssize_t i, j, k=0
    cdef cvector[bint] bres
    cdef cvector[cvector[bint]] bres_
    bres.reserve(m1.size)
    if m2.ndim == 0:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] <= m2.dDat[0]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] <= m2.dDat[k]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    else:
        raise ValueError('<= : shape不匹配')

def eq(cyMatrix m1, cyMatrix m2): # ==
    cdef Py_ssize_t i, j, k=0
    cdef cvector[bint] bres
    cdef cvector[cvector[bint]] bres_
    bres.reserve(m1.size)
    if m2.ndim == 0:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] == m2.dDat[0]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] == m2.dDat[k]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    else:
        raise ValueError('== : shape不匹配')

def ne(cyMatrix m1, cyMatrix m2): # !=
    cdef Py_ssize_t i, j, k=0
    cdef cvector[bint] bres
    cdef cvector[cvector[bint]] bres_
    bres.reserve(m1.size)
    if m2.ndim == 0:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] != m2.dDat[0]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] != m2.dDat[k]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    else:
        raise ValueError('!= : shape不匹配')

def gt(cyMatrix m1, cyMatrix m2): # >
    cdef Py_ssize_t i, j, k=0
    cdef cvector[bint] bres
    cdef cvector[cvector[bint]] bres_
    bres.reserve(m1.size)
    if m2.ndim == 0:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] > m2.dDat[0]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] > m2.dDat[k]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    else:
        raise ValueError('> : shape不匹配')

def ge(cyMatrix m1, cyMatrix m2): # >=
    cdef Py_ssize_t i, j, k=0
    cdef cvector[bint] bres
    cdef cvector[cvector[bint]] bres_
    bres.reserve(m1.size)
    if m2.ndim == 0:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] >= m2.dDat[0]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    elif m1.naxis0 == m2.naxis0 and m1.naxis1 == m2.naxis1:
        bres_.reserve(m1.naxis0)
        for i in range(m1.naxis0):
            bres.resize(m1.naxis1)
            for j in range(m1.naxis1):
                bres[j] = m1.dDat[k] >= m2.dDat[k]
                k += 1
            bres_.push_back(bres)
        if m1.ndim == 0:
            return bres[0]
        if m1.ndim == 1:
            return bres_[0]
        return bres_
    else:
        raise ValueError('>= : shape不匹配')


def add_concatenate(cyMatrix m1, cyMatrix m2, bint axis0, bint isColVector): # 矩阵拼接 将mb添加到ma中 inplace
    cdef Py_ssize_t i1, i2, k11 = 0, k12, k21 = 0, k22, m2naxis0 = m2.naxis0, m2naxis1 = m2.naxis1
    cdef cvector[double] dDat
    if isColVector: # 当做列向量
        if m1.ndim == 1:
            m1.naxis0 = m1.size
            m1.naxis1 = 1
        if m2.ndim == 1:
            m2naxis0 = m2.size
            m2naxis1 = 1
    dDat.reserve(m1.size + m2.size)
    if axis0: # 按行拼接
        if m1.naxis1 != m2naxis1:
            raise ValueError('concatenate: m1 and m2 must have the same number of columns')
        dDat.assign(m1.dDat.begin(), m1.dDat.end())
        dDat.insert(dDat.end(), m2.dDat.begin(), m2.dDat.end())
    else: # 按列拼接
        if m1.naxis0 != m2naxis0:
            raise ValueError('concatenate: m1 and m2 must have the same number of rows')
        k12 = k11 + m1.naxis1
        k22 = k21 + m2naxis1
        for i1 in range(m1.naxis0):
            dDat.insert(dDat.end(), m1.dDat.begin() + k11, m1.dDat.begin() + k12)
            dDat.insert(dDat.end(), m2.dDat.begin() + k21, m2.dDat.begin() + k22)
            k11 += m1.naxis1
            k21 += m2naxis1
            k12 += m1.naxis1
            k22 += m2naxis1
    if axis0:
        m1.naxis0 += m2naxis0
        m1.ndim = 2
    else:
        m1.naxis1 += m2naxis1
        if (not isColVector) and m1.ndim == 1 and m2.ndim == 1:
            m1.ndim = 1
        else:
            m1.ndim = 2
    m1.size = m1.naxis0 * m1.naxis1
    m1.dDat.reserve(m1.size)
    m1.dDat.assign(dDat.begin(), dDat.end())
    return m1
def swap_row(cyMatrix m, Py_ssize_t r1, Py_ssize_t r2):
    cdef Py_ssize_t i
    r1 *= m.naxis1
    r2 *= m.naxis1
    if r1 > m.naxis0 or r2 > m.naxis0:
        raise ValueError('超出索引 swap_row: r1 > m.naxis0 or r2 > m.naxis0')
    for i in range(m.naxis1):
        m.dDat[r1 + i], m.dDat[r2 + i] = m.dDat[r2 + i], m.dDat[r1 + i]
    return m
def swap_col(cyMatrix m, Py_ssize_t c1, Py_ssize_t c2):
    cdef Py_ssize_t i
    if c1 > m.naxis1 or c2 > m.naxis1:
        raise ValueError('超出索引 swap_col: c1 > m.naxis1 or c2 > m.naxis1')
    for i in range(m.naxis0):
        m.dDat[c1], m.dDat[c2] = m.dDat[c2], m.dDat[c1]
        c1 += m.naxis1
        c2 += m.naxis1
    return m

def rad2deg(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = m.dDat[i] * 180. * d1_PI
    return res
def deg2rad(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = m.dDat[i] / 180. * dPI
    return res
def sin(cyMatrix m, bint isRad): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    cdef double num
    if isRad:
        for i in range(m.size):
            res.dDat[i] = math.sin(m.dDat[i])
    else:
        for i in range(m.size):
            num = m.dDat[i]
            res.dDat[i] = 0. if math.fabs(num) % 180 == 0 else math.sin(num / 180. * dPI)
    return res
def cos(cyMatrix m, bint isRad): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    cdef double num
    if isRad:
        for i in range(m.size):
            res.dDat[i] = math.cos(m.dDat[i])
    else:
        for i in range(m.size):
            num = m.dDat[i]
            res.dDat[i] = 0. if math.fabs(num) // 90 % 2 == 1 else math.cos(num / 180. * dPI)
    return res
def tan(cyMatrix m, bint isRad): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    cdef double num
    if isRad:
        for i in range(m.size):
            res.dDat[i] = math.tan(m.dDat[i])
    else:
        for i in range(m.size):
            num = m.dDat[i]
            res.dDat[i] = 0. if math.fabs(num) % 180 == 0 else math.tan(m.dDat[i] / 180. * dPI)
    return res
def asin(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.asin(m.dDat[i])
    return res
def acos(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.acos(m.dDat[i])
    return res
def atan(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.atan(m.dDat[i])
    return res
def atan2(cyMatrix m1, cyMatrix m2): # copy
    cdef Py_ssize_t i
    cdef cyMatrix m1_ = copy(m1), m2_ = copy(m2)
    _check_broadcast(m1_, m2_)
    cdef cyMatrix res = cyMatrix(m1_.naxis0, m1_.naxis1, m1_.ndim)
    for i in range(m1_.size):
        res.dDat[i] = math.atan2(m1_.dDat[i], m2_.dDat[i])
    return res

def exp(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.exp(m.dDat[i])
    return res
def exp2(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.exp2(m.dDat[i])
    return res
def expm1(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.expm1(m.dDat[i])
    return res
def log1p(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.log1p(m.dDat[i])
    return res
def log(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.log(m.dDat[i])
    return res
def log2(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.log2(m.dDat[i])
    return res
def log10(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.log10(m.dDat[i])
    return res

def sqrt(cyMatrix m): # **1/2 copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.sqrt(m.dDat[i])
    return res
def cbrt(cyMatrix m): # **1/3 copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.cbrt(m.dDat[i])
    return res
def numrt(cyMatrix m, long num): # **1/num copy
    if num == 0:
        raise ValueError('阶数不能是0 num == 0')
    cdef Py_ssize_t i
    cdef double dNum, po = 1./num
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        dNum = m.dDat[i]
        if dNum < 0:
            if num % 2 == 0:
                res.dDat[i] = dNan
            else:
                res.dDat[i] = -(-dNum) ** po
        else:
            res.dDat[i] = dNum ** po
    return res
def hypot(cyMatrix m1, cyMatrix m2): # copy
    cdef Py_ssize_t i
    cdef cyMatrix m1_ = copy(m1), m2_ = copy(m2)
    _check_broadcast(m1_, m2_)
    cdef cyMatrix res = cyMatrix(m1_.naxis0, m1_.naxis1, m1_.ndim)
    for i in range(m1_.size):
        res.dDat[i] = math.hypot(m1_.dDat[i], m2_.dDat[i])
    return res
def abs(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.fabs(m.dDat[i])
    return res

def floor(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.floor(m.dDat[i])
    return res
def ceil(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        res.dDat[i] = math.ceil(m.dDat[i])
    return res
def round(cyMatrix m, int digits): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    cdef double d = 10**digits
    for i in range(m.size):
        res.dDat[i] = math.round(m.dDat[i] * d) / d
    return res

def sign(cyMatrix m): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        if m.dDat[i] > 0:
            res.dDat[i] = 1.
        elif m.dDat[i] < 0:
            res.dDat[i] = -1.
        else:
            res.dDat[i] = 0.
    return res
def nan2num(cyMatrix m): # nan -> 0 inplace
    cdef Py_ssize_t i
    # cdef Matrix res = Matrix(m.naxis0, m.naxis1, m.ndim)
    for i in range(m.size):
        if math.isnan(m.dDat[i]):
            m.dDat[i] = 0.
    return m

cdef cvector[Py_ssize_t] _bubblesort(cvector[double] a): # 冒泡排序 输出序号
    cdef Py_ssize_t i, j, size = a.size()
    cdef cvector[Py_ssize_t] idx
    cdef bint notswapped
    idx.reserve(size)
    for i in range(size):
        idx.push_back(i)
    for i in range(size - 1):
        notswapped = True
        for j in range(size - i - 1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                idx[j], idx[j+1] = idx[j+1], idx[j]
                notswapped = False
        if notswapped:
            break
    return idx
cdef double median_vec(cvector[double] a):
    cdef cvector[double] temp = a
    cdef int size = temp.size()
    _sort(temp.begin(), temp.end())
    if size % 2 == 1:
        return temp[(size - 1) / 2]
    else:
        return (temp[size / 2 - 1] + temp[size / 2]) / 2

def sort(cyMatrix m, int axis): # copy
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    cdef cvector[double] temp
    if axis == 1:
        res.dDat.clear()
        k2 = m.naxis1
        temp.reserve(k2)
        for i in range(m.naxis0):
            temp.assign(m.dDat.begin() + k1, m.dDat.begin() + k2)
            _sort(temp.begin(), temp.end())
            res.dDat.insert(res.dDat.end(), temp.begin(), temp.end())
            k1 += m.naxis1
            k2 += m.naxis1
    elif axis == 0:
        temp.resize(m.naxis0)
        for i in range(m.naxis1):
            k1 = 0
            for j in range(m.naxis0):
                temp[j] = m.dDat[i + k1]
                k1 += m.naxis1
            _sort(temp.begin(), temp.end())
            k1 = 0
            for j in range(m.naxis0):
                res.dDat[i + k1] = temp[j]
                k1 += m.naxis1
    else:
        temp.reserve(m.size)
        temp.assign(m.dDat.begin(), m.dDat.end())
        _sort(temp.begin(), temp.end())
        res.dDat.assign(temp.begin(), temp.end())
    return res
cpdef cvector[cvector[Py_ssize_t]] index(cyMatrix m, double num):
    cdef Py_ssize_t i
    cdef cvector[cvector[Py_ssize_t]] idx
    cdef cvector[Py_ssize_t] idx1, idx2
    idx1.reserve(1)
    idx2.reserve(1)
    idx.reserve(2)
    idx.push_back(idx1)
    idx.push_back(idx2)
    for i in range(m.size):
        if m.dDat[i] == num:
            idx[0].push_back(i // m.naxis1)
            idx[1].push_back(i % m.naxis1)
    return idx

def argsort(cyMatrix m, bint axis): # axis == True: 按行, False: 按列
    cdef:
        Py_ssize_t i, j
        cvector[double] temp
        cvector[cvector[Py_ssize_t]] idx
    if axis:
        idx.reserve(m.naxis0)
        temp.reserve(m.naxis1)
        for i in range(m.naxis0):
            temp.assign(m.dDat.begin() + i*m.naxis1, m.dDat.begin() + (i+1)*m.naxis1)
        idx.push_back(_bubblesort(temp))
    else:
        idx.reserve(m.naxis1)
        temp.resize(m.naxis0)
        for i in range(m.naxis1):
            for j in range(m.naxis0):
                temp[j] = m.dDat[i+j*m.naxis1]
        idx.push_back(_bubblesort(temp))
    return idx

cpdef cyMatrix max(cyMatrix m, int axis): # copy
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef double dNum, maxNum
    cdef cyMatrix res
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        for i in range(m.naxis0):
            maxNum = m.dDat[k1]
            for j in range(m.naxis1):
                dNum = m.dDat[k1 + j]
                if dNum > maxNum:
                    maxNum = dNum
            res.dDat[i] = maxNum
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        for i in range(m.naxis1):
            maxNum = m.dDat[i]
            k1 = i
            for j in range(m.naxis0):
                dNum = m.dDat[k1]
                if dNum > maxNum:
                    maxNum = dNum
                k1 += m.naxis1
            res.dDat[i] = maxNum
    else:
        res = cyMatrix(1, 1, 0)
        dNum = m.dDat[0]
        for i in range(m.size):
            if m.dDat[i] > dNum:
                dNum = m.dDat[i]
        res.dDat[0] = dNum
    return res
cpdef cyMatrix min(cyMatrix m, int axis):
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef double dNum, minNum
    cdef cyMatrix res
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        for i in range(m.naxis0):
            minNum = m.dDat[k1]
            for j in range(m.naxis1):
                dNum = m.dDat[k1 + j]
                if dNum < minNum:
                    minNum = dNum
            res.dDat[i] = minNum
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        for i in range(m.naxis1):
            minNum = m.dDat[i]
            k1 = i
            for j in range(m.naxis0):
                dNum = m.dDat[k1]
                if dNum < minNum:
                    minNum = dNum
                k1 += m.naxis1
            res.dDat[i] = minNum
    else:
        res = cyMatrix(1, 1, 0)
        dNum = m.dDat[0]
        for i in range(m.size):
            if m.dDat[i] < dNum:
                dNum = m.dDat[i]
        res.dDat[0] = dNum
    return res
cpdef cyMatrix sum(cyMatrix m, int axis):
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef double dNum = 0.
    cdef cyMatrix res
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        for i in range(m.naxis0):
            dNum = 0.
            for j in range(m.naxis1):
                dNum += m.dDat[k1 + j]
            res.dDat[i] = dNum
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        for i in range(m.naxis1):
            k1 = i
            dNum = 0.
            for j in range(m.naxis0):
                dNum += m.dDat[k1]
                k1 += m.naxis1
            res.dDat[i] = dNum
    else:
        res = cyMatrix(1, 1, 0)
        for i in range(m.size):
            dNum += m.dDat[i]
        res.dDat[0] = dNum
    return res
def prod(cyMatrix m, int axis):
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef double dNum = 1.
    cdef cyMatrix res
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        for i in range(m.naxis0):
            dNum = 1.
            for j in range(m.naxis1):
                dNum *= m.dDat[k1 + j]
            res.dDat[i] = dNum
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        for i in range(m.naxis1):
            k1 = i
            dNum = 1.
            for j in range(m.naxis0):
                dNum *= m.dDat[k1]
                k1 += m.naxis1
            res.dDat[i] = dNum
    else:
        res = cyMatrix(1, 1, 0)
        for i in range(m.size):
            dNum *= m.dDat[i]
        res.dDat[0] = dNum
    return res
cpdef cyMatrix mean(cyMatrix m, int axis):
    cdef cyMatrix res = sum(m, axis)
    if axis == 1:
        itruediv_number(res, m.naxis1, True)
    elif axis == 0:
        itruediv_number(res, m.naxis0, True)
    else:
        res.dDat[0] /= m.size
    return res
def median(cyMatrix m, int axis):
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef cvector[double] v
    cdef cyMatrix res
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        v.resize(m.naxis1)
        for i in range(m.naxis0):
            for j in range(m.naxis1):
                v[j] = m.dDat[k1 + j]
            res.dDat[i] = median_vec(v)
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        v.resize(m.naxis0)
        for i in range(m.naxis1):
            k1 = i
            for j in range(m.naxis0):
                v[j] = m.dDat[k1]
                k1 += m.naxis1
            res.dDat[i] = median_vec(v)
    else:
        res = cyMatrix(1, 1, 0)
        v.reserve(m.size)
        v.assign(m.dDat.begin(), m.dDat.end())
        res.dDat[0] = median_vec(v)
    return res
cpdef cyMatrix sum_squares(cyMatrix m, int axis):
    cdef Py_ssize_t i, k1 = 0, k2, j
    cdef double dNum = 0.
    cdef cyMatrix res
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        for i in range(m.naxis0):
            dNum = 0.
            for j in range(m.naxis1):
                dNum += m.dDat[k1 + j]**2
            res.dDat[i] = dNum
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        for i in range(m.naxis1):
            k1 = i
            dNum = 0.
            for j in range(m.naxis0):
                dNum += m.dDat[k1]**2
                k1 += m.naxis1
            res.dDat[i] = dNum
    else:
        res = cyMatrix(1, 1, 0)
        for i in range(m.size):
            dNum += m.dDat[i]**2
        res.dDat[0] = dNum
    return res
def norm(cyMatrix m, int axis):
    cdef cyMatrix res = sum_squares(m, axis)
    return _ipow_number(res, 0.5)
def rms(cyMatrix m, int axis):
    cdef cyMatrix res = sum_squares(m, axis)
    if axis == 1:
        itruediv_number(res, m.naxis1, True)
    elif axis == 0:
        itruediv_number(res, m.naxis0, True)
    else:
        res.dDat[0] /= m.size
    return _ipow_number(res, 0.5)
def var(cyMatrix m, int axis):
    cdef cyMatrix mean_ = mean(m, axis), res
    cdef Py_ssize_t i, j, k1 = 0
    cdef double dNum, mean_dDat
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
        for i in range(m.naxis0):
            dNum = 0.
            mean_dDat = mean_.dDat[i]
            for j in range(m.naxis1):
                dNum += (m.dDat[k1 + j] - mean_dDat)**2
            res.dDat[i] = dNum / m.naxis1
            k1 += m.naxis1
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
        for i in range(m.naxis1):
            k1 = i
            dNum = 0.
            mean_dDat = mean_.dDat[i]
            for j in range(m.naxis0):
                dNum += (m.dDat[k1] - mean_dDat)**2
                k1 += m.naxis1
            res.dDat[i] = dNum / m.naxis0
    else:
        res = cyMatrix(1, 1, 0)
        dNum = 0.
        mean_dDat = mean_.dDat[0]
        for i in range(m.size):
            dNum += (m.dDat[i]- mean_dDat)**2
        res.dDat[0] = dNum / m.size
    return res
def std(cyMatrix m, int axis):
    cdef cyMatrix res = var(m, axis)
    return _ipow_number(res, 0.5)
def ptp (cyMatrix m, int axis): # copy
    cdef Py_ssize_t i
    cdef cyMatrix res, max_ = max(m, axis), min_ = min(m, axis)
    if axis == 1:
        res = cyMatrix(m.naxis0, 1, 1)
    elif axis == 0:
        res = cyMatrix(1, m.naxis1, 1)
    else:
        res = cyMatrix(1, 1, 0)
    for i in range(res.size):
        res.dDat[i] = max_.dDat[i] - min_.dDat[i]
    return res

def diff(cyMatrix m, int axis): # copy
    cdef cyMatrix res
    cdef Py_ssize_t i1, i2, k1, k2
    if m.ndim == 1:
        res = cyMatrix(m.size-1, 1, 1)
        for i1 in range(m.size - 1):
            res.dDat[i1] = m.dDat[i1+1] - m.dDat[i1]
    elif axis == 1:
        res = cyMatrix(m.naxis0, m.naxis1-1, m.ndim)
        for i1 in range(m.naxis0):
            k1 = i1*res.naxis1
            k2 = i1*m.naxis1
            for i2 in range(res.naxis1-1):
                res.dDat[k1] = m.dDat[k2 + 1] - m.dDat[k2]
                k1 += 1
                k2 += 1

    elif axis == 0:
        res = cyMatrix(m.naxis0-1, m.naxis1, m.ndim)
        for i1 in range(m.naxis1):
            k1 = i1
            k2 = i1
            for i2 in range(res.naxis0-1):
                res.dDat[k1] = m.dDat[k2 + m.naxis1] - m.dDat[k2]
                k1 += res.naxis1
                k2 += m.naxis1
    return res
def  trace(cyMatrix m): # copy
    cdef Py_ssize_t i, j = 0, n = m.naxis0 if m.naxis0 < m.naxis1 else m.naxis1
    cdef double dNum = 0.
    cdef cyMatrix res = cyMatrix(1, 1, 0)
    for i in range(n):
        dNum += m.dDat[j]
        j += m.naxis1 + 1
    res.dDat[0] = dNum
    return res
def  diag(cyMatrix m): # copy
    cdef Py_ssize_t i, j = 0
    cdef cyMatrix res
    if m.naxis0 <= m.naxis1:
        res = cyMatrix(m.naxis0, 1, 1)
    else:
        res = cyMatrix(m.naxis1, 1, 1)
    for i in range(res.size):
        res.dDat[i] = m.dDat[j]
        j += m.naxis1 + 1
    return res
cdef double _det_fun(cvector[double] x, cvector[int] index, Py_ssize_t nn):
    cdef double dNum = 1.
    cdef Py_ssize_t i, j=0
    for i in range(index.size()):
        dNum *= x[j + index[i]]
        j += nn
    return dNum
def det(cyMatrix m): # copy
    if m.ndim != 2 or m.naxis0 != m.naxis1:
        raise ValueError("det: 非方阵")
    cdef:
        Py_ssize_t i, j, mnaxis0 = m.naxis0
        double dNum = 0.
        cyMatrix res = cyMatrix(1,1,0)
        cvector[int] index1, index2
    index1.reserve(mnaxis0+1)
    index2.reserve(mnaxis0+1)
    for i in range(mnaxis0):
        index1.push_back(i)
        index2.push_back(mnaxis0 - i - 1)
    for i in range(mnaxis0):
        dNum += _det_fun(m.dDat, index1, mnaxis0)
        index1.push_back(index1[0])
        index1.erase(index1.begin())
    for i in range(mnaxis0):
        dNum -= _det_fun(m.dDat, index2, mnaxis0)
        index2.insert(index2.begin(), index2.back())
        index2.pop_back()
    res.dDat[0] = 0. if dNum < 3e-14 else dNum
    return res

cpdef cyMatrix inv(cyMatrix m): # 逆 高斯-约当消元法 copy
    if m.ndim != 2 or m.naxis0 != m.naxis1:
        raise ValueError("inv: 非方阵")
    cdef:
        Py_ssize_t i1, i2, i3, k1, k2, mnaxis0 = m.naxis0, mnaxis1 = m.naxis1
        cyMatrix augmented = cyMatrix(mnaxis0, mnaxis1*2, 2), res = cyMatrix(mnaxis0, mnaxis1, 2)
        cvector[double] temp, temp0
        double scale
    # 和单位矩阵组成增广矩阵
    temp.reserve(augmented.size)
    temp0.resize(mnaxis0)
    k1 = 0
    k2 = mnaxis1
    for i1 in range(mnaxis0):
        temp0.assign(mnaxis0, 0.) # [0,0,...]
        temp0[i1] = 1. # [1,0,0...]
        temp.insert(temp.end(), m.dDat.begin() + k1, m.dDat.begin() + k2)
        temp.insert(temp.end(), temp0.begin(), temp0.end())
        k1 += mnaxis1
        k2 += mnaxis1
    augmented.dDat.assign(temp.begin(), temp.end())
    # 高斯-约当消元法
    for i1 in range(mnaxis0):
        scale = augmented.dDat[i1 * augmented.naxis1 + i1]
        if scale == 0:
            raise ValueError("inv: 不可逆")
        k1 = i1 * augmented.naxis1
        for i2 in range(augmented.naxis1):
            augmented.dDat[k1 + i2] /= scale
        for i2 in range(mnaxis0):
            if i2 != i1:
                scale = augmented.dDat[i2 * augmented.naxis1 + i1]
                # scale = augmented.get_item(i2, i1)
                k2 = i2 * augmented.naxis1
                for i3 in range(augmented.naxis1):
                    augmented.dDat[k2 + i3] -= augmented.dDat[k1 + i3] * scale
    # 分离
    k1 = 0
    k2 = mnaxis1
    for i1 in range(mnaxis0):
        for i2 in range(mnaxis1):
            res.dDat[k1] = augmented.dDat[k2]
            k1 += 1
            k2 += 1
        k2 += m.naxis1
    return res

# cpdef Py_ssize_t rank(Matrix m):
#     cdef:
#         Py_ssize_t i1, i2, i3, k = 0, mnaxis0 = m.naxis0, mnaxis1 = m.naxis1, rank_ = 0
#         bint isZero = True
#         double factor
#         Matrix m_ = copy(m)
#     if m.ndim == 0:
#         return 0
#     if m.ndim == 1:
#         return 1
#     for i1 in range(mnaxis1):
#         isZero = True
#         for i2 in range(mnaxis0):
#             if m_.get_item(i2, i1) != 0:
#                 isZero = False
#                 k = i2 # 当前列中非零元素的行索引
#                 break
#         if isZero: # 如果当前列的所有元素都为0，则继续下一列
#             continue
#         if k > rank_: # 如果找到的行索引大于当前秩，则交换这两行
#             swap_row(m_, k, rank_)
#         for i2 in range(rank_+1, mnaxis0): # 使用当前行来消元其他行
#             factor = m_.get_item(i2, i1) / m_.get_item(rank_, i1)
#             for i3 in range(mnaxis1):
#                 m_.dDat[i2 * mnaxis1 + i3] -= factor * m_.dDat[rank_ * mnaxis1 + i3]
#         rank_ += 1
#         if rank_ == mnaxis0:
#             break
#     return rank_


def interp1d_2_mco(cyMatrix x, cyMatrix y, int dtype):
    cdef Py_ssize_t i, n = x.size, j, n_1 = n-1
    cdef cyMatrix res = cyMatrix(n_1, 3, 2), mcoe = cyMatrix(n, n, 2), yy = cyMatrix(n, 1, 1)
    cdef cvector[double] a, xx, dx, dy, b, c
    dx.reserve(n_1)
    dy.reserve(n_1)
    a.reserve(n_1)
    xx.reserve(n_1)
    b.reserve(n_1)
    c.reserve(n_1)
    for i in range(n_1):
        dx.push_back(x.dDat[i+1] - x.dDat[i])
        dy.push_back(y.dDat[i+1] - y.dDat[i])

    for i in range(1, n):
        j = i*n
        mcoe.dDat[j + i-1] = 1.
        mcoe.dDat[j + i] = 1.
    if dtype == 0:
        mcoe.dDat[0] = 1.
    elif dtype == 1:
        mcoe.dDat[0] = dx[1]
        mcoe.dDat[1] = -(dx[0] + dx[1])
        mcoe.dDat[2] = dx[0]
    else:
        mcoe.dDat[0] = 1.
        mcoe.dDat[n_1] = -1.

    a.assign(y.dDat.begin(), y.dDat.end()-1)
    xx.assign(x.dDat.begin(), x.dDat.end()-1)
    yy.dDat[0] = 0.
    for i in range(n_1):
        yy.dDat[i] = dy[i] / dx[i] * 2
    cdef cyMatrix b_ = matmul(inv(mcoe), yy)
    b.assign(b_.dDat.begin(), b_.dDat.end()-1)
    for i in range(n_1):
        c.push_back((b_.dDat[i+1] - b_.dDat[i]) / dx[i] / 2)
    for i in range(n_1):
        j = i*3
        res.dDat[j]   = a[i] - b[i] * xx[i] + c[i] * xx[i]**2
        res.dDat[j+1] = b[i] - c[i] * xx[i] * 2
        res.dDat[j+2] = c[i]
    return res

def interp1d_3_mco(cyMatrix x, cyMatrix y, int dtype):
    cdef Py_ssize_t i, n = x.size, j, nn = n*n, n_1 = n-1
    cdef cyMatrix res = cyMatrix(n_1, 4, 2), mcoe = cyMatrix(n, n, 2), yy = cyMatrix(n, 1, 1)
    cdef cvector[double] a, xx, dx, dy, b, c, k, d
    dx.reserve(n_1)
    dy.reserve(n_1)
    a.reserve(n_1)
    xx.reserve(n_1)
    b.reserve(n_1)
    c.reserve(n_1)
    d.reserve(n_1)
    k.reserve(n_1)
    for i in range(n_1):
        dx.push_back(x.dDat[i+1] - x.dDat[i])
        dy.push_back(y.dDat[i+1] - y.dDat[i])
    for i in range(n_1):
        k.push_back(dy[i] / dx[i])
    j = 1
    for i in range(n-2):
        yy.dDat[j] = (k[j] - k[i])*3
        j += 1

    for i in range(1, n - 1):
        j = i*n
        mcoe.dDat[j + i - 1] = dx[i - 1]
        mcoe.dDat[j + i] = 2 * (dx[i] + dx[i - 1])
        mcoe.dDat[j + i + 1] = dx[i]
    if dtype == 0:
        mcoe.dDat[0] = 1.
        mcoe.dDat[nn-1] = 1.
    elif dtype == 1:
        mcoe.dDat[0] = -dx[1]
        mcoe.dDat[1] = dx[0] + dx[1]
        mcoe.dDat[2] = -dx[0]
        mcoe.dDat[nn-1] = -dx[n-2]
        mcoe.dDat[nn-2] = dx.back() + dx[n-2]
        mcoe.dDat[nn-3] = -dx.back()
    else:
        mcoe.dDat[0] = 1.
        mcoe.dDat[nn-1] = -1.
        yy.dDat[n_1] = dy.back() - dy[0]
        mcoe.dDat[nn-1] = dx.back() / 6
        mcoe.dDat[nn-2] = dx.back() / 3
        mcoe.dDat[nn-n+1] = -dx[0] / 6
        mcoe.dDat[nn-n] = -dx[0] / 3

    a.assign(y.dDat.begin(), y.dDat.end()-1)
    xx.assign(x.dDat.begin(), x.dDat.end()-1)
    cdef cyMatrix c_ = matmul(inv(mcoe), yy)
    c.assign(c_.dDat.begin(), c_.dDat.end()-1)
    for i in range(n_1):
        d.push_back((c_.dDat[i+1] - c_.dDat[i]) / dx[i] / 3)
        b.push_back(k[i] - dx[i] * c[i] * 2 / 3 - dx[i] * c_.dDat[i+1] / 3)
    for i in range(n_1):
        j = i*4
        res.dDat[j]   = a[i] - b[i] * xx[i]     + c[i] * xx[i]**2 - d[i] * xx[i]**3
        res.dDat[j+1] = b[i] - c[i] * xx[i] * 2 + d[i] * xx[i]**2 * 3
        res.dDat[j+2] = c[i] - d[i] * xx[i] * 3
        res.dDat[j+3] = d[i]
    return res


def flip(cyMatrix m, int axis):
    cdef Py_ssize_t i, j, k1=0, k2=0
    cdef cyMatrix res = cyMatrix(m.naxis0, m.naxis1, m.ndim)
    if axis == 0: # 翻转行
        for i in range(m.naxis0):
            k1 = i * m.naxis1
            k2 = m.size - k1 - m.naxis1
            for j in range(m.naxis1):
                res.dDat[k1 + j] = m.dDat[k2 + j]
    elif axis == 1: # 翻转列
        for i in range(m.naxis1):
            k1 = i
            k2 = m.naxis1 - k1 - 1
            for j in range(m.naxis0):
                res.dDat[k1] = m.dDat[k2]
                k1 += m.naxis1
                k2 += m.naxis1
    else:
        raise ValueError('filp: axis 只能为 0 或 1')
    return res


def repeat(cyMatrix m, Py_ssize_t n, int axis):
    cdef Py_ssize_t i, j, k, kk, kk1
    cdef cyMatrix res
    cdef double num
    if axis == 1:
        res = cyMatrix(m.naxis0, m.naxis1*n, m.ndim)
        for i in range(m.naxis0):
            kk = i*m.naxis1
            for j in range(m.naxis1):
                num = m.dDat[kk]
                kk1 = kk*n
                kk += 1
                for k in range(n):
                    res.dDat[kk1 + k] = num
    elif axis == 0:
        res = cyMatrix(m.naxis0*n, m.naxis1, m.ndim)
        for j in range(m.naxis1):
            for i in range(m.naxis0):
                kk = i*m.naxis1 + j
                num = m.dDat[kk]
                kk = i*m.naxis1*n + j
                for k in range(n):
                    res.dDat[kk] = num
                    kk += m.naxis1
    else:
        res = cyMatrix(1, m.size*n, 1)
        for i in range(m.size):
            num = m.dDat[i]
            k = i*n
            for j in range(n):
                res.dDat[k] = num
                k += 1
    return res

def  maximum(cyMatrix m1, cyMatrix m2):
    cdef Py_ssize_t i
    cdef cyMatrix m1_ = copy(m1), m2_ = copy(m2)
    _check_broadcast(m1_, m2_)
    cdef cyMatrix res = cyMatrix(m1_.naxis0, m1_.naxis1, m1_.ndim)
    for i in range(m1_.size):
        if m1_.dDat[i] >= m2_.dDat[i]:
            res.dDat[i] = m1_.dDat[i]
        else:
            res.dDat[i] = m2_.dDat[i]
    return res

cpdef cyMatrix minimum(cyMatrix m1, cyMatrix m2):
    cdef Py_ssize_t i
    cdef cyMatrix m1_ = copy(m1), m2_ = copy(m2)
    _check_broadcast(m1_, m2_)
    cdef cyMatrix res = cyMatrix(m1_.naxis0, m1_.naxis1, m1_.ndim)
    for i in range(m1_.size):
        if m1_.dDat[i] <= m2_.dDat[i]:
            res.dDat[i] = m1_.dDat[i]
        else:
            res.dDat[i] = m2_.dDat[i]
    return res

def argmin(cyMatrix m, int axis):
    cdef cvector[Py_ssize_t] res
    cdef cvector[double] temp
    cdef Py_ssize_t i, j
    if axis == 1:
        res.reserve(m.naxis0)
        temp.resize(m.naxis1)
        for i in range(m.naxis0):
            temp.assign(m.dDat.begin() + i*m.naxis1, m.dDat.begin() + (i+1)*m.naxis1)
            res.push_back(_bubblesort(temp)[0])
            
    elif axis == 0:
        res.reserve(m.naxis1)
        temp.resize(m.naxis0)
        for i in range(m.naxis1):
            for j in range(m.naxis0):
                temp[j] = m.dDat[i+j*m.naxis1]
            res.push_back(_bubblesort(temp)[0])
    else:
        res.reserve(1)
        temp.reserve(m.size)
        temp.assign(m.dDat.begin(), m.dDat.end())
        res.push_back(_bubblesort(temp)[0])
    return res

def argmax(cyMatrix m, int axis):
    cdef cvector[Py_ssize_t] res
    cdef cvector[double] temp
    cdef Py_ssize_t i, j
    if axis == 1:
        res.reserve(m.naxis0)
        temp.resize(m.naxis1)
        for i in range(m.naxis0):
            temp.assign(m.dDat.begin() + i*m.naxis1, m.dDat.begin() + (i+1)*m.naxis1)
            res.push_back(_bubblesort(temp).back())
    elif axis == 0:
        res.reserve(m.naxis1)
        temp.resize(m.naxis0)
        for i in range(m.naxis1):
            for j in range(m.naxis0):
                temp[j] = m.dDat[i+j*m.naxis1]
            res.push_back(_bubblesort(temp).back())
    else:
        res.reserve(1)
        temp.reserve(m.size)
        temp.assign(m.dDat.begin(), m.dDat.end())
        res.push_back(_bubblesort(temp).back())
    return res


def extract(Parray condition, cyMatrix m):
    cdef Py_ssize_t i, n = 0, k = 0
    cdef char[:] con = condition
    cdef cyMatrix res
    if con.shape[0] != m.size:
        raise ValueError('extract: condition.size != m.size')
    for i in range(con.shape[0]):
        if con[i]:
            n += 1
    res = cyMatrix(n, 1, 1)
    for i in range(m.size):
        if con[i]:
            res.dDat[k] = m.dDat[i]
            k += 1
    return res

def compress(Parray condition, cyMatrix m, int axis):
    cdef Py_ssize_t i, j, n = 0, k = 0, ir
    cdef char[:] con = condition
    cdef cyMatrix res
    for i in range(con.shape[0]):
        if con[i]:
            n += 1
    if axis == 1:
        if con.shape[0] > m.naxis1:
            raise ValueError('compress: axis == 1,  condition.size > m.naxis1')
        res = cyMatrix(m.naxis0, n, m.ndim)
        for i in range(m.naxis0):
            ir = i * m.naxis1
            for j in range(m.naxis1):
                if con[j]:
                    res.dDat[k] = m.dDat[ir + j]
                    k += 1
    elif axis == 0:
        if con.shape[0] > m.naxis0:
            raise ValueError('compress: axis == 0,  condition.size > m.naxis0')
        res = cyMatrix(n, m.naxis1, m.ndim)
        for i in range(m.naxis0):
            if con[i]:
                ir = i * m.naxis1
                for j in range(m.naxis1):
                    res.dDat[k] = m.dDat[ir + j]
                    k += 1
    else:
        res = cyMatrix(n, 1, 1)
        for i in range(con.shape[0]):
            if con[i]:
                res.dDat[k] = m.dDat[i]
                k += 1
    return res

def compress_setitem(Parray condition, cyMatrix m, cyMatrix val, int axis):
    cdef Py_ssize_t i, j, k = 0, n = 0, ir
    cdef char[:] con = condition
    cdef cyMatrix temp
    for i in range(con.shape[0]):
        if con[i]:
            n += 1
    if axis == 1:
        if con.shape[0] > m.naxis1:
            raise ValueError('compress_setitem: axis == 1,  condition.size > m.naxis1')
        temp = cyMatrix(m.naxis0, n, m.ndim)
        broadcast(val, temp)
        for i in range(m.naxis0):
            ir = i * m.naxis1
            for j in range(m.naxis1):
                if con[j]:
                    m.dDat[ir + j] = val.dDat[k]
                    k += 1
    elif axis == 0:
        if con.shape[0] > m.naxis0:
            raise ValueError('compress_setitem: axis == 0,  condition.size > m.naxis0')
        temp = cyMatrix(n, m.naxis1, m.ndim)
        broadcast(val, temp)
        for i in range(m.naxis0):
            if con[i]:
                ir = i * m.naxis1
                for j in range(m.naxis1):
                    m.dDat[ir + j] = val.dDat[k]
                    k += 1
    else:
        temp = cyMatrix(n, 1, 1)
        broadcast(val, temp)
        for i in range(con.shape[0]):
            if con[i]:
                m.dDat[i] = val.dDat[k]
                k += 1
    return m


def append(cyMatrix m1, cyMatrix m2):
    m1.dDat.insert(m1.dDat.end(), m2.dDat.begin(), m2.dDat.end())
    m1.size += m2.size
    if m1.ndim == 0:
        m1.reshape(1, 1, 2)
    elif m1.ndim == 1:
        if m2.ndim == 0:
            m1.reshape(1, 1, 2)
        elif m1.naxis1 == m2.naxis1:
            m1.reshape(2, m1.naxis0 + m2.naxis0, m1.naxis1)
        else:
            m1.reshape(1, 1, 2)
    else:
        if m1.naxis1 != m2.naxis1:
            raise ValueError('append: m1 and m2 must have the same number of columns')
        m1.reshape(2, m1.naxis0 + m2.naxis0, m1.naxis1)
    return m1




##### ----- 对Python接口 ----- #####
def get_shape(cyMatrix m):
    if m.ndim == 0:
        return ()
    elif m.ndim == 1:
        return (m.size,)
    else:
        return (m.naxis0, m.naxis1)
def get_ndim(cyMatrix m): return m.ndim
def get_size(cyMatrix m): return m.size
def get_value(cyMatrix m):
    cdef:
        cvector[cvector[double]] dDat
        cvector[double] dDat_
        Py_ssize_t i, j, k = 0, IR = m.naxis0, IC = m.naxis1
    if m.ndim == 0:
        return m.dDat[0]
    if m.ndim == 1:
        return m.dDat
    
    dDat.reserve(IR)
    dDat_.resize(IC)
    for i in range(IR):
        # dDat_ = cvector[double]()
        # dDat_.reserve(IC)
        for j in range(IC):
            dDat_[j] = m.dDat[k]
            k += 1
        dDat.push_back(dDat_)
    return dDat
def get_value_int(cyMatrix m):
    cdef:
        cvector[cvector[long]] lDat
        cvector[long] lDat_
        Py_ssize_t i, j, k = 0, IR = m.naxis0, IC = m.naxis1
    if m.ndim == 0:
        return <long>m.dDat[0]
    if m.ndim == 1:
        lDat_.reserve(m.size)
        for i in range(m.size):
            lDat_.push_back(<long>m.dDat[i])
        return lDat_
    
    lDat.reserve(IR)
    lDat_.resize(IC)
    for i in range(IR):
        for j in range(IC):
            lDat_[j] = <long>m.dDat[k]
            k += 1
        lDat.push_back(lDat_)
    return lDat
def get_value_bool(cyMatrix m):
    cdef:
        cvector[cvector[bint]] bDat
        cvector[bint] bDat_
        Py_ssize_t i, j, k = 0, IR = m.naxis0, IC = m.naxis1
    if m.ndim == 0:
        return <bint>m.dDat[0]
    if m.ndim == 1:
        bDat_.reserve(m.size)
        for i in range(m.size):
            bDat_.push_back(<bint>m.dDat[i])
        return bDat_
    
    bDat.reserve(IR)
    bDat_.resize(IC)
    for i in range(IR):
        for j in range(IC):
            bDat_[j] = <bint>m.dDat[k]
            k += 1
        bDat.push_back(bDat_)
    return bDat


def reshape(cyMatrix m, Py_ssize_t ndim, Py_ssize_t nrows, Py_ssize_t ncols):
    m.reshape(ndim, nrows, ncols)
    return m
def set_ndim(cyMatrix m, Py_ssize_t ndim): m.ndim = ndim
def fill(cyMatrix m, numberType value): m.fill(value)
def get_nan(): return dNan
def get_inf(): return dInf
def isnan(double x): return math.isnan(x)
def ravel(cyMatrix m): return m.dDat

def buildFromArr(Parray a, Py_ssize_t nrow, Py_ssize_t ncol, Py_ssize_t ndim):
    cdef cyMatrix res = cyMatrix(nrow, ncol, ndim)
    cdef double[:] dArr_v = a
    cdef Py_ssize_t i
    # dArr_v = a
    for i in range(res.size):
        res.dDat[i] = dArr_v[i]
    return res

def eye(Py_ssize_t n):
    cdef cyMatrix res = cyMatrix(n, n, 2)
    cdef Py_ssize_t i
    for i in range(n):
        res.dDat[i*n+i] = 1.
    return res

def get_matrix_row(cyMatrix m, Py_ssize_t n):
    cdef cyMatrix res
    if m.ndim == 0:
        raise ValueError('get_row: m.ndim == 0')
    if m.ndim == 1:
        res = cyMatrix(1,1,0)
        res.dDat[0] = m.dDat[n]
    else:
        res = cyMatrix(1, m.naxis1, 1)
        res.dDat.assign(m.dDat.begin() + n * m.naxis1, m.dDat.begin() + (n + 1) * m.naxis1)
    return res

def matrix_byindex(cyMatrix m, Parray idx, bint axis): # copy
    cdef int[:] idx_v = idx
    cdef Py_ssize_t i, j, n = idx_v.shape[0], k, k0 = 0, ndim = m.ndim
    cdef cyMatrix res
    if ndim == 0:
        return copy(m)
    if ndim == 1:
        res = cyMatrix(1, n, 1)
        for i in range(n):
            res.dDat[i] = m.dDat[idx_v[i]]
        return res
    if axis: # axis == 1 在行内进行操作
        res = cyMatrix(m.naxis0, n, 2)
        for i in range(m.naxis0):
            k = i*m.naxis1
            for j in range(n):
                res.dDat[k0] = m.dDat[k + idx_v[j]]
                k0 += 1
    else: # axis == 0 在列内进行操作
        res = cyMatrix(n, m.naxis1, 2)
        for i in range(n):
            k = idx_v[i]*m.naxis1
            for j in range(m.naxis1):
                res.dDat[k0] = m.dDat[k + j]
                k0 += 1
    return res


