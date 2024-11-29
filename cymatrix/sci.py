# -*- coding: utf-8 -*-
"""
基于 Matrix 的插值 滤波 拟合
打包低空间占用
"""
from typing import Callable, Dict, Literal, Sequence, Tuple, Union, overload

from .matrix import (INF, Array, Matrix, NumberR, abs, arange, argsort, concat,
                     eye, interp1d_2_mco, interp1d_3_mco, inv, linv, mean,
                     median, norm, sum_squares, zeros)


def _poly_xbuild(m: Matrix, n: int) -> Matrix:
    # res.shape[0] == m.size res.shape[1] == n+1
    if m.ndim != 1:
        m = m.reshape(m.size)
    return concat([m**ii for ii in range(n + 1)], axis=1, isColVector=True)


def ndim_check(x: Matrix) -> Matrix:
    # 将二维向量转一维 其他不变 vector
    if x.ndim == 2 and 1 in x.shape:
        x.reshape(x.size, inplace=True)
    return x


# %% interpolate
class interp1d:
    # kind_str = ('linear', 'quadratic', 'cubic')
    boundary_str = ('natural', 'not-a-knot', 'periodic')
    @overload
    def __init__(self, x: Union[Array, Matrix], y: Union[Array, Matrix], kind:Literal[1, 2, 3], boundary: Literal['natural', 'periodic', 'not-a-knot'])->None:...
    def __init__(self, x: Union[Array, Matrix], y: Union[Array, Matrix], kind: int=3, boundary: str='not-a-knot'):
        self.x, self.y, self.kind = Matrix(x), Matrix(y), kind
        self.boundary = self.boundary_str.index(boundary.lower())
        self.input_check()

        if self.kind == 1:
            fun = self.mcoe_cal1
        elif self.kind == 3:
            fun = self.mcoe_cal3
        elif self.kind == 2:
            fun = self.mcoe_cal2
        if self.y.ndim == 1:
            self.mcoe = fun(self.y)
        else:
            self.mcoe = [fun(ii) for ii in self.y]

    def input_check(self):
        x, y = self.x.copy(), self.y.copy()
        # x 一维向量
        if x.ndim != 1 and (x.shape[0] - 1) * (x.shape[1] - 1) != 0:
            raise ValueError('x is not vector')
        x = ndim_check(x)
        x_sort_index = argsort(x)[0]
        self.x = x[x_sort_index]

        # y 一维向量 | 二维行向量的堆叠
        y = ndim_check(y)
        if y.ndim == 2:
            if y.shape[0] == x.size: # y是列向量堆叠
                self.y = y[x_sort_index].transpos() # -> 行向量堆叠
            elif y.shape[1] == x.size: # y是行向量堆叠
                self.y = y.transpos()[x_sort_index].transpos() # -> 行向量堆叠
            else:
                raise ValueError('y.shape和x不匹配')
        elif y.ndim == 1:
            assert y.size == x.size
            self.y = y[x_sort_index]
        else:
            raise ValueError('y.ndim > 2')

        assert isinstance(self.kind, int)
        assert self.kind > 0
        assert self.kind <= 3

    def xin_index(self, x_in: Matrix):
        xin_ind, start = [], 0
        for i1 in x_in:
            if i1 >= self.x[-1]:
                xin_ind.append(self.x.size - 2)
                continue
            for i2 in range(start, self.x.size - 1):
                if i1 < self.x[i2 + 1]:
                    xin_ind.append(i2)
                    start = i2
                    break
        return xin_ind

    def __call__(self, x: Union[Array, Matrix]) -> Matrix:
        x = Matrix(x).sort()
        # x_in = ndim_change(x_in)
        if x.ndim == 0:
            x_in = x.reshape(1)
        else:
            x_in = x
        assert x_in.ndim == 1
        x_in_ind = self.xin_index(x_in)
        X = _poly_xbuild(x_in, self.kind)
        if self.y.ndim == 1:
            res = [self.mcoe[val] @ X[ii] for ii, val in enumerate(x_in_ind)]
            return concat(res, 1)[0]
        else:
            res = [concat([coe[val] @ X[ii] for ii, val in enumerate(x_in_ind)], 1).vector() for coe in self.mcoe]
            return  concat(res, 1, True)

    def mcoe_cal1(self, y: Matrix) -> Matrix:
        # y.ndim == 1
        dx, dy = self.x.diff(), y.diff()
        k = dy / dx
        a, xx = y[:-1], self.x[:-1]
        return concat((a - k * xx, k), axis=1, isColVector=True)

    def mcoe_cal2(self, y: Matrix) -> Matrix:
        # y.ndim == 1
        return Matrix(interp1d_2_mco(self.x.base, y.base, self.boundary))

    def mcoe_cal3(self, y: Matrix) -> Matrix:
        # y.ndim == 1
        return Matrix(interp1d_3_mco(self.x.base, y.base, self.boundary))


# %% polyfit
def _polyfit(x: Matrix, y: Matrix, n: int) -> Matrix:
    # x.ndim == 1
    # y.ndim in (1, 2) y.shape[0] == x.szie
    # retern res.shape[0] == n+1  res.shape[1] == y.shape[1]
    X = _poly_xbuild(x, n)
    return linv(X) @ y

def polyfit(x: Union[Array, Matrix], y: Union[Array, Matrix], n: int) -> Matrix:
    #
    assert isinstance(n, int)
    assert n > 0
    x, y = Matrix(x), Matrix(y)
    assert n <= x.size
    # x 一维向量
    if x.ndim != 1 and (x.shape[0] - 1) * (x.shape[1] - 1) != 0:
        raise ValueError('x is not vector')
    x = ndim_check(x)
    x_sort_index = argsort(x)[0]
    x = x[x_sort_index]
    # y 一维向量 | 二维列向量的堆叠
    yndim = y.ndim
    if yndim == 0:
        raise ValueError('polyfit: y.ndim == 0')
    elif yndim == 1:
        assert y.size == x.size
        y = y[x_sort_index]
    else:
        y = ndim_check(y)
        if yndim == 2:
            if y.shape[0] == x.size:
                y = y[x_sort_index]
            elif y.shape[1] == x.size:
                y = y.transpos()[x_sort_index]
            else:
                raise ValueError('y.shape和x不匹配')
    return _polyfit(x, y, n)

def polyval(p: Matrix, x: Matrix) -> Matrix:
    return _poly_xbuild(x, p.shape[0] - 1) @ p


# %% filter
def _filter1(fun: Callable[[Matrix], Matrix], x: Matrix, axis=0) -> Matrix:
    x = Matrix(x)
    assert x.ndim in (1, 2)
    if x.ndim == 1:
        return fun(x)
    if axis == 0:  # 对列内
        return concat([fun(ii) for ii in x.transpos()], 1, True)
    if axis == 1:  # 对行内
        return concat([fun(ii) for ii in x], 0)
    raise ValueError('axis should be 0 or 1')

def _filter2(x: Matrix, length:int=3, axis:int=0, mod:Callable=mean) -> Matrix:
    assert length % 2 == 1
    _n = length // 2
    def fun(m: Matrix):
        m_ = m.copy()
        for ii in range(_n, m.size - _n):
            m_[ii] = mod(m[ii - _n:ii + _n + 1])
        for ii in range(_n):
            m_[ii] = mod(m[:ii + _n + 1])
            m_[-1 - ii] = mod(m[-2 - ii:])
        return m_
    return _filter1(fun, x, axis)


def avg_filter(x: Matrix, length:int=3, axis:int=0) -> Matrix:
    """滑动窗口平均值滤波
    axis=0 在列内操作
    axis=1 在行内操作"""
    return _filter2(x, length, axis, mean)


def med_filter(x: Matrix, length:int=3, axis:int=0) -> Matrix:
    """滑动窗口中位数滤波
    axis=0 在列内操作
    axis=1 在行内操作"""
    return _filter2(x, length, axis, median)


def savgol_filter(x: Matrix, length:int=5, poly:int=3, axis:int=0) -> Matrix:
    """滑动窗口SG滤波
    axis=0 在列内操作
    axis=1 在行内操作"""
    assert isinstance(poly, int)
    assert poly > 0 and poly <= length
    assert length % 2 == 1
    _n = length // 2

    xx = arange(length)
    def fun(m: Matrix)-> Matrix:
        m_ = m.copy()
        for ii in range(_n, m.size - _n):
            coe = polyfit(xx, m[ii - _n:ii + _n + 1], poly)
            m_[ii] = polyval(coe, _n)
        coe0 = polyfit(xx, m[:length], poly)
        coe1 = polyfit(xx, m[-1:-1 - length:-1], poly)
        x = arange(_n)
        m_[:_n] = polyval(coe0, x)
        m_[-1-_n:] = polyval(coe1, x).flip()
        return m_

    return _filter1(fun, x, axis)


# %% optimize
EPS = Matrix(1e-8)
EPS2 = Matrix(2e-8)
EPS22 = Matrix(4e-16)
def _func_dy(f: Callable, x_in: Matrix, *args, **kwargs) -> Matrix:
    """近似导数"""
    return (f(x_in + EPS, *args, **kwargs) -
            f(x_in - EPS, *args, **kwargs)) / EPS2


def _dfdx(f: Callable, x_in: Matrix, ii: int, *args, **kwargs) -> Matrix:
    """近似一阶偏导"""
    dx = zeros(x_in.size)
    dx[ii] = EPS
    return (f(x_in + dx, *args, **kwargs) -
            f(x_in - dx, *args, **kwargs)) / EPS2


def _ddfdxx(f: Callable, x_in: Matrix, i1: int, i2: int, *args, **kwargs) -> Matrix:
    """近似二阶偏导"""
    dx1 = zeros(x_in.size)
    dx2 = zeros(x_in.size)
    dx1[i1], dx2[i2] = EPS, EPS
    return (f(x_in + dx1 + dx2, *args, **kwargs) +
            f(x_in - dx1 - dx2, *args, **kwargs) -
            f(x_in + dx1 - dx2, *args, **kwargs) -
            f(x_in - dx1 + dx2, *args, **kwargs)) / EPS22


def jacobian(f: Callable, x_in: Matrix, *args, **kwargs) -> Matrix:
    """
    近似雅克比矩阵 
    ----------
    ::
    
        f 函数
        x_in 自变量
        *args, **kwargs 函数的其他输入
    
    ----------
    return
    ::
    
        如果 f是标量函数 f -> float64 那么return一维向量 梯度
        如果 f是向量函数 f -> ndarray 那么return二维矩阵 雅可比矩阵
    """
    jac = [_dfdx(f, x_in, ii, *args, **kwargs) for ii in range(x_in.size)]
    return concat(jac, 1, True).vector()


def hessian_1y(f: Callable, x_in: Matrix, *args, **kwargs) -> Matrix:
    """近似海森矩阵 1y:f->float"""
    hes = zeros(x_in.size, x_in.size)
    hes_ = hes.copy()
    for i1 in range(x_in.size):
        hes_[i1, i1] = _ddfdxx(f, x_in, i1, i1, *args, **kwargs)
        for i2 in range(i1 + 1, x_in.size):
            hes[i1, i2] = _ddfdxx(f, x_in, i1, i2, *args, **kwargs)
    return hes + hes.transpos() + hes_


# %%% 二分
def bisection(
        func: Callable[[Union[NumberR, Matrix]], Matrix],
        bounds: Sequence[Union[NumberR, Matrix]],
        ftol: NumberR = 1e-08,
        max_nfev: NumberR = 1e3,
        args: list = [],
        kwargs: dict = {}) -> Tuple[Matrix, Dict[str, Union[NumberR, Matrix]]]:
    """二分法 计算单变量func零点"""
    assert len(bounds) == 2
    if func(bounds[0], *args, **kwargs) < func(bounds[1], *args, **kwargs):
        k1, k2 = bounds
    else:
        k2, k1 = bounds
    for n in range(max_nfev):
        kk = (k1 + k2) / 2
        y_ = func(kk, *args, **kwargs)
        if y_.abs() < ftol:
            break
        if y_ < 0:
            k1 = kk
        else:
            k2 = kk
    return kk, {'ftol': y_, 'max_nfev': n + 1}


# %%% grad
def newton(
        func: Callable[[Union[NumberR, Matrix]], Matrix],
        x0: Union[Matrix, NumberR, Array],
        err: NumberR = 1e-12,
        ftol: NumberR = 1e-8,
        xtol: NumberR = 1e-8,
        gtol: NumberR = 1e-8,
        max_nfev: NumberR = 1e3,
        args: list = [],
        kwargs: dict = {}) -> Tuple[Matrix, Dict[str, Union[NumberR, Matrix]]]:
    """牛顿迭代法 计算单变量func零点"""
    x_0 = Matrix(x0)

    y_0 , grad_ = func(x_0, *args, **kwargs), INF
    for n in range(int(max_nfev)):
        grad = _func_dy(func, x_0, *args, **kwargs)
        x_1 = x_0 - y_0 / grad
        y_1 = func(x_1, *args, **kwargs)
        ftol_, xtol_, gtol_ = abs(y_1 - y_0), abs(x_1 - x_0), abs(grad - grad_)
        if y_1.abs() < err or ftol_ < ftol or xtol_ < xtol or gtol_ < gtol:
            break
        grad_, x_0, y_0 = grad, x_1, y_1
    return x_1, {
        'err': y_1,
        'ftol': ftol_,
        'xtol': xtol_,
        'gtol': gtol_,
        'nfev': n + 1
    }


def gradient_descent(
        func: Callable[[Union[NumberR, Matrix]], Matrix],
        x0: Union[Matrix, NumberR, Array],
        err: NumberR = 1e-8,
        ftol: NumberR = 1e-8,
        xtol: NumberR = 1e-8,
        gtol: NumberR = 1e-8,
        max_nfev: NumberR = 1e3,
        alpha: NumberR = 0.01,
        args: list = [],
        kwargs: dict = {}) -> Tuple[Matrix, Dict[str, Union[NumberR, Matrix]]]:
    """梯度下降法 计算多变量func极值"""
    x_0, grad_norm_1 = Matrix(x0), INF
    y_0 = func(x_0, *args, **kwargs)
    for n in range(int(max_nfev)):
        grad = jacobian(func, x_0, *args, **kwargs)
        grad_norm = grad.norm()
        xtol_arr = grad * alpha  # (alpha*(n+1)**0.1)
        x_1 = x_0 - xtol_arr
        y_1 = func(x_1, *args, **kwargs)
        # alpha = abs(xtol_arr) / ()
        ftol_, gtol_ = abs(y_1 - y_0), abs(grad_norm_1 - grad_norm)
        xtol_ = xtol_arr.norm()
        if y_1.abs() < err or ftol_ < ftol or xtol_ < xtol or gtol_ < gtol:
            break
        grad_norm_1, x_0, y_0 = grad_norm, x_1, y_1
    return x_1, {
        'err': y_1,
        'ftol': ftol_,
        'xtol': xtol_,
        'gtol': gtol_,
        'nfev': n + 1
    }


# %%% Levenberg-Marquardt模型 最小二乘
def levenberg_marquardt(
        func: Callable[[Union[NumberR, Matrix]], Matrix],
        x0: Union[Matrix, NumberR, Array],
        y0: Union[NumberR, Matrix] = 0.,
        stol: NumberR = 1e-12,
        ftol: NumberR = 1e-8,
        xtol: NumberR = 1e-8,
        gtol: NumberR = 1e-8,
        max_nfev: NumberR = 100,
        lam: NumberR = 1.,
        args: list = [],
        kwargs: dict = {}) -> Tuple[Matrix, Dict[str, Union[NumberR, Matrix]]]:
    """Levenberg-Marquardt(列文伯格-马夸尔特)(lm)算法模型计算最小二乘"""
    x_0, y0 = Matrix(x0), Matrix(y0)

    def se(x_in: Matrix, *args, **kwargs) -> Matrix:
        return y0 - func(x_in, *args, **kwargs)  # 残差

    def sse(x_in: Matrix, *args, **kwargs) -> Matrix:
        a = se(x_in, *args, **kwargs)
        return sum_squares(a)  # 和方差

    grad_, eye0 = INF, eye(x_0.size)
    y_0 = func(x_0, *args, **kwargs)
    sse_0 = sse(x_0, *args, **kwargs)

    for n in range(int(max_nfev)):
        jac = jacobian(se, x_0, *args, **kwargs)
        jact = jac.transpos()
        h_lm = inv(jact @ jac + lam * eye0) @ jact @ se(x_0, *args, **kwargs)
        x_1 = x_0 - h_lm
        sse_1, y_1 = sse(x_1, *args, **kwargs), func(x_1, *args, **kwargs)
        stol_, ftol_ = abs(sse_1 - sse_0), norm(y_1 - y_0)
        xtol_, gtol_ = norm(h_lm), norm(jac - grad_)

        if stol_ < stol or ftol_ < ftol or xtol_ < xtol or gtol_ < gtol:
            break
        if sse_1 > sse_0:
            lam *= 2
        else:
            lam /= 3
        x_0, sse_0, y_0, grad_ = x_1, sse_1, y_1, jac
    return x_1, {
        'nfev': n + 1,
        'stol': stol_,
        'ftol': ftol_,
        'xtol': xtol_,
        'gtol': gtol_,
        'sse': sse_1,
        'y': y_1,
        'y0': y0,
        'se': y0 - y_1,
        'rmse': (sse_1 / y0.size)**0.5,
        'R-square': 1 - sse_1 / sum_squares(y_1 - y0.mean())
    }


if __name__ == '__main__':
    
    data = ((450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510,
             515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575,
             580, 585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650),
            (146.571474, 145.823075, 145.491652, 144.740026, 144.628817,
             143.965309, 143.492719, 143.015041, 141.78422, 141.800691,
             141.506711, 140.843265, 140.60691, 140.329312, 140.051493,
             139.608754, 138.948754, 137.757343, 137.403268, 137.609801,
             137.258783, 136.727531, 136.443899, 136.575577, 135.741262,
             135.72312, 135.438998, 135.603508, 134.313808, 134.044323,
             134.305536, 134.096924, 133.354517, 132.983888, 133.142939,
             133.28946, 132.804627, 132.204945, 131.894646, 131.733284, 132.873662))
    res = {
        'x': Matrix([1.17479537e+02, 6.15118989, -4.72717596e-02]),
        'se':
        Matrix([
            -0.13151581, -0.26581629, -0.00200823, -0.17651866, 0.27198804,
            0.15147718, 0.2058147, 0.2396129, -0.49459384, 0.00419153,
            0.17876214, -0.02938534, 0.17679457, 0.32943438, 0.4700018,
            0.43422381, 0.17016671, -0.63592997, -0.61494652, -0.04325421,
            -0.03867052, -0.22355132, -0.16972921, 0.29078619, -0.22302016,
            0.07129393, 0.0918404, 0.55348549, -0.44636871, -0.43306284,
            0.10411119, 0.16484669, -0.31461846, -0.42851143, -0.01873802,
            0.37267663, 0.1270864, -0.2388325, -0.32068326, -0.25875314,
            1.09991373
        ]),
        'sse': 2.394890682886707 * 2,
        'nfev': 3,
        'grad': Matrix([2.45430354e-08, 9.92523326e-08, -2.19310754e-06]),
        'optimality': 2.193107544457007e-06
    }
    
    # x = Matrix(data[0])
    # y = Matrix([data[1], data[1]]).transpos()
    # y = Matrix(data[1])
    # cs = interp1d(x, y, 3)
    # yy = cs(x)
    def cauchy(p, x) -> Matrix:
        return p[0] + p[1] / x**2 + p[2] / x**4

    result = levenberg_marquardt(cauchy, [150., 0., 0.],
                                  Matrix(data[1]),
                                  args=[Matrix(data[0]) / 1000])
    print(result)
