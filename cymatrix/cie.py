# -*- coding: utf-8 -*-
"""
CIE for matrix
"""
from typing import List, Literal, Tuple, Union

from .cie_data import (Mrgb, Mrgb2, aKabHunter, aStandardIlluminant,
                       aWhitePoint, aWhitePointHunter, axyzL, aSIValue)
from .matrix import (Array, Matrix, NumberR, arange, argsort, arrayFromMatrix,
                     asarray, atan2, ceil, floor, hypot, nan2num, rad2deg)
from .sci import interp1d, ndim_check

_POW2 = Matrix(3)
_A1, _A2 = Matrix(216 / 24389), Matrix(6 / 29)
_B1, _B2, _B0 = Matrix(841 / 108), Matrix(108 / 841), Matrix(4 / 29)
_380 = Matrix(380.)


def _ff(t: Matrix):
    a = Matrix(t > _A1)
    return t.numrt(3) * a + (t * _B1 + _B0) * (1 - a)


def _ff_(t: Matrix):
    a = Matrix(t > _A2)
    return t**_POW2 * a + (t - _B0) * _B2 * (1 - a)


def _input_check(s: Matrix) -> Matrix:
    s = Matrix(s)
    if s.ndim not in (1, 2):
        raise ValueError('输入参数格式不正确')
    s = ndim_check(s)
    if s.shape[0] != 3:
        raise ValueError('输入参数格式不正确')
    return s


class CIEHueTransform:
    """
        CIE1931|1964|1976色度转换
        -----
        
        输入  格式 CIEHueTransform(SI,vi).*(x)
        -----
            SI : 光源 'A','D65','C','D50','D55','D75' 默认'D65'
            vi : 视场角 2或者10 默认2
            x : 相应的输入参数
            
        方法
        -----
            - xyz2lab(XYZ):   CIE XYZ to CIE 1976(L*a*b*)colour space
            - xyz2lab_h(XYZ): CIE XYZ to HunterLab
            - xyz2yuv(XYZ):   CIE XYZ to CIE Yu'v'
            - xyz2yxy(XYZ):   CIE XYZ to CIE Yxy
            - xyz2luv(XYZ):   CIE XYZ to CIE 1976 (L*u*v*) colour space
            - xyz2rgb(XYZ):   CIE XYZ to sRGB
            - lab2xyz(Lab):   CIE 1976 (L*a*b*) colour space to CIE XYZ
            - lab_h2xyz(Lab): HunterLab to CIE XYZ
            - yuv2xyz(Yuv):   CIE Yu'v' to CIE XYZ
            - yxy2xyz(Yxy):   CIE Yxy to CIE XYZ
            - luv2xyz(Luv):   CIE 1976 (L*u*v*) colour space to CIE XYZ
            - rgb2xyz(sRGB):  sRGB to CIE XYZ
            - chs(Lab|Luv):   CIE 1976 (L*a*b*)|(L*u*v*) colour space to Chs
            - lab2rgb(Lab):   CIE 1976 (L*a*b*) colour space to sRGB
            - rgb2lab(sRGB):  sRGB to CIE 1976 (L*a*b*) colour space
            - rgb16(sRGB):    sRGB to 16-sRGB
            - rbg16_(16-sRGB): 16-sRGB to sRGB
        """

    def __init__(self,
                 si: Literal['A', 'D65', 'C', 'D50', 'D55', 'D75'] = 'D65',
                 vi: Literal[2, 10] = 2):
        si = si.upper()
        if si in aSIValue:
            self.si_v = aSIValue.index(si)
        else:
            raise ValueError('err: 光源种类错误')
        if vi == 2:
            self.vi_v = 0
        elif vi == 10:
            self.vi_v = 1
        else:
            raise ValueError('err: 视场角错误')
        # 当前光源下的白点
        self.wp = aWhitePoint[self.vi_v][:, self.si_v]
        self.wp_h = aWhitePointHunter[self.vi_v][:, self.si_v]
        self.kab = aKabHunter[self.vi_v][:, self.si_v]

        self._wp_t = (self.wp, self.wp.reshape(self.wp.size, 1, inplace=False))
        self._wp_h_t = (self.wp_h, self.wp_h.reshape(self.wp_h.size, 1, inplace=False))

    def _xyz2lab(self, s: Matrix):
        wp = self._wp_t[s.ndim - 1]
        x1, y1, z1 = _ff(s / wp)
        L = y1 * 116 - 16
        a = (x1 - y1) * 500
        b = (y1 - z1) * 200
        return asarray((L, a, b))

    def _xyz2lab_h(self, s: Matrix):
        x1, y1, z1 = s / self._wp_h_t[s.ndim - 1]
        y2 = y1.numrt(2)
        L = y2 * 100
        a = self.kab[0] * (x1 - y1) / y2
        b = self.kab[1] * (y1 - z1) / y2
        return asarray((L, a, b))

    def _xyz2yuv(self, s: Matrix):
        x, y, z = s
        fm = x + y * 15 + z * 3
        return asarray((y, x / fm * 4, y / fm * 9))

    def _xyz2yxy(self, s: Matrix) -> Matrix:
        temp = s[:-1] / s.sum(axis=0)
        return asarray((s[1], *temp))

    def _xyz2luv(self, s: Matrix) -> Matrix:
        uv = self._xyz2yuv(s)[1:]
        uvn = self._xyz2yuv(self._wp_t[s.ndim - 1])[1:]
        L = _ff(s[1] / 100) * 116 - 16
        return asarray((L, (uv - uvn) * L * 13))

    def _lab2xyz(self, s: Matrix):
        L, a, b = s
        y1 = (L + 16) / 116
        x1 = a / 500 + y1
        z1 = y1 - b / 200
        r = asarray((x1, y1, z1))
        return _ff_(r) * self._wp_t[s.ndim - 1]

    def _lab_h2xyz(self, s: Matrix):
        L, a, b = s
        y0 = L**2 / 10000
        x0 = a / self.kab[0] * L / 100 + y0
        z0 = y0 - b / self.kab[1] * L / 100
        r = asarray((x0, y0, z0))
        return r * self._wp_h_t[s.ndim - 1]

    def _luv2xyz(self, s: Matrix):
        uvn = self._xyz2yuv(self._wp_t[s.ndim - 1])
        L = s[0]
        u_, v_ = s[1:] / 13 / L + uvn
        Y = _ff_((L + 16) / 116) * 100
        X = 9 / 4 * u_ / v_ * Y
        Z = Y / v_ * 3 - X / 3 - Y * 5
        return asarray((X, Y, Z))

    def _yxy2xyz(self, s: Matrix) -> Matrix:
        all_sum = s[0] / s[2]
        return asarray((s[1] * all_sum, s[0], (1 - s[1] - s[2]) * all_sum))

    def _yuv2xyz(self, s: Matrix) -> Matrix:
        # (y, 4 * x / fm, 9 * y / fm)
        y = s[0]
        fm = 9 * y / s[2] # x + 15 * y + 3 * z
        x = s[1] * fm / 4
        return asarray((x, y, (fm - x - 15 * y) / 3))

    def _chs(self, s: Matrix) -> Matrix:
        L, u, v = s
        C = hypot(u, v)
        h = rad2deg(atan2(v, u))
        s = C / L
        return asarray((C, h, s))

    def _xyz2rgb(self, s: Matrix) -> Matrix:
        """
        XYZ to sRGB
        ----------

        输入:
            s : XYZ矩阵, 行为 XYZ, 列为输入项
        ----------

        输出:
            sRGB矩阵 (RGB, 输入项)
        """
        y1 = Mrgb @ s / 100
        y1_ = y1**(1 / 2.4)
        nan2num(y1_)
        y2 = (y1_ - 0.055) * 1.055
        y3 = y1 * 12.92
        # http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html
        a = Matrix(y1 > 0.0031308)
        res = y2 * a + y3 * (1 - a)
        res[res > 1] = 1.
        res[res < 0] = 0.
        return res

    def _rgb2xyz(self, s: Matrix):
        s[s > 1] = 1
        s[s < 0] = 0
        y1 = s / 12.92
        y2 = ((s + 0.055) / 1.055)**2.4
        yy = y1 * Matrix(s <= 0.04045) + y2 * Matrix(s > 0.04045)
        return Mrgb2 @ yy * 100


    def xyz2lab(self, s: Matrix) -> Matrix:
        """
        XYZ to CIELab
        ----------

        输入：
            s : XYZ矩阵 行为 XYZ 列为输入项
        ----------

        输出：
            Lab矩阵 (CIELab, 输入项)
        """
        return self._xyz2lab(_input_check(s))

    def lab2xyz(self, s: Matrix) -> Matrix:
        """
        CIELab to XYZ
        ----------

        输入：
            s : Lab矩阵 行为 CIELab 列为输入项
        ----------

        输出：
            XYZ矩阵 (XYZ, 输入项)
        """
        return self._lab2xyz(_input_check(s))

    def xyz2lab_h(self, s: Matrix) -> Matrix:
        """XYZ to HunterLab"""
        return self._xyz2lab_h(_input_check(s))

    def lab_h2xyz(self, s: Matrix) -> Matrix:
        """HunterLab to XYZ"""
        return self._lab_h2xyz(_input_check(s))

    def xyz2yuv(self, s: Matrix) -> Matrix:
        """XYZ to Yu'v'"""
        return self._xyz2yuv(_input_check(s))

    def yuv2xyz(self, s: Matrix) -> Matrix:
        """Yu'v' to XYZ"""
        return self._yuv2xyz(_input_check(s))

    def xyz2luv(self, s: Matrix) -> Matrix:
        """XYZ to CIELuv"""
        return self._xyz2luv(_input_check(s))

    def luv2xyz(self, s: Matrix) -> Matrix:
        """CIELuv to XYZ"""
        return self._luv2xyz(_input_check(s))

    def chs(self, s: Matrix) -> Matrix:
        return self._chs(_input_check(s))

    def xyz2yxy(self, s: Matrix) -> Matrix:
        """XYZ to Yxy"""
        return self._xyz2yxy(_input_check(s))

    def yxy2xyz(self, s: Matrix) -> Matrix:
        """Yxy to XYZ"""
        return self._yxy2xyz(_input_check(s))

    def xyz2rgb(self, s: Matrix) -> Matrix:
        """
        XYZ to sRGB
        ----------

        输入:
            s : XYZ矩阵, 行为 XYZ, 列为输入项
        ----------

        输出：
            sRGB矩阵 (RGB, 输入项)
        """
        # assert np.all(s >= 0 & s <= 100)
        return self._xyz2rgb(s)

    def rgb2xyz(self, s: Matrix) -> Matrix:
        """sRGB to XYZ"""
        return self._rgb2xyz(_input_check(s))

    def lab2rgb(self, s: Matrix) -> Matrix:
        """CIELab to sRGB"""
        return self._xyz2rgb(self._lab2xyz(_input_check(s)))

    def rgb2lab(self, s: Matrix) -> Matrix:
        """sRGB to CIELab"""
        return self._xyz2lab(self._rgb2xyz(_input_check(s)))

    def rgb16(self, s: Matrix, upper: NumberR = 255) -> List[str]:
        """RGB[int,int,int] 转 16进制"""
        if upper != 255:
            s = (s/upper*255).round()
        s[s > 255] = 255
        s[s < 0] = 0
        # s = s.astype(int)
        if s.ndim == 1 or s.shape[0] == 1:
            s.reshape(3, 1, inplace=True)
        sshape0 = s.shape[0]
        s_ = []
        for ic in range(s.shape[1]):
            ss_ = '#'
            for ir in range(sshape0):
                s0 = hex(int(s[ir, ic]))[2:]
                if len(s0) == 1:
                    s0 = '0'+s0
                ss_ += s0
            s_.append(ss_)
        return s_

    def rgb16_(self, rgbtxt: Union[List[str], Tuple[str], str], upper: NumberR=255) -> Matrix:
        """输出[0,1]的RGB"""
        def f(val:str):
            if val[0] != '#' or len(val) != 7:
                raise ValueError('sRGB格式错误, 请输入带#的16进制颜色码')
            return [int('0x'+val[1:3], 16), int('0x'+val[3:5], 16), int('0x'+val[5:7], 16)]

        if isinstance(rgbtxt, str):
            rgb = Matrix(f(rgbtxt))
        else:
            rgblst = [f(val) for val in rgbtxt]
            rgb = Matrix(rgblst).transpos()
        return rgb/upper


class CIE(CIEHueTransform):
    """
    CIE1931|1964|1976色度计算
    -----

    输入  格式 CIE(lam0,SI,vi).*(x)
    ::
    
        lam0 : 波长
        SI : 光源 'A','D65','C','D50','D55','D75' 默认'D65'
        vi : 视场角 2或者10 默认2
        x : 相应的输入参数
    
    -----

    方法
    ::
    
        .xyz2lab(XYZ) CIE XYZ to CIE 1976(L*a*b*)colour space
        .xyz2lab_h(XYZ) CIE XYZ to HunterLab
        .lab2xyz(Lab) CIE 1976 (L*a*b*) colour space to CIE XYZ
        .lab_h2xyz(Lab) HunterLab to CIE XYZ
        .xyz2uvw(XYZ) CIE XYZ to CIE u'v'w'
        .spe2uvw(spe) spectrum to CIE u'v'w'
        .xyz2luv(XYZ) CIE XYZ to CIE 1976 (L*u*v*) colour space
        .luv2xyz(Luv) CIE 1976 (L*u*v*) colour space to CIE XYZ
        .xyz2rgb(XYZ) CIE XYZ to sRGB
        .lab2rgb(Lab) CIE 1976 (L*a*b*) colour space to sRGB
    """

    def __init__(self,
                 data: Union[Matrix, Array],
                 si: Literal['A', 'D65', 'C', 'D50', 'D55', 'D75'] = 'D65',
                 vi: Literal[2, 10] = 2):
        super().__init__(si=si, vi=vi)
        if isinstance(data, (Matrix, list, tuple)):
            data = Matrix(data)
            data = data[argsort(data[:,0])[0]]
            w = data[:, 0]
            data = data[:, 1:]
        else:
            raise ValueError('data格式错误')

        if w.max() < 3:
            w *= 1000
        wn = w.min()
        wm = w.max()
        cs = interp1d(w, data, kind=3)

        dy = w.diff()
        if all(dy == dy[0]):
            step = int(dy[0])
            if step >= 10:
                step = 10
            elif step <= 1:
                step = 1
        else:
            dy_mean = dy.mean().round()
            if dy_mean <= 1:
                step = 1
            elif dy_mean >= 10:
                step = 10
            else:
                step = int(dy_mean)
        if wn > 400:
            raise ValueError('光谱最小波长不能大于400nm')
        wn = int(floor(wn / step) * step) if wn > 380 else 380
        if wm < 700:
            raise ValueError('光谱最大波长不能小于700nm')
        wm = int(ceil(wm / step) * step) if wm < 780 else 780
        lam = arange(wn, wm + step, step)

        self.data = cs(lam)
        self.si0 = aStandardIlluminant[lam-_380][:, 1 + self.si_v]  # 当前波长下的光源功率分布
        self.xyzl0 = axyzL[self.vi_v][lam-_380].transpos()[1:]  # 当前波长和光源下的色匹配函数
        self.sxyzl = self.si0 * self.xyzl0

        self.data_ex = lambda x: cs(x)

    def spe2xyz(self) -> Matrix:
        """
        spectrum to XYZ
        ----------

        输出：
            XYZ矩阵 (XYZ, 输入项)
        """
        # sumXYZ = self.sxyzl@t/100
        # k = 100/np.sum(sxyz[1])
        return self.sxyzl @ self.data / self.sxyzl[1].sum()

    def spe2uv_(self) -> Matrix:
        """spectrum to Yuv"""
        return self._xyz2yuv(self.spe2xyz())

    def spe2lab(self) -> Matrix:
        """spectrum to CIELAB"""
        return self._xyz2lab(self.spe2xyz())

    def spe2lab_h(self) -> Matrix:
        """spectrum to HunterLAB"""
        return self._xyz2lab_h(self.spe2xyz())

    def spe2rgb(self) -> Matrix:
        """spectrum to sRGB"""
        return self.xyz2rgb(self.spe2xyz())
