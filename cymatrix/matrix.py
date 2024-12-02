# -*- coding: utf-8 -*-
"""
矩阵 cython
"""

from abc import ABC, abstractmethod
from array import array as parray
from random import random as _random
from typing import List, Literal, Sequence, Tuple, Union, overload

from . import cymatrix as _cm
from .cymatrix import cyMatrix

NumberR = Union[int, float, bool]
NumberR_ = (int, float, bool)
Array = Union[list, tuple, parray, range]
Array_ = (list, tuple, parray, range)

interp1d_2_mco = _cm.interp1d_2_mco
interp1d_3_mco = _cm.interp1d_3_mco


# %% matrix class
def obj2cm(obj: Union[NumberR, Array, cyMatrix]) -> cyMatrix:
    if isinstance(obj, cyMatrix):
        return obj
    if isinstance(obj, NumberR_):
        return _cm.buildFromArr(parray('d', [obj]), 1, 1, 0)
    if isinstance(obj, Array_):
        if isinstance(obj[0], Array_):
            lenl = [len(ii) for ii in obj]
            lenll = len(lenl)
            if len(set(lenl)) != 1:
                raise ValueError('matrix.obj2cm: matrix 的每行元素数必须相同')
            res = parray('d')
            for ii in obj:
                for jj in ii:
                    res.append(jj)
            return _cm.buildFromArr(res, lenll, lenl[0], 2)
        return _cm.buildFromArr(parray('d', obj), 1, len(obj), 1)
    raise ValueError('matrix.obj2cm: obj 必须是 Real or list or tuple or cyMatrix')


def list2iarr(lst: Union[float, int, list, tuple]):
    if isinstance(lst, (float, int)):
        return parray('i', [int(lst)])
    if isinstance(lst, (list, tuple)):
        return parray('i', [int(i) for i in lst])
    raise ValueError('list2iarr: lst 必须是 float or int or list or tuple, list 和 tuple 中的元素必须是 float or int')


def list2barr(lst: Union[bool, int, list, tuple]):
    if isinstance(lst, (bool, int)):
        return parray('b', [lst])
    if isinstance(lst[0], (list, tuple)):
        res = parray('b')
        for ii in lst:
            for jj in ii:
                res.append(jj)
        return res
    return parray('b', lst)


class Matrix(object):
    """
    基于`cython`的矩阵类  用法和`numpy`基本一致
    ----------
    只支持`ndim <= 2`的矩阵
    ::
    
        ndim == 0 表示数字
        ndim == 1 表示一维向量
        ndim == 2 表示二维矩阵
    
    只支持`float`类型
    
    输入参数可以是数字，一维向量，二维矩阵
    
    ----------
    实例变量

    ::

        .base  (cyMatrix)                                   cython 的矩阵对象
        .dtype (str)                                        'f8'
        .value (list[list[float]] | list[float] | float)    矩阵各个元素组成的list property
        .ndim  (int)                                        矩阵维度 property
        .shape (tuple[int] | tuple[()])                     矩阵形状 property
        .size  (int)                                        矩阵元素个数 property
    
    ----------
    
    方法

    ::

        transpos() -> Matrix
            求转置
        reshape(*shape, inplace: bool) -> Matrix | None
            变形
        value1d() -> list[float]
            所有元素组成的 1 维 list
        tolist(dtype: str) -> list[list[NumberR]]] | list[NumberR] | NumberR
            将矩阵转换为 dtype格式的 list, dtype in ('float', 'int', 'bool')
        copy() -> Matrix
            本实例的拷贝
        _copy() -> cyMatrix
            本实例的 base 的拷贝
        fill(value: float | int | bool) -> Matrix
            用 value 填充矩阵
        flip(axis: 0 | 1) -> Matrix
            翻转矩阵
            axis==0 翻转行, axis==1 翻转列
        vector(axis: None | 0 | 1) -> Matrix
            将矩阵变成向量
                axis is None:
                    ndim==2 and 1 in shape: 二维向量 --> 一维向量
                axis==0:
                    ndim==1: 一维向量 --> 二维行向量
                axis==1:
                    ndim==1: 一维向量 --> 二维列向量
                其余格式不变
        index(value: float) -> list[int] | list[list[int]] | [] | [[]]
            找出矩阵中 value 的位置, 如果 value 不存在, 返回空列表
                ndim==1: list[int]
                ndim==2: list[list[int], list[int]] 第一个list[int] 表示行，第二个list[int] 表示列
        extract(condition: list[[bool]]] | list[bool] | bool) -> Matrix
            提取矩阵中满足 condition 元素, 返回一个一维矩阵
        compress(condition: list[[bool]]] | list[bool] | bool, axis: None | 0 | 1) -> Matrix
            沿给定 axis 返回数组的选定切片
        numrt(num: int) -> Matrix
            对矩阵开 num 次方
        sin(isRad: bool) -> Matrix
            对矩阵求正弦
                isRad: 是否为弧度
        cos(isRad: bool) -> Matrix
            对矩阵求余弦
                isRad: 是否为弧度
        tan(isRad: bool) -> Matrix
            对矩阵求正切
                isRad: 是否为弧度
        abs() -> Matrix
            对矩阵求绝对值
        floor() -> Matrix
            对矩阵每个元素向下取整
        ceil() -> Matrix
            对矩阵每个元素向上取整
        round(digits : int) -> Matrix
            对矩阵每个元素四舍五入 digits 位小数
        sort(axis: 0 | 1 | other int) -> Matrix
            沿着 axis 对矩阵排序
                axis==0: 按行排序
                axis==1: 按列排序
                axis==other int: 当成一维矩阵排序
        max(axis: 0 | 1 | other int) -> Matrix
            求矩阵的最大值
                axis==0: 求每行的最大值
                axis==1: 求每列的最大值
                axis==other int: 计算全部元素的最大值
        min(axis: 0 | 1 | other int) -> Matrix
            求矩阵的最小值
                axis==0: 求每行的最小值
                axis==1: 求每列的最小值
                axis==other int: 计算全部元素的最小值
        mean(axis: 0 | 1 | other int) -> Matrix
            求矩阵的均值
                axis==0: 求每行的均值
                axis==1: 求每列的均值
                axis==other int: 计算全部元素的均值
        median(axis: 0 | 1 | other int) -> Matrix
            求矩阵的中位数
                axis==0: 求每行的中位数
                axis==1: 求每列的中位数
                axis==other int: 计算全部元素的中位数
        sum(axis: 0 | 1 | other int) -> Matrix
            求矩阵的累加
                axis==0: 求每行的累加
                axis==1: 求每列的累加
                axis==other int: 计算全部元素的累加
        prod(axis: 0 | 1 | other int) -> Matrix
            求矩阵的累乘
                axis==0: 求每行的累乘
                axis==1: 求每列的累乘
                axis==other int: 计算全部元素的累乘
        norm(axis: 0 | 1 | other int) -> Matrix
            求矩阵的二阶范数
                axis==0: 求每行的二阶范数
                axis==1: 求每列的二阶范数
                axis==other int: 计算全部元素的二阶范数
        rms(axis: 0 | 1 | other int) -> Matrix
            求矩阵的均方差
                axis==0: 求每行的均方差
                axis==1: 求每列的均方差
                axis==other int: 计算全部元素的均方差
        var(axis: 0 | 1 | other int) -> Matrix
            求矩阵的方差
                axis==0: 求每行的方差
                axis==1: 求每列的方差
                axis==other int: 计算全部元素的方差
        std(axis: 0 | 1 | other int) -> Matrix
            求矩阵的标准差
                axis==0: 求每行的标准差
                axis==1: 求每列的标准差
                axis==other int: 计算全部元素的标准差
        ptp(axis: 0 | 1 | other int) -> Matrix
            求矩阵的极差
                axis==0: 求每行的极差
                axis==1: 求每列的极差
                axis==other int: 计算全部元素的极差
        diff(axis: 0 | 1 | other int) -> Matrix
            求矩阵的差分
                axis==0: 求每行的差分
                axis==1: 求每列的差分
                axis==other int: 计算全部元素的差分
    """

    base: cyMatrix
    dtype: str = 'f8'

    def __new__(cls, obj: Union[NumberR, Array, cyMatrix, 'Matrix']):
        if isinstance(obj, Matrix):
            # 如果第一个参数是一个 Matrix，直接返回这个 Matrix 的引用作为该实例的引用
            return obj
        else:
            clas = super().__new__(cls)
            clas.base = obj2cm(obj)
            return clas
        # return super().__new__(cls)
    # def __init__(self, base: Union[NumberR, Array, cyMatrix]) -> None:
    #     self.base = obj2cm(base)

    def __class_getitem__(cls, item_types):

        return cls

    @property
    def shape(self) -> Union[Tuple[int, int], Tuple[int], Tuple[()]]:
        return _cm.get_shape(self.base)

    @property
    def ndim(self) -> int:
        return _cm.get_ndim(self.base)

    @property
    def size(self) -> int:
        return _cm.get_size(self.base)

    @property
    def value(self) -> Union[List[List[float]], List[float], float]:
        return _cm.get_value(self.base)

    @overload
    def tolist(self, dtype:Literal['float']) -> Union[List[List[float]], List[float], float]:...
    @overload
    def tolist(self, dtype:Literal['int']) -> Union[List[List[int]], List[int], int]:...
    @overload
    def tolist(self, dtype:Literal['bool']) -> Union[List[List[bool]], List[bool], bool]:...
    def tolist(self, dtype:Literal['float', 'int', 'bool']='float'):
        if dtype not in ('float', 'int', 'bool'):
            raise ValueError('tolist: dtype 必须是 float or int or bool')
        if dtype == 'float':
            return _cm.get_value(self.base)
        if dtype == 'int':
            return _cm.get_value_int(self.base)
        if dtype == 'bool':
            return _cm.get_value_bool(self.base)

    def value1d(self) -> List[float]:
        return _cm.ravel(self.base)

    def copy(self) -> 'Matrix':
        return Matrix(self._copy())

    def __copy__(self) -> 'Matrix':
        return Matrix(self._copy())

    def _copy(self) -> cyMatrix:
        return _cm.copy(self.base)

    def fill(self, value: NumberR) -> 'Matrix':
        _cm.fill(self.base, value)
        return self

    def flip(self, axis:int=0) -> 'Matrix':
        # axis == 0 翻转行 axis == 1 翻转列
        if self.ndim == 1:
            return Matrix(_cm.flip(self.base, 1))
        return Matrix(_cm.flip(self.base, axis))

    def transpos(self) -> 'Matrix':
        return Matrix(_cm.transpos(self.base))

    @overload
    def reshape(self, *shape: int, inplace: Literal[True]) -> None:...
    @overload
    def reshape(self, *shape: int, inplace: Literal[False]) -> 'Matrix':...
    def reshape(self, *shape: int, inplace: bool = False):
        len_ = len(shape)
        if inplace:
            if len_ == 1:
                _cm.reshape(self.base, 1, 1, shape[0])
            elif len_ == 2:
                _cm.reshape(self.base, 2, shape[0], shape[1])
            else:
                raise ValueError('reshape: 参数数量错误')
        else:
            if len_ == 1:
                return Matrix(_cm.reshape(self._copy(), 1, 1, shape[0]))
            if len_ == 2:
                return Matrix(_cm.reshape(self._copy(), 2, shape[0], shape[1]))
            raise ValueError('reshape: 参数数量错误')

    def vector(self, axis: Union[Literal[0, 1], None] = None) -> 'Matrix':
        """
        向量转换 : ndim == 1 or 1 in shape
        -----
        axis
            - `None`  二维向量 --> 一维向量
            
            - `0`     一维向量 --> 二维行向量
            
            - `1`     一维向量 --> 二维列向量
            
        非向量的格式保持不变
        """
        ndim = self.ndim
        size: int = self.size
        if ndim == 0:
            return self
        if axis is None:
            if ndim == 2 and 1 in self.shape:
                # 二维向量 转 一维向量
                return self.reshape(size, inplace=False)
        elif ndim == 1:
            if axis == 0:
                # 一维向量 转 二维行向量
                return self.reshape(1, size, inplace=False)
            elif axis == 1:
                # 一维向量 转 二维列向量
                return self.reshape(size, 1, inplace=False)
            else:
                raise ValueError('vector: 参数错误')
        return self

    def __str__(self) -> str:
        return str(self.value).replace('],', '],\n')

    def __repr__(self) -> str:

        def fun(s: str):
            return 'Matrix(' + s + ')'

        if isinstance(self.value, float):
            return str(self.value)
        if not isinstance(self.value[0], list):
            return fun(str(self.value))

        value: List[List[float]] = self.value
        num = 0
        for ir in value:
            for ic in ir:
                num_ = len(str(ic))
                if num_ > num:
                    num = num_
        tex = '['
        for ir in value:
            tex += '['
            for ic in ir:
                tex += ' ' + str(ic).ljust(num) + ','
            tex = tex[:-1]
            tex += '],\n' + ' ' * 8
        tex = tex[:-10]
        tex += ']'
        return fun(tex)

    def __hash__(self) -> int:
        return hash(self.base)

    def __reduce__(self):
        return (self.__class__, (self.value,))

    def __len__(self) -> int:
        if self.ndim == 0:
            return 0
        return self.shape[0]

    def _getitem_tuple(self, key0, key1) -> 'Matrix':
        is_key0_int = isinstance(key0, int)
        is_key0_slice = isinstance(key0, slice)
        is_key0_list = isinstance(key0, list)
        is_key1_int = isinstance(key1, int)
        is_key1_slice = isinstance(key1, slice)
        is_key1_list = isinstance(key1, list)
        if not (is_key0_int | is_key0_slice | is_key0_list) and not (is_key1_int | is_key1_slice | is_key1_list):
            raise ValueError('key 只能是 int 或 slice 或 list')
        if is_key0_int & is_key1_int:
            ndim = 0
            key0 = slice(key0, key0 + 1)
            key1 = slice(key1, key1 + 1)
            return Matrix(_cm.getitems(self.base, ndim, key0, key1))
        elif is_key0_int & is_key1_slice:
            key0 = slice(key0, key0 + 1)
            ndim = 1
            return Matrix(_cm.getitems(self.base, ndim, key0, key1))
        elif is_key1_int & is_key0_slice:
            key1 = slice(key1, key1 + 1)
            ndim = 1
            return Matrix(_cm.getitems(self.base, ndim, key0, key1))
        elif is_key0_slice & is_key1_slice:
            ndim = 2
            return Matrix(_cm.getitems(self.base, ndim, key0, key1))

        if is_key0_list & is_key1_list:
            ndim = 2
            is_key00_bool = isinstance(key0[0], bool)
            is_key00_num = isinstance(key0[0], (int, float))
            is_key10_bool = isinstance(key1[0], bool)
            is_key10_num = isinstance(key1[0], (int, float))
            if is_key00_num:
                _cym = _cm.matrix_byindex(self.base, list2iarr(key0), 0)
                if is_key10_num:
                    return Matrix(_cm.matrix_byindex(_cym, list2iarr(key1), 1))
                if is_key10_bool:
                    return Matrix(_cm.compress(_cym, list2barr(key1), 1))
            if is_key00_bool:
                _cym = _cm.compress(self.base, list2barr(key0), 0)
                if is_key10_num:
                    return Matrix(_cm.matrix_byindex(_cym, list2iarr(key1), 1))
                if is_key10_bool:
                    return Matrix(_cm.compress(_cym, list2barr(key1), 1))

        if is_key0_list:
            is_key00_bool = isinstance(key0[0], bool)
            is_key00_num = isinstance(key0[0], (int, float))
            if is_key00_num:
                _cym = _cm.matrix_byindex(self.base, list2iarr(key0), 0)
            elif is_key00_bool:
                _cym = _cm.compress(self.base, list2barr(key0), 0)
            else:
                raise ValueError('key0[0] 必须是 int 或 bool')

            if is_key1_list:
                is_key10_bool = isinstance(key1[0], bool)
                is_key10_num = isinstance(key1[0], (int, float))
                ndim = 2
                if is_key10_bool:
                    return Matrix(_cm.compress(_cym, list2barr(key1), 1))
                if is_key10_num:
                    return Matrix(_cm.matrix_byindex(_cym, list2iarr(key1), 1))
            else:
                ndim = 1
                key0_ = slice(0, None)
                key1_ = slice(key1, key1 + 1) if is_key1_int else key1
                return Matrix(_cm.getitems(_cym, ndim, key0_, key1_))
        else:
            ndim = 2
            key1_ = slice(0, None)
            key0_ = slice(key0, key0 + 1) if is_key0_int else key0
            _cym = _cm.getitems(self.base, ndim, key0_, key1_)

            is_key10_bool = isinstance(key1[0], bool)
            is_key10_num = isinstance(key1[0], (int, float))
            if is_key10_bool:
                return Matrix(_cm.compress(_cym, list2barr(key1), 1))
            if is_key10_num:
                return Matrix(_cm.matrix_byindex(_cym, list2iarr(key1), 1))
            raise ValueError('key1[0] 必须是 int 或 bool')

    def __getitem__(self, key) -> 'Matrix':
        selfndim = self.ndim
        if selfndim == 0:
            raise IndexError('ndim == 0 时，不能索引')
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('key 的长度必须是 2')
            if selfndim != 2:
                raise ValueError('矩阵维度不匹配索引: len(key) == 2 but ndim != 2')
            key0, key1 = key
            return self._getitem_tuple(key0, key1)

        if isinstance(key, int):
            if selfndim == 1:
                ndim = 0
                key1, key0 = slice(key, key + 1), slice(0, 1, 1)
            else:
                ndim = 1
                key0, key1 = slice(key, key + 1), slice(0, None, 1)
        elif isinstance(key, slice):  # type(key) == slice
            if selfndim == 1:
                ndim = 1
                key1 = key
                key0 = slice(0, 1, 1)
            else:
                ndim = 2
                key0 = key
                key1 = slice(0, None, 1)
        elif isinstance(key, list):
            if isinstance(key[0], (float, int)):
                key = list2iarr(key)
                return Matrix(_cm.matrix_byindex(self.base, key, 0))
            if selfndim == 1 and isinstance(key[0], bool):
                key = list2barr(key)
                return Matrix(_cm.extract(key, self.base))
            if isinstance(key[0][0], bool):
                key = list2barr(key)
                return Matrix(_cm.extract(key, self.base))
            raise ValueError('key 类型错误')
        elif isinstance(key, Matrix):
            key = list2iarr(_cm.ravel(key.base))
            return Matrix(_cm.matrix_byindex(self.base, key, 0))
        else:
            raise ValueError('key 类型错误')
        return Matrix(_cm.getitems(self.base, ndim, key0, key1))

    def __setitem__(self, key, value: Union[NumberR, Array, cyMatrix, 'Matrix']) -> None:
        value = Matrix(value)
        if isinstance(key, tuple):
            key0, key1 = key
            isk0s, isk0i = isinstance(key0, slice), isinstance(key0, int)
            isk1s, isk1i = isinstance(key1, slice), isinstance(key1, int)
            if isk0s:
                step0 = 1 if key0.step is None else key0.step
                key0 = 0 if key0.start is None else key0.start
                if isk1i and value.ndim == 1:
                    value = value.reshape(value.size, 1)
            elif isk0i:
                step0 = 1
            else:
                if isinstance(key0, list):
                    if isinstance(key0[0], bool):
                        if isk1s:
                            if key1.start is None or key1.start == 0:
                                _cm.compress_setitem(list2barr(key0), self.base, value.base, 0)
                                return None
                        raise ValueError('使用bool作为key时,只能进行整行整列的赋值')
                raise ValueError('key 类型错误')
            if isk1s:
                step1 = 1 if key1.step is None else key1.step
                key1 = 0 if key1.start is None else key1.start
            elif isk1i:
                step1 = 1
            else:
                if isinstance(key1, list):
                    if isinstance(key1[0], bool):
                        if isk0s:
                            if key0.start is None or key0.start == 0:
                                _cm.compress_setitem(list2barr(key1), self.base, value.base, 1)
                                return None
                        raise ValueError('使用bool作为key时,只能进行整行整列的赋值')
                raise ValueError('key 类型错误')
        elif isinstance(key, list):
            if self.ndim == 1:
                if isinstance(key[0], bool):
                    _cm.compress_setitem(list2barr(key), self.base, value.base, -1)
                    return None
            elif isinstance(key[0][0], bool):
                _cm.compress_setitem(list2barr(key), self.base, value.base, -1)
                return None
        else:
            if isinstance(key, int):
                step = 1
            elif isinstance(key, slice):
                step = 1 if key.step is None else key.step
                key = 0 if key.start is None else key.start
            else:
                raise ValueError('key 类型错误')

            if self.ndim == 1:
                key0, step0 = 0, 1
                key1, step1 = key, step
            else:
                key0, step0 = key, step
                key1, step1 = 0, 1

        _cm.setitems(self.base, value.base, key0, key1, step0, step1)

    def __iter__(self):
        # 迭代器
        if self.ndim == 0:
            raise ValueError('0维 Matrix 不能形成迭代器')
        return MatrixIter(self)

    def __float__(self) -> float:
        if self.ndim == 0:
            return self.value
        else:
            raise ValueError('float: Matrix 不是 0 维')

    def __int__(self) -> int:
        if self.ndim == 0:
            return int(self.value)
        else:
            raise ValueError('int: Matrix 不是 0 维')

    def index(self, value:float)-> Union[List[list], list]:
        ind = _cm.index(self.base, value)
        if self.ndim == 1:
            ind = ind[1]
        return ind

    def __abs__(self) -> 'Matrix':
        return Matrix(_cm.abs(self.base))

    def extract(self, condition: Union[bool, List[bool], List[List[bool]]]) -> 'Matrix':
        return Matrix(_cm.extract(list2barr(condition), self.base))

    def compress(self, condition: Union[bool, List[bool], List[List[bool]]], axis: int = -1) -> 'Matrix':
        return Matrix(_cm.compress(list2barr(condition), self.base, axis))

    def _compress_setitem(self, condition: Union[bool, List[bool], List[List[bool]]], value: cyMatrix, axis: int = -1) -> None:
        _cm.compress_setitem(list2barr(condition), self.base, value, axis)

    def __pos__(self) -> 'Matrix':
        """+self"""
        return self

    def __neg__(self) -> 'Matrix':
        """-self"""
        res = cyMatrix(1, 1, 0)
        _cm.setitem(res, -1., 0, 0)
        return Matrix(_cm.imul(self._copy(), res))
    # def _operator_build(func):
    #     def f(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
    #         res: cyMatrix = func(self, other)
    #         return Matrix(res)
    #     return f

    # @_operator_build
    def __add__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self + other"""
        return Matrix(_cm.iadd(self._copy(), Matrix(other).base))

    def __sub__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self - other"""
        return Matrix(_cm.isub(self._copy(), Matrix(other).base, True))

    def __mul__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self * other"""
        return Matrix(_cm.imul(self._copy(), Matrix(other).base))

    def __truediv__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self / other"""
        return Matrix(_cm.itruediv(self._copy(), Matrix(other).base, True))

    def __floordiv__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self // other"""
        return Matrix(_cm.ifloordiv(self._copy(), Matrix(other).base, True))

    def __mod__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self % other"""
        return Matrix(_cm.imod(self._copy(), Matrix(other).base, True))

    def __pow__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self ** other"""
        oth = Matrix(other)
        if oth.ndim != 0 and self.ndim != 0:
            raise ValueError('没有定义矩阵指数')
        return Matrix(_cm.ipow(self._copy(), oth.base))

    def __matmul__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """self @ other"""
        return Matrix(_cm.matmul(self.base, Matrix(other).base))

    def __rmatmul__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other @ self"""
        return Matrix(_cm.matmul(Matrix(other).base, self.base))

    def __rpow__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other ** self"""
        oth = Matrix(other)
        if oth.ndim != 0 and self.ndim != 0:
            raise ValueError('没有定义矩阵指数')
        return Matrix(_cm.ipow(oth.base, self._copy()))

    def __radd__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other + self"""
        return Matrix(_cm.iadd(self._copy(), Matrix(other).base))

    def __rsub__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other - self"""
        return Matrix(_cm.isub(self._copy(), Matrix(other).base, False))

    def __rmul__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other * self"""
        return Matrix(_cm.imul(self._copy(), Matrix(other).base))

    def __rtruediv__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other / self"""
        return Matrix(_cm.itruediv(self._copy(), Matrix(other).base, False))

    def __rfloordiv__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other // self"""
        return Matrix(_cm.ifloordiv(self._copy(), Matrix(other).base, False))

    def __rmod__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']) -> 'Matrix':
        """other % self"""
        return Matrix(_cm.imod(self._copy(), Matrix(other).base, False))

    def __iadd__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self + other"""
        _cm.iadd(self.base, Matrix(other).base)
        return self

    def __isub__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self - other"""
        _cm.isub(self.base, Matrix(other).base, True)
        return self

    def __imul__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self * other"""
        _cm.imul(self.base, Matrix(other).base)
        return self

    def __itruediv__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self / other"""
        _cm.itruediv(self.base, Matrix(other).base, True)
        return self

    def __ifloordiv__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self // other"""
        _cm.ifloordiv(self.base, Matrix(other).base, True)
        return self

    def __imod__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self % other"""
        _cm.imod(self.base, Matrix(other).base, True)
        return self

    def __ipow__(self, other: Union[NumberR, Array, cyMatrix, 'Matrix']):
        """self ** other"""
        oth = Matrix(other)
        if oth.ndim != 0 and self.ndim != 0:
            raise ValueError('没有定义矩阵指数')
        _cm.ipow(self.base, oth.base)
        return self

    def __lt__(self, o) -> Union[bool, List[bool], List[List[bool]]]:
        """self < o"""
        return _cm.lt(self.base, Matrix(o).base)

    def __gt__(self, o) -> Union[bool, List[bool], List[List[bool]]]:
        """self > o"""
        return _cm.gt(self.base, Matrix(o).base)

    def __le__(self, o) -> Union[bool, List[bool], List[List[bool]]]:
        """self <= o"""
        return _cm.le(self.base, Matrix(o).base)

    def __ge__(self, o) -> Union[bool, List[bool], List[List[bool]]]:
        """self >= o"""
        return _cm.ge(self.base, Matrix(o).base)

    def __eq__(self, o) -> Union[bool, List[bool], List[List[bool]]]:
        """self == o"""
        return _cm.eq(self.base, Matrix(o).base)

    def __ne__(self, o) -> Union[bool, List[bool], List[List[bool]]]:
        """self != o"""
        return _cm.ne(self.base, Matrix(o).base)
    # def _method_build(func):
    #     def f(self, *args, **kwargs) -> 'Matrix':
    #         res: cyMatrix = func(self, *args, **kwargs)
    #         return Matrix(res)
    #     return f

    def numrt(self, num:int=2) -> 'Matrix': return Matrix(_cm.numrt(self.base, num))
    def sin(self, isRad=True) -> 'Matrix': return Matrix(_cm.sin(self.base, isRad))
    def cos(self, isRad=True) -> 'Matrix': return Matrix(_cm.cos(self.base, isRad))
    def tan(self, isRad=True) -> 'Matrix': return Matrix(_cm.tan(self.base, isRad))
    # def asin(self) -> 'Matrix': return Matrix(_asin(self.base))
    # def acos(self) -> 'Matrix': return Matrix(_acos(self.base))
    # def atan(self) -> 'Matrix': return Matrix(_atan(self.base))
    # def exp(self) -> 'Matrix': return Matrix(_exp(self.base))
    # def log(self) -> 'Matrix': return Matrix(_log(self.base))
    def abs(self) -> 'Matrix': return Matrix(_cm.abs(self.base))
    def floor(self) -> 'Matrix': return Matrix(_cm.floor(self.base))
    def ceil(self) -> 'Matrix': return Matrix(_cm.ceil(self.base))
    def round(self, digits: int = 0) -> 'Matrix': return Matrix(_cm.round(self.base, digits))
    def sort(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.sort(self.base, axis))
    def max(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.max(self.base, axis))
    def min(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.min(self.base, axis))
    def mean(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.mean(self.base, axis))
    def median(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.median(self.base, axis))
    def sum(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.sum(self.base, axis))
    def prod(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.prod(self.base, axis))
    def norm(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.norm(self.base, axis))
    def rms(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.rms(self.base, axis))
    def var(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.var(self.base, axis))
    def std(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.std(self.base, axis))
    def ptp(self, axis: int = -1) -> 'Matrix': return Matrix(_cm.ptp(self.base, axis))
    def diff(self, axis: int = 1) -> 'Matrix': return Matrix(_cm.diff(self.base, axis))
    # def diag(self) -> 'Matrix': return Matrix(_diag(self.base))
    # def trace(self) -> 'Matrix': return Matrix(_trace(self.base))
    # def det(self) -> 'Matrix': return Matrix(_det(self.base))
    # @_method_build
    # def inv(self) -> 'Matrix': return Matrix(_inv(self.base))


class MatrixIter:

    def __init__(self, matrix: Matrix) -> None:
        self.base = matrix.base
        self._count = 0
        self._count_end = matrix.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self._count < self._count_end:
            result = Matrix(_cm.get_matrix_row(self.base, self._count))
            self._count += 1
            return result
        raise StopIteration


# %% build matrix
def obj2cm_(obj: Union[NumberR, Array, cyMatrix, Matrix]) -> cyMatrix:
    if isinstance(obj, Matrix):
        return obj.base
    return obj2cm(obj)


def zeros(*shape: int) -> Matrix:
    len_ = len(shape)
    if len_ == 1:
        return Matrix(cyMatrix(1, shape[0], 1))
    if len_ == 2:
        return Matrix(cyMatrix(shape[0], shape[1], 2))
    raise ValueError('zeros: 参数数量错误')


def ones(*shape: int) -> Matrix:
    len_ = len(shape)
    if len_ == 1:
        res = cyMatrix(1, shape[0], 1)
    elif len_ == 2:
        res = cyMatrix(shape[0], shape[1], 2)
    else:
        raise ValueError('ones: 参数数量错误')
    _cm.fill(res, 1.)
    return Matrix(res)


def eye(n: int) -> Matrix:
    return Matrix(_cm.eye(n))


def _check_shape(shape: Union[tuple, list, None], num: int):
    if shape is None:
        nr = 1
        nc = num
        ndim = 1
    elif len(shape) == 2:
        nr = shape[0]
        nc = shape[1]
        if nr * nc != num:
            raise ValueError('shape: 参数数量错误')
        ndim = 2
    elif len(shape) == 1:
        nr = 1
        nc = num
        ndim = 1
    else:
        raise ValueError('shape: 参数数量错误')
    return nr, nc, ndim


def arange(*args:int, shape: Union[tuple, list, None]=None) -> Matrix:
    # start=0, end=10, step=1
    len_ = len(args)
    if len_ == 0:
        start, end, step = 0, 10, 1
    elif len_ == 1:
        start, end, step = 0, args[0], 1
    elif len_ == 3:
        start, end, step = args
    else:
        raise ValueError('arange: 参数数量错误')
    parr = parray('d', range(start, end, step))
    size = len(parr)
    return Matrix(_cm.buildFromArr(parr, *_check_shape(shape, size)))


def linspace(*args: NumberR, shape: Union[tuple, list, None]=None) -> Matrix:
    len_ = len(args)
    if len_ == 0:
        start, end, num = 0, 9, 10
    elif len_ == 1:
        start, end, num = 0, args[0], 10
    elif len_ == 3:
        start, end, num = args
        num = int(num)
    else:
        raise ValueError('linspace: 参数数量错误')
    num_ = num - 1
    step = (end - start) / num_
    parr = parray('d')
    for _ in range(num_):
        parr.append(start)
        start += step
    parr.append(end)
    return Matrix(_cm.buildFromArr(parr, *_check_shape(shape, num)))


def rand(*shape: int) -> Matrix:
    len_ = len(shape)
    if len_ == 1:
        parr = parray('d', [_random() for _ in range(shape[0])])
        _cm.buildFromArr(parr, 1, shape[0], 1)
        return Matrix(_cm.buildFromArr(parr, 1, len(parr), 1))
    if len_ == 2:
        parr = parray('d', [_random() for _ in range(shape[0] * shape[1])])
        return Matrix(_cm.buildFromArr(parr, shape[0], shape[1], 2))
    raise ValueError('rand: 参数数量错误')


def concat(seq: Sequence[Matrix],
           axis: int = 0,
           isColVector: bool = False) -> Matrix:
    res = seq[0]._copy()
    # next(seq)
    baxis = True if axis == 0 else False
    for it in range(1, len(seq)):
        _cm.add_concatenate(res, seq[it].base, baxis, isColVector)
    return Matrix(res)


def matrixByIndex(m: Union[NumberR, Array, cyMatrix, Matrix], index: Array) -> Matrix:
    if not isinstance(index, Array_):
        raise ValueError('matrixByIndex: index 必须是 list, tuple, array')
    key = parray('i')
    for i in index:
        key.append(int(i))
    return Matrix(_cm.matrix_byindex(Matrix(m).base, key))


# %% math functions
INF = Matrix(_cm.get_inf())
NAN = Matrix(_cm.get_nan())
PI = Matrix(3.1415926535897932384626433832795)
E = Matrix(2.7182818284590452353602874713527)
TOU = Matrix(6.28318530717958647692528676655901)
PI_2 = Matrix(1.57079632679489661923132169163975)
PI_3 = Matrix(1.04719755119659774615421446109317)
PI_4 = Matrix(0.785398163397448309615660845819876)
PI_6 = Matrix(0.523598775598298873077107230546584)
_1_PI = Matrix(0.318309886183790671537767526745029)
_2_PI = Matrix(0.636619772367581343075535053490057)
_3_PI = Matrix(0.954929658551372014613302580235086)
_4_PI = Matrix(1.27323954473516268615107010698011)
_6_PI = Matrix(1.64493406684822643647241516664682)
SQRT2 = Matrix(1.41421356237309504880)
SQRT3 = Matrix(1.73205080756887729352)
LN2 = Matrix(0.69314718055994530942)
LN10 = Matrix(2.30258509299404568402)
LOG2E = Matrix(1.44269504088896340736)
LOG10E = Matrix(0.43429448190325182765)
SQRT1_2 = Matrix(0.70710678118654752440)


# --- class内不包含 --- #
def nan2num(m: Matrix) -> None: # inplace
    _cm.nan2num(obj2cm_(m))
def isnan(num: NumberR) -> bool: return _cm.isnan(num)
def rad2deg(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.rad2deg(obj2cm_(m)))
def deg2rad(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.deg2rad(obj2cm_(m)))
def asin(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.asin(obj2cm_(m)))
def acos(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.acos(obj2cm_(m)))
def atan(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.atan(obj2cm_(m)))
def atan2(mx: Union[NumberR, Array, cyMatrix, Matrix], my: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.atan2(obj2cm_(mx), obj2cm_(my)))
def exp(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.exp(obj2cm_(m)))
def exp2(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.exp2(obj2cm_(m)))
def expm1(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.expm1(obj2cm_(m)))
def log(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.log(obj2cm_(m)))
def log2(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.log2(obj2cm_(m)))
def log10(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.log10(obj2cm_(m)))
def log1p(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.log1p(obj2cm_(m)))
def sqrt(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.sqrt(obj2cm_(m)))
def cbrt(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.cbrt(obj2cm_(m)))
def hypot(mx: Union[NumberR, Array, cyMatrix, Matrix], my: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.hypot(obj2cm_(mx), obj2cm_(my)))
def sign(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.sign(obj2cm_(m)))
def diag(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.diag(obj2cm_(m)))
def trace(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.trace(obj2cm_(m)))
def det(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.det(obj2cm_(m)))
def inv(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.inv(obj2cm_(m)))
def sum_squares(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.sum_squares(obj2cm_(m), axis))
def repeat(m: Union[NumberR, Array, cyMatrix, Matrix], repeats:int, axis:int=0) -> Matrix: return Matrix(_cm.repeat(obj2cm_(m), repeats, axis))
def maximum(m1: Union[NumberR, Array, cyMatrix, Matrix], m2: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.maximum(obj2cm_(m1), obj2cm_(m2)))
def minimum(m1: Union[NumberR, Array, cyMatrix, Matrix], m2: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.minimum(obj2cm_(m1), obj2cm_(m2)))
def extract(condition: Union[bool, List[bool], List[List[bool]]], m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.extract(list2barr(condition), obj2cm_(m)))
def compress(condition: Union[bool, List[bool], List[List[bool]]], m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.compress(list2barr(condition), obj2cm_(m), axis))
def percentile(m: Union[NumberR, Array, cyMatrix, Matrix], q: Union[int, float], axis:int=-1) -> Matrix:
    temp = Matrix(m).sort(axis)
    temp.reshape(temp.size, inplace=True)
    n = temp.size / 100 * q
    n_ = str(n).rpartition('.')[-1]
    if n_ == '' or int(n_) == 0:
        return temp[int(n) - 1]
    else:
        n_f = int(floor(n))
        n_c = int(ceil(n))
        return temp[n_c] * (n_c - n) + temp[n_f] * (n - n_f)

def argsort(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=1) -> List[List[int]]:
    return _cm.argsort(obj2cm_(m), True if axis == 1 else False)
def argmin(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Union[List[int], int]:
    res = _cm.argmin(obj2cm_(m), axis)
    return res if axis in (0, 1) else res[0]
def argmax(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Union[List[int], int]:
    res = _cm.argmax(obj2cm_(m), axis)
    return res if axis in (0, 1) else res[0]

def linv(m: Matrix) -> Matrix:
    """左伪逆"""
    m = Matrix(m)
    if m.ndim == 1:
        m.reshape(m.size, 1, inplace=True)
    mt = m.transpos()
    return inv(mt @ m) @ mt

def rinv(m: Matrix) -> Matrix:
    """右伪逆"""
    m = Matrix(m)
    if m.ndim == 1:
        m.reshape(1, m.size, inplace=True)
    mt = m.transpos()
    return mt @ inv(m @ mt)

def lstsq(ma: Matrix, my: Matrix):
    """返回线性方程组矩阵的最小二乘解"""
    return linv(ma) @ my


def arrayFromMatrix(seq: Sequence[Matrix], isColVector:bool=False) -> Matrix:
    q1 = seq[0]
    if q1.ndim == 0:
        return concat(seq, 1)[0]
    if isColVector and q1.ndim == 1:
        return concat(seq, 1, True)
    return concat(seq, 0)

def _asarray_f(m):
    if isinstance(m, Matrix):
        return m._copy()
    if isinstance(m, cyMatrix):
        return _cm.copy(m)
    if isinstance(m, (float, int, bool)):
        return _cm.buildFromArr(parray('d', [m]), 1, 1, 0)
    if isinstance(m, (list, tuple, range)):
        return _cm.buildFromArr(parray('d', m), 1, len(m), 1)
    if isinstance(m, parray):
        return _cm.buildFromArr(m, 1, len(m), 1)
    raise ValueError('asarray.f: m 必须是 Matrix or cyMatrix or tuple or list or parray or NumberR, 现在是 {}'.format(type(m)))

def asarray(arr) -> Matrix:
    if isinstance(arr, Matrix):
        return arr
    if isinstance(arr, cyMatrix):
        return Matrix(arr)
    if isinstance(arr, (list, tuple)):
        arr_ = [_asarray_f(i) for i in arr]
        res = arr_[0]
        if _cm.get_ndim(res) == 1:
            _cm.reshape(res, 2, 1, _cm.get_size(res))
        for i in range(1, len(arr_)):
            _cm.append(res, arr_[i])
        return Matrix(res)
    if isinstance(arr, range):
        return Matrix(_cm.buildFromArr(parray('d', arr), 1, len(arr), 1))
    if isinstance(arr, (int, float, bool)):
        return Matrix(_cm.buildFromArr(parray('d', [arr]), 1, 1, 0))
    raise ValueError('asarray: arr 必须是 NumberR or list or tuple or cyMatrix or Matrix, 现在是 {}'.format(type(arr)))



# --- class内包含 --- #
def numrt(m: Union[NumberR, Array, cyMatrix, Matrix], num:int=2) -> Matrix: return Matrix(_cm.numrt(obj2cm_(m), num))
def abs(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.abs(obj2cm_(m)))
def ceil(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.ceil(obj2cm_(m)))
def floor(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.floor(obj2cm_(m)))
def round(m: Union[NumberR, Array, cyMatrix, Matrix], digits:int=0) -> Matrix: return Matrix(_cm.round(obj2cm_(m), digits))
def sin(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.sin(obj2cm_(m), True))
def cos(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.cos(obj2cm_(m), True))
def tan(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.tan(obj2cm_(m), True))
def sind(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.sin(obj2cm_(m), False))
def cosd(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.cos(obj2cm_(m), False))
def tand(m: Union[NumberR, Array, cyMatrix, Matrix]) -> Matrix: return Matrix(_cm.tan(obj2cm_(m), False))
def max(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.max(obj2cm_(m), axis))
def min(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.min(obj2cm_(m), axis))
def sum(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.sum(obj2cm_(m), axis))
def mean(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.mean(obj2cm_(m), axis))
def prod(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.prod(obj2cm_(m), axis))
def median(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.median(obj2cm_(m), axis))
def std(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.std(obj2cm_(m), axis))
def var(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.var(obj2cm_(m), axis))
def rms(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.rms(obj2cm_(m), axis))
def norm(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.norm(obj2cm_(m), axis))
def ptp(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.ptp(obj2cm_(m), axis))
def diff(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=0) -> Matrix: return Matrix(_cm.diff(obj2cm_(m), axis))
def flip(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=0) -> Matrix: return Matrix(m).flip(axis)
def sort(m: Union[NumberR, Array, cyMatrix, Matrix], axis:int=-1) -> Matrix: return Matrix(_cm.sort(obj2cm_(m), axis))




# %% test
if __name__ == '__main__':
    a = [30., -3., 90., -180.]
    m = Matrix(a)
    # print(m.cos(False) + m)
    print(concat((m, m-1, m+1), 1, True))
    print(m)
    print(hash(m))
    print(hash(Matrix(m.base)))
