import sys

from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize

pyversion = sys.version_info
python_requires = f'>={pyversion.major}.{pyversion.minor},\
<{pyversion.major}.{pyversion.minor + 1}'

extensions = [Extension('cymatrix.cymatrix', ['cymatrix/cymatrix.pyx'])]
# python setup.py build_ext --inplace
# python setup.py bdist_wheel
setup(
    name='cymatrix',  # 你的包名
    version='0.1.1',  # 版本号
    author='Yifan Hong',  # 作者名字
    author_email='hong.yf@qq.com',  # 作者邮箱
    description='A simple matrix implementation in `Cython`',  # 简短描述
    long_description=open('README.md').read(),  # 长描述，通常是README的内容
    long_description_content_type='text/markdown',  # 长描述的内容类型
    url='https://github.com/H00Hong/cymatrix.git',  # 项目主页
    packages=find_packages(),  # 自动发现所有包和模块
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=python_requires,  # 所需的最低Python版本
    ext_modules=cythonize(extensions),
)
