# This works >>>>>>>>>>>

# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy as np
#
# setup(
#   name='interp test',
#   ext_modules=cythonize("interp.pyx"),
#   include_dirs=[np.get_include()],
# )

#<<<<<<<<<<<<<

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("cython_vinth2p_4d",
              sources=["cython_vinth2p_4d.pyx"],
              extra_compile_args = ["-O0", "-fopenmp"],
              extra_link_args=['-fopenmp'],
              include_dirs=[np.get_include()],
              libraries=["m"]
             )
]

setup(
  name="MyStuff",
  ext_modules=cythonize(ext_modules, language_level = "3")
)
