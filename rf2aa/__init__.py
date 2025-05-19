import contextlib
import os

projdir = os.path.realpath(os.path.dirname(__file__))

#model = lazyimport('rf2aa.model')
#motif = lazyimport('rf2aa.motif')
#sym = lazyimport('rf2aa.sym')
#tests = lazyimport('rf2aa.tests')
#tools = lazyimport('rf2aa.tools')
#util = lazyimport('rf2aa.util')

with contextlib.suppress(ImportError):
    from icecream import ic
    ic.configureOutput(includeContext=True)

