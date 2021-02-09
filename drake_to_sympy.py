#!/usr/bin/env python

from pydrake.all import *
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def drake_to_sympy(v):
    """
    Convert a vector v which contains drake symbolic expressions to 
    an equivalent matrix with sympy symbolic expressions. This is a bit hacky,
    but sympy provides better tools for processing symbolic expressions.
    """
    assert v.ndim == 1, "the vector v must be a 1d numpy array"
   
    # Sympy allows us to do parsing based on strings, so we'll use
    # the to_string() method of drake expressions generate such strings.
    v_sympy = []
    [ v_sympy.append(parse_expr(str(row))) for row in v ]

    return np.array(v_sympy)


a = Variable('a')
b = Variable('b')
c = Variable('c')
v = np.array([a+c,b,a+b,a*b])


v_sym = drake_to_sympy(v)
print(v_sym)
print(v_sym[2])
print(type(v_sym[2]))

