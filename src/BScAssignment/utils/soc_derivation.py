import sympy as sp
from collections import namedtuple, defaultdict
from typing import Tuple, Dict
from sympy import zeros
from numpy import abs
from sympy.interactive import init_printing

init_printing(use_unicode=True)

hbar = sp.Symbol('hbar')
alpha = sp.Symbol('alpha')

State = namedtuple("State", "l m_l s m_s")
OperatorRet = Tuple[sp.Expr, State]

def Lp(state: State) -> OperatorRet:
    coeff = hbar * sp.sqrt(state.l * (state.l + 1) - state.m_l * (state.m_l + 1))
    new_state = State(state.l, state.m_l + 1, state.s, state.m_s)
    return (coeff, new_state)

def Lm(state: State) -> OperatorRet:
    coeff = hbar * sp.sqrt(state.l * (state.l + 1) - state.m_l * (state.m_l - 1))
    new_state = State(state.l, state.m_l - 1, state.s, state.m_s)
    return (coeff, new_state)

def Lz(state: State) -> OperatorRet:
    coeff = hbar * state.m_l
    return (coeff, state)

def Sp(state: State) -> OperatorRet:
    coeff = hbar * sp.sqrt(state.s * (state.s + 1) - state.m_s * (state.m_s + 1))
    new_state = State(state.l, state.m_l, state.s, state.m_s + 1)
    return (coeff, new_state)

def Sm(state: State) -> OperatorRet:
    coeff = hbar * sp.sqrt(state.s * (state.s + 1) - state.m_s * (state.m_s - 1))
    new_state = State(state.l, state.m_l, state.s, state.m_s - 1)
    return (coeff, new_state)

def Sz(state: State) -> OperatorRet:
    coeff = hbar * state.m_s
    return (coeff, state)

def SpLm(state: State) -> OperatorRet:
    l_coeff, state = Lm(state)
    s_coeff, state = Sp(state)

    return (l_coeff * s_coeff, state)

def SmLp(state: State) -> OperatorRet:
    l_coeff, state = Lp(state)
    s_coeff, state = Sm(state)

    return (l_coeff * s_coeff, state)

def SzLz(state: State) -> OperatorRet:
    l_coeff, state = Lz(state)
    s_coeff, state = Sz(state)
    return (l_coeff * s_coeff, state)


basis = [
    State(1, -1, 0.5, 0.5),
    State(1, -1, 0.5, -0.5),

    State(1,  0, 0.5, 0.5),
    State(1,  0, 0.5, -0.5),

    State(1,  1, 0.5, 0.5),
    State(1,  1, 0.5, -0.5),
]

def legal_state(state: State) -> bool:
    return state.l >= abs(state.m_l) and state.s >= abs(state.m_s)

H_SOC_sh = sp.zeros(len(basis), len(basis))

for i, state in enumerate(basis):
    coeffs: Dict[State, sp.Expr] = defaultdict(lambda: sp.Integer(0))

    coeff, new_state = SpLm(state)
    if legal_state(new_state):
        coeffs[new_state] += coeff

    coeff, new_state = SmLp(state)
    if legal_state(new_state):
        coeffs[new_state] += coeff

    coeff, new_state = SzLz(state)
    if legal_state(new_state):
        coeffs[new_state] += 2*coeff

    for new_state, coeff in coeffs.items():
        H_SOC_sh[i, basis.index(new_state)] = alpha / 2 * coeff

print(H_SOC_sh)
