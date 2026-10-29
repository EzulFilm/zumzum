"""Zumzum: spectral audio editing as a small checked language.

A program selects regions in time and frequency, applies ops to them, and states measured checks. The checker
(`sound.checker`) admits a program only if every name, type, unit, range and size limit holds; the NumPy reference
engine (`sound.engine`) renders it; `sound.measure` evaluates the checks on the before / after pair.
"""

VERSION = "0.1"
