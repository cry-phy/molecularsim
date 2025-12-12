%chk=water_cartesian.chk
#p HF/6-31G(d) Opt

Water molecule (Cartesian form)
@charges = [
  -0.834, 0.417, 0.417
]

0 1
O   0.000000   0.000000   0.000000
H   0.000000   0.757160   0.586260
H   0.000000  -0.757160   0.586260
