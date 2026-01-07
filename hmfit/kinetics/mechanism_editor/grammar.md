# Mechanism Editor Grammar (MVP)

## Sections

- Species:

```
species: A, B, C
```

- Fixed species (optional):

```
fixed: H2O, H_plus
```

- Reactions (mass-action by default):

```
A -> B ; k1
B <-> C ; k2, k_2
2A + B -> C ; k4
```

- Temperature models (optional):

```
arrhenius: k1(A=A1, Ea=Ea1)

arrhenius:
  k2(A=A2, Ea=Ea2)

# Eyring model

eyring: k3(dH=dH3, dS=dS3)
```

## Notes

- Comments start with `#` and are ignored.
- Identifiers are `[A-Za-z_][A-Za-z0-9_]*`.
- Coefficients are optional integers (e.g., `2A`, `3 B`).
- Reversible reactions use `<->` and require two parameters.
