# vflog
vertical + gpu + free join + datalog

# Dependency
- CMake >= 3.20
- Nvidia HPC sdk >= 24.1
- OpenMP
- OneTBB

# Build
```bash
cmake -Bbuild . && cd build && make -j
```

# Datalog queries

Test queries in paper are under `test` folder. After compile, the binary for queries will under `build/test`.
