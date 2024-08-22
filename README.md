# vflog
vertical + gpu + free join + datalog

# Dependency
- CMake >= 3.20
- Nvidia HPC sdk >= 24.1
- OpenMP
- OneTBB

# Build
```bash
cmake -DCMAKE_BUILD_TYPE=Release -Bbuild . && cd build && make -j
```

# Datalog queries

Test queries in paper are under `test` folder. After compile, the binary for queries will under `build/test`.


# Supported Relational Algebra Machine(RAM) operators

- `print_size` : print the size of relation
- `cache_update`: This operator is used to update the cache of a relation. The cache map var name to the surrogated ID values of a relation and is used to store join/copy results and specify subsets of the relation used inside a relational algebra (RA) operator. 
- `cache_init`: This operator is used to initialize a cache slot with all IDs of a relation.
- `cache_clear`: This operator is used to clear the cache
- `prepare_materialization`: Materialization means project column to a relation based on cached IDs. This operator is used to prepare the materialization of a relation, (usually for join/filter). It allocates the required memory at the end of the newt, which has a size equal to the cached variable's ID size.
- `end_materialization`: This operator is used to end the materialization of a relation. This will update the relation's size and the number of tuples.
- `project_op`: This operator is used to project a column from one relation to another.
- `join_op`: This operator is used to join two relations column. (WARNNING: join will change cache to facillitate k-way join, but this will be changed in future soon, when real k-way join is implemented)
- `fixpoint_op`: This operator is used to run a fixpoint operator. It will loop over all RA op list until the fixpoint is reached.
- `persistent`: This operator is used to populating DELTA and clear NEWT of a relation. Generated DELTA will be merged into full immediately.


# Example : TC
```
    auto ram = vflog::RelationalAlgebraMachine();

    auto edge = ram.create_rel("edge", data_path, 2);
    auto path = ram.create_rel("path", nullptr, 2);

    // All surrogated column buffer must be allocated before running,
    // its similar to "register", we will reuse these buffer in compiled RA ops 
    auto input_indices_ptr = std::make_shared<vflog::device_indices_t>();
    auto tmp_id0 = std::make_shared<vflog::device_indices_t>();
    auto tmp_id1 = std::make_shared<vflog::device_indices_t>();

    using namespace vflog;

    ram.add_operator(
        {// path(a, b) :- edge(a, b).
         // cache can temporary store surrogate column 
         cache_update("edge", input_indices_ptr), cache_init("edge", edge, FULL),
         prepare_materialization(path, "edge"),
         project_op(column_t(edge, 0, FULL), column_t(path, 0, FULL), "edge"),
         project_op(column_t(edge, 1, FULL), column_t(path, 1, FULL), "edge"),
         end_materialization(path, "edge"), persistent(path),
         cache_clear(),
         // path(a, c) :- path(a, b), edge(b, c).
         fixpoint_op(
             {
                 cache_update("path", tmp_id0),
                 cache_init("path", path, DELTA),
                 join_op(column_t(edge, 0, FULL), column_t(path, 1, DELTA),
                         "path", tmp_id1),
                 cache_update("edge", tmp_id1),
                 prepare_materialization(path, "path"),
                 project_op(column_t(path, 0, DELTA), column_t(path, 0, DELTA),
                            "path"),
                 project_op(column_t(edge, 1, FULL), column_t(path, 1, DELTA),
                            "edge"),
                 end_materialization(path, "path"),
                 persistent(path),
                 print_size("path", path),
             },
             {path})});

    std::cout << "Start executing" << std::endl;
    KernelTimer timer;
    timer.start_timer();
    ram.execute();
    timer.stop_timer();
    auto elapsed = timer.get_spent_time();
    std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
```

