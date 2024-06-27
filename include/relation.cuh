
#include "hisa.cuh"
#include <string>
#include <thrust/count.h>

namespace fvlog {

using device_data_ptr = thrust::device_ptr<hisa::internal_data_type>;

/**
 * @brief Relation built with hisa
 */
class relation {
    int arity;
    std::string name;
    // cpu data
    HOST_VECTOR<hisa::VetricalColumnCpu> cpu_columns;
    // internal data
    std::shared_ptr<hisa::multi_hisa> hisa_data = nullptr;
    bool static_flag = false;
    bool tmp_flag = false;

  public:
    // constructor
    relation(int arity, std::string name) : arity(arity), name(name) {
        cpu_columns.resize(arity);
        hisa_data = std::make_shared<hisa::multi_hisa>(arity);
    }

    // destructor
    ~relation() = default;

    // get arity
    int get_arity() const { return arity; }

    // get name
    std::string get_name() const { return name; }

    // get hisa data
    std::shared_ptr<hisa::multi_hisa> get_hisa_data() const {
        return hisa_data;
    }

    void load_data(const std::vector<std::vector<int>> &data,
                   bool dup_flag = false);

    /**
     * @brief Allocate space newt verison
     */
    void allocate_newt(size_t size);

    /**
     * @brief Get the pointer to the head of current newt of a column i
     */
    device_data_ptr get_newt_head(int i);

    hisa::VerticalColumnGpu &get_column(RelationVersion ver, int i) {
        switch (ver) {
        case RelationVersion::DELTA:
            return hisa_data->delta_columns[i];
        case RelationVersion::FULL:
            return hisa_data->full_columns[i];
        case RelationVersion::NEWT:
            return hisa_data->newt_columns[i];
        default:
            throw std::runtime_error("Unknown version");
        }
    }

    // get size of each version
    size_t get_size(RelationVersion ver) {
        switch (ver) {
        case RelationVersion::DELTA:
            return hisa_data->delta_size;
        case RelationVersion::FULL:
            return hisa_data->full_size;
        case RelationVersion::NEWT:
            return hisa_data->newt_size;
        default:
            throw std::runtime_error("Unknown version");
        }
    }
};

using relational_ptr = std::shared_ptr<relation>;

/**
 * @brief Slice of a relation
 *  Slice is a part of a relation
 *  `indices` is the sorted indices of the tuples included in this slice
 */
struct slice {
    relational_ptr relation;
    RelationVersion version;
    hisa::device_bitmap_t bitmap;
    size_t matched_size = 0;

    /**
     * @brief Create a slice with indices, move the indices into the slice
     */
    slice(relational_ptr relation, RelationVersion version,
          hisa::device_bitmap_t &bitmap)
        : relation(relation), version(version) {
        this->bitmap.swap(bitmap);
    }

    /**
     * @brief Create a slice with empty indices
     */
    slice(relational_ptr relation, RelationVersion version)
        : relation(relation), version(version) {}

    /**
     * @brief Copy constructor
     */
    slice(const slice &s) {
        relation = s.relation;
        version = s.version;
        bitmap = s.bitmap;
    }

    // reload = operator
    slice &operator=(const slice &s) {
        relation = s.relation;
        version = s.version;
        bitmap = s.bitmap;
        return *this;
    }

    /**
     * @brief Move constructor
     */
    slice(slice &&s) {
        relation = s.relation;
        version = s.version;
        bitmap.swap(s.bitmap);
    }

    /**
     * @brief Move the indices into the slice
     *  this will empty the input indices, and lost the original data
     */
    void move_indices(hisa::device_bitmap_t &indices) {
        indices.resize(0);
        indices.shrink_to_fit();
        this->bitmap.swap(bitmap);
    }

    /**
     * @brief Get the size of the slice
     */
    size_t size() const { return matched_size; }

    void count_matched() {
        matched_size = thrust::count(DEFAULT_DEVICE_POLICY, bitmap.begin(),
                                     bitmap.end(), true);
    }
};

/**
 * @brief Environment for relational algebra
 * "Env" of a Relational Algebra Machine
 * it is the "stack" like temporal storage
 * similar to an unmaterilized view
 * Each datalog rule will create a new environment
 * all slice in environment must has same length
 */
struct RelationalEnvironment {
    std::map<std::string, relational_ptr> relations;
    std::map<std::string, slice> slices;

    // move constructor
    RelationalEnvironment(RelationalEnvironment &&env) {
        relations.swap(env.relations);
        slices.swap(env.slices);
    }
};

} // namespace fvlog
