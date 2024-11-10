#define SOUFFLE_GENERATOR_VERSION "2.4"
#include "souffle/CompiledSouffle.h"
#include "souffle/SignalHandler.h"
#include "souffle/SouffleInterface.h"
#include "souffle/datastructure/BTree.h"
#include "souffle/io/IOSystem.h"
#include <any>
namespace functors {
extern "C" {
}
} //namespace functors
namespace souffle::t_btree_ii__0_1__11__10 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_ii__0_1__11__10 
namespace souffle::t_btree_ii__0_1__11__10 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_10(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_10(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_ii__0_1__11__10 
namespace souffle::t_btree_ii__0_1__11 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 2;
using t_tuple = Tuple<RamDomain, 2>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :(0));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_ii__0_1__11 
namespace souffle::t_btree_ii__0_1__11 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[2];
std::copy(ramDomain, ramDomain + 2, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1) {
RamDomain data[2] = {a0,a1};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_00(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_11(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_11(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 2 direct b-tree index 0 lex-order [0,1]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_ii__0_1__11 
namespace souffle::t_btree_i__0__1 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 1;
using t_tuple = Tuple<RamDomain, 1>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :(0);
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_i__0__1 
namespace souffle::t_btree_i__0__1 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[1];
std::copy(ramDomain, ramDomain + 1, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0) {
RamDomain data[1] = {a0};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_0(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_1(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_1(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 1 direct b-tree index 0 lex-order [0]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_i__0__1 
namespace souffle::t_btree_iii__0_1_2__111 {
using namespace souffle;
struct Type {
static constexpr Relation::arity_type Arity = 3;
using t_tuple = Tuple<RamDomain, 3>;
struct t_comparator_0{
 int operator()(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0])) ? -1 : (ramBitCast<RamSigned>(a[0]) > ramBitCast<RamSigned>(b[0])) ? 1 :((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1])) ? -1 : (ramBitCast<RamSigned>(a[1]) > ramBitCast<RamSigned>(b[1])) ? 1 :((ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2])) ? -1 : (ramBitCast<RamSigned>(a[2]) > ramBitCast<RamSigned>(b[2])) ? 1 :(0)));
 }
bool less(const t_tuple& a, const t_tuple& b) const {
  return (ramBitCast<RamSigned>(a[0]) < ramBitCast<RamSigned>(b[0]))|| ((ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0])) && ((ramBitCast<RamSigned>(a[1]) < ramBitCast<RamSigned>(b[1]))|| ((ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1])) && ((ramBitCast<RamSigned>(a[2]) < ramBitCast<RamSigned>(b[2]))))));
 }
bool equal(const t_tuple& a, const t_tuple& b) const {
return (ramBitCast<RamSigned>(a[0]) == ramBitCast<RamSigned>(b[0]))&&(ramBitCast<RamSigned>(a[1]) == ramBitCast<RamSigned>(b[1]))&&(ramBitCast<RamSigned>(a[2]) == ramBitCast<RamSigned>(b[2]));
 }
};
using t_ind_0 = btree_set<t_tuple,t_comparator_0>;
t_ind_0 ind_0;
using iterator = t_ind_0::iterator;
struct context {
t_ind_0::operation_hints hints_0_lower;
t_ind_0::operation_hints hints_0_upper;
};
context createContext() { return context(); }
bool insert(const t_tuple& t);
bool insert(const t_tuple& t, context& h);
bool insert(const RamDomain* ramDomain);
bool insert(RamDomain a0,RamDomain a1,RamDomain a2);
bool contains(const t_tuple& t, context& h) const;
bool contains(const t_tuple& t) const;
std::size_t size() const;
iterator find(const t_tuple& t, context& h) const;
iterator find(const t_tuple& t) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const;
range<iterator> lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const;
range<t_ind_0::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const;
range<t_ind_0::iterator> lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const;
bool empty() const;
std::vector<range<iterator>> partition() const;
void purge();
iterator begin() const;
iterator end() const;
void printStatistics(std::ostream& o) const;
};
} // namespace souffle::t_btree_iii__0_1_2__111 
namespace souffle::t_btree_iii__0_1_2__111 {
using namespace souffle;
using t_ind_0 = Type::t_ind_0;
using iterator = Type::iterator;
using context = Type::context;
bool Type::insert(const t_tuple& t) {
context h;
return insert(t, h);
}
bool Type::insert(const t_tuple& t, context& h) {
if (ind_0.insert(t, h.hints_0_lower)) {
return true;
} else return false;
}
bool Type::insert(const RamDomain* ramDomain) {
RamDomain data[3];
std::copy(ramDomain, ramDomain + 3, data);
const t_tuple& tuple = reinterpret_cast<const t_tuple&>(data);
context h;
return insert(tuple, h);
}
bool Type::insert(RamDomain a0,RamDomain a1,RamDomain a2) {
RamDomain data[3] = {a0,a1,a2};
return insert(data);
}
bool Type::contains(const t_tuple& t, context& h) const {
return ind_0.contains(t, h.hints_0_lower);
}
bool Type::contains(const t_tuple& t) const {
context h;
return contains(t, h);
}
std::size_t Type::size() const {
return ind_0.size();
}
iterator Type::find(const t_tuple& t, context& h) const {
return ind_0.find(t, h.hints_0_lower);
}
iterator Type::find(const t_tuple& t) const {
context h;
return find(t, h);
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */, context& /* h */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<iterator> Type::lowerUpperRange_000(const t_tuple& /* lower */, const t_tuple& /* upper */) const {
return range<iterator>(ind_0.begin(),ind_0.end());
}
range<t_ind_0::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper, context& h) const {
t_comparator_0 comparator;
int cmp = comparator(lower, upper);
if (cmp == 0) {
    auto pos = ind_0.find(lower, h.hints_0_lower);
    auto fin = ind_0.end();
    if (pos != fin) {fin = pos; ++fin;}
    return make_range(pos, fin);
}
if (cmp > 0) {
    return make_range(ind_0.end(), ind_0.end());
}
return make_range(ind_0.lower_bound(lower, h.hints_0_lower), ind_0.upper_bound(upper, h.hints_0_upper));
}
range<t_ind_0::iterator> Type::lowerUpperRange_111(const t_tuple& lower, const t_tuple& upper) const {
context h;
return lowerUpperRange_111(lower,upper,h);
}
bool Type::empty() const {
return ind_0.empty();
}
std::vector<range<iterator>> Type::partition() const {
return ind_0.getChunks(400);
}
void Type::purge() {
ind_0.clear();
}
iterator Type::begin() const {
return ind_0.begin();
}
iterator Type::end() const {
return ind_0.end();
}
void Type::printStatistics(std::ostream& o) const {
o << " arity 3 direct b-tree index 0 lex-order [0,1,2]\n";
ind_0.printStats(o);
}
} // namespace souffle::t_btree_iii__0_1_2__111 
namespace  souffle {
using namespace souffle;
class Stratum_ancestor_path_c0112b3bbe21adc3 {
public:
 Stratum_ancestor_path_c0112b3bbe21adc3(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_ancestor_path_bcaf8780da306af1,t_btree_ii__0_1__11::Type& rel_new_ancestor_path_4d88f0281b6b2a54,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11__10::Type& rel_child_path_d9b38651563c0c3d);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_ancestor_path_bcaf8780da306af1;
t_btree_ii__0_1__11::Type* rel_new_ancestor_path_4d88f0281b6b2a54;
t_btree_ii__0_1__11__10::Type* rel_ancestor_path_56e99f4d0bc91e6e;
t_btree_ii__0_1__11__10::Type* rel_child_path_d9b38651563c0c3d;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_ancestor_path_c0112b3bbe21adc3::Stratum_ancestor_path_c0112b3bbe21adc3(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_ancestor_path_bcaf8780da306af1,t_btree_ii__0_1__11::Type& rel_new_ancestor_path_4d88f0281b6b2a54,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11__10::Type& rel_child_path_d9b38651563c0c3d):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_ancestor_path_bcaf8780da306af1(&rel_delta_ancestor_path_bcaf8780da306af1),
rel_new_ancestor_path_4d88f0281b6b2a54(&rel_new_ancestor_path_4d88f0281b6b2a54),
rel_ancestor_path_56e99f4d0bc91e6e(&rel_ancestor_path_56e99f4d0bc91e6e),
rel_child_path_d9b38651563c0c3d(&rel_child_path_d9b38651563c0c3d){
}

void Stratum_ancestor_path_c0112b3bbe21adc3::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(ancestor_path(x,y) :- 
   child_path(x,y).
in file polonius.dl [162:1-162:41])_");
if(!(rel_child_path_d9b38651563c0c3d->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
CREATE_OP_CONTEXT(rel_child_path_d9b38651563c0c3d_op_ctxt,rel_child_path_d9b38651563c0c3d->createContext());
for(const auto& env0 : *rel_child_path_d9b38651563c0c3d) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_ancestor_path_56e99f4d0bc91e6e->insert(tuple,READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_ancestor_path_bcaf8780da306af1_op_ctxt,rel_delta_ancestor_path_bcaf8780da306af1->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
for(const auto& env0 : *rel_ancestor_path_56e99f4d0bc91e6e) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_ancestor_path_bcaf8780da306af1->insert(tuple,READ_OP_CONTEXT(rel_delta_ancestor_path_bcaf8780da306af1_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(ancestor_path(Grandparent,Child) :- 
   ancestor_path(Parent,Child),
   child_path(Parent,Grandparent).
in file polonius.dl [167:1-169:37])_");
if(!(rel_delta_ancestor_path_bcaf8780da306af1->empty()) && !(rel_child_path_d9b38651563c0c3d->empty())) {
[&](){
auto part = rel_delta_ancestor_path_bcaf8780da306af1->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_ancestor_path_bcaf8780da306af1_op_ctxt,rel_delta_ancestor_path_bcaf8780da306af1->createContext());
CREATE_OP_CONTEXT(rel_new_ancestor_path_4d88f0281b6b2a54_op_ctxt,rel_new_ancestor_path_4d88f0281b6b2a54->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
CREATE_OP_CONTEXT(rel_child_path_d9b38651563c0c3d_op_ctxt,rel_child_path_d9b38651563c0c3d->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_child_path_d9b38651563c0c3d->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_child_path_d9b38651563c0c3d_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_ancestor_path_56e99f4d0bc91e6e->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[1]),ramBitCast(env0[1])}};
rel_new_ancestor_path_4d88f0281b6b2a54->insert(tuple,READ_OP_CONTEXT(rel_new_ancestor_path_4d88f0281b6b2a54_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_ancestor_path_4d88f0281b6b2a54->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_ancestor_path_4d88f0281b6b2a54_op_ctxt,rel_new_ancestor_path_4d88f0281b6b2a54->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
for(const auto& env0 : *rel_new_ancestor_path_4d88f0281b6b2a54) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_ancestor_path_56e99f4d0bc91e6e->insert(tuple,READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt));
}
}
();std::swap(rel_delta_ancestor_path_bcaf8780da306af1, rel_new_ancestor_path_4d88f0281b6b2a54);
rel_new_ancestor_path_4d88f0281b6b2a54->purge();
iter++;
}
iter = 0;
rel_delta_ancestor_path_bcaf8780da306af1->purge();
rel_new_ancestor_path_4d88f0281b6b2a54->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","ancestor_path"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ancestor_path_56e99f4d0bc91e6e);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_child_path_d9b38651563c0c3d->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_cfg_edge_6a02c2f8aa89b902 {
public:
 Stratum_cfg_edge_6a02c2f8aa89b902(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_cfg_edge_113c4ec5f576f8cf;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_cfg_edge_6a02c2f8aa89b902::Stratum_cfg_edge_6a02c2f8aa89b902(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_cfg_edge_113c4ec5f576f8cf(&rel_cfg_edge_113c4ec5f576f8cf){
}

void Stratum_cfg_edge_6a02c2f8aa89b902::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","cfg_edge"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_cfg_edge_113c4ec5f576f8cf);
} catch (std::exception& e) {std::cerr << "Error loading cfg_edge data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_cfg_node_bd032d689549d42e {
public:
 Stratum_cfg_node_bd032d689549d42e(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_i__0__1::Type& rel_cfg_node_c61b5c6f77f80834);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_cfg_edge_113c4ec5f576f8cf;
t_btree_i__0__1::Type* rel_cfg_node_c61b5c6f77f80834;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_cfg_node_bd032d689549d42e::Stratum_cfg_node_bd032d689549d42e(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_i__0__1::Type& rel_cfg_node_c61b5c6f77f80834):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_cfg_edge_113c4ec5f576f8cf(&rel_cfg_edge_113c4ec5f576f8cf),
rel_cfg_node_c61b5c6f77f80834(&rel_cfg_node_c61b5c6f77f80834){
}

void Stratum_cfg_node_bd032d689549d42e::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(cfg_node(point1) :- 
   cfg_edge(point1,_).
in file polonius.dl [133:1-134:30])_");
if(!(rel_cfg_edge_113c4ec5f576f8cf->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt,rel_cfg_edge_113c4ec5f576f8cf->createContext());
CREATE_OP_CONTEXT(rel_cfg_node_c61b5c6f77f80834_op_ctxt,rel_cfg_node_c61b5c6f77f80834->createContext());
for(const auto& env0 : *rel_cfg_edge_113c4ec5f576f8cf) {
Tuple<RamDomain,1> tuple{{ramBitCast(env0[0])}};
rel_cfg_node_c61b5c6f77f80834->insert(tuple,READ_OP_CONTEXT(rel_cfg_node_c61b5c6f77f80834_op_ctxt));
}
}
();}
signalHandler->setMsg(R"_(cfg_node(point2) :- 
   cfg_edge(_,point2).
in file polonius.dl [133:1-134:30])_");
if(!(rel_cfg_edge_113c4ec5f576f8cf->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt,rel_cfg_edge_113c4ec5f576f8cf->createContext());
CREATE_OP_CONTEXT(rel_cfg_node_c61b5c6f77f80834_op_ctxt,rel_cfg_node_c61b5c6f77f80834->createContext());
for(const auto& env0 : *rel_cfg_edge_113c4ec5f576f8cf) {
Tuple<RamDomain,1> tuple{{ramBitCast(env0[1])}};
rel_cfg_node_c61b5c6f77f80834->insert(tuple,READ_OP_CONTEXT(rel_cfg_node_c61b5c6f77f80834_op_ctxt));
}
}
();}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x"},{"auxArity","0"},{"name","cfg_node"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 1, \"params\": [\"x\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 1, \"types\": [\"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_cfg_node_c61b5c6f77f80834);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_child_path_a638e16652f89241 {
public:
 Stratum_child_path_a638e16652f89241(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_child_path_d9b38651563c0c3d);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_child_path_d9b38651563c0c3d;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_child_path_a638e16652f89241::Stratum_child_path_a638e16652f89241(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_child_path_d9b38651563c0c3d):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_child_path_d9b38651563c0c3d(&rel_child_path_d9b38651563c0c3d){
}

void Stratum_child_path_a638e16652f89241::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","child_path"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_child_path_d9b38651563c0c3d);
} catch (std::exception& e) {std::cerr << "Error loading child_path data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_errors_3f7449d7aa9e4967 {
public:
 Stratum_errors_3f7449d7aa9e4967(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_errors_01d6f43aaabedcef);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_errors_01d6f43aaabedcef;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_errors_3f7449d7aa9e4967::Stratum_errors_3f7449d7aa9e4967(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_errors_01d6f43aaabedcef):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_errors_01d6f43aaabedcef(&rel_errors_01d6f43aaabedcef){
}

void Stratum_errors_3f7449d7aa9e4967::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","errors"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_errors_01d6f43aaabedcef);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_known_placeholder_subset_10b7319424a0a314 {
public:
 Stratum_known_placeholder_subset_10b7319424a0a314(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_known_placeholder_subset_d3aa7e46869bc78d);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_known_placeholder_subset_d3aa7e46869bc78d;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_known_placeholder_subset_10b7319424a0a314::Stratum_known_placeholder_subset_10b7319424a0a314(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_known_placeholder_subset_d3aa7e46869bc78d):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_known_placeholder_subset_d3aa7e46869bc78d(&rel_known_placeholder_subset_d3aa7e46869bc78d){
}

void Stratum_known_placeholder_subset_10b7319424a0a314::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","known_placeholder_subset"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_known_placeholder_subset_d3aa7e46869bc78d);
} catch (std::exception& e) {std::cerr << "Error loading known_placeholder_subset data: " << e.what() << '\n';
exit(1);
}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","known_placeholder_subset"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_known_placeholder_subset_d3aa7e46869bc78d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_loan_invalidated_at_263f4508732ca10a {
public:
 Stratum_loan_invalidated_at_263f4508732ca10a(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_loan_invalidated_at_3d4d106967a8a5da);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_loan_invalidated_at_3d4d106967a8a5da;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_loan_invalidated_at_263f4508732ca10a::Stratum_loan_invalidated_at_263f4508732ca10a(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_loan_invalidated_at_3d4d106967a8a5da):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_loan_invalidated_at_3d4d106967a8a5da(&rel_loan_invalidated_at_3d4d106967a8a5da){
}

void Stratum_loan_invalidated_at_263f4508732ca10a::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","loan_invalidated_at"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_loan_invalidated_at_3d4d106967a8a5da);
} catch (std::exception& e) {std::cerr << "Error loading loan_invalidated_at data: " << e.what() << '\n';
exit(1);
}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","loan_invalidated_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_loan_invalidated_at_3d4d106967a8a5da);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_loan_live_at_78d49072431de823 {
public:
 Stratum_loan_live_at_78d49072431de823(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_loan_live_at_5761440badf0c5c7);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_loan_live_at_5761440badf0c5c7;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_loan_live_at_78d49072431de823::Stratum_loan_live_at_78d49072431de823(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_loan_live_at_5761440badf0c5c7):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_loan_live_at_5761440badf0c5c7(&rel_loan_live_at_5761440badf0c5c7){
}

void Stratum_loan_live_at_78d49072431de823::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","loan_live_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_loan_live_at_5761440badf0c5c7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_move_error_71e89c25c297481d {
public:
 Stratum_move_error_71e89c25c297481d(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_move_error_b60914d9ec0fbce7,t_btree_ii__0_1__11::Type& rel_path_maybe_uninitialized_on_exit_7353693d35108c78);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_cfg_edge_113c4ec5f576f8cf;
t_btree_ii__0_1__11::Type* rel_move_error_b60914d9ec0fbce7;
t_btree_ii__0_1__11::Type* rel_path_maybe_uninitialized_on_exit_7353693d35108c78;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_move_error_71e89c25c297481d::Stratum_move_error_71e89c25c297481d(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_move_error_b60914d9ec0fbce7,t_btree_ii__0_1__11::Type& rel_path_maybe_uninitialized_on_exit_7353693d35108c78):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_cfg_edge_113c4ec5f576f8cf(&rel_cfg_edge_113c4ec5f576f8cf),
rel_move_error_b60914d9ec0fbce7(&rel_move_error_b60914d9ec0fbce7),
rel_path_maybe_uninitialized_on_exit_7353693d35108c78(&rel_path_maybe_uninitialized_on_exit_7353693d35108c78){
}

void Stratum_move_error_71e89c25c297481d::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(move_error(Path,TargetNode) :- 
   path_maybe_uninitialized_on_exit(Path,SourceNode),
   cfg_edge(SourceNode,TargetNode).
in file polonius.dl [198:1-200:38])_");
if(!(rel_path_maybe_uninitialized_on_exit_7353693d35108c78->empty()) && !(rel_cfg_edge_113c4ec5f576f8cf->empty())) {
[&](){
auto part = rel_path_maybe_uninitialized_on_exit_7353693d35108c78->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt,rel_cfg_edge_113c4ec5f576f8cf->createContext());
CREATE_OP_CONTEXT(rel_move_error_b60914d9ec0fbce7_op_ctxt,rel_move_error_b60914d9ec0fbce7->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt,rel_path_maybe_uninitialized_on_exit_7353693d35108c78->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_cfg_edge_113c4ec5f576f8cf->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt));
for(const auto& env1 : range) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_move_error_b60914d9ec0fbce7->insert(tuple,READ_OP_CONTEXT(rel_move_error_b60914d9ec0fbce7_op_ctxt));
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","move_error"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_move_error_b60914d9ec0fbce7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_origin_contains_loan_on_entry_c07e2f79c52c4867 {
public:
 Stratum_origin_contains_loan_on_entry_c07e2f79c52c4867(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_iii__0_1_2__111::Type& rel_origin_contains_loan_on_entry_179b9d324743ed9c);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_iii__0_1_2__111::Type* rel_origin_contains_loan_on_entry_179b9d324743ed9c;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_origin_contains_loan_on_entry_c07e2f79c52c4867::Stratum_origin_contains_loan_on_entry_c07e2f79c52c4867(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_iii__0_1_2__111::Type& rel_origin_contains_loan_on_entry_179b9d324743ed9c):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_origin_contains_loan_on_entry_179b9d324743ed9c(&rel_origin_contains_loan_on_entry_179b9d324743ed9c){
}

void Stratum_origin_contains_loan_on_entry_c07e2f79c52c4867::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"name","origin_contains_loan_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_origin_contains_loan_on_entry_179b9d324743ed9c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_origin_live_on_entry_755e37540122d60b {
public:
 Stratum_origin_live_on_entry_755e37540122d60b(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_i__0__1::Type& rel_cfg_node_c61b5c6f77f80834,t_btree_ii__0_1__11::Type& rel_origin_live_on_entry_4253bcb785a6d929,t_btree_i__0__1::Type& rel_universal_region_a10614e796c69aef);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_i__0__1::Type* rel_cfg_node_c61b5c6f77f80834;
t_btree_ii__0_1__11::Type* rel_origin_live_on_entry_4253bcb785a6d929;
t_btree_i__0__1::Type* rel_universal_region_a10614e796c69aef;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_origin_live_on_entry_755e37540122d60b::Stratum_origin_live_on_entry_755e37540122d60b(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_i__0__1::Type& rel_cfg_node_c61b5c6f77f80834,t_btree_ii__0_1__11::Type& rel_origin_live_on_entry_4253bcb785a6d929,t_btree_i__0__1::Type& rel_universal_region_a10614e796c69aef):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_cfg_node_c61b5c6f77f80834(&rel_cfg_node_c61b5c6f77f80834),
rel_origin_live_on_entry_4253bcb785a6d929(&rel_origin_live_on_entry_4253bcb785a6d929),
rel_universal_region_a10614e796c69aef(&rel_universal_region_a10614e796c69aef){
}

void Stratum_origin_live_on_entry_755e37540122d60b::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(origin_live_on_entry(origin,point) :- 
   cfg_node(point),
   universal_region(origin).
in file polonius.dl [128:1-130:30])_");
if(!(rel_cfg_node_c61b5c6f77f80834->empty()) && !(rel_universal_region_a10614e796c69aef->empty())) {
[&](){
auto part = rel_cfg_node_c61b5c6f77f80834->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_cfg_node_c61b5c6f77f80834_op_ctxt,rel_cfg_node_c61b5c6f77f80834->createContext());
CREATE_OP_CONTEXT(rel_origin_live_on_entry_4253bcb785a6d929_op_ctxt,rel_origin_live_on_entry_4253bcb785a6d929->createContext());
CREATE_OP_CONTEXT(rel_universal_region_a10614e796c69aef_op_ctxt,rel_universal_region_a10614e796c69aef->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
for(const auto& env1 : *rel_universal_region_a10614e796c69aef) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[0]),ramBitCast(env0[0])}};
rel_origin_live_on_entry_4253bcb785a6d929->insert(tuple,READ_OP_CONTEXT(rel_origin_live_on_entry_4253bcb785a6d929_op_ctxt));
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","origin_live_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_origin_live_on_entry_4253bcb785a6d929);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_universal_region_a10614e796c69aef->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_accessed_at_c65016814ce89194 {
public:
 Stratum_path_accessed_at_c65016814ce89194(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_accessed_at_6d6ad78564a19774,t_btree_ii__0_1__11::Type& rel_new_path_accessed_at_533da89e85764930,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11::Type& rel_path_accessed_at_0c5de7f55ac3352d,t_btree_ii__0_1__11::Type& rel_path_accessed_at_base_5750398152328293);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_path_accessed_at_6d6ad78564a19774;
t_btree_ii__0_1__11::Type* rel_new_path_accessed_at_533da89e85764930;
t_btree_ii__0_1__11__10::Type* rel_ancestor_path_56e99f4d0bc91e6e;
t_btree_ii__0_1__11::Type* rel_path_accessed_at_0c5de7f55ac3352d;
t_btree_ii__0_1__11::Type* rel_path_accessed_at_base_5750398152328293;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_accessed_at_c65016814ce89194::Stratum_path_accessed_at_c65016814ce89194(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_accessed_at_6d6ad78564a19774,t_btree_ii__0_1__11::Type& rel_new_path_accessed_at_533da89e85764930,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11::Type& rel_path_accessed_at_0c5de7f55ac3352d,t_btree_ii__0_1__11::Type& rel_path_accessed_at_base_5750398152328293):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_path_accessed_at_6d6ad78564a19774(&rel_delta_path_accessed_at_6d6ad78564a19774),
rel_new_path_accessed_at_533da89e85764930(&rel_new_path_accessed_at_533da89e85764930),
rel_ancestor_path_56e99f4d0bc91e6e(&rel_ancestor_path_56e99f4d0bc91e6e),
rel_path_accessed_at_0c5de7f55ac3352d(&rel_path_accessed_at_0c5de7f55ac3352d),
rel_path_accessed_at_base_5750398152328293(&rel_path_accessed_at_base_5750398152328293){
}

void Stratum_path_accessed_at_c65016814ce89194::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(path_accessed_at(x,y) :- 
   path_accessed_at_base(x,y).
in file polonius.dl [165:1-165:55])_");
if(!(rel_path_accessed_at_base_5750398152328293->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt,rel_path_accessed_at_0c5de7f55ac3352d->createContext());
CREATE_OP_CONTEXT(rel_path_accessed_at_base_5750398152328293_op_ctxt,rel_path_accessed_at_base_5750398152328293->createContext());
for(const auto& env0 : *rel_path_accessed_at_base_5750398152328293) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_accessed_at_0c5de7f55ac3352d->insert(tuple,READ_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_path_accessed_at_6d6ad78564a19774_op_ctxt,rel_delta_path_accessed_at_6d6ad78564a19774->createContext());
CREATE_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt,rel_path_accessed_at_0c5de7f55ac3352d->createContext());
for(const auto& env0 : *rel_path_accessed_at_0c5de7f55ac3352d) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_path_accessed_at_6d6ad78564a19774->insert(tuple,READ_OP_CONTEXT(rel_delta_path_accessed_at_6d6ad78564a19774_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(path_accessed_at(Child,point) :- 
   path_accessed_at(Parent,point),
   ancestor_path(Parent,Child).
in file polonius.dl [176:1-178:34])_");
if(!(rel_delta_path_accessed_at_6d6ad78564a19774->empty()) && !(rel_ancestor_path_56e99f4d0bc91e6e->empty())) {
[&](){
auto part = rel_delta_path_accessed_at_6d6ad78564a19774->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_path_accessed_at_6d6ad78564a19774_op_ctxt,rel_delta_path_accessed_at_6d6ad78564a19774->createContext());
CREATE_OP_CONTEXT(rel_new_path_accessed_at_533da89e85764930_op_ctxt,rel_new_path_accessed_at_533da89e85764930->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
CREATE_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt,rel_path_accessed_at_0c5de7f55ac3352d->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_ancestor_path_56e99f4d0bc91e6e->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_path_accessed_at_0c5de7f55ac3352d->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[1]),ramBitCast(env0[1])}};
rel_new_path_accessed_at_533da89e85764930->insert(tuple,READ_OP_CONTEXT(rel_new_path_accessed_at_533da89e85764930_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_path_accessed_at_533da89e85764930->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_path_accessed_at_533da89e85764930_op_ctxt,rel_new_path_accessed_at_533da89e85764930->createContext());
CREATE_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt,rel_path_accessed_at_0c5de7f55ac3352d->createContext());
for(const auto& env0 : *rel_new_path_accessed_at_533da89e85764930) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_accessed_at_0c5de7f55ac3352d->insert(tuple,READ_OP_CONTEXT(rel_path_accessed_at_0c5de7f55ac3352d_op_ctxt));
}
}
();std::swap(rel_delta_path_accessed_at_6d6ad78564a19774, rel_new_path_accessed_at_533da89e85764930);
rel_new_path_accessed_at_533da89e85764930->purge();
iter++;
}
iter = 0;
rel_delta_path_accessed_at_6d6ad78564a19774->purge();
rel_new_path_accessed_at_533da89e85764930->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_accessed_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_accessed_at_0c5de7f55ac3352d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_path_accessed_at_base_5750398152328293->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_accessed_at_base_3fb237503c193020 {
public:
 Stratum_path_accessed_at_base_3fb237503c193020(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_accessed_at_base_5750398152328293);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_path_accessed_at_base_5750398152328293;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_accessed_at_base_3fb237503c193020::Stratum_path_accessed_at_base_3fb237503c193020(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_accessed_at_base_5750398152328293):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_path_accessed_at_base_5750398152328293(&rel_path_accessed_at_base_5750398152328293){
}

void Stratum_path_accessed_at_base_3fb237503c193020::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_accessed_at_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_accessed_at_base_5750398152328293);
} catch (std::exception& e) {std::cerr << "Error loading path_accessed_at_base data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_assigned_at_621142eb80792795 {
public:
 Stratum_path_assigned_at_621142eb80792795(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_assigned_at_c33236013564d9a6,t_btree_ii__0_1__11::Type& rel_new_path_assigned_at_3751a63a80813906,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11::Type& rel_path_assigned_at_110ef50c11512ca7,t_btree_ii__0_1__11::Type& rel_path_assigned_at_base_baa182de12fa38e4);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_path_assigned_at_c33236013564d9a6;
t_btree_ii__0_1__11::Type* rel_new_path_assigned_at_3751a63a80813906;
t_btree_ii__0_1__11__10::Type* rel_ancestor_path_56e99f4d0bc91e6e;
t_btree_ii__0_1__11::Type* rel_path_assigned_at_110ef50c11512ca7;
t_btree_ii__0_1__11::Type* rel_path_assigned_at_base_baa182de12fa38e4;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_assigned_at_621142eb80792795::Stratum_path_assigned_at_621142eb80792795(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_assigned_at_c33236013564d9a6,t_btree_ii__0_1__11::Type& rel_new_path_assigned_at_3751a63a80813906,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11::Type& rel_path_assigned_at_110ef50c11512ca7,t_btree_ii__0_1__11::Type& rel_path_assigned_at_base_baa182de12fa38e4):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_path_assigned_at_c33236013564d9a6(&rel_delta_path_assigned_at_c33236013564d9a6),
rel_new_path_assigned_at_3751a63a80813906(&rel_new_path_assigned_at_3751a63a80813906),
rel_ancestor_path_56e99f4d0bc91e6e(&rel_ancestor_path_56e99f4d0bc91e6e),
rel_path_assigned_at_110ef50c11512ca7(&rel_path_assigned_at_110ef50c11512ca7),
rel_path_assigned_at_base_baa182de12fa38e4(&rel_path_assigned_at_base_baa182de12fa38e4){
}

void Stratum_path_assigned_at_621142eb80792795::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(path_assigned_at(x,y) :- 
   path_assigned_at_base(x,y).
in file polonius.dl [164:1-164:55])_");
if(!(rel_path_assigned_at_base_baa182de12fa38e4->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt,rel_path_assigned_at_110ef50c11512ca7->createContext());
CREATE_OP_CONTEXT(rel_path_assigned_at_base_baa182de12fa38e4_op_ctxt,rel_path_assigned_at_base_baa182de12fa38e4->createContext());
for(const auto& env0 : *rel_path_assigned_at_base_baa182de12fa38e4) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_assigned_at_110ef50c11512ca7->insert(tuple,READ_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_path_assigned_at_c33236013564d9a6_op_ctxt,rel_delta_path_assigned_at_c33236013564d9a6->createContext());
CREATE_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt,rel_path_assigned_at_110ef50c11512ca7->createContext());
for(const auto& env0 : *rel_path_assigned_at_110ef50c11512ca7) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_path_assigned_at_c33236013564d9a6->insert(tuple,READ_OP_CONTEXT(rel_delta_path_assigned_at_c33236013564d9a6_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(path_assigned_at(Child,point) :- 
   path_assigned_at(Parent,point),
   ancestor_path(Parent,Child).
in file polonius.dl [173:1-175:34])_");
if(!(rel_delta_path_assigned_at_c33236013564d9a6->empty()) && !(rel_ancestor_path_56e99f4d0bc91e6e->empty())) {
[&](){
auto part = rel_delta_path_assigned_at_c33236013564d9a6->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_path_assigned_at_c33236013564d9a6_op_ctxt,rel_delta_path_assigned_at_c33236013564d9a6->createContext());
CREATE_OP_CONTEXT(rel_new_path_assigned_at_3751a63a80813906_op_ctxt,rel_new_path_assigned_at_3751a63a80813906->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
CREATE_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt,rel_path_assigned_at_110ef50c11512ca7->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_ancestor_path_56e99f4d0bc91e6e->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_path_assigned_at_110ef50c11512ca7->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[1]),ramBitCast(env0[1])}};
rel_new_path_assigned_at_3751a63a80813906->insert(tuple,READ_OP_CONTEXT(rel_new_path_assigned_at_3751a63a80813906_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_path_assigned_at_3751a63a80813906->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_path_assigned_at_3751a63a80813906_op_ctxt,rel_new_path_assigned_at_3751a63a80813906->createContext());
CREATE_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt,rel_path_assigned_at_110ef50c11512ca7->createContext());
for(const auto& env0 : *rel_new_path_assigned_at_3751a63a80813906) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_assigned_at_110ef50c11512ca7->insert(tuple,READ_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt));
}
}
();std::swap(rel_delta_path_assigned_at_c33236013564d9a6, rel_new_path_assigned_at_3751a63a80813906);
rel_new_path_assigned_at_3751a63a80813906->purge();
iter++;
}
iter = 0;
rel_delta_path_assigned_at_c33236013564d9a6->purge();
rel_new_path_assigned_at_3751a63a80813906->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_assigned_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_assigned_at_110ef50c11512ca7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_path_assigned_at_base_baa182de12fa38e4->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_assigned_at_base_fb2821af12987abc {
public:
 Stratum_path_assigned_at_base_fb2821af12987abc(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_assigned_at_base_baa182de12fa38e4);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_path_assigned_at_base_baa182de12fa38e4;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_assigned_at_base_fb2821af12987abc::Stratum_path_assigned_at_base_fb2821af12987abc(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_assigned_at_base_baa182de12fa38e4):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_path_assigned_at_base_baa182de12fa38e4(&rel_path_assigned_at_base_baa182de12fa38e4){
}

void Stratum_path_assigned_at_base_fb2821af12987abc::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_assigned_at_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_assigned_at_base_baa182de12fa38e4);
} catch (std::exception& e) {std::cerr << "Error loading path_assigned_at_base data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_begins_with_var_2db6b88b797a3f4b {
public:
 Stratum_path_begins_with_var_2db6b88b797a3f4b(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_begins_with_var_e7c1d67c31c31e22,t_btree_ii__0_1__11::Type& rel_new_path_begins_with_var_1c04b903c23b96d0,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11__10::Type& rel_path_begins_with_var_31b7afffef8c9178,t_btree_ii__0_1__11::Type& rel_path_is_var_cd572d441f221f1c);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_path_begins_with_var_e7c1d67c31c31e22;
t_btree_ii__0_1__11::Type* rel_new_path_begins_with_var_1c04b903c23b96d0;
t_btree_ii__0_1__11__10::Type* rel_ancestor_path_56e99f4d0bc91e6e;
t_btree_ii__0_1__11__10::Type* rel_path_begins_with_var_31b7afffef8c9178;
t_btree_ii__0_1__11::Type* rel_path_is_var_cd572d441f221f1c;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_begins_with_var_2db6b88b797a3f4b::Stratum_path_begins_with_var_2db6b88b797a3f4b(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_begins_with_var_e7c1d67c31c31e22,t_btree_ii__0_1__11::Type& rel_new_path_begins_with_var_1c04b903c23b96d0,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11__10::Type& rel_path_begins_with_var_31b7afffef8c9178,t_btree_ii__0_1__11::Type& rel_path_is_var_cd572d441f221f1c):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_path_begins_with_var_e7c1d67c31c31e22(&rel_delta_path_begins_with_var_e7c1d67c31c31e22),
rel_new_path_begins_with_var_1c04b903c23b96d0(&rel_new_path_begins_with_var_1c04b903c23b96d0),
rel_ancestor_path_56e99f4d0bc91e6e(&rel_ancestor_path_56e99f4d0bc91e6e),
rel_path_begins_with_var_31b7afffef8c9178(&rel_path_begins_with_var_31b7afffef8c9178),
rel_path_is_var_cd572d441f221f1c(&rel_path_is_var_cd572d441f221f1c){
}

void Stratum_path_begins_with_var_2db6b88b797a3f4b::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(path_begins_with_var(x,var) :- 
   path_is_var(x,var).
in file polonius.dl [166:1-166:53])_");
if(!(rel_path_is_var_cd572d441f221f1c->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt,rel_path_begins_with_var_31b7afffef8c9178->createContext());
CREATE_OP_CONTEXT(rel_path_is_var_cd572d441f221f1c_op_ctxt,rel_path_is_var_cd572d441f221f1c->createContext());
for(const auto& env0 : *rel_path_is_var_cd572d441f221f1c) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_begins_with_var_31b7afffef8c9178->insert(tuple,READ_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_path_begins_with_var_e7c1d67c31c31e22_op_ctxt,rel_delta_path_begins_with_var_e7c1d67c31c31e22->createContext());
CREATE_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt,rel_path_begins_with_var_31b7afffef8c9178->createContext());
for(const auto& env0 : *rel_path_begins_with_var_31b7afffef8c9178) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_path_begins_with_var_e7c1d67c31c31e22->insert(tuple,READ_OP_CONTEXT(rel_delta_path_begins_with_var_e7c1d67c31c31e22_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(path_begins_with_var(Child,Var) :- 
   path_begins_with_var(Parent,Var),
   ancestor_path(Parent,Child).
in file polonius.dl [179:1-181:34])_");
if(!(rel_delta_path_begins_with_var_e7c1d67c31c31e22->empty()) && !(rel_ancestor_path_56e99f4d0bc91e6e->empty())) {
[&](){
auto part = rel_delta_path_begins_with_var_e7c1d67c31c31e22->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_path_begins_with_var_e7c1d67c31c31e22_op_ctxt,rel_delta_path_begins_with_var_e7c1d67c31c31e22->createContext());
CREATE_OP_CONTEXT(rel_new_path_begins_with_var_1c04b903c23b96d0_op_ctxt,rel_new_path_begins_with_var_1c04b903c23b96d0->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
CREATE_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt,rel_path_begins_with_var_31b7afffef8c9178->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_ancestor_path_56e99f4d0bc91e6e->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_path_begins_with_var_31b7afffef8c9178->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[1]),ramBitCast(env0[1])}};
rel_new_path_begins_with_var_1c04b903c23b96d0->insert(tuple,READ_OP_CONTEXT(rel_new_path_begins_with_var_1c04b903c23b96d0_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_path_begins_with_var_1c04b903c23b96d0->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_path_begins_with_var_1c04b903c23b96d0_op_ctxt,rel_new_path_begins_with_var_1c04b903c23b96d0->createContext());
CREATE_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt,rel_path_begins_with_var_31b7afffef8c9178->createContext());
for(const auto& env0 : *rel_new_path_begins_with_var_1c04b903c23b96d0) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_begins_with_var_31b7afffef8c9178->insert(tuple,READ_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt));
}
}
();std::swap(rel_delta_path_begins_with_var_e7c1d67c31c31e22, rel_new_path_begins_with_var_1c04b903c23b96d0);
rel_new_path_begins_with_var_1c04b903c23b96d0->purge();
iter++;
}
iter = 0;
rel_delta_path_begins_with_var_e7c1d67c31c31e22->purge();
rel_new_path_begins_with_var_1c04b903c23b96d0->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_begins_with_var"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_begins_with_var_31b7afffef8c9178);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_path_is_var_cd572d441f221f1c->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_is_var_2dc04b24a90ca307 {
public:
 Stratum_path_is_var_2dc04b24a90ca307(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_is_var_cd572d441f221f1c);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_path_is_var_cd572d441f221f1c;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_is_var_2dc04b24a90ca307::Stratum_path_is_var_2dc04b24a90ca307(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_is_var_cd572d441f221f1c):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_path_is_var_cd572d441f221f1c(&rel_path_is_var_cd572d441f221f1c){
}

void Stratum_path_is_var_2dc04b24a90ca307::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_is_var"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_is_var_cd572d441f221f1c);
} catch (std::exception& e) {std::cerr << "Error loading path_is_var data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_maybe_initialized_on_exit_8501b0421286a7e8 {
public:
 Stratum_path_maybe_initialized_on_exit_8501b0421286a7e8(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea,t_btree_ii__0_1__11::Type& rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_path_assigned_at_110ef50c11512ca7,t_btree_ii__0_1__11::Type& rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5,t_btree_ii__0_1__11::Type& rel_path_moved_at_db26ad17e580a28b);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea;
t_btree_ii__0_1__11::Type* rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999;
t_btree_ii__0_1__11__10::Type* rel_cfg_edge_113c4ec5f576f8cf;
t_btree_ii__0_1__11::Type* rel_path_assigned_at_110ef50c11512ca7;
t_btree_ii__0_1__11::Type* rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5;
t_btree_ii__0_1__11::Type* rel_path_moved_at_db26ad17e580a28b;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_maybe_initialized_on_exit_8501b0421286a7e8::Stratum_path_maybe_initialized_on_exit_8501b0421286a7e8(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea,t_btree_ii__0_1__11::Type& rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_path_assigned_at_110ef50c11512ca7,t_btree_ii__0_1__11::Type& rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5,t_btree_ii__0_1__11::Type& rel_path_moved_at_db26ad17e580a28b):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea(&rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea),
rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999(&rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999),
rel_cfg_edge_113c4ec5f576f8cf(&rel_cfg_edge_113c4ec5f576f8cf),
rel_path_assigned_at_110ef50c11512ca7(&rel_path_assigned_at_110ef50c11512ca7),
rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5(&rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5),
rel_path_moved_at_db26ad17e580a28b(&rel_path_moved_at_db26ad17e580a28b){
}

void Stratum_path_maybe_initialized_on_exit_8501b0421286a7e8::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(path_maybe_initialized_on_exit(path,point) :- 
   path_assigned_at(path,point).
in file polonius.dl [183:1-184:35])_");
if(!(rel_path_assigned_at_110ef50c11512ca7->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt,rel_path_assigned_at_110ef50c11512ca7->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt,rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->createContext());
for(const auto& env0 : *rel_path_assigned_at_110ef50c11512ca7) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->insert(tuple,READ_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea_op_ctxt,rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt,rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->createContext());
for(const auto& env0 : *rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea->insert(tuple,READ_OP_CONTEXT(rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(path_maybe_initialized_on_exit(path,point2) :- 
   path_maybe_initialized_on_exit(path,point1),
   cfg_edge(point1,point2),
   !path_moved_at(path,point2).
in file polonius.dl [187:1-190:34])_");
if(!(rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea->empty()) && !(rel_cfg_edge_113c4ec5f576f8cf->empty())) {
[&](){
auto part = rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea_op_ctxt,rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea->createContext());
CREATE_OP_CONTEXT(rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999_op_ctxt,rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999->createContext());
CREATE_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt,rel_cfg_edge_113c4ec5f576f8cf->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt,rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->createContext());
CREATE_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt,rel_path_moved_at_db26ad17e580a28b->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_cfg_edge_113c4ec5f576f8cf->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_path_moved_at_db26ad17e580a28b->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt))) && !(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999->insert(tuple,READ_OP_CONTEXT(rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999_op_ctxt,rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt,rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->createContext());
for(const auto& env0 : *rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->insert(tuple,READ_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt));
}
}
();std::swap(rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea, rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999);
rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999->purge();
iter++;
}
iter = 0;
rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea->purge();
rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_maybe_initialized_on_exit"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_maybe_uninitialized_on_exit_8c2a6781b6de6211 {
public:
 Stratum_path_maybe_uninitialized_on_exit_8c2a6781b6de6211(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77,t_btree_ii__0_1__11::Type& rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_path_assigned_at_110ef50c11512ca7,t_btree_ii__0_1__11::Type& rel_path_maybe_uninitialized_on_exit_7353693d35108c78,t_btree_ii__0_1__11::Type& rel_path_moved_at_db26ad17e580a28b);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77;
t_btree_ii__0_1__11::Type* rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782;
t_btree_ii__0_1__11__10::Type* rel_cfg_edge_113c4ec5f576f8cf;
t_btree_ii__0_1__11::Type* rel_path_assigned_at_110ef50c11512ca7;
t_btree_ii__0_1__11::Type* rel_path_maybe_uninitialized_on_exit_7353693d35108c78;
t_btree_ii__0_1__11::Type* rel_path_moved_at_db26ad17e580a28b;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_maybe_uninitialized_on_exit_8c2a6781b6de6211::Stratum_path_maybe_uninitialized_on_exit_8c2a6781b6de6211(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77,t_btree_ii__0_1__11::Type& rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_path_assigned_at_110ef50c11512ca7,t_btree_ii__0_1__11::Type& rel_path_maybe_uninitialized_on_exit_7353693d35108c78,t_btree_ii__0_1__11::Type& rel_path_moved_at_db26ad17e580a28b):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77(&rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77),
rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782(&rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782),
rel_cfg_edge_113c4ec5f576f8cf(&rel_cfg_edge_113c4ec5f576f8cf),
rel_path_assigned_at_110ef50c11512ca7(&rel_path_assigned_at_110ef50c11512ca7),
rel_path_maybe_uninitialized_on_exit_7353693d35108c78(&rel_path_maybe_uninitialized_on_exit_7353693d35108c78),
rel_path_moved_at_db26ad17e580a28b(&rel_path_moved_at_db26ad17e580a28b){
}

void Stratum_path_maybe_uninitialized_on_exit_8c2a6781b6de6211::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(path_maybe_uninitialized_on_exit(path,point) :- 
   path_moved_at(path,point).
in file polonius.dl [185:1-186:32])_");
if(!(rel_path_moved_at_db26ad17e580a28b->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt,rel_path_maybe_uninitialized_on_exit_7353693d35108c78->createContext());
CREATE_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt,rel_path_moved_at_db26ad17e580a28b->createContext());
for(const auto& env0 : *rel_path_moved_at_db26ad17e580a28b) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_maybe_uninitialized_on_exit_7353693d35108c78->insert(tuple,READ_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77_op_ctxt,rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt,rel_path_maybe_uninitialized_on_exit_7353693d35108c78->createContext());
for(const auto& env0 : *rel_path_maybe_uninitialized_on_exit_7353693d35108c78) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77->insert(tuple,READ_OP_CONTEXT(rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(path_maybe_uninitialized_on_exit(path,point2) :- 
   path_maybe_uninitialized_on_exit(path,point1),
   cfg_edge(point1,point2),
   !path_assigned_at(path,point2).
in file polonius.dl [191:1-194:37])_");
if(!(rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77->empty()) && !(rel_cfg_edge_113c4ec5f576f8cf->empty())) {
[&](){
auto part = rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77_op_ctxt,rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77->createContext());
CREATE_OP_CONTEXT(rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782_op_ctxt,rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782->createContext());
CREATE_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt,rel_cfg_edge_113c4ec5f576f8cf->createContext());
CREATE_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt,rel_path_assigned_at_110ef50c11512ca7->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt,rel_path_maybe_uninitialized_on_exit_7353693d35108c78->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_cfg_edge_113c4ec5f576f8cf->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_path_assigned_at_110ef50c11512ca7->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_path_assigned_at_110ef50c11512ca7_op_ctxt))) && !(rel_path_maybe_uninitialized_on_exit_7353693d35108c78->contains(Tuple<RamDomain,2>{{ramBitCast(env0[0]),ramBitCast(env1[1])}},READ_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782->insert(tuple,READ_OP_CONTEXT(rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782_op_ctxt,rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt,rel_path_maybe_uninitialized_on_exit_7353693d35108c78->createContext());
for(const auto& env0 : *rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_maybe_uninitialized_on_exit_7353693d35108c78->insert(tuple,READ_OP_CONTEXT(rel_path_maybe_uninitialized_on_exit_7353693d35108c78_op_ctxt));
}
}
();std::swap(rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77, rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782);
rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782->purge();
iter++;
}
iter = 0;
rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77->purge();
rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_maybe_uninitialized_on_exit"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_maybe_uninitialized_on_exit_7353693d35108c78);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_moved_at_577bd1dfc2c609a8 {
public:
 Stratum_path_moved_at_577bd1dfc2c609a8(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_moved_at_8f857592b06b7171,t_btree_ii__0_1__11::Type& rel_new_path_moved_at_4019859f9b547927,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11::Type& rel_path_moved_at_db26ad17e580a28b,t_btree_ii__0_1__11::Type& rel_path_moved_at_base_3824abd0ff20d508);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_delta_path_moved_at_8f857592b06b7171;
t_btree_ii__0_1__11::Type* rel_new_path_moved_at_4019859f9b547927;
t_btree_ii__0_1__11__10::Type* rel_ancestor_path_56e99f4d0bc91e6e;
t_btree_ii__0_1__11::Type* rel_path_moved_at_db26ad17e580a28b;
t_btree_ii__0_1__11::Type* rel_path_moved_at_base_3824abd0ff20d508;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_moved_at_577bd1dfc2c609a8::Stratum_path_moved_at_577bd1dfc2c609a8(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_delta_path_moved_at_8f857592b06b7171,t_btree_ii__0_1__11::Type& rel_new_path_moved_at_4019859f9b547927,t_btree_ii__0_1__11__10::Type& rel_ancestor_path_56e99f4d0bc91e6e,t_btree_ii__0_1__11::Type& rel_path_moved_at_db26ad17e580a28b,t_btree_ii__0_1__11::Type& rel_path_moved_at_base_3824abd0ff20d508):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_delta_path_moved_at_8f857592b06b7171(&rel_delta_path_moved_at_8f857592b06b7171),
rel_new_path_moved_at_4019859f9b547927(&rel_new_path_moved_at_4019859f9b547927),
rel_ancestor_path_56e99f4d0bc91e6e(&rel_ancestor_path_56e99f4d0bc91e6e),
rel_path_moved_at_db26ad17e580a28b(&rel_path_moved_at_db26ad17e580a28b),
rel_path_moved_at_base_3824abd0ff20d508(&rel_path_moved_at_base_3824abd0ff20d508){
}

void Stratum_path_moved_at_577bd1dfc2c609a8::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(path_moved_at(x,y) :- 
   path_moved_at_base(x,y).
in file polonius.dl [163:1-163:49])_");
if(!(rel_path_moved_at_base_3824abd0ff20d508->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt,rel_path_moved_at_db26ad17e580a28b->createContext());
CREATE_OP_CONTEXT(rel_path_moved_at_base_3824abd0ff20d508_op_ctxt,rel_path_moved_at_base_3824abd0ff20d508->createContext());
for(const auto& env0 : *rel_path_moved_at_base_3824abd0ff20d508) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_moved_at_db26ad17e580a28b->insert(tuple,READ_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt));
}
}
();}
[&](){
CREATE_OP_CONTEXT(rel_delta_path_moved_at_8f857592b06b7171_op_ctxt,rel_delta_path_moved_at_8f857592b06b7171->createContext());
CREATE_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt,rel_path_moved_at_db26ad17e580a28b->createContext());
for(const auto& env0 : *rel_path_moved_at_db26ad17e580a28b) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_delta_path_moved_at_8f857592b06b7171->insert(tuple,READ_OP_CONTEXT(rel_delta_path_moved_at_8f857592b06b7171_op_ctxt));
}
}
();iter = 0;
for(;;) {
signalHandler->setMsg(R"_(path_moved_at(Child,Point) :- 
   path_moved_at(Parent,Point),
   ancestor_path(Parent,Child).
in file polonius.dl [170:1-172:34])_");
if(!(rel_delta_path_moved_at_8f857592b06b7171->empty()) && !(rel_ancestor_path_56e99f4d0bc91e6e->empty())) {
[&](){
auto part = rel_delta_path_moved_at_8f857592b06b7171->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_delta_path_moved_at_8f857592b06b7171_op_ctxt,rel_delta_path_moved_at_8f857592b06b7171->createContext());
CREATE_OP_CONTEXT(rel_new_path_moved_at_4019859f9b547927_op_ctxt,rel_new_path_moved_at_4019859f9b547927->createContext());
CREATE_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt,rel_ancestor_path_56e99f4d0bc91e6e->createContext());
CREATE_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt,rel_path_moved_at_db26ad17e580a28b->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_ancestor_path_56e99f4d0bc91e6e->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_ancestor_path_56e99f4d0bc91e6e_op_ctxt));
for(const auto& env1 : range) {
if( !(rel_path_moved_at_db26ad17e580a28b->contains(Tuple<RamDomain,2>{{ramBitCast(env1[1]),ramBitCast(env0[1])}},READ_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt)))) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[1]),ramBitCast(env0[1])}};
rel_new_path_moved_at_4019859f9b547927->insert(tuple,READ_OP_CONTEXT(rel_new_path_moved_at_4019859f9b547927_op_ctxt));
}
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if(rel_new_path_moved_at_4019859f9b547927->empty()) break;
[&](){
CREATE_OP_CONTEXT(rel_new_path_moved_at_4019859f9b547927_op_ctxt,rel_new_path_moved_at_4019859f9b547927->createContext());
CREATE_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt,rel_path_moved_at_db26ad17e580a28b->createContext());
for(const auto& env0 : *rel_new_path_moved_at_4019859f9b547927) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_path_moved_at_db26ad17e580a28b->insert(tuple,READ_OP_CONTEXT(rel_path_moved_at_db26ad17e580a28b_op_ctxt));
}
}
();std::swap(rel_delta_path_moved_at_8f857592b06b7171, rel_new_path_moved_at_4019859f9b547927);
rel_new_path_moved_at_4019859f9b547927->purge();
iter++;
}
iter = 0;
rel_delta_path_moved_at_8f857592b06b7171->purge();
rel_new_path_moved_at_4019859f9b547927->purge();
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_moved_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_moved_at_db26ad17e580a28b);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_path_moved_at_base_3824abd0ff20d508->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_path_moved_at_base_29cf06c442229de3 {
public:
 Stratum_path_moved_at_base_29cf06c442229de3(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_moved_at_base_3824abd0ff20d508);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_path_moved_at_base_3824abd0ff20d508;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_path_moved_at_base_29cf06c442229de3::Stratum_path_moved_at_base_29cf06c442229de3(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_path_moved_at_base_3824abd0ff20d508):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_path_moved_at_base_3824abd0ff20d508(&rel_path_moved_at_base_3824abd0ff20d508){
}

void Stratum_path_moved_at_base_29cf06c442229de3::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_moved_at_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_moved_at_base_3824abd0ff20d508);
} catch (std::exception& e) {std::cerr << "Error loading path_moved_at_base data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_placeholder_origin_288e65506ee4cf7c {
public:
 Stratum_placeholder_origin_288e65506ee4cf7c(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_i__0__1::Type& rel_placeholder_origin_b6a06634cbe5f6b9);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_i__0__1::Type* rel_placeholder_origin_b6a06634cbe5f6b9;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_placeholder_origin_288e65506ee4cf7c::Stratum_placeholder_origin_288e65506ee4cf7c(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_i__0__1::Type& rel_placeholder_origin_b6a06634cbe5f6b9):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_placeholder_origin_b6a06634cbe5f6b9(&rel_placeholder_origin_b6a06634cbe5f6b9){
}

void Stratum_placeholder_origin_288e65506ee4cf7c::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x"},{"auxArity","0"},{"name","placeholder_origin"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 1, \"params\": [\"x\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 1, \"types\": [\"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_placeholder_origin_b6a06634cbe5f6b9);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_subset_base_25e48f147a6a85b9 {
public:
 Stratum_subset_base_25e48f147a6a85b9(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_iii__0_1_2__111::Type& rel_subset_base_b78bf93877fd780c);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_iii__0_1_2__111::Type* rel_subset_base_b78bf93877fd780c;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_subset_base_25e48f147a6a85b9::Stratum_subset_base_25e48f147a6a85b9(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_iii__0_1_2__111::Type& rel_subset_base_b78bf93877fd780c):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_subset_base_b78bf93877fd780c(&rel_subset_base_b78bf93877fd780c){
}

void Stratum_subset_base_25e48f147a6a85b9::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","subset_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_subset_base_b78bf93877fd780c);
} catch (std::exception& e) {std::cerr << "Error loading subset_base data: " << e.what() << '\n';
exit(1);
}
}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"name","subset_base"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_subset_base_b78bf93877fd780c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_subset_error_29f91fb918a1c7d2 {
public:
 Stratum_subset_error_29f91fb918a1c7d2(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_iii__0_1_2__111::Type& rel_subset_error_b2b916084fa03516);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_iii__0_1_2__111::Type* rel_subset_error_b2b916084fa03516;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_subset_error_29f91fb918a1c7d2::Stratum_subset_error_29f91fb918a1c7d2(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_iii__0_1_2__111::Type& rel_subset_error_b2b916084fa03516):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_subset_error_b2b916084fa03516(&rel_subset_error_b2b916084fa03516){
}

void Stratum_subset_error_29f91fb918a1c7d2::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"name","subset_error"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_subset_error_b2b916084fa03516);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_universal_region_487e9539ebce1ca4 {
public:
 Stratum_universal_region_487e9539ebce1ca4(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_i__0__1::Type& rel_universal_region_a10614e796c69aef);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_i__0__1::Type* rel_universal_region_a10614e796c69aef;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_universal_region_487e9539ebce1ca4::Stratum_universal_region_487e9539ebce1ca4(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_i__0__1::Type& rel_universal_region_a10614e796c69aef):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_universal_region_a10614e796c69aef(&rel_universal_region_a10614e796c69aef){
}

void Stratum_universal_region_487e9539ebce1ca4::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","universal_region"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 1, \"params\": [\"x\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 1, \"types\": [\"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_universal_region_a10614e796c69aef);
} catch (std::exception& e) {std::cerr << "Error loading universal_region data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_var_drop_live_on_entry_f3cb1f8d79709ee7 {
public:
 Stratum_var_drop_live_on_entry_f3cb1f8d79709ee7(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_var_drop_live_on_entry_8196af42d8d10787);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_var_drop_live_on_entry_8196af42d8d10787;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_var_drop_live_on_entry_f3cb1f8d79709ee7::Stratum_var_drop_live_on_entry_f3cb1f8d79709ee7(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_var_drop_live_on_entry_8196af42d8d10787):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_var_drop_live_on_entry_8196af42d8d10787(&rel_var_drop_live_on_entry_8196af42d8d10787){
}

void Stratum_var_drop_live_on_entry_f3cb1f8d79709ee7::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_drop_live_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_drop_live_on_entry_8196af42d8d10787);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_var_live_on_entry_728d1d06a4a4c39a {
public:
 Stratum_var_live_on_entry_728d1d06a4a4c39a(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_var_live_on_entry_844bfb36987186e4,t_btree_ii__0_1__11::Type& rel_var_used_at_b2462f789cb6e6a2);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_var_live_on_entry_844bfb36987186e4;
t_btree_ii__0_1__11::Type* rel_var_used_at_b2462f789cb6e6a2;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_var_live_on_entry_728d1d06a4a4c39a::Stratum_var_live_on_entry_728d1d06a4a4c39a(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_var_live_on_entry_844bfb36987186e4,t_btree_ii__0_1__11::Type& rel_var_used_at_b2462f789cb6e6a2):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_var_live_on_entry_844bfb36987186e4(&rel_var_live_on_entry_844bfb36987186e4),
rel_var_used_at_b2462f789cb6e6a2(&rel_var_used_at_b2462f789cb6e6a2){
}

void Stratum_var_live_on_entry_728d1d06a4a4c39a::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(var_live_on_entry(var,point) :- 
   var_used_at(var,point).
in file polonius.dl [137:1-137:58])_");
if(!(rel_var_used_at_b2462f789cb6e6a2->empty())) {
[&](){
CREATE_OP_CONTEXT(rel_var_live_on_entry_844bfb36987186e4_op_ctxt,rel_var_live_on_entry_844bfb36987186e4->createContext());
CREATE_OP_CONTEXT(rel_var_used_at_b2462f789cb6e6a2_op_ctxt,rel_var_used_at_b2462f789cb6e6a2->createContext());
for(const auto& env0 : *rel_var_used_at_b2462f789cb6e6a2) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env0[1])}};
rel_var_live_on_entry_844bfb36987186e4->insert(tuple,READ_OP_CONTEXT(rel_var_live_on_entry_844bfb36987186e4_op_ctxt));
}
}
();}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_live_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_live_on_entry_844bfb36987186e4);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_var_used_at_b2462f789cb6e6a2->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_var_maybe_partly_initialized_on_entry_07e9d53803911812 {
public:
 Stratum_var_maybe_partly_initialized_on_entry_07e9d53803911812(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d,t_btree_ii__0_1__11::Type& rel_var_maybe_partly_initialized_on_exit_794531425885c5e5);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_cfg_edge_113c4ec5f576f8cf;
t_btree_ii__0_1__11::Type* rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d;
t_btree_ii__0_1__11::Type* rel_var_maybe_partly_initialized_on_exit_794531425885c5e5;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_var_maybe_partly_initialized_on_entry_07e9d53803911812::Stratum_var_maybe_partly_initialized_on_entry_07e9d53803911812(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_cfg_edge_113c4ec5f576f8cf,t_btree_ii__0_1__11::Type& rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d,t_btree_ii__0_1__11::Type& rel_var_maybe_partly_initialized_on_exit_794531425885c5e5):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_cfg_edge_113c4ec5f576f8cf(&rel_cfg_edge_113c4ec5f576f8cf),
rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d(&rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d),
rel_var_maybe_partly_initialized_on_exit_794531425885c5e5(&rel_var_maybe_partly_initialized_on_exit_794531425885c5e5){
}

void Stratum_var_maybe_partly_initialized_on_entry_07e9d53803911812::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(var_maybe_partly_initialized_on_entry(var,point2) :- 
   var_maybe_partly_initialized_on_exit(var,point1),
   cfg_edge(point1,point2).
in file polonius.dl [138:1-140:30])_");
if(!(rel_var_maybe_partly_initialized_on_exit_794531425885c5e5->empty()) && !(rel_cfg_edge_113c4ec5f576f8cf->empty())) {
[&](){
auto part = rel_var_maybe_partly_initialized_on_exit_794531425885c5e5->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt,rel_cfg_edge_113c4ec5f576f8cf->createContext());
CREATE_OP_CONTEXT(rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d_op_ctxt,rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d->createContext());
CREATE_OP_CONTEXT(rel_var_maybe_partly_initialized_on_exit_794531425885c5e5_op_ctxt,rel_var_maybe_partly_initialized_on_exit_794531425885c5e5->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_cfg_edge_113c4ec5f576f8cf->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[1]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_cfg_edge_113c4ec5f576f8cf_op_ctxt));
for(const auto& env1 : range) {
Tuple<RamDomain,2> tuple{{ramBitCast(env0[0]),ramBitCast(env1[1])}};
rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d->insert(tuple,READ_OP_CONTEXT(rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d_op_ctxt));
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_maybe_partly_initialized_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
if (pruneImdtRels) rel_cfg_edge_113c4ec5f576f8cf->purge();
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_var_maybe_partly_initialized_on_exit_8a8a0186389a8f24 {
public:
 Stratum_var_maybe_partly_initialized_on_exit_8a8a0186389a8f24(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_path_begins_with_var_31b7afffef8c9178,t_btree_ii__0_1__11::Type& rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5,t_btree_ii__0_1__11::Type& rel_var_maybe_partly_initialized_on_exit_794531425885c5e5);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11__10::Type* rel_path_begins_with_var_31b7afffef8c9178;
t_btree_ii__0_1__11::Type* rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5;
t_btree_ii__0_1__11::Type* rel_var_maybe_partly_initialized_on_exit_794531425885c5e5;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_var_maybe_partly_initialized_on_exit_8a8a0186389a8f24::Stratum_var_maybe_partly_initialized_on_exit_8a8a0186389a8f24(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11__10::Type& rel_path_begins_with_var_31b7afffef8c9178,t_btree_ii__0_1__11::Type& rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5,t_btree_ii__0_1__11::Type& rel_var_maybe_partly_initialized_on_exit_794531425885c5e5):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_path_begins_with_var_31b7afffef8c9178(&rel_path_begins_with_var_31b7afffef8c9178),
rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5(&rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5),
rel_var_maybe_partly_initialized_on_exit_794531425885c5e5(&rel_var_maybe_partly_initialized_on_exit_794531425885c5e5){
}

void Stratum_var_maybe_partly_initialized_on_exit_8a8a0186389a8f24::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
signalHandler->setMsg(R"_(var_maybe_partly_initialized_on_exit(var,point) :- 
   path_maybe_initialized_on_exit(path,point),
   path_begins_with_var(path,var).
in file polonius.dl [195:1-197:37])_");
if(!(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->empty()) && !(rel_path_begins_with_var_31b7afffef8c9178->empty())) {
[&](){
auto part = rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->partition();
PARALLEL_START
CREATE_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt,rel_path_begins_with_var_31b7afffef8c9178->createContext());
CREATE_OP_CONTEXT(rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5_op_ctxt,rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5->createContext());
CREATE_OP_CONTEXT(rel_var_maybe_partly_initialized_on_exit_794531425885c5e5_op_ctxt,rel_var_maybe_partly_initialized_on_exit_794531425885c5e5->createContext());

                   #if defined _OPENMP && _OPENMP < 200805
                           auto count = std::distance(part.begin(), part.end());
                           auto base = part.begin();
                           pfor(int index  = 0; index < count; index++) {
                               auto it = base + index;
                   #else
                           pfor(auto it = part.begin(); it < part.end(); it++) {
                   #endif
                   try{
for(const auto& env0 : *it) {
auto range = rel_path_begins_with_var_31b7afffef8c9178->lowerUpperRange_10(Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MIN_RAM_SIGNED)}},Tuple<RamDomain,2>{{ramBitCast(env0[0]), ramBitCast<RamDomain>(MAX_RAM_SIGNED)}},READ_OP_CONTEXT(rel_path_begins_with_var_31b7afffef8c9178_op_ctxt));
for(const auto& env1 : range) {
Tuple<RamDomain,2> tuple{{ramBitCast(env1[1]),ramBitCast(env0[1])}};
rel_var_maybe_partly_initialized_on_exit_794531425885c5e5->insert(tuple,READ_OP_CONTEXT(rel_var_maybe_partly_initialized_on_exit_794531425885c5e5_op_ctxt));
}
}
} catch(std::exception &e) { signalHandler->error(e.what());}
}
PARALLEL_END
}
();}
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_maybe_partly_initialized_on_exit"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (outputDirectory == "-"){directiveMap["IO"] = "stdout"; directiveMap["headers"] = "true";}
else if (!outputDirectory.empty()) {directiveMap["output-dir"] = outputDirectory;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_maybe_partly_initialized_on_exit_794531425885c5e5);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Stratum_var_used_at_baf1468fd1e6d5a4 {
public:
 Stratum_var_used_at_baf1468fd1e6d5a4(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_var_used_at_b2462f789cb6e6a2);
void run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret);
private:
SymbolTable& symTable;
RecordTable& recordTable;
ConcurrentCache<std::string,std::regex>& regexCache;
bool& pruneImdtRels;
bool& performIO;
SignalHandler*& signalHandler;
std::atomic<std::size_t>& iter;
std::atomic<RamDomain>& ctr;
std::string& inputDirectory;
std::string& outputDirectory;
t_btree_ii__0_1__11::Type* rel_var_used_at_b2462f789cb6e6a2;
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Stratum_var_used_at_baf1468fd1e6d5a4::Stratum_var_used_at_baf1468fd1e6d5a4(SymbolTable& symTable,RecordTable& recordTable,ConcurrentCache<std::string,std::regex>& regexCache,bool& pruneImdtRels,bool& performIO,SignalHandler*& signalHandler,std::atomic<std::size_t>& iter,std::atomic<RamDomain>& ctr,std::string& inputDirectory,std::string& outputDirectory,t_btree_ii__0_1__11::Type& rel_var_used_at_b2462f789cb6e6a2):
symTable(symTable),
recordTable(recordTable),
regexCache(regexCache),
pruneImdtRels(pruneImdtRels),
performIO(performIO),
signalHandler(signalHandler),
iter(iter),
ctr(ctr),
inputDirectory(inputDirectory),
outputDirectory(outputDirectory),
rel_var_used_at_b2462f789cb6e6a2(&rel_var_used_at_b2462f789cb6e6a2){
}

void Stratum_var_used_at_baf1468fd1e6d5a4::run([[maybe_unused]] const std::vector<RamDomain>& args,[[maybe_unused]] std::vector<RamDomain>& ret){
if (performIO) {
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","var_used_at"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectory.empty()) {directiveMap["fact-dir"] = inputDirectory;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_var_used_at_b2462f789cb6e6a2);
} catch (std::exception& e) {std::cerr << "Error loading var_used_at data: " << e.what() << '\n';
exit(1);
}
}
}

} // namespace  souffle

namespace  souffle {
using namespace souffle;
class Sf_test: public SouffleProgram {
public:
 Sf_test();
 ~Sf_test();
void run();
void runAll(std::string inputDirectoryArg = "",std::string outputDirectoryArg = "",bool performIOArg = true,bool pruneImdtRelsArg = true);
void printAll([[maybe_unused]] std::string outputDirectoryArg = "");
void loadAll([[maybe_unused]] std::string inputDirectoryArg = "");
void dumpInputs();
void dumpOutputs();
SymbolTable& getSymbolTable();
RecordTable& getRecordTable();
void setNumThreads(std::size_t numThreadsValue);
void executeSubroutine(std::string name,const std::vector<RamDomain>& args,std::vector<RamDomain>& ret);
private:
void runFunction(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg);
SymbolTableImpl symTable;
SpecializedRecordTable<0> recordTable;
ConcurrentCache<std::string,std::regex> regexCache;
Own<t_btree_ii__0_1__11__10::Type> rel_child_path_d9b38651563c0c3d;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_child_path_d9b38651563c0c3d;
Own<t_btree_ii__0_1__11__10::Type> rel_ancestor_path_56e99f4d0bc91e6e;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_ancestor_path_56e99f4d0bc91e6e;
Own<t_btree_ii__0_1__11::Type> rel_delta_ancestor_path_bcaf8780da306af1;
Own<t_btree_ii__0_1__11::Type> rel_new_ancestor_path_4d88f0281b6b2a54;
Own<t_btree_ii__0_1__11__10::Type> rel_cfg_edge_113c4ec5f576f8cf;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_cfg_edge_113c4ec5f576f8cf;
Own<t_btree_i__0__1::Type> rel_cfg_node_c61b5c6f77f80834;
souffle::RelationWrapper<t_btree_i__0__1::Type> wrapper_rel_cfg_node_c61b5c6f77f80834;
Own<t_btree_ii__0_1__11::Type> rel_errors_01d6f43aaabedcef;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_errors_01d6f43aaabedcef;
Own<t_btree_ii__0_1__11::Type> rel_known_placeholder_subset_d3aa7e46869bc78d;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_known_placeholder_subset_d3aa7e46869bc78d;
Own<t_btree_ii__0_1__11::Type> rel_loan_invalidated_at_3d4d106967a8a5da;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_loan_invalidated_at_3d4d106967a8a5da;
Own<t_btree_ii__0_1__11::Type> rel_loan_live_at_5761440badf0c5c7;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_loan_live_at_5761440badf0c5c7;
Own<t_btree_ii__0_1__11::Type> rel_path_assigned_at_base_baa182de12fa38e4;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_assigned_at_base_baa182de12fa38e4;
Own<t_btree_ii__0_1__11::Type> rel_path_assigned_at_110ef50c11512ca7;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_assigned_at_110ef50c11512ca7;
Own<t_btree_ii__0_1__11::Type> rel_delta_path_assigned_at_c33236013564d9a6;
Own<t_btree_ii__0_1__11::Type> rel_new_path_assigned_at_3751a63a80813906;
Own<t_btree_ii__0_1__11::Type> rel_path_moved_at_base_3824abd0ff20d508;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_moved_at_base_3824abd0ff20d508;
Own<t_btree_ii__0_1__11::Type> rel_path_moved_at_db26ad17e580a28b;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_moved_at_db26ad17e580a28b;
Own<t_btree_ii__0_1__11::Type> rel_delta_path_moved_at_8f857592b06b7171;
Own<t_btree_ii__0_1__11::Type> rel_new_path_moved_at_4019859f9b547927;
Own<t_btree_ii__0_1__11::Type> rel_path_maybe_uninitialized_on_exit_7353693d35108c78;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_maybe_uninitialized_on_exit_7353693d35108c78;
Own<t_btree_ii__0_1__11::Type> rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77;
Own<t_btree_ii__0_1__11::Type> rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782;
Own<t_btree_ii__0_1__11::Type> rel_move_error_b60914d9ec0fbce7;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_move_error_b60914d9ec0fbce7;
Own<t_btree_ii__0_1__11::Type> rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5;
Own<t_btree_ii__0_1__11::Type> rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea;
Own<t_btree_ii__0_1__11::Type> rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999;
Own<t_btree_iii__0_1_2__111::Type> rel_origin_contains_loan_on_entry_179b9d324743ed9c;
souffle::RelationWrapper<t_btree_iii__0_1_2__111::Type> wrapper_rel_origin_contains_loan_on_entry_179b9d324743ed9c;
Own<t_btree_i__0__1::Type> rel_universal_region_a10614e796c69aef;
souffle::RelationWrapper<t_btree_i__0__1::Type> wrapper_rel_universal_region_a10614e796c69aef;
Own<t_btree_ii__0_1__11::Type> rel_origin_live_on_entry_4253bcb785a6d929;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_origin_live_on_entry_4253bcb785a6d929;
Own<t_btree_ii__0_1__11::Type> rel_path_accessed_at_base_5750398152328293;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_accessed_at_base_5750398152328293;
Own<t_btree_ii__0_1__11::Type> rel_path_accessed_at_0c5de7f55ac3352d;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_accessed_at_0c5de7f55ac3352d;
Own<t_btree_ii__0_1__11::Type> rel_delta_path_accessed_at_6d6ad78564a19774;
Own<t_btree_ii__0_1__11::Type> rel_new_path_accessed_at_533da89e85764930;
Own<t_btree_ii__0_1__11::Type> rel_path_is_var_cd572d441f221f1c;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_path_is_var_cd572d441f221f1c;
Own<t_btree_ii__0_1__11__10::Type> rel_path_begins_with_var_31b7afffef8c9178;
souffle::RelationWrapper<t_btree_ii__0_1__11__10::Type> wrapper_rel_path_begins_with_var_31b7afffef8c9178;
Own<t_btree_ii__0_1__11::Type> rel_delta_path_begins_with_var_e7c1d67c31c31e22;
Own<t_btree_ii__0_1__11::Type> rel_new_path_begins_with_var_1c04b903c23b96d0;
Own<t_btree_ii__0_1__11::Type> rel_var_maybe_partly_initialized_on_exit_794531425885c5e5;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_var_maybe_partly_initialized_on_exit_794531425885c5e5;
Own<t_btree_ii__0_1__11::Type> rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d;
Own<t_btree_i__0__1::Type> rel_placeholder_origin_b6a06634cbe5f6b9;
souffle::RelationWrapper<t_btree_i__0__1::Type> wrapper_rel_placeholder_origin_b6a06634cbe5f6b9;
Own<t_btree_iii__0_1_2__111::Type> rel_subset_base_b78bf93877fd780c;
souffle::RelationWrapper<t_btree_iii__0_1_2__111::Type> wrapper_rel_subset_base_b78bf93877fd780c;
Own<t_btree_iii__0_1_2__111::Type> rel_subset_error_b2b916084fa03516;
souffle::RelationWrapper<t_btree_iii__0_1_2__111::Type> wrapper_rel_subset_error_b2b916084fa03516;
Own<t_btree_ii__0_1__11::Type> rel_var_drop_live_on_entry_8196af42d8d10787;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_var_drop_live_on_entry_8196af42d8d10787;
Own<t_btree_ii__0_1__11::Type> rel_var_used_at_b2462f789cb6e6a2;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_var_used_at_b2462f789cb6e6a2;
Own<t_btree_ii__0_1__11::Type> rel_var_live_on_entry_844bfb36987186e4;
souffle::RelationWrapper<t_btree_ii__0_1__11::Type> wrapper_rel_var_live_on_entry_844bfb36987186e4;
Stratum_ancestor_path_c0112b3bbe21adc3 stratum_ancestor_path_7d6842c0932d45a4;
Stratum_cfg_edge_6a02c2f8aa89b902 stratum_cfg_edge_6da88fdab5ff7aeb;
Stratum_cfg_node_bd032d689549d42e stratum_cfg_node_8fb6476d99764e5c;
Stratum_child_path_a638e16652f89241 stratum_child_path_e18e9caee3034b58;
Stratum_errors_3f7449d7aa9e4967 stratum_errors_7c0052b91dbc8cd3;
Stratum_known_placeholder_subset_10b7319424a0a314 stratum_known_placeholder_subset_d192c370c3ce0463;
Stratum_loan_invalidated_at_263f4508732ca10a stratum_loan_invalidated_at_bbd99fb323264f33;
Stratum_loan_live_at_78d49072431de823 stratum_loan_live_at_0beaf7c2d5ad6da8;
Stratum_move_error_71e89c25c297481d stratum_move_error_65abe743b26b282f;
Stratum_origin_contains_loan_on_entry_c07e2f79c52c4867 stratum_origin_contains_loan_on_entry_b91cec7954f6f494;
Stratum_origin_live_on_entry_755e37540122d60b stratum_origin_live_on_entry_4c43ac180f7a72f6;
Stratum_path_accessed_at_c65016814ce89194 stratum_path_accessed_at_6e9e7fa8666de132;
Stratum_path_accessed_at_base_3fb237503c193020 stratum_path_accessed_at_base_d3a452ff09842af9;
Stratum_path_assigned_at_621142eb80792795 stratum_path_assigned_at_a9bda03914f546f6;
Stratum_path_assigned_at_base_fb2821af12987abc stratum_path_assigned_at_base_371a92758c476db0;
Stratum_path_begins_with_var_2db6b88b797a3f4b stratum_path_begins_with_var_d07b7ce5ed71449c;
Stratum_path_is_var_2dc04b24a90ca307 stratum_path_is_var_74a5bd75a67a99d2;
Stratum_path_maybe_initialized_on_exit_8501b0421286a7e8 stratum_path_maybe_initialized_on_exit_f6dd1659bda7e7a6;
Stratum_path_maybe_uninitialized_on_exit_8c2a6781b6de6211 stratum_path_maybe_uninitialized_on_exit_431ab0b2a625e42e;
Stratum_path_moved_at_577bd1dfc2c609a8 stratum_path_moved_at_4313bf2761b58d3e;
Stratum_path_moved_at_base_29cf06c442229de3 stratum_path_moved_at_base_247abbd6fa9b0c20;
Stratum_placeholder_origin_288e65506ee4cf7c stratum_placeholder_origin_dcf2670f8e05fe12;
Stratum_subset_base_25e48f147a6a85b9 stratum_subset_base_9b67feff35d2a796;
Stratum_subset_error_29f91fb918a1c7d2 stratum_subset_error_425f2847addf0c89;
Stratum_universal_region_487e9539ebce1ca4 stratum_universal_region_482b38b70960b64d;
Stratum_var_drop_live_on_entry_f3cb1f8d79709ee7 stratum_var_drop_live_on_entry_089ef33a3094079c;
Stratum_var_live_on_entry_728d1d06a4a4c39a stratum_var_live_on_entry_6395fd2725759bf3;
Stratum_var_maybe_partly_initialized_on_entry_07e9d53803911812 stratum_var_maybe_partly_initialized_on_entry_7c4c860a2a614299;
Stratum_var_maybe_partly_initialized_on_exit_8a8a0186389a8f24 stratum_var_maybe_partly_initialized_on_exit_39fe0c629ff9c78e;
Stratum_var_used_at_baf1468fd1e6d5a4 stratum_var_used_at_9304780eefd9d03a;
std::string inputDirectory;
std::string outputDirectory;
SignalHandler* signalHandler{SignalHandler::instance()};
std::atomic<RamDomain> ctr{};
std::atomic<std::size_t> iter{};
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
 Sf_test::Sf_test():
symTable(),
recordTable(),
regexCache(),
rel_child_path_d9b38651563c0c3d(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_child_path_d9b38651563c0c3d(0, *rel_child_path_d9b38651563c0c3d, *this, "child_path", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_ancestor_path_56e99f4d0bc91e6e(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_ancestor_path_56e99f4d0bc91e6e(1, *rel_ancestor_path_56e99f4d0bc91e6e, *this, "ancestor_path", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_ancestor_path_bcaf8780da306af1(mk<t_btree_ii__0_1__11::Type>()),
rel_new_ancestor_path_4d88f0281b6b2a54(mk<t_btree_ii__0_1__11::Type>()),
rel_cfg_edge_113c4ec5f576f8cf(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_cfg_edge_113c4ec5f576f8cf(2, *rel_cfg_edge_113c4ec5f576f8cf, *this, "cfg_edge", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_cfg_node_c61b5c6f77f80834(mk<t_btree_i__0__1::Type>()),
wrapper_rel_cfg_node_c61b5c6f77f80834(3, *rel_cfg_node_c61b5c6f77f80834, *this, "cfg_node", std::array<const char *,1>{{"i:number"}}, std::array<const char *,1>{{"x"}}, 0),
rel_errors_01d6f43aaabedcef(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_errors_01d6f43aaabedcef(4, *rel_errors_01d6f43aaabedcef, *this, "errors", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_known_placeholder_subset_d3aa7e46869bc78d(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_known_placeholder_subset_d3aa7e46869bc78d(5, *rel_known_placeholder_subset_d3aa7e46869bc78d, *this, "known_placeholder_subset", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_loan_invalidated_at_3d4d106967a8a5da(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_loan_invalidated_at_3d4d106967a8a5da(6, *rel_loan_invalidated_at_3d4d106967a8a5da, *this, "loan_invalidated_at", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_loan_live_at_5761440badf0c5c7(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_loan_live_at_5761440badf0c5c7(7, *rel_loan_live_at_5761440badf0c5c7, *this, "loan_live_at", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_assigned_at_base_baa182de12fa38e4(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_assigned_at_base_baa182de12fa38e4(8, *rel_path_assigned_at_base_baa182de12fa38e4, *this, "path_assigned_at_base", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_assigned_at_110ef50c11512ca7(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_assigned_at_110ef50c11512ca7(9, *rel_path_assigned_at_110ef50c11512ca7, *this, "path_assigned_at", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_path_assigned_at_c33236013564d9a6(mk<t_btree_ii__0_1__11::Type>()),
rel_new_path_assigned_at_3751a63a80813906(mk<t_btree_ii__0_1__11::Type>()),
rel_path_moved_at_base_3824abd0ff20d508(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_moved_at_base_3824abd0ff20d508(10, *rel_path_moved_at_base_3824abd0ff20d508, *this, "path_moved_at_base", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_moved_at_db26ad17e580a28b(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_moved_at_db26ad17e580a28b(11, *rel_path_moved_at_db26ad17e580a28b, *this, "path_moved_at", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_path_moved_at_8f857592b06b7171(mk<t_btree_ii__0_1__11::Type>()),
rel_new_path_moved_at_4019859f9b547927(mk<t_btree_ii__0_1__11::Type>()),
rel_path_maybe_uninitialized_on_exit_7353693d35108c78(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_maybe_uninitialized_on_exit_7353693d35108c78(12, *rel_path_maybe_uninitialized_on_exit_7353693d35108c78, *this, "path_maybe_uninitialized_on_exit", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77(mk<t_btree_ii__0_1__11::Type>()),
rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782(mk<t_btree_ii__0_1__11::Type>()),
rel_move_error_b60914d9ec0fbce7(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_move_error_b60914d9ec0fbce7(13, *rel_move_error_b60914d9ec0fbce7, *this, "move_error", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5(14, *rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5, *this, "path_maybe_initialized_on_exit", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea(mk<t_btree_ii__0_1__11::Type>()),
rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999(mk<t_btree_ii__0_1__11::Type>()),
rel_origin_contains_loan_on_entry_179b9d324743ed9c(mk<t_btree_iii__0_1_2__111::Type>()),
wrapper_rel_origin_contains_loan_on_entry_179b9d324743ed9c(15, *rel_origin_contains_loan_on_entry_179b9d324743ed9c, *this, "origin_contains_loan_on_entry", std::array<const char *,3>{{"i:number","i:number","i:number"}}, std::array<const char *,3>{{"x","y","z"}}, 0),
rel_universal_region_a10614e796c69aef(mk<t_btree_i__0__1::Type>()),
wrapper_rel_universal_region_a10614e796c69aef(16, *rel_universal_region_a10614e796c69aef, *this, "universal_region", std::array<const char *,1>{{"i:number"}}, std::array<const char *,1>{{"x"}}, 0),
rel_origin_live_on_entry_4253bcb785a6d929(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_origin_live_on_entry_4253bcb785a6d929(17, *rel_origin_live_on_entry_4253bcb785a6d929, *this, "origin_live_on_entry", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_accessed_at_base_5750398152328293(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_accessed_at_base_5750398152328293(18, *rel_path_accessed_at_base_5750398152328293, *this, "path_accessed_at_base", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_accessed_at_0c5de7f55ac3352d(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_accessed_at_0c5de7f55ac3352d(19, *rel_path_accessed_at_0c5de7f55ac3352d, *this, "path_accessed_at", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_path_accessed_at_6d6ad78564a19774(mk<t_btree_ii__0_1__11::Type>()),
rel_new_path_accessed_at_533da89e85764930(mk<t_btree_ii__0_1__11::Type>()),
rel_path_is_var_cd572d441f221f1c(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_path_is_var_cd572d441f221f1c(20, *rel_path_is_var_cd572d441f221f1c, *this, "path_is_var", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_path_begins_with_var_31b7afffef8c9178(mk<t_btree_ii__0_1__11__10::Type>()),
wrapper_rel_path_begins_with_var_31b7afffef8c9178(21, *rel_path_begins_with_var_31b7afffef8c9178, *this, "path_begins_with_var", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_delta_path_begins_with_var_e7c1d67c31c31e22(mk<t_btree_ii__0_1__11::Type>()),
rel_new_path_begins_with_var_1c04b903c23b96d0(mk<t_btree_ii__0_1__11::Type>()),
rel_var_maybe_partly_initialized_on_exit_794531425885c5e5(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_var_maybe_partly_initialized_on_exit_794531425885c5e5(22, *rel_var_maybe_partly_initialized_on_exit_794531425885c5e5, *this, "var_maybe_partly_initialized_on_exit", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d(23, *rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d, *this, "var_maybe_partly_initialized_on_entry", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_placeholder_origin_b6a06634cbe5f6b9(mk<t_btree_i__0__1::Type>()),
wrapper_rel_placeholder_origin_b6a06634cbe5f6b9(24, *rel_placeholder_origin_b6a06634cbe5f6b9, *this, "placeholder_origin", std::array<const char *,1>{{"i:number"}}, std::array<const char *,1>{{"x"}}, 0),
rel_subset_base_b78bf93877fd780c(mk<t_btree_iii__0_1_2__111::Type>()),
wrapper_rel_subset_base_b78bf93877fd780c(25, *rel_subset_base_b78bf93877fd780c, *this, "subset_base", std::array<const char *,3>{{"i:number","i:number","i:number"}}, std::array<const char *,3>{{"x","y","z"}}, 0),
rel_subset_error_b2b916084fa03516(mk<t_btree_iii__0_1_2__111::Type>()),
wrapper_rel_subset_error_b2b916084fa03516(26, *rel_subset_error_b2b916084fa03516, *this, "subset_error", std::array<const char *,3>{{"i:number","i:number","i:number"}}, std::array<const char *,3>{{"x","y","z"}}, 0),
rel_var_drop_live_on_entry_8196af42d8d10787(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_var_drop_live_on_entry_8196af42d8d10787(27, *rel_var_drop_live_on_entry_8196af42d8d10787, *this, "var_drop_live_on_entry", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_var_used_at_b2462f789cb6e6a2(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_var_used_at_b2462f789cb6e6a2(28, *rel_var_used_at_b2462f789cb6e6a2, *this, "var_used_at", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
rel_var_live_on_entry_844bfb36987186e4(mk<t_btree_ii__0_1__11::Type>()),
wrapper_rel_var_live_on_entry_844bfb36987186e4(29, *rel_var_live_on_entry_844bfb36987186e4, *this, "var_live_on_entry", std::array<const char *,2>{{"i:number","i:number"}}, std::array<const char *,2>{{"x","y"}}, 0),
stratum_ancestor_path_7d6842c0932d45a4(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_ancestor_path_bcaf8780da306af1,*rel_new_ancestor_path_4d88f0281b6b2a54,*rel_ancestor_path_56e99f4d0bc91e6e,*rel_child_path_d9b38651563c0c3d),
stratum_cfg_edge_6da88fdab5ff7aeb(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_cfg_edge_113c4ec5f576f8cf),
stratum_cfg_node_8fb6476d99764e5c(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_cfg_edge_113c4ec5f576f8cf,*rel_cfg_node_c61b5c6f77f80834),
stratum_child_path_e18e9caee3034b58(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_child_path_d9b38651563c0c3d),
stratum_errors_7c0052b91dbc8cd3(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_errors_01d6f43aaabedcef),
stratum_known_placeholder_subset_d192c370c3ce0463(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_known_placeholder_subset_d3aa7e46869bc78d),
stratum_loan_invalidated_at_bbd99fb323264f33(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_loan_invalidated_at_3d4d106967a8a5da),
stratum_loan_live_at_0beaf7c2d5ad6da8(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_loan_live_at_5761440badf0c5c7),
stratum_move_error_65abe743b26b282f(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_cfg_edge_113c4ec5f576f8cf,*rel_move_error_b60914d9ec0fbce7,*rel_path_maybe_uninitialized_on_exit_7353693d35108c78),
stratum_origin_contains_loan_on_entry_b91cec7954f6f494(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_origin_contains_loan_on_entry_179b9d324743ed9c),
stratum_origin_live_on_entry_4c43ac180f7a72f6(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_cfg_node_c61b5c6f77f80834,*rel_origin_live_on_entry_4253bcb785a6d929,*rel_universal_region_a10614e796c69aef),
stratum_path_accessed_at_6e9e7fa8666de132(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_path_accessed_at_6d6ad78564a19774,*rel_new_path_accessed_at_533da89e85764930,*rel_ancestor_path_56e99f4d0bc91e6e,*rel_path_accessed_at_0c5de7f55ac3352d,*rel_path_accessed_at_base_5750398152328293),
stratum_path_accessed_at_base_d3a452ff09842af9(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_path_accessed_at_base_5750398152328293),
stratum_path_assigned_at_a9bda03914f546f6(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_path_assigned_at_c33236013564d9a6,*rel_new_path_assigned_at_3751a63a80813906,*rel_ancestor_path_56e99f4d0bc91e6e,*rel_path_assigned_at_110ef50c11512ca7,*rel_path_assigned_at_base_baa182de12fa38e4),
stratum_path_assigned_at_base_371a92758c476db0(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_path_assigned_at_base_baa182de12fa38e4),
stratum_path_begins_with_var_d07b7ce5ed71449c(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_path_begins_with_var_e7c1d67c31c31e22,*rel_new_path_begins_with_var_1c04b903c23b96d0,*rel_ancestor_path_56e99f4d0bc91e6e,*rel_path_begins_with_var_31b7afffef8c9178,*rel_path_is_var_cd572d441f221f1c),
stratum_path_is_var_74a5bd75a67a99d2(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_path_is_var_cd572d441f221f1c),
stratum_path_maybe_initialized_on_exit_f6dd1659bda7e7a6(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_path_maybe_initialized_on_exit_fe14363d8b0f66ea,*rel_new_path_maybe_initialized_on_exit_4376a5b9ad489999,*rel_cfg_edge_113c4ec5f576f8cf,*rel_path_assigned_at_110ef50c11512ca7,*rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5,*rel_path_moved_at_db26ad17e580a28b),
stratum_path_maybe_uninitialized_on_exit_431ab0b2a625e42e(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_path_maybe_uninitialized_on_exit_0347d6e03f972d77,*rel_new_path_maybe_uninitialized_on_exit_ac0c89a281ff0782,*rel_cfg_edge_113c4ec5f576f8cf,*rel_path_assigned_at_110ef50c11512ca7,*rel_path_maybe_uninitialized_on_exit_7353693d35108c78,*rel_path_moved_at_db26ad17e580a28b),
stratum_path_moved_at_4313bf2761b58d3e(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_delta_path_moved_at_8f857592b06b7171,*rel_new_path_moved_at_4019859f9b547927,*rel_ancestor_path_56e99f4d0bc91e6e,*rel_path_moved_at_db26ad17e580a28b,*rel_path_moved_at_base_3824abd0ff20d508),
stratum_path_moved_at_base_247abbd6fa9b0c20(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_path_moved_at_base_3824abd0ff20d508),
stratum_placeholder_origin_dcf2670f8e05fe12(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_placeholder_origin_b6a06634cbe5f6b9),
stratum_subset_base_9b67feff35d2a796(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_subset_base_b78bf93877fd780c),
stratum_subset_error_425f2847addf0c89(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_subset_error_b2b916084fa03516),
stratum_universal_region_482b38b70960b64d(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_universal_region_a10614e796c69aef),
stratum_var_drop_live_on_entry_089ef33a3094079c(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_var_drop_live_on_entry_8196af42d8d10787),
stratum_var_live_on_entry_6395fd2725759bf3(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_var_live_on_entry_844bfb36987186e4,*rel_var_used_at_b2462f789cb6e6a2),
stratum_var_maybe_partly_initialized_on_entry_7c4c860a2a614299(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_cfg_edge_113c4ec5f576f8cf,*rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d,*rel_var_maybe_partly_initialized_on_exit_794531425885c5e5),
stratum_var_maybe_partly_initialized_on_exit_39fe0c629ff9c78e(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_path_begins_with_var_31b7afffef8c9178,*rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5,*rel_var_maybe_partly_initialized_on_exit_794531425885c5e5),
stratum_var_used_at_9304780eefd9d03a(symTable,recordTable,regexCache,pruneImdtRels,performIO,signalHandler,iter,ctr,inputDirectory,outputDirectory,*rel_var_used_at_b2462f789cb6e6a2){
addRelation("child_path", wrapper_rel_child_path_d9b38651563c0c3d, true, false);
addRelation("ancestor_path", wrapper_rel_ancestor_path_56e99f4d0bc91e6e, false, true);
addRelation("cfg_edge", wrapper_rel_cfg_edge_113c4ec5f576f8cf, true, false);
addRelation("cfg_node", wrapper_rel_cfg_node_c61b5c6f77f80834, false, true);
addRelation("errors", wrapper_rel_errors_01d6f43aaabedcef, false, true);
addRelation("known_placeholder_subset", wrapper_rel_known_placeholder_subset_d3aa7e46869bc78d, true, true);
addRelation("loan_invalidated_at", wrapper_rel_loan_invalidated_at_3d4d106967a8a5da, true, true);
addRelation("loan_live_at", wrapper_rel_loan_live_at_5761440badf0c5c7, false, true);
addRelation("path_assigned_at_base", wrapper_rel_path_assigned_at_base_baa182de12fa38e4, true, false);
addRelation("path_assigned_at", wrapper_rel_path_assigned_at_110ef50c11512ca7, false, true);
addRelation("path_moved_at_base", wrapper_rel_path_moved_at_base_3824abd0ff20d508, true, false);
addRelation("path_moved_at", wrapper_rel_path_moved_at_db26ad17e580a28b, false, true);
addRelation("path_maybe_uninitialized_on_exit", wrapper_rel_path_maybe_uninitialized_on_exit_7353693d35108c78, false, true);
addRelation("move_error", wrapper_rel_move_error_b60914d9ec0fbce7, false, true);
addRelation("path_maybe_initialized_on_exit", wrapper_rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5, false, true);
addRelation("origin_contains_loan_on_entry", wrapper_rel_origin_contains_loan_on_entry_179b9d324743ed9c, false, true);
addRelation("universal_region", wrapper_rel_universal_region_a10614e796c69aef, true, false);
addRelation("origin_live_on_entry", wrapper_rel_origin_live_on_entry_4253bcb785a6d929, false, true);
addRelation("path_accessed_at_base", wrapper_rel_path_accessed_at_base_5750398152328293, true, false);
addRelation("path_accessed_at", wrapper_rel_path_accessed_at_0c5de7f55ac3352d, false, true);
addRelation("path_is_var", wrapper_rel_path_is_var_cd572d441f221f1c, true, false);
addRelation("path_begins_with_var", wrapper_rel_path_begins_with_var_31b7afffef8c9178, false, true);
addRelation("var_maybe_partly_initialized_on_exit", wrapper_rel_var_maybe_partly_initialized_on_exit_794531425885c5e5, false, true);
addRelation("var_maybe_partly_initialized_on_entry", wrapper_rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d, false, true);
addRelation("placeholder_origin", wrapper_rel_placeholder_origin_b6a06634cbe5f6b9, false, true);
addRelation("subset_base", wrapper_rel_subset_base_b78bf93877fd780c, true, true);
addRelation("subset_error", wrapper_rel_subset_error_b2b916084fa03516, false, true);
addRelation("var_drop_live_on_entry", wrapper_rel_var_drop_live_on_entry_8196af42d8d10787, false, true);
addRelation("var_used_at", wrapper_rel_var_used_at_b2462f789cb6e6a2, true, false);
addRelation("var_live_on_entry", wrapper_rel_var_live_on_entry_844bfb36987186e4, false, true);
}

 Sf_test::~Sf_test(){
}

void Sf_test::runFunction(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg){

    this->inputDirectory  = std::move(inputDirectoryArg);
    this->outputDirectory = std::move(outputDirectoryArg);
    this->performIO       = performIOArg;
    this->pruneImdtRels   = pruneImdtRelsArg;

    // set default threads (in embedded mode)
    // if this is not set, and omp is used, the default omp setting of number of cores is used.
#if defined(_OPENMP)
    if (0 < getNumThreads()) { omp_set_num_threads(static_cast<int>(getNumThreads())); }
#endif

    signalHandler->set();
// -- query evaluation --
{
 std::vector<RamDomain> args, ret;
stratum_child_path_e18e9caee3034b58.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_ancestor_path_7d6842c0932d45a4.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_cfg_edge_6da88fdab5ff7aeb.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_cfg_node_8fb6476d99764e5c.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_errors_7c0052b91dbc8cd3.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_known_placeholder_subset_d192c370c3ce0463.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_loan_invalidated_at_bbd99fb323264f33.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_loan_live_at_0beaf7c2d5ad6da8.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_assigned_at_base_371a92758c476db0.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_assigned_at_a9bda03914f546f6.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_moved_at_base_247abbd6fa9b0c20.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_moved_at_4313bf2761b58d3e.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_maybe_uninitialized_on_exit_431ab0b2a625e42e.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_move_error_65abe743b26b282f.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_maybe_initialized_on_exit_f6dd1659bda7e7a6.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_origin_contains_loan_on_entry_b91cec7954f6f494.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_universal_region_482b38b70960b64d.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_origin_live_on_entry_4c43ac180f7a72f6.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_accessed_at_base_d3a452ff09842af9.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_accessed_at_6e9e7fa8666de132.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_is_var_74a5bd75a67a99d2.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_path_begins_with_var_d07b7ce5ed71449c.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_var_maybe_partly_initialized_on_exit_39fe0c629ff9c78e.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_var_maybe_partly_initialized_on_entry_7c4c860a2a614299.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_placeholder_origin_dcf2670f8e05fe12.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_subset_base_9b67feff35d2a796.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_subset_error_425f2847addf0c89.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_var_drop_live_on_entry_089ef33a3094079c.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_var_used_at_9304780eefd9d03a.run(args, ret);
}
{
 std::vector<RamDomain> args, ret;
stratum_var_live_on_entry_6395fd2725759bf3.run(args, ret);
}

// -- relation hint statistics --
signalHandler->reset();
}

void Sf_test::run(){
runFunction("", "", false, false);
}

void Sf_test::runAll(std::string inputDirectoryArg,std::string outputDirectoryArg,bool performIOArg,bool pruneImdtRelsArg){
runFunction(inputDirectoryArg, outputDirectoryArg, performIOArg, pruneImdtRelsArg);
}

void Sf_test::printAll([[maybe_unused]] std::string outputDirectoryArg){
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","ancestor_path"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_ancestor_path_56e99f4d0bc91e6e);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","errors"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_errors_01d6f43aaabedcef);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x"},{"auxArity","0"},{"name","cfg_node"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 1, \"params\": [\"x\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 1, \"types\": [\"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_cfg_node_c61b5c6f77f80834);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","known_placeholder_subset"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_known_placeholder_subset_d3aa7e46869bc78d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","loan_invalidated_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_loan_invalidated_at_3d4d106967a8a5da);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","loan_live_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_loan_live_at_5761440badf0c5c7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_assigned_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_assigned_at_110ef50c11512ca7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_moved_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_moved_at_db26ad17e580a28b);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_maybe_uninitialized_on_exit"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_maybe_uninitialized_on_exit_7353693d35108c78);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","move_error"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_move_error_b60914d9ec0fbce7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_maybe_initialized_on_exit"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"name","origin_contains_loan_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_origin_contains_loan_on_entry_179b9d324743ed9c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","origin_live_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_origin_live_on_entry_4253bcb785a6d929);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_accessed_at"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_accessed_at_0c5de7f55ac3352d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","path_begins_with_var"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_path_begins_with_var_31b7afffef8c9178);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_maybe_partly_initialized_on_exit"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_maybe_partly_initialized_on_exit_794531425885c5e5);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_maybe_partly_initialized_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x"},{"auxArity","0"},{"name","placeholder_origin"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 1, \"params\": [\"x\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 1, \"types\": [\"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_placeholder_origin_b6a06634cbe5f6b9);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"name","subset_base"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_subset_base_b78bf93877fd780c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"name","subset_error"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_subset_error_b2b916084fa03516);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_drop_live_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_drop_live_on_entry_8196af42d8d10787);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> directiveMap({{"IO","stdoutprintsize"},{"attributeNames","x\ty"},{"auxArity","0"},{"name","var_live_on_entry"},{"operation","printsize"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!outputDirectoryArg.empty()) {directiveMap["output-dir"] = outputDirectoryArg;}
IOSystem::getInstance().getWriter(directiveMap, symTable, recordTable)->writeAll(*rel_var_live_on_entry_844bfb36987186e4);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

void Sf_test::loadAll([[maybe_unused]] std::string inputDirectoryArg){
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","child_path"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_child_path_d9b38651563c0c3d);
} catch (std::exception& e) {std::cerr << "Error loading child_path data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","cfg_edge"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_cfg_edge_113c4ec5f576f8cf);
} catch (std::exception& e) {std::cerr << "Error loading cfg_edge data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","known_placeholder_subset"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_known_placeholder_subset_d3aa7e46869bc78d);
} catch (std::exception& e) {std::cerr << "Error loading known_placeholder_subset data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","loan_invalidated_at"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_loan_invalidated_at_3d4d106967a8a5da);
} catch (std::exception& e) {std::cerr << "Error loading loan_invalidated_at data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_assigned_at_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_assigned_at_base_baa182de12fa38e4);
} catch (std::exception& e) {std::cerr << "Error loading path_assigned_at_base data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_moved_at_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_moved_at_base_3824abd0ff20d508);
} catch (std::exception& e) {std::cerr << "Error loading path_moved_at_base data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","universal_region"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 1, \"params\": [\"x\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 1, \"types\": [\"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_universal_region_a10614e796c69aef);
} catch (std::exception& e) {std::cerr << "Error loading universal_region data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_accessed_at_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_accessed_at_base_5750398152328293);
} catch (std::exception& e) {std::cerr << "Error loading path_accessed_at_base data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","path_is_var"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_path_is_var_cd572d441f221f1c);
} catch (std::exception& e) {std::cerr << "Error loading path_is_var data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty\tz"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","subset_base"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 3, \"params\": [\"x\", \"y\", \"z\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 3, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_subset_base_b78bf93877fd780c);
} catch (std::exception& e) {std::cerr << "Error loading subset_base data: " << e.what() << '\n';
exit(1);
}
try {std::map<std::string, std::string> directiveMap({{"IO","file"},{"attributeNames","x\ty"},{"auxArity","0"},{"fact-dir","../../data/polonius/clap-rs/"},{"name","var_used_at"},{"operation","input"},{"params","{\"records\": {}, \"relation\": {\"arity\": 2, \"params\": [\"x\", \"y\"]}}"},{"types","{\"ADTs\": {}, \"records\": {}, \"relation\": {\"arity\": 2, \"types\": [\"i:number\", \"i:number\"]}}"}});
if (!inputDirectoryArg.empty()) {directiveMap["fact-dir"] = inputDirectoryArg;}
IOSystem::getInstance().getReader(directiveMap, symTable, recordTable)->readAll(*rel_var_used_at_b2462f789cb6e6a2);
} catch (std::exception& e) {std::cerr << "Error loading var_used_at data: " << e.what() << '\n';
exit(1);
}
}

void Sf_test::dumpInputs(){
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "child_path";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_child_path_d9b38651563c0c3d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "cfg_edge";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_cfg_edge_113c4ec5f576f8cf);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "known_placeholder_subset";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_known_placeholder_subset_d3aa7e46869bc78d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "loan_invalidated_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_loan_invalidated_at_3d4d106967a8a5da);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_assigned_at_base";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_assigned_at_base_baa182de12fa38e4);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_moved_at_base";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_moved_at_base_3824abd0ff20d508);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "universal_region";
rwOperation["types"] = "{\"relation\": {\"arity\": 1, \"auxArity\": 0, \"types\": [\"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_universal_region_a10614e796c69aef);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_accessed_at_base";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_accessed_at_base_5750398152328293);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_is_var";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_is_var_cd572d441f221f1c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "subset_base";
rwOperation["types"] = "{\"relation\": {\"arity\": 3, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_subset_base_b78bf93877fd780c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "var_used_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_var_used_at_b2462f789cb6e6a2);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

void Sf_test::dumpOutputs(){
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "ancestor_path";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_ancestor_path_56e99f4d0bc91e6e);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "errors";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_errors_01d6f43aaabedcef);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "cfg_node";
rwOperation["types"] = "{\"relation\": {\"arity\": 1, \"auxArity\": 0, \"types\": [\"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_cfg_node_c61b5c6f77f80834);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "known_placeholder_subset";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_known_placeholder_subset_d3aa7e46869bc78d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "loan_invalidated_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_loan_invalidated_at_3d4d106967a8a5da);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "loan_live_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_loan_live_at_5761440badf0c5c7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_assigned_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_assigned_at_110ef50c11512ca7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_moved_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_moved_at_db26ad17e580a28b);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_maybe_uninitialized_on_exit";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_maybe_uninitialized_on_exit_7353693d35108c78);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "move_error";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_move_error_b60914d9ec0fbce7);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_maybe_initialized_on_exit";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_maybe_initialized_on_exit_3fb1ed4ef11ac3e5);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "origin_contains_loan_on_entry";
rwOperation["types"] = "{\"relation\": {\"arity\": 3, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_origin_contains_loan_on_entry_179b9d324743ed9c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "origin_live_on_entry";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_origin_live_on_entry_4253bcb785a6d929);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_accessed_at";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_accessed_at_0c5de7f55ac3352d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "path_begins_with_var";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_path_begins_with_var_31b7afffef8c9178);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "var_maybe_partly_initialized_on_exit";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_var_maybe_partly_initialized_on_exit_794531425885c5e5);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "var_maybe_partly_initialized_on_entry";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_var_maybe_partly_initialized_on_entry_760f6126ecaa1a1d);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "placeholder_origin";
rwOperation["types"] = "{\"relation\": {\"arity\": 1, \"auxArity\": 0, \"types\": [\"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_placeholder_origin_b6a06634cbe5f6b9);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "subset_base";
rwOperation["types"] = "{\"relation\": {\"arity\": 3, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_subset_base_b78bf93877fd780c);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "subset_error";
rwOperation["types"] = "{\"relation\": {\"arity\": 3, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_subset_error_b2b916084fa03516);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "var_drop_live_on_entry";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_var_drop_live_on_entry_8196af42d8d10787);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
try {std::map<std::string, std::string> rwOperation;
rwOperation["IO"] = "stdout";
rwOperation["name"] = "var_live_on_entry";
rwOperation["types"] = "{\"relation\": {\"arity\": 2, \"auxArity\": 0, \"types\": [\"i:number\", \"i:number\"]}}";
IOSystem::getInstance().getWriter(rwOperation, symTable, recordTable)->writeAll(*rel_var_live_on_entry_844bfb36987186e4);
} catch (std::exception& e) {std::cerr << e.what();exit(1);}
}

SymbolTable& Sf_test::getSymbolTable(){
return symTable;
}

RecordTable& Sf_test::getRecordTable(){
return recordTable;
}

void Sf_test::setNumThreads(std::size_t numThreadsValue){
SouffleProgram::setNumThreads(numThreadsValue);
symTable.setNumLanes(getNumThreads());
recordTable.setNumLanes(getNumThreads());
regexCache.setNumLanes(getNumThreads());
}

void Sf_test::executeSubroutine(std::string name,const std::vector<RamDomain>& args,std::vector<RamDomain>& ret){
if (name == "ancestor_path") {
stratum_ancestor_path_7d6842c0932d45a4.run(args, ret);
return;}
if (name == "cfg_edge") {
stratum_cfg_edge_6da88fdab5ff7aeb.run(args, ret);
return;}
if (name == "cfg_node") {
stratum_cfg_node_8fb6476d99764e5c.run(args, ret);
return;}
if (name == "child_path") {
stratum_child_path_e18e9caee3034b58.run(args, ret);
return;}
if (name == "errors") {
stratum_errors_7c0052b91dbc8cd3.run(args, ret);
return;}
if (name == "known_placeholder_subset") {
stratum_known_placeholder_subset_d192c370c3ce0463.run(args, ret);
return;}
if (name == "loan_invalidated_at") {
stratum_loan_invalidated_at_bbd99fb323264f33.run(args, ret);
return;}
if (name == "loan_live_at") {
stratum_loan_live_at_0beaf7c2d5ad6da8.run(args, ret);
return;}
if (name == "move_error") {
stratum_move_error_65abe743b26b282f.run(args, ret);
return;}
if (name == "origin_contains_loan_on_entry") {
stratum_origin_contains_loan_on_entry_b91cec7954f6f494.run(args, ret);
return;}
if (name == "origin_live_on_entry") {
stratum_origin_live_on_entry_4c43ac180f7a72f6.run(args, ret);
return;}
if (name == "path_accessed_at") {
stratum_path_accessed_at_6e9e7fa8666de132.run(args, ret);
return;}
if (name == "path_accessed_at_base") {
stratum_path_accessed_at_base_d3a452ff09842af9.run(args, ret);
return;}
if (name == "path_assigned_at") {
stratum_path_assigned_at_a9bda03914f546f6.run(args, ret);
return;}
if (name == "path_assigned_at_base") {
stratum_path_assigned_at_base_371a92758c476db0.run(args, ret);
return;}
if (name == "path_begins_with_var") {
stratum_path_begins_with_var_d07b7ce5ed71449c.run(args, ret);
return;}
if (name == "path_is_var") {
stratum_path_is_var_74a5bd75a67a99d2.run(args, ret);
return;}
if (name == "path_maybe_initialized_on_exit") {
stratum_path_maybe_initialized_on_exit_f6dd1659bda7e7a6.run(args, ret);
return;}
if (name == "path_maybe_uninitialized_on_exit") {
stratum_path_maybe_uninitialized_on_exit_431ab0b2a625e42e.run(args, ret);
return;}
if (name == "path_moved_at") {
stratum_path_moved_at_4313bf2761b58d3e.run(args, ret);
return;}
if (name == "path_moved_at_base") {
stratum_path_moved_at_base_247abbd6fa9b0c20.run(args, ret);
return;}
if (name == "placeholder_origin") {
stratum_placeholder_origin_dcf2670f8e05fe12.run(args, ret);
return;}
if (name == "subset_base") {
stratum_subset_base_9b67feff35d2a796.run(args, ret);
return;}
if (name == "subset_error") {
stratum_subset_error_425f2847addf0c89.run(args, ret);
return;}
if (name == "universal_region") {
stratum_universal_region_482b38b70960b64d.run(args, ret);
return;}
if (name == "var_drop_live_on_entry") {
stratum_var_drop_live_on_entry_089ef33a3094079c.run(args, ret);
return;}
if (name == "var_live_on_entry") {
stratum_var_live_on_entry_6395fd2725759bf3.run(args, ret);
return;}
if (name == "var_maybe_partly_initialized_on_entry") {
stratum_var_maybe_partly_initialized_on_entry_7c4c860a2a614299.run(args, ret);
return;}
if (name == "var_maybe_partly_initialized_on_exit") {
stratum_var_maybe_partly_initialized_on_exit_39fe0c629ff9c78e.run(args, ret);
return;}
if (name == "var_used_at") {
stratum_var_used_at_9304780eefd9d03a.run(args, ret);
return;}
fatal(("unknown subroutine " + name).c_str());
}

} // namespace  souffle
namespace souffle {
SouffleProgram *newInstance_test(){return new  souffle::Sf_test;}
SymbolTable *getST_test(SouffleProgram *p){return &reinterpret_cast<souffle::Sf_test*>(p)->getSymbolTable();}
} // namespace souffle

#ifndef __EMBEDDED_SOUFFLE__
#include "souffle/CompiledOptions.h"
int main(int argc, char** argv)
{
try{
souffle::CmdOptions opt(R"(../../test/souffle/polonius.dl)",
R"()",
R"()",
false,
R"()",
12);
if (!opt.parse(argc,argv)) return 1;
souffle::Sf_test obj;
#if defined(_OPENMP) 
obj.setNumThreads(opt.getNumJobs());

#endif
obj.runAll(opt.getInputFileDir(), opt.getOutputFileDir());
return 0;
} catch(std::exception &e) { souffle::SignalHandler::instance()->error(e.what());}
}
#endif

namespace  souffle {
using namespace souffle;
class factory_Sf_test: souffle::ProgramFactory {
public:
souffle::SouffleProgram* newInstance();
 factory_Sf_test();
private:
};
} // namespace  souffle
namespace  souffle {
using namespace souffle;
souffle::SouffleProgram* factory_Sf_test::newInstance(){
return new  souffle::Sf_test();
}

 factory_Sf_test::factory_Sf_test():
souffle::ProgramFactory("test"){
}

} // namespace  souffle
namespace souffle {

#ifdef __EMBEDDED_SOUFFLE__
extern "C" {
souffle::factory_Sf_test __factory_Sf_test_instance;
}
#endif
} // namespace souffle

