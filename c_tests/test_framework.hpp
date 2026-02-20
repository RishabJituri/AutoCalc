// tests/test_framework.hpp
#pragma once
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

namespace tfw {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

#define CONCAT_INNER(a,b) a##b
#define CONCAT(a,b) CONCAT_INNER(a,b)

#define TEST(name) \
    static void CONCAT(test_fn_,__LINE__)(); \
    static ::tfw::Registrar CONCAT(test_reg_,__LINE__)(name, CONCAT(test_fn_,__LINE__)); \
    static void CONCAT(test_fn_,__LINE__)()

struct Failure : public std::exception {
    std::string msg;
    explicit Failure(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

inline std::string loc(const char* file, int line) {
    std::ostringstream oss;
    oss << file << ":" << line;
    return oss.str();
}

inline void assert_true(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::ostringstream oss;
        oss << "[ASSERT_TRUE FAILED] " << expr << " at " << loc(file, line);
        throw Failure(oss.str());
    }
}
#define ASSERT_TRUE(x) ::tfw::assert_true((x), #x, __FILE__, __LINE__)

inline void assert_near(float a, float b, float eps, const char* exprA, const char* exprB, float e, const char* file, int line) {
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b) || std::fabs(a - b) > eps) {
        std::ostringstream oss;
        oss << "[ASSERT_NEAR FAILED] |" << exprA << " - " << exprB << "| = " << std::fabs(a-b)
            << " > " << e << " at " << loc(file, line) << "\n  " << exprA << " = " << a << "\n  " << exprB << " = " << b;
        throw Failure(oss.str());
    }
}
#define ASSERT_NEAR(a,b,eps) ::tfw::assert_near((float)(a),(float)(b),(float)(eps), #a, #b, (float)(eps), __FILE__, __LINE__)

} // namespace tfw


// ===== tfw extensions: richer assertions & non-fatal EXPECTs =====
#ifndef TFW_EXTENSIONS_ADDED
#define TFW_EXTENSIONS_ADDED
#include <type_traits>
#include <exception>

namespace tfw {

// ---- scoped trace (optional context in failure messages) ----
struct TraceStack {
    std::vector<std::string> items;
};
inline thread_local TraceStack _trace_stack;

struct ScopedTrace {
    explicit ScopedTrace(std::string s) { _trace_stack.items.emplace_back(std::move(s)); }
    ~ScopedTrace() { if (!_trace_stack.items.empty()) _trace_stack.items.pop_back(); }
};
inline std::string _trace_suffix() {
    if (_trace_stack.items.empty()) return {};
    std::ostringstream oss;
    oss << "\n  Trace:";
    for (const auto& s : _trace_stack.items) oss << "\n    - " << s;
    return oss.str();
}
} // namespace tfw

#define SCOPED_TRACE(...) do { std::ostringstream __oss; __oss << __VA_ARGS__; ::tfw::ScopedTrace __tfw_scoped_trace(__oss.str()); } while(0)

namespace tfw {
// ---- non-fatal EXPECT infrastructure ----
struct TestContext { int expect_failures = 0; };
inline thread_local TestContext* _ctx_ptr = nullptr;

struct CtxGuard {
    TestContext ctx;
    CtxGuard() { _ctx_ptr = &ctx; }
    ~CtxGuard() {
        if (ctx.expect_failures) {
            std::ostringstream oss;
            oss << ctx.expect_failures << " expectation(s) failed";
            throw Failure(oss.str());
        }
        _ctx_ptr = nullptr;
    }
};
inline void _record_expect_failure(const std::string& msg) {
    if (_ctx_ptr) {
        _ctx_ptr->expect_failures++;
        std::cerr << msg << "\n";
    } else {
        // If used outside a test or without guard, make it fatal.
        throw Failure(msg);
    }
}
} // namespace tfw

// Helper macro to enable non-fatal expectations inside a TEST body.
// If your TEST macro can be updated, you can inject this automatically.
#define TFW_WITH_CONTEXT ::tfw::CtxGuard __tfw_ctx_guard__;

// ---- value-reporting helpers ----
namespace tfw {
template <class A, class B>
inline void assert_eq(const A& a, const B& b, const char* ea, const char* eb, const char* file, int line) {
    if (!(a == b)) {
        std::ostringstream oss;
        oss << "[ASSERT_EQ FAILED] " << ea << " == " << eb << " at " << loc(file,line)
            << "\n  " << ea << " = " << a << "\n  " << eb << " = " << b
            << _trace_suffix();
        throw Failure(oss.str());
    }
}
template <class A, class B>
inline void expect_eq(const A& a, const B& b, const char* ea, const char* eb, const char* file, int line) {
    if (!(a == b)) {
        std::ostringstream oss;
        oss << "[EXPECT_EQ FAILED] " << ea << " == " << eb << " at " << loc(file,line)
            << "\n  " << ea << " = " << a << "\n  " << eb << " = " << b
            << _trace_suffix();
        _record_expect_failure(oss.str());
    }
}
// generic comparators
#define TFW_GEN_CMP(NAME, OPSTR, OP) \
template <class A, class B> \
inline void assert_##NAME(const A& a, const B& b, const char* ea,const char* eb,const char* file,int line){ \
    if (!(a OP b)) { std::ostringstream oss; \
      oss << "[ASSERT_" #NAME " FAILED] " << ea << " " OPSTR " " << eb << " at " << loc(file,line) \
          << "\n  " << ea << " = " << a << "\n  " << eb << " = " << b << _trace_suffix(); \
      throw Failure(oss.str()); } } \
template <class A, class B> \
inline void expect_##NAME(const A& a, const B& b, const char* ea,const char* eb,const char* file,int line){ \
    if (!(a OP b)) { std::ostringstream oss; \
      oss << "[EXPECT_" #NAME " FAILED] " << ea << " " OPSTR " " << eb << " at " << loc(file,line) \
          << "\n  " << ea << " = " << a << "\n  " << eb << " = " << b << _trace_suffix(); \
      _record_expect_failure(oss.str()); } }

TFW_GEN_CMP(ne, "!=" , !=)
TFW_GEN_CMP(le, "<=", <=)
TFW_GEN_CMP(lt, "<" , < )
TFW_GEN_CMP(ge, ">=", >=)
TFW_GEN_CMP(gt, ">" , > )
#undef TFW_GEN_CMP

// float relative-near
inline void assert_near_rel(float a,float b,float rtol,const char* ea,const char* eb,float rt,const char* file,int line){
    float diff = std::fabs(a-b);
    float tol  = rt * std::max(std::fabs(a), std::fabs(b));
    if (!(diff <= tol)) {
        std::ostringstream oss;
        oss << "[ASSERT_NEAR_REL FAILED] |" << ea << " - " << eb << "| = " << diff
            << " > rtol*max(|a|,|b|) = " << tol << " (rtol="<<rt<<") at " << loc(file,line)
            << "\n  " << ea << " = " << a << "\n  " << eb << " = " << b
            << _trace_suffix();
        throw Failure(oss.str());
    }
}
inline void expect_near_rel(float a,float b,float rtol,const char* ea,const char* eb,float rt,const char* file,int line){
    float diff = std::fabs(a-b);
    float tol  = rt * std::max(std::fabs(a), std::fabs(b));
    if (!(diff <= tol)) {
        std::ostringstream oss;
        oss << "[EXPECT_NEAR_REL FAILED] |" << ea << " - " << eb << "| = " << diff
            << " > rtol*max(|a|,|b|) = " << tol << " (rtol="<<rt<<") at " << loc(file,line)
            << "\n  " << ea << " = " << a << "\n  " << eb << " = " << b
            << _trace_suffix();
        _record_expect_failure(oss.str());
    }
}

// array/vector allclose
template <class T>
inline void assert_allclose(const T* a, const T* b, std::size_t n, float atol, float rtol,
                            const char* ea,const char* eb,const char* file,int line){
    for (std::size_t i=0;i<n;++i){
        float ai=a[i], bi=b[i];
        float diff = std::fabs(ai-bi);
        float tol  = atol + rtol*std::fabs(bi);
        if (!(diff <= tol)) {
            std::ostringstream oss;
            oss << "[ASSERT_ALLCLOSE FAILED] at " << loc(file,line)
                << "\n  first mismatch i="<<i<<": |"<<ea<<"["<<i<<"] - "<<eb<<"["<<i<<"]| = "<<diff
                << " > " << (atol + rtol*std::fabs(bi))
                << "\n  " << ea << "["<<i<<"] = " << ai
                << "\n  " << eb << "["<<i<<"] = " << bi
                << _trace_suffix();
            throw Failure(oss.str());
        }
    }
}
template <class T>
inline void assert_allclose(const std::vector<T>& a, const std::vector<T>& b, float atol, float rtol,
                            const char* ea,const char* eb,const char* file,int line){
    if (a.size()!=b.size()){
        std::ostringstream oss;
        oss << "[ASSERT_ALLCLOSE FAILED] size mismatch: " << ea << ".size()="<<a.size()
            << " vs " << eb << ".size()="<<b.size() << " at " << loc(file,line) << _trace_suffix();
        throw Failure(oss.str());
    }
    assert_allclose(a.data(), b.data(), a.size(), atol, rtol, ea, eb, file, line);
}

// exception helpers
template <class Fn>
inline void assert_throws(Fn&& fn, const char* expr, const char* file, int line){
    bool thrown=false;
    try { fn(); } catch (...) { thrown=true; }
    if (!thrown) {
        std::ostringstream oss;
        oss << "[ASSERT_THROWS FAILED] expected exception: " << expr << " at " << loc(file,line) << _trace_suffix();
        throw Failure(oss.str());
    }
}
template <class Fn>
inline void assert_no_throw(Fn&& fn, const char* expr, const char* file, int line){
    try { fn(); } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "[ASSERT_NO_THROW FAILED] threw: " << e.what() << " in " << expr << " at " << loc(file,line) << _trace_suffix();
        throw Failure(oss.str());
    } catch (...) {
        std::ostringstream oss;
        oss << "[ASSERT_NO_THROW FAILED] threw unknown exception in " << expr << " at " << loc(file,line) << _trace_suffix();
        throw Failure(oss.str());
    }
}

} // namespace tfw

// ---- public macros ----
#define EXPECT_EQ(a,b) ::tfw::expect_eq((a),(b), #a, #b, __FILE__, __LINE__)
#define ASSERT_EQ(a,b) ::tfw::assert_eq((a),(b), #a, #b, __FILE__, __LINE__)
#define EXPECT_NE(a,b) ::tfw::expect_ne((a),(b), #a, #b, __FILE__, __LINE__)
#define ASSERT_NE(a,b) ::tfw::assert_ne((a),(b), #a, #b, __FILE__, __LINE__)
#define EXPECT_LE(a,b) ::tfw::expect_le((a),(b), #a, #b, __FILE__, __LINE__)
#define ASSERT_LE(a,b) ::tfw::assert_le((a),(b), #a, #b, __FILE__, __LINE__)
#define EXPECT_LT(a,b) ::tfw::expect_lt((a),(b), #a, #b, __FILE__, __LINE__)
#define ASSERT_LT(a,b) ::tfw::assert_lt((a),(b), #a, #b, __FILE__, __LINE__)
#define EXPECT_GE(a,b) ::tfw::expect_ge((a),(b), #a, #b, __FILE__, __LINE__)
#define ASSERT_GE(a,b) ::tfw::assert_ge((a),(b), #a, #b, __FILE__, __LINE__)
#define EXPECT_GT(a,b) ::tfw::expect_gt((a),(b), #a, #b, __FILE__, __LINE__)
#define ASSERT_GT(a,b) ::tfw::assert_gt((a),(b), #a, #b, __FILE__, __LINE__)

#define ASSERT_NEAR_REL(a,b,rtol) ::tfw::assert_near_rel((float)(a),(float)(b),(float)(rtol), #a, #b, (float)(rtol), __FILE__, __LINE__)
#define EXPECT_NEAR_REL(a,b,rtol) ::tfw::expect_near_rel((float)(a),(float)(b),(float)(rtol), #a, #b, (float)(rtol), __FILE__, __LINE__)

#define ASSERT_ALLCLOSE_PTR(a,b,n,atol,rtol) ::tfw::assert_allclose((a),(b),(std::size_t)(n),(float)(atol),(float)(rtol), #a, #b, __FILE__, __LINE__)
#define ASSERT_ALLCLOSE_VEC(a,b,atol,rtol)   ::tfw::assert_allclose((a),(b),(float)(atol),(float)(rtol), #a, #b, __FILE__, __LINE__)

#define ASSERT_THROWS(expr) ::tfw::assert_throws([&](){ (void)(expr); }, #expr, __FILE__, __LINE__)
#define ASSERT_NO_THROW(expr) ::tfw::assert_no_throw([&](){ (void)(expr); }, #expr, __FILE__, __LINE__)

#endif // TFW_EXTENSIONS_ADDED
