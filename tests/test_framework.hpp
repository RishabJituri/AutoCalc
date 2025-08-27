
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

inline void assert_near(double a, double b, double eps, const char* exprA, const char* exprB, double e, const char* file, int line) {
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b) || std::fabs(a - b) > eps) {
        std::ostringstream oss;
        oss << "[ASSERT_NEAR FAILED] |" << exprA << " - " << exprB << "| = " << std::fabs(a-b)
            << " > " << e << " at " << loc(file, line) << "\n  " << exprA << " = " << a << "\n  " << exprB << " = " << b;
        throw Failure(oss.str());
    }
}
#define ASSERT_NEAR(a,b,eps) ::tfw::assert_near((double)(a),(double)(b),(double)(eps), #a, #b, (double)(eps), __FILE__, __LINE__)

} // namespace tfw

