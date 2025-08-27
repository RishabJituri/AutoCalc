
// tests/test_main.cpp
#include "test_framework.hpp"

// No tests here; all tests are registered via static initializers
// in the other compilation units. This TU just provides main().
int main() {
    const auto& tests = tfw::registry();
    int passed = 0;
    for (const auto& test : tests) {
        std::cout << "[ RUN      ] " << test.name << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        try {
            test.fn();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "[       OK ] " << test.name << " (" << elapsed.count() << "s)" << std::endl;
            ++passed;
        } catch (const tfw::Failure& e) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << e.what() << std::endl;
            std::cout << "[  FAILED  ] " << test.name << " (" << elapsed.count() << "s)" << std::endl;
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "[  EXCEPTION  ] " << test.name << ": " << e.what() << " (" << elapsed.count() << "s)" << std::endl;
        } catch (...) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "[  UNKNOWN EXCEPTION  ] " << test.name << " (" << elapsed.count() << "s)" << std::endl;
        }
    }
    std::cout << "[==========] " << tests.size() << " tests ran. " << passed << " passed, " << (tests.size() - passed) << " failed." << std::endl;
    return (passed == tests.size()) ? 0 : 1;
}
