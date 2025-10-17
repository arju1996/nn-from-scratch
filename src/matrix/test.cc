#include "mat.h"  // Assuming your mynn::Mat class is in mat.h
#include <iostream>
#include <cmath>
#include <iomanip>

#define ASSERT_TEST(condition, test_name) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << test_name << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        if (test_func()) { \
            std::cout << "PASSED" << std::endl; \
            passed++; \
        } else { \
            failed++; \
        } \
        total++; \
    } while(0)

bool approx_equal(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

// Test constructors
bool test_constructor_basic() {
    mynn::Mat m(3, 4);
    ASSERT_TEST(m.size().rows == 3, "rows should be 3");
    ASSERT_TEST(m.size().cols == 4, "cols should be 4");
    return true;
}

bool test_constructor_with_data() {
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    mynn::Mat m(2, 3, data);
    ASSERT_TEST(m.size().rows == 2, "rows should be 2");
    ASSERT_TEST(m.size().cols == 3, "cols should be 3");
    ASSERT_TEST(approx_equal(m(0, 0), 1.0), "m(0,0) should be 1");
    ASSERT_TEST(approx_equal(m(1, 2), 6.0), "m(1,2) should be 6");
    return true;
}

bool test_constructor_2d_vector() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    mynn::Mat m(data);
    ASSERT_TEST(m.size().rows == 3, "rows should be 3");
    ASSERT_TEST(m.size().cols == 3, "cols should be 3");
    ASSERT_TEST(approx_equal(m(0, 0), 1.0), "m(0,0) should be 1");
    ASSERT_TEST(approx_equal(m(1, 1), 5.0), "m(1,1) should be 5");
    ASSERT_TEST(approx_equal(m(2, 2), 9.0), "m(2,2) should be 9");
    return true;
}

// Test element access
bool test_element_access() {
    mynn::Mat m(3, 3);
    m(0, 0) = 1.5;
    m(1, 2) = 2.5;
    m(2, 1) = 3.5;
    
    ASSERT_TEST(approx_equal(m(0, 0), 1.5), "m(0,0) should be 1.5");
    ASSERT_TEST(approx_equal(m(1, 2), 2.5), "m(1,2) should be 2.5");
    ASSERT_TEST(approx_equal(m(2, 1), 3.5), "m(2,1) should be 3.5");
    return true;
}

bool test_element_access_const() {
    std::vector<std::vector<double>> data = {{1, 2}, {3, 4}};
    const mynn::Mat m(data);
    
    ASSERT_TEST(approx_equal(m(0, 0), 1.0), "const m(0,0) should be 1");
    ASSERT_TEST(approx_equal(m(1, 1), 4.0), "const m(1,1) should be 4");
    return true;
}

bool test_out_of_bounds() {
    mynn::Mat m(3, 3);
    bool caught = false;
    
    try {
        m(5, 0) = 1.0;  // Should throw
    } catch (const std::out_of_range&) {
        caught = true;
    }
    ASSERT_TEST(caught, "Should throw out_of_range for row");
    
    caught = false;
    try {
        m(0, 5) = 1.0;  // Should throw
    } catch (const std::out_of_range&) {
        caught = true;
    }
    ASSERT_TEST(caught, "Should throw out_of_range for col");
    
    caught = false;
    try {
        m(-1, 0) = 1.0;  // Should throw
    } catch (const std::out_of_range&) {
        caught = true;
    }
    ASSERT_TEST(caught, "Should throw out_of_range for negative index");
    
    return true;
}

// Test row and col extraction
bool test_row_extraction() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    mynn::Mat m(data);
    mynn::Mat r = m.row(1);
    
    ASSERT_TEST(r.size().rows == 1, "row should have 1 row");
    ASSERT_TEST(r.size().cols == 3, "row should have 3 cols");
    ASSERT_TEST(approx_equal(r(0, 0), 4.0), "r(0,0) should be 4");
    ASSERT_TEST(approx_equal(r(0, 1), 5.0), "r(0,1) should be 5");
    ASSERT_TEST(approx_equal(r(0, 2), 6.0), "r(0,2) should be 6");
    return true;
}

bool test_col_extraction() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    mynn::Mat m(data);
    mynn::Mat c = m.col(1);
    
    ASSERT_TEST(c.size().rows == 3, "col should have 3 rows");
    ASSERT_TEST(c.size().cols == 1, "col should have 1 col");
    ASSERT_TEST(approx_equal(c(0, 0), 2.0), "c(0,0) should be 2");
    ASSERT_TEST(approx_equal(c(1, 0), 5.0), "c(1,0) should be 5");
    ASSERT_TEST(approx_equal(c(2, 0), 8.0), "c(2,0) should be 8");
    return true;
}

// Test transpose
bool test_transpose_square() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    mynn::Mat m(data);
    mynn::Mat t = m.transpose();
    
    ASSERT_TEST(t.size().rows == 3, "transpose rows");
    ASSERT_TEST(t.size().cols == 3, "transpose cols");
    ASSERT_TEST(approx_equal(t(0, 0), 1.0), "t(0,0)");
    ASSERT_TEST(approx_equal(t(0, 1), 4.0), "t(0,1)");
    ASSERT_TEST(approx_equal(t(1, 0), 2.0), "t(1,0)");
    ASSERT_TEST(approx_equal(t(2, 1), 6.0), "t(2,1)");
    return true;
}

bool test_transpose_rectangular() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };
    mynn::Mat m(data);
    mynn::Mat t = m.transpose();
    
    ASSERT_TEST(t.size().rows == 4, "transpose should have 4 rows");
    ASSERT_TEST(t.size().cols == 2, "transpose should have 2 cols");
    ASSERT_TEST(approx_equal(t(0, 0), 1.0), "t(0,0) should be 1");
    ASSERT_TEST(approx_equal(t(0, 1), 5.0), "t(0,1) should be 5");
    ASSERT_TEST(approx_equal(t(3, 1), 8.0), "t(3,1) should be 8");
    return true;
}

bool test_double_transpose() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6}
    };
    mynn::Mat m(data);
    mynn::Mat t = m.transpose().transpose();
    
    ASSERT_TEST(t.size().rows == 2, "double transpose rows");
    ASSERT_TEST(t.size().cols == 3, "double transpose cols");
    ASSERT_TEST(approx_equal(t(0, 0), 1.0), "t(0,0)");
    ASSERT_TEST(approx_equal(t(1, 2), 6.0), "t(1,2)");
    return true;
}

// Test dot product
bool test_dot_row_vectors() {
    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5, 6};
    mynn::Mat v1(1, 3, data1);
    mynn::Mat v2(1, 3, data2);
    
    double result = v1.dot(v2);
    ASSERT_TEST(approx_equal(result, 32.0), "dot product should be 32 (1*4 + 2*5 + 3*6)");
    return true;
}

bool test_dot_col_vectors() {
    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5, 6};
    mynn::Mat v1(3, 1, data1);
    mynn::Mat v2(3, 1, data2);
    
    double result = v1.dot(v2);
    ASSERT_TEST(approx_equal(result, 32.0), "dot product should be 32");
    return true;
}

bool test_dot_mixed_vectors() {
    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5, 6};
    mynn::Mat v1(1, 3, data1);
    mynn::Mat v2(3, 1, data2);
    
    double result = v1.dot(v2);
    ASSERT_TEST(approx_equal(result, 32.0), "dot product should be 32");
    return true;
}

bool test_dot_size_mismatch() {
    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5};
    mynn::Mat v1(1, 3, data1);
    mynn::Mat v2(1, 2, data2);
    
    bool caught = false;
    try {
        v1.dot(v2);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT_TEST(caught, "Should throw for size mismatch");
    return true;
}

bool test_dot_non_vector() {
    mynn::Mat m1(2, 2);
    mynn::Mat m2(2, 2);
    
    bool caught = false;
    try {
        m1.dot(m2);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT_TEST(caught, "Should throw for non-1d matrices");
    return true;
}

// Test sum
bool test_sum_simple() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6}
    };
    mynn::Mat m(data);
    double s = m.sum();
    ASSERT_TEST(approx_equal(s, 21.0), "sum should be 21");
    return true;
}

bool test_sum_single_element() {
    mynn::Mat m(1, 1);
    m(0, 0) = 42.0;
    ASSERT_TEST(approx_equal(m.sum(), 42.0), "sum should be 42");
    return true;
}

bool test_sum_negatives() {
    std::vector<std::vector<double>> data = {
        {1, -2, 3},
        {-4, 5, -6}
    };
    mynn::Mat m(data);
    double s = m.sum();
    ASSERT_TEST(approx_equal(s, -3.0), "sum should be -3");
    return true;
}

// Test matrix multiplication
bool test_multiply_square() {
    std::vector<std::vector<double>> data1 = {
        {1, 2},
        {3, 4}
    };
    std::vector<std::vector<double>> data2 = {
        {5, 6},
        {7, 8}
    };
    mynn::Mat m1(data1);
    mynn::Mat m2(data2);
    mynn::Mat result = m1.multiply(m2);
    
    ASSERT_TEST(result.size().rows == 2, "result rows");
    ASSERT_TEST(result.size().cols == 2, "result cols");
    ASSERT_TEST(approx_equal(result(0, 0), 19.0), "result(0,0) = 1*5 + 2*7 = 19");
    ASSERT_TEST(approx_equal(result(0, 1), 22.0), "result(0,1) = 1*6 + 2*8 = 22");
    ASSERT_TEST(approx_equal(result(1, 0), 43.0), "result(1,0) = 3*5 + 4*7 = 43");
    ASSERT_TEST(approx_equal(result(1, 1), 50.0), "result(1,1) = 3*6 + 4*8 = 50");
    return true;
}

bool test_multiply_rectangular() {
    std::vector<std::vector<double>> data1 = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<double>> data2 = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    mynn::Mat m1(data1);
    mynn::Mat m2(data2);
    mynn::Mat result = m1.multiply(m2);
    
    ASSERT_TEST(result.size().rows == 2, "result should be 2x2");
    ASSERT_TEST(result.size().cols == 2, "result should be 2x2");
    ASSERT_TEST(approx_equal(result(0, 0), 58.0), "result(0,0) = 1*7 + 2*9 + 3*11");
    ASSERT_TEST(approx_equal(result(0, 1), 64.0), "result(0,1) = 1*8 + 2*10 + 3*12");
    ASSERT_TEST(approx_equal(result(1, 0), 139.0), "result(1,0) = 4*7 + 5*9 + 6*11");
    ASSERT_TEST(approx_equal(result(1, 1), 154.0), "result(1,1) = 4*8 + 5*10 + 6*12");
    return true;
}

bool test_multiply_identity() {
    std::vector<std::vector<double>> data = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    std::vector<std::vector<double>> identity = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    mynn::Mat m(data);
    mynn::Mat I(identity);
    mynn::Mat result = m.multiply(I);
    
    ASSERT_TEST(approx_equal(result(0, 0), 1.0), "should be unchanged");
    ASSERT_TEST(approx_equal(result(1, 1), 5.0), "should be unchanged");
    ASSERT_TEST(approx_equal(result(2, 2), 9.0), "should be unchanged");
    return true;
}

bool test_multiply_dimension_mismatch() {
    mynn::Mat m1(2, 3);
    mynn::Mat m2(2, 2);
    
    bool caught = false;
    try {
        m1.multiply(m2);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT_TEST(caught, "Should throw for dimension mismatch");
    return true;
}

bool test_multiply_vector() {
    std::vector<std::vector<double>> matrix = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<double> vec_data = {7, 8, 9};
    mynn::Mat m(matrix);
    mynn::Mat v(3, 1, vec_data);
    mynn::Mat result = m.multiply(v);
    
    ASSERT_TEST(result.size().rows == 2, "result should be 2x1");
    ASSERT_TEST(result.size().cols == 1, "result should be 2x1");
    ASSERT_TEST(approx_equal(result(0, 0), 50.0), "result(0,0) = 1*7 + 2*8 + 3*9");
    ASSERT_TEST(approx_equal(result(1, 0), 122.0), "result(1,0) = 4*7 + 5*8 + 6*9");
    return true;
}

// Test mynn::Size struct
bool test_size_equality() {
    mynn::Size s1{3, 4};
    mynn::Size s2{3, 4};
    mynn::Size s3{3, 5};
    
    ASSERT_TEST(s1 == s2, "equal sizes should be equal");
    ASSERT_TEST(s1 != s3, "different sizes should not be equal");
    return true;
}

bool test_size_count() {
    mynn::Size s{3, 4};
    ASSERT_TEST(s.count() == 12, "count should be rows * cols");
    return true;
}

// Edge cases
bool test_single_element_matrix() {
    mynn::Mat m(1, 1);
    m(0, 0) = 3.14;
    ASSERT_TEST(approx_equal(m(0, 0), 3.14), "single element access");
    ASSERT_TEST(approx_equal(m.sum(), 3.14), "single element sum");
    
    mynn::Mat t = m.transpose();
    ASSERT_TEST(approx_equal(t(0, 0), 3.14), "single element transpose");
    return true;
}

bool test_large_matrix() {
    mynn::Mat m(100, 100);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            m(i, j) = i + j;
        }
    }
    ASSERT_TEST(m.size().rows == 100, "large matrix rows");
    ASSERT_TEST(m.size().cols == 100, "large matrix cols");
    ASSERT_TEST(approx_equal(m(50, 50), 100.0), "large matrix access");
    return true;
}

// Main test runner
int main() {
    int total = 0, passed = 0, failed = 0;
    
    std::cout << "========================================" << std::endl;
    std::cout << "       mynn::Mat Class Test Suite" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    std::cout << "Constructor Tests:" << std::endl;
    RUN_TEST(test_constructor_basic);
    RUN_TEST(test_constructor_with_data);
    RUN_TEST(test_constructor_2d_vector);
    std::cout << std::endl;
    
    std::cout << "Element Access Tests:" << std::endl;
    RUN_TEST(test_element_access);
    RUN_TEST(test_element_access_const);
    RUN_TEST(test_out_of_bounds);
    std::cout << std::endl;
    
    std::cout << "Row/Col Extraction Tests:" << std::endl;
    RUN_TEST(test_row_extraction);
    RUN_TEST(test_col_extraction);
    std::cout << std::endl;
    
    std::cout << "Transpose Tests:" << std::endl;
    RUN_TEST(test_transpose_square);
    RUN_TEST(test_transpose_rectangular);
    RUN_TEST(test_double_transpose);
    std::cout << std::endl;
    
    std::cout << "Dot Product Tests:" << std::endl;
    RUN_TEST(test_dot_row_vectors);
    RUN_TEST(test_dot_col_vectors);
    RUN_TEST(test_dot_mixed_vectors);
    RUN_TEST(test_dot_size_mismatch);
    RUN_TEST(test_dot_non_vector);
    std::cout << std::endl;
    
    std::cout << "Sum Tests:" << std::endl;
    RUN_TEST(test_sum_simple);
    RUN_TEST(test_sum_single_element);
    RUN_TEST(test_sum_negatives);
    std::cout << std::endl;
    
    std::cout << "Matrix Multiplication Tests:" << std::endl;
    RUN_TEST(test_multiply_square);
    RUN_TEST(test_multiply_rectangular);
    RUN_TEST(test_multiply_identity);
    RUN_TEST(test_multiply_dimension_mismatch);
    RUN_TEST(test_multiply_vector);
    std::cout << std::endl;
    
    std::cout << "mynn::Size Struct Tests:" << std::endl;
    RUN_TEST(test_size_equality);
    RUN_TEST(test_size_count);
    std::cout << std::endl;
    
    std::cout << "Edge Case Tests:" << std::endl;
    RUN_TEST(test_single_element_matrix);
    RUN_TEST(test_large_matrix);
    std::cout << std::endl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed";
    if (failed > 0) {
        std::cout << " (" << failed << " failed)";
    }
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    
    return failed == 0 ? 0 : 1;
}