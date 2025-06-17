// utils.h
#pragma once

#include <iostream>
#include <vector>
#include <type_traits>
#include <string>

namespace print {

// Base case: scalar values
template <typename T>
typename std::enable_if<!std::is_class<T>::value>::type
PrintVector(const T& val) {
    std::cout << val << " ";
}

// Recursive case: vector of vectors
template <typename T>
void PrintVector(const std::vector<T>& vec) {
    std::cout << "[ ";
    for (const auto& item : vec) {
        PrintVector(item);  // recurse
    }
    std::cout << "] ";
}

template <typename T>
void PrintVectorWithLabel(const T& vec, const std::string& label) {
    std::cout << label << ": ";
    PrintVector(vec);
    std::cout << "\n";
}

} // namespace util