#pragma once

#include <vector>
#include <cassert>
#include <stdexcept>
#include <iostream>

namespace mynn {
    struct Size {
        int rows;
        int cols;
        bool operator==(const Size& Other){
            return rows==Other.rows && cols==Other.cols;
        }

        bool operator!=(const Size& Other){
            return rows!=Other.rows || cols!=Other.cols;
        }

        inline int count() const { return rows * cols; }
    };

    class Mat {
        int rows_, cols_;
        std::vector<double> data_;
    public:
        Mat();
        Mat(int rows, int cols);
        Mat(const std::vector<double>& data); // by default ill create row matrix 1d
        // should i override (int i) to access in a row matrix, instead of (0, 4) ?
        Mat(int rows, int cols, const std::vector<double>& data);
        Mat(const std::vector<std::vector<double>>& data);
        double& operator()(int i, int j);
        const double& operator()(int i, int j) const;
        Mat operator+(const Mat& Other);
        Mat operator+(const double number);
        Mat operator/(const Mat& Other);
        Mat operator*(double scale);
        friend std::ostream& operator<<(std::ostream& os, const Mat& m);
        Mat row(int i) const;
        Mat col(int i) const;
        Mat transpose() const;
        Size size() const;
        double dot(const Mat& Other) const;
        double sum() const;
        Mat multiply(const Mat& other) const;
        Mat multiplyasthetic(const Mat& other) const;
        Mat multiplybasic(const Mat& other) const;
        Mat multiplyElementWise(const Mat& other) const;
        double meanof1d() const;
        Mat sq() const;
        Mat sqrt() const;
    };

    std::ostream& operator<<(std::ostream& os, const Mat& m);
}
