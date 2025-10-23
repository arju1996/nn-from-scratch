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
        void FlattenFrom(const std::vector<std::vector<double>>& data);
    public:
        Mat();
        Mat(Mat&& input);
        Mat(const Mat& input);
        Mat(int rows, int cols);
        Mat(const std::vector<double>& data); // by default ill create row matrix 1d
        Mat(std::vector<double>&& data) noexcept; // by default ill create row matrix 1d
        // should i override (int i) to access in a row matrix, instead of (0, 4) ?
        Mat(int rows, int cols, const std::vector<double>& data);
        Mat(int rows, int cols, std::vector<double>&& data) noexcept;
        Mat(const std::vector<std::vector<double>>& data);
        Mat(std::vector<std::vector<double>>&& data) noexcept;
        /*
        double& operator()(int i, int j);
        const double& operator()(int i, int j) const;
        */
        inline double& mynn::Mat::operator()(int i, int j) {
            if (i < 0 || i >= rows_)
                throw std::out_of_range("Row index out of bound.");
            if ( j < 0 || j >= cols_)
                throw std::out_of_range("Col index out of bound.");
            return data_[i * cols_ + j];
        }
        inline const double& mynn::Mat::operator()(int i, int j) const{
            if ( i < 0 || i >= rows_)
                throw std::out_of_range("Row index out of bound.");
            if ( j < 0 || j >= cols_)
                throw std::out_of_range("Col index out of bound.");
            return data_[i * cols_ + j];
        }

        Mat operator+(const Mat& Other);
        Mat operator+(const double number);
        Mat operator/(const Mat& Other);
        Mat operator*(double scale);
        Mat& operator=(const Mat& Other);
        Mat& operator=(Mat&& Other);
        friend std::ostream& operator<<(std::ostream& os, const Mat& m);
        Mat row(int i) const;
        void setRow(int i, Mat);
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
#ifndef RAW_ACCESS
        const double* data() const { return data_.data();}
        double* data() { return data_.data();}
#endif
        mynn::Mat getMulRows(size_t start, size_t end);
        std::pair<mynn::Mat, mynn::Mat> splitRows(size_t splitindex);
    };

    std::ostream& operator<<(std::ostream& os, const Mat& m);
}
