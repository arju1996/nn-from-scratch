#include "Mat.h"

#include <cmath>



mynn::Mat::Mat() : rows_(0), cols_(0), data_() {

};

mynn::Mat::Mat(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {

};

mynn::Mat::Mat(int rows, int cols, const std::vector<double>& data) : rows_(rows), cols_(cols), data_(data) {

};

mynn::Mat::Mat(int rows, int cols,std::vector<double>&& data) noexcept : rows_(rows), cols_(cols), data_(std::move(data)) {
};

mynn::Mat::Mat(const std::vector<double>& data) : rows_(1), cols_(data.size()), data_(data) {}
mynn::Mat::Mat(std::vector<double>&& data) noexcept : rows_(1), cols_(data.size()), data_(std::move(data)) {}

void mynn::Mat::FlattenFrom(const std::vector<std::vector<double>>& data) {
    // funnily this doesnt give any improvements cause move iterator works same as copy
    // for primitive data types
    int rows = data.size();
    rows_ = rows;
    int cols = data.at(0).size();
    cols_ = cols;
    data_.reserve(rows_ * cols_);

    for (auto &row : data) {
        assert(cols == row.size());
        data_.insert(data_.end(),
            std::make_move_iterator(row.begin()),
            std::make_move_iterator(row.end())
        );
    }
}


mynn::Mat::Mat(const std::vector<std::vector<double>>& data) {
    int rows = data.size();
    rows_ = rows;
    int cols = data.at(0).size();
    cols_ = cols;
    data_.reserve(rows_ * cols_);
    for(int i = 0; i < rows; ++i) {
        assert(cols == data[i].size());
        data_.insert(data_.end(), data[i].begin(), data[i].end());
    }
}

mynn::Mat::Mat(Mat&& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::move(other.data_);
}
mynn::Mat::Mat(const Mat& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {

}

mynn::Mat& mynn::Mat::operator=(const Mat& other) {
    if (this == &other) return *this;
    // delete other;
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
    return *this;
}

mynn::Mat& mynn::Mat::operator=(Mat&& other) {
    if (this == &other) return *this;
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::move(other.data_);
}




mynn::Mat::Mat(std::vector<std::vector<double>>&& data) noexcept {
    FlattenFrom(std::move(data));
}

/*
double& mynn::Mat::operator()(int i, int j) {
    if (i < 0 || i >= rows_)
        throw std::out_of_range("Row index out of bound.");
    if ( j < 0 || j >= cols_)
        throw std::out_of_range("Col index out of bound.");
    return data_[i * cols_ + j];
}
const double& mynn::Mat::operator()(int i, int j) const{
    if ( i < 0 || i >= rows_)
        throw std::out_of_range("Row index out of bound.");
    if ( j < 0 || j >= cols_)
        throw std::out_of_range("Col index out of bound.");
    return data_[i * cols_ + j];
}
*/
mynn::Mat mynn::Mat::operator+(const mynn::Mat& Other) {
    mynn::Mat result (this->rows_, this->cols_);
    if (this->size() == Other.size()) {
        for(int i = 0; i < this->rows_; ++i) {
            for(int j=0; j < this->cols_; ++j) {
                result(i,j) = this->operator()(i,j) + Other(i,j);
            }
        }
    } 

    // if 150*2 + 1*2 vannal add 1*2 to all 150*2
    else if (Other.rows_ == 1 && Other.cols_ == this->cols_) {
        for(int i = 0; i < this->rows_; ++i) {
            for(int j=0; j < this->cols_; ++j) {
                result(i,j) = this->operator()(i,j) + Other(0,j);
            }
        }
    }

    // if 2*50 + 2*1 vannal add 2*1 to all 2*50
    else if(Other.cols_ == 1 && Other.rows_ == this->rows_) {
        for(int i = 0; i < this->rows_; ++i) {
            for(int j=0; j < this->cols_; ++j) {
                result(i,j) = this->operator()(i,j) + Other(i,0);
            }
        }
    } else {
        throw std::invalid_argument("Mats are of diff dimentions that cant be added or broadcasted.");
    }

    return result;
}

mynn::Mat mynn::Mat::operator*(double scale) {
    mynn::Mat result (this->rows_, this->cols_);
    for(int i = 0; i < this->rows_; ++i) {
        for(int j=0; j < this->cols_; ++j) {
            result(i,j) = this->operator()(i,j) * scale;
        }
    }
    return result;
}

std::ostream& mynn::operator<<(std::ostream& os, const mynn::Mat& m) {
    for(int i = 0; i < m.rows_; ++i) {
        for(int j = 0; j < m.cols_; ++j) {
            os << m(i,j)<<" ";
        }
        os << "\n";
    }
    return os;
}

mynn::Mat mynn::Mat::row(int i) const {
    // if this could not create any new memory it would be good;
    // cause i think this row op is gonna be slow;
    Mat row1(1, cols_);
    for (int j = 0; j<cols_; ++j) {
        row1.data_[j] = data_[i * cols_ + j];
    }
    return row1;
}

void mynn::Mat::setRow(int i, Mat row) {
    for (int j = 0; j<cols_; ++j) {
        data_[i * cols_ + j] = row.data_[j];
    }
}


mynn::Mat mynn::Mat::col(int i) const {
    // if this could not create any new memory it would be good;
    // cause i think this row op is gonna be slow;

    // cols is not optimized locally, so may be transpose and row might be faster
    Mat col1(rows_, 1);
    for (int j = 0; j<rows_; ++j) {
        col1.data_[j] = data_[j * cols_ + i];
    }
    return col1;
}

mynn::Mat mynn::Mat::transpose() const {
    int transposeRows = cols_;
    int transposeCols = rows_;
    Mat tr (transposeRows, transposeCols);

    for(int i = 0; i < transposeRows; ++i) {
        for(int j = 0; j < transposeCols; ++j) {
            tr(i, j) = this->operator()(j, i);
        }
    }
    return tr;
}

mynn::Size mynn::Mat::size() const {
    return {rows_, cols_};
}

double mynn::Mat::dot(const Mat& Other) const {
    Size currentSize = this->size();
    Size otherSize = Other.size();

    if(currentSize.cols != 1 && currentSize.rows != 1 &&
        otherSize.cols != 1 && otherSize.rows != 1
    )
        throw std::invalid_argument("Mat is not 1d.");

    if(currentSize.count() != otherSize.count())
        throw std::invalid_argument("Size is not same.");
    
    double result = 0;
    for (int i = 0, n = currentSize.count(); i < n; ++i) {
        result += this->data_[i] * Other.data_[i];
    }
    return result;
}

double mynn::Mat::sum() const {
    double result = 0;
    for(int i = 0, n = data_.size(); i < n; ++i) {
        result +=  data_[i];
    }
    return result;
}

mynn::Mat mynn::Mat::multiply(const Mat& other) const {
    return multiplybasic(other);
}


mynn::Mat mynn::Mat::multiplyasthetic(const Mat& other) const {
    // this is the slowest code that can be written ig;
    
    if (this->size().cols != other.size().rows)
        throw std::invalid_argument("Cannot multiply.");

    // auto otherTranspose = other.transpose();
    Mat result(this->size().rows, other.size().cols);

    for(int i = 0, r = this->size().rows; i < r; ++i) {
        for(int j = 0, c = other.size().cols; j < c; ++j) {
            result(i, j) = this->row(i).dot(other.col(j));
        }
    }

    return result;
}

mynn::Mat mynn::Mat::multiplybasic(const Mat& other) const {
    // this is the most basic logic

    
    if (this->size().cols != other.size().rows)
        throw std::invalid_argument("Cannot multiply.");

    Mat result(this->size().rows, other.size().cols);

    for(int i = 0, r = this->size().rows; i < r; ++i) {
        for(int j = 0, c = other.size().cols; j < c; ++j) {
            double resultk = 0;
            for(int k = 0, cc = this->size().cols; k<cc; ++k) {
                resultk += this->data_[i*cols_ + k] * other.operator()(k, j);
                // resultk += this->operator()(i, k) * other.operator()(k, j);
            }
            result.operator()(i,j) = resultk;
        }
    }

    return result;
}

mynn::Mat mynn::Mat::multiplyElementWise(const Mat& other) const {
    if (this->size() != other.size())
        throw std::invalid_argument("Cannot multiply.");

    Mat result(this->size().rows, other.size().cols);

    for(int i = 0, r = this->size().rows; i < r; ++i) {
        for(int j = 0, c = other.size().cols; j < c; ++j) {
            result(i,j) = this->operator()(i,j) * other(i,j);
        }
    }
    return result;
}


double mynn::Mat::meanof1d() const {
    // hi
    if(this->size().rows == 1 || this->size().cols != 1) {
        throw std::runtime_error("mean logic currenly work only for 1d matrix");
    }

    double sum = 0;
    int count = this->size().count();
    for(int i = 0; i < count; ++i) {
        sum += this->data_[i];
    }
    sum /= count;
    return sum;
}

// mynn::Mat mynn::Mat::operator*(double scale) {
//     mynn::Mat result (this->rows_, this->cols_);
//     for(int i = 0; i < this->rows_; ++i) {
//         for(int j=0; j < this->cols_; ++j) {
//             result(i,j) = this->operator()(i,j) * scale;
//         }
//     }
//     return result;
// }

mynn::Mat mynn::Mat::sq() const {
    mynn::Mat result (this->rows_, this->cols_);
    for(int i = 0; i < this->rows_; ++i) {
        for(int j=0; j < this->cols_; ++j) {
            result(i,j) = this->operator()(i,j) * this->operator()(i,j);
        }
    }
    return result;
}

mynn::Mat mynn::Mat::sqrt() const {
    mynn::Mat result (this->rows_, this->cols_);
    for(int i = 0; i < this->rows_; ++i) {
        for(int j=0; j < this->cols_; ++j) {
            result(i,j) = std::sqrt(this->operator()(i,j));
        }
    }
    return result;
}

mynn::Mat mynn::Mat::operator/(const mynn::Mat& Other) {
    mynn::Mat result (this->rows_, this->cols_);
    if (this->size() == Other.size()) {
        for(int i = 0; i < this->rows_; ++i) {
            for(int j=0; j < this->cols_; ++j) {
                result(i,j) = this->operator()(i,j) / Other(i,j);
            }
        }
    } 
/*
    // if 150*2 + 1*2 vannal add 1*2 to all 150*2
    else if (Other.rows_ == 1 && Other.cols_ == this->cols_) {
        for(int i = 0; i < this->rows_; ++i) {
            for(int j=0; j < this->cols_; ++j) {
                result(i,j) = this->operator()(i,j) + Other(0,j);
            }
        }
    }

    // if 2*50 + 2*1 vannal add 2*1 to all 2*50
    else if(Other.cols_ == 1 && Other.rows_ == this->rows_) {
        for(int i = 0; i < this->rows_; ++i) {
            for(int j=0; j < this->cols_; ++j) {
                result(i,j) = this->operator()(i,j) + Other(i,0);
            }
        }
    } 
*/
    else {
        throw std::invalid_argument("Mats are of diff dimentions that cant be added or broadcasted.");
    }

    return result;
}

mynn::Mat mynn::Mat::operator+(const double number) {
    mynn::Mat result (this->rows_, this->cols_);
    for(int i = 0; i < this->rows_; ++i) {
        for(int j=0; j < this->cols_; ++j) {
            result(i,j) = this->operator()(i,j) + number;
        }
    }
    return result;
}

mynn::Mat mynn::Mat::getMulRows(size_t start, size_t end) {
    mynn::Mat result(end-start, cols_);
    for(size_t i = start; i < end; ++i) {
        result.setRow(i - start, row(i));
    }
    return result;
}

std::pair<mynn::Mat, mynn::Mat> mynn::Mat::splitRows(size_t splitindex) {
    return {getMulRows(0, splitindex), getMulRows(splitindex, rows_)};
}