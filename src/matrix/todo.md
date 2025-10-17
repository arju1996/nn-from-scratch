.make row return without creating new memory;
.cols(i) is not optimized locally, so may be transpose and row might be faster
cause mat is row major
check which is faster
.make multiply with 3 level loop, simplest and the fastest === done

.simd
.cache localization
.openmp






====
apprendix
class RowView {
    double* ptr_;
    int size_;
public:
    RowView(double* p, int s) : ptr_(p), size_(s) {}
    double& operator[](int j) { return ptr_[j]; }
    double* begin() { return ptr_; }
    double* end() { return ptr_ + size_; }
    int size() const { return size_; }
};

RowView row(int i) { return RowView(data_.data() + i*cols_, cols_); }

    
isnt this better than returning a mat?
and may be making it return a const aslo might be better idea
dont forget to think of already existing algos like max_element from softmax 