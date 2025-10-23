#include <nn.h>
#include "random.h"


#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>


std::pair< std::vector<std::vector<double>>, std::vector<double>> spiltToInputAndOutputs(std::vector<std::tuple<double, double, int>> data) {
    std::vector<std::vector<double>> inputs;
    std::vector<double> outputs;
    for(auto item: data) {
        inputs.push_back(
            {
            std::get<0>(item),
            std::get<1>(item)
            }
        );
        outputs.push_back(std::get<2>(item));
    }
    return {inputs, outputs};
}

void benchmark_access_patterns() {
    const size_t rows = 1000;
    const size_t cols = 1000;
    const size_t iterations = 100;
    
    mynn::Mat mat(rows, cols);
    // Fill matrix
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            mat(i, j) = i * j;
        }
    }
    
    // Test 1: operator()
    auto start1 = std::chrono::high_resolution_clock::now();
    double sum1 = 0;
    for(size_t iter = 0; iter < iterations; ++iter) {
        for(size_t i = 0; i < rows; ++i) {
            for(size_t j = 0; j < cols; ++j) {
                sum1 += mat(i, j);
            }
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    
    // Test 2: Raw pointer via friend access
    auto start2 = std::chrono::high_resolution_clock::now();
    double sum2 = 0;
    const double* ptr = mat.data();  // Direct access to vector
    for(size_t iter = 0; iter < iterations; ++iter) {
        for(size_t i = 0; i < rows; ++i) {
            for(size_t j = 0; j < cols; ++j) {
                sum2 += ptr[i * cols + j];
            }
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    
    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    
    std::cout << "operator(): " << d1 << " μs (sum=" << sum1 << ")\n";
    std::cout << "raw access: " << d2 << " μs (sum=" << sum2 << ")\n";
    std::cout << "Speedup: " << (double)d1 / d2 << "x\n";
}

class Benchmark {
public:
    struct Result {
        double mean_ms;
        double median_ms;
        double min_ms;
        double max_ms;
        double std_dev_ms;
        size_t iterations;
    };

    // Benchmark a function with multiple iterations
    template<typename Func>
    static Result run(Func&& func, size_t warmup_iters = 10, size_t bench_iters = 100) {
        std::vector<double> timings;
        timings.reserve(bench_iters);

        // Warmup phase - important for cache warming and branch prediction
        for (size_t i = 0; i < warmup_iters; ++i) {
            func();
        }

        // Actual benchmark
        for (size_t i = 0; i < bench_iters; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            timings.push_back(duration.count() / 1e6); // Convert to milliseconds
        }

        return calculate_statistics(timings);
    }

    static void print_result(const std::string& name, const Result& result) {
        std::cout << "\n=== " << name << " ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Mean:       " << result.mean_ms << " ms" << std::endl;
        std::cout << "Median:     " << result.median_ms << " ms" << std::endl;
        std::cout << "Min:        " << result.min_ms << " ms" << std::endl;
        std::cout << "Max:        " << result.max_ms << " ms" << std::endl;
        std::cout << "Std Dev:    " << result.std_dev_ms << " ms" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
    }

    static void compare_results(const Result& baseline, const Result& optimized) {
        double speedup = baseline.mean_ms / optimized.mean_ms;
        double improvement = ((baseline.mean_ms - optimized.mean_ms) / baseline.mean_ms) * 100.0;
        
        std::cout << "\n=== COMPARISON ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Speedup:     " << speedup << "x" << std::endl;
        std::cout << "Improvement: " << improvement << "%" << std::endl;
        
        if (speedup > 1.05) {
            std::cout << "✓ Fused version is FASTER" << std::endl;
        } else if (speedup < 0.95) {
            std::cout << "✗ Fused version is SLOWER" << std::endl;
        } else {
            std::cout << "≈ Performance is SIMILAR" << std::endl;
        }
    }

private:
    static Result calculate_statistics(std::vector<double>& timings) {
        Result result;
        result.iterations = timings.size();

        // Sort for median calculation
        std::sort(timings.begin(), timings.end());

        // Mean
        result.mean_ms = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();

        // Median
        size_t mid = timings.size() / 2;
        if (timings.size() % 2 == 0) {
            result.median_ms = (timings[mid - 1] + timings[mid]) / 2.0;
        } else {
            result.median_ms = timings[mid];
        }

        // Min and Max
        result.min_ms = timings.front();
        result.max_ms = timings.back();

        // Standard deviation
        double sq_sum = 0.0;
        for (const auto& time : timings) {
            sq_sum += (time - result.mean_ms) * (time - result.mean_ms);
        }
        result.std_dev_ms = std::sqrt(sq_sum / timings.size());

        return result;
    }
};

void benchmarkDenseLayerForward() {
    std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSpiralData(3, 100, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> inputandoutput =
            spiltToInputAndOutputs(data);
    auto inputbatch = inputandoutput.first;
    mynn::Mat input(inputbatch);
    
    DenseLayer L1 {2, 64, 0, 0, 5e-4, 5e-4};
    
    std::cout << "Benchmarking Dense Layer Forward Pass..." << std::endl;
    std::cout << "Input size: " << input.size().rows << "x" << input.size().cols << std::endl;
    std::cout << "Layer size: " << L1.getWeights().size().rows << "x" << L1.getWeights().size().cols << std::endl;
    
    // Benchmark standard implementation
    auto result_standard = Benchmark::run([&]() {
        mynn::Mat result = input.multiply(L1.getWeights().transpose()) + L1.getBias();
        // Prevent compiler from optimizing away the computation
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    // Benchmark fused implementation
    auto result_fused = Benchmark::run([&]() {
        mynn::Mat result = mynn::fast::dense_forward(input, L1.getWeights(), L1.getBias());
        // Prevent compiler from optimizing away the computation
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    Benchmark::print_result("Standard Implementation", result_standard);
    Benchmark::print_result("Fused Implementation", result_fused);
    Benchmark::compare_results(result_standard, result_fused);


    // functionality checks
    mynn::Mat result1 = input.multiply(L1.getWeights().transpose()) + L1.getBias();
    mynn::Mat result2 = mynn::fast::dense_forward(input, L1.getWeights(), L1.getBias());

    // Check if identical
    double max_diff = 0.0;
    for (int i = 0; i < result1.size().rows; ++i) {
        for (int j = 0; j < result1.size().cols; ++j) {
        double diff = std::abs(result1(i,j) - result2(i,j));
        max_diff = std::max(max_diff, diff);
        }
    }

    std::cout << "Max difference: " << max_diff << std::endl;
}

void benchmarkActivationForward() {
    // hi
    std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSpiralData(3, 100, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> inputandoutput =
            spiltToInputAndOutputs(data);
    auto inputbatch = inputandoutput.first;
    mynn::Mat input(inputbatch);
    
    DenseLayer L1 {2, 64, 0, 0, 5e-4, 5e-4};
    ReLU activation1;

    
    std::cout << "Benchmarking activation Layer Forward Pass..." << std::endl;
    std::cout << "Input size: " << input.size().rows << "x" << input.size().cols << std::endl;
    std::cout << "Layer size: " << L1.getWeights().size().rows << "x" << L1.getWeights().size().cols << std::endl;

    // Benchmark standard implementation
    auto result_standard = Benchmark::run([&]() {
        // mynn::Mat result = input.multiply(L1.getWeights().transpose()) + L1.getBias();
        // Prevent compiler from optimizing away the computation
        mynn::Mat result = activation1.run(input);
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    // Benchmark fused implementation
    auto result_fused = Benchmark::run([&]() {
        mynn::Mat result = mynn::fast::relu_forward(input);
        // Prevent compiler from optimizing away the computation
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    Benchmark::print_result("Standard Implementation", result_standard);
    Benchmark::print_result("raw access", result_fused);
    Benchmark::compare_results(result_standard, result_fused);


    // functionality checks
    mynn::Mat result1 = activation1.run(input);
    mynn::Mat result2 = mynn::fast::relu_forward(input);

    // Check if identical
    double max_diff = 0.0;
    for (int i = 0; i < result1.size().rows; ++i) {
        for (int j = 0; j < result1.size().cols; ++j) {
        double diff = std::abs(result1(i,j) - result2(i,j));
        max_diff += std::max(max_diff, diff);
        }
    }

    std::cout << "Max difference: " << max_diff << std::endl;

}

void benchmarkDropForward() {
std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSpiralData(3, 100, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> inputandoutput =
            spiltToInputAndOutputs(data);
    auto inputbatch = inputandoutput.first;
    mynn::Mat input(inputbatch);
    
    DenseLayer L1 {2, 64, 0, 0, 5e-4, 5e-4};
    ReLU activation1;
    
    
    double drop_rate_ = 0.1;
    double success_rate_ = 1 - drop_rate_; 
    DropoutLayer D1 {drop_rate_};
    
    std::random_device rd;
    // unsigned int seed = 234;
    unsigned int seed = rd();
    auto binary_mask_ = mynn::Mat(math::random::GetBernoulliSamples(success_rate_, seed, input.size().rows, input.size().cols)) * (1/success_rate_);

    
    std::cout << "Benchmarking activation Layer Forward Pass..." << std::endl;
    std::cout << "Input size: " << input.size().rows << "x" << input.size().cols << std::endl;
    std::cout << "Layer size: " << L1.getWeights().size().rows << "x" << L1.getWeights().size().cols << std::endl;

    // Benchmark standard implementation
    auto result_standard = Benchmark::run([&]() {
        // mynn::Mat result = input.multiply(L1.getWeights().transpose()) + L1.getBias();
        // Prevent compiler from optimizing away the computation
        // auto result = D1.forward(input);

        mynn::Mat result = input.multiplyElementWise(binary_mask_);

        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    // Benchmark fused implementation
    auto result_fused = Benchmark::run([&]() {
        mynn::Mat result = mynn::fast::drop_forward(input, binary_mask_);
        // Prevent compiler from optimizing away the computation
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    Benchmark::print_result("Standard Implementation", result_standard);
    Benchmark::print_result("raw access", result_fused);
    Benchmark::compare_results(result_standard, result_fused);


    // functionality checks
    mynn::Mat result1 = input.multiplyElementWise(binary_mask_);
    mynn::Mat result2 = mynn::fast::drop_forward(input, binary_mask_);

    // Check if identical
    double max_diff = 0.0;
    for (int i = 0; i < result1.size().rows; ++i) {
        for (int j = 0; j < result1.size().cols; ++j) {
        double diff = std::abs(result1(i,j) - result2(i,j));
        max_diff += std::max(max_diff, diff);
        }
    }

    std::cout << "Max difference: " << max_diff << std::endl;
}

void benchmarkSoftmaxForward() {
    std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSpiralData(3, 100, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> inputandoutput =
            spiltToInputAndOutputs(data);
    auto inputbatch = inputandoutput.first;
    mynn::Mat input(inputbatch);
    Softmax a1;

    auto result_standard = Benchmark::run([&]() {
        mynn::Mat result = a1.forward(input);
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    // Benchmark fused implementation
    auto result_fused = Benchmark::run([&]() {
        mynn::Mat result = mynn::fast::softmax_forward(input);
        // Prevent compiler from optimizing away the computation
        volatile double dummy = result(0, 0);
        (void)dummy;
    }, 10, 100);

    Benchmark::print_result("Standard Implementation", result_standard);
    Benchmark::print_result("raw access", result_fused);
    Benchmark::compare_results(result_standard, result_fused);


    // functionality checks
    mynn::Mat result1 = a1.forward(input);
    mynn::Mat result2 = mynn::fast::softmax_forward(input);

    // Check if identical
    double max_diff = 0.0;
    for (int i = 0; i < result1.size().rows; ++i) {
        for (int j = 0; j < result1.size().cols; ++j) {
        double diff = std::abs(result1(i,j) - result2(i,j));
        max_diff += std::max(max_diff, diff);
        }
    }
}

int main() {
    // benchmark_access_patterns();
    // benchmarkDenseLayerForward();
    // benchmarkActivationForward();
    // benchmarkDropForward();
    benchmarkSoftmaxForward();
    return 0;
}