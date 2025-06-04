#pragma once

#include <random>

namespace math {
    namespace random {

        // uniform in distribution
        inline int GetRandomInterger(int min, int max) {
            static std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<> dist(min, max);
            return dist(gen);
        }

        inline std::vector<int> GetRandomIntVector(int min, int max, int size) {
            std::vector<int> result;
            result.reserve(size);
            for(int i = 0; i < size; ++i)
                result.push_back(GetRandomInterger(min, max));
            return result;
        }

        // normal_distribution
        inline double GetRandomNormal(double mu, double sigma) {
            static std::mt19937 gen(std::random_device{}());
            auto dist = std::normal_distribution<double>{mu, sigma};
            auto value = dist(gen);
            return value;
        }

        inline double GetRandomNormalWithSeed(double mu, double sigma) {
            auto const seed = 123;
            static std::mt19937 urbg(seed);
            auto dist = std::normal_distribution<double>{mu, sigma};
            auto value = dist(urbg);
            return value;
        }

        inline std::vector<double> GetRandomNormVector(double mu, double sigma, int size) {
            std::vector<double> result;
            result.reserve(size);
            for(int i = 0; i < size; ++i)
                result.push_back(GetRandomNormalWithSeed(mu, sigma));
            return result;
        }
    }
}