#pragma once

#include <random>
#include <tuple>

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

    namespace dataset {
        constexpr double PI = 3.141592;
        inline std::vector<std::pair<double, double>> GenerateLinearData(
            int num_classes,
            int points_per_class,
            double noise = 0.2) {

                double m = 2.0;
                double c = 4.0;
                std::vector<std::pair<double, double>> result;
                for(int i = 0; i < num_classes*points_per_class; ++i) {
                    double x = i;
                    double y = m * x + c;
                    result.push_back({x, y});
                }
                
                return result;
        }
        inline std::vector<std::vector<double>> GenerateLinearDataV(
            int num_classes,
            int points_per_class,
            double noise = 0.2) {
                std::vector<std::vector<double>> result;
                std::vector<std::pair<double, double>> res = GenerateLinearData(num_classes, points_per_class, noise);
                result.reserve(res.size());
                // for(auto const& [x, y]: res) {
                for(auto const& item: res) {
                    result.push_back({item.first, item.second});
                }
                return result;
            }
        
        inline std::vector<std::tuple<double, double, int>> GenerateSineDataV(
            int num_classes,
            int points_per_class,
            double noise = 0.2
        ) {
            //
            double one_degree = PI/180;
            std::vector<std::tuple<double, double, int>> result;
            for(int i = 0 ; i < num_classes; ++i) {
                for (int j = 0; j < points_per_class; ++j) {
                    double x = i * 2 * PI + j * one_degree;
                    double y = std::sin(x);
                    // noise = random::GetRandomNormalWithSeed(0, noise); // no idea why this isnt working
                    noise = random::GetRandomNormalWithSeed(0, 0.2);
                    y += noise;

                    // result.push_back({x, y, i});
                    result.push_back(std::make_tuple(x, y, i));
                    // no idea about class,
                    // right now let it be like range of x values, 
                }
            }
            return result;
        }

        // inline std::vector<std::tuple<double, double, int>> GenerateSineDataV(
        //     int num_classes,
        //     int points_per_class,
        //     double noise = 0.2
        // ) {
        // }
    }

    // namespace dataset {
    //     constexpr double M_PI = 3.1415;
    //     std::vector<Point> GenerateSpiralData(int num_classes, int points_per_class, double noise = 0.2) {
    //         std::vector<Point> data;
    //         data.reserve(num_classes * points_per_class);

    //         std::mt19937 gen(std::random_device{}());
    //         std::normal_distribution<double> noise_dist(0.0, noise);

    //         double angle_step = 2 * M_PI / num_classes;

    //         for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
    //             for (int i = 0; i < points_per_class; ++i) {
    //                 double ratio = static_cast<double>(i) / points_per_class;
    //                 double r = ratio;
    //                 double theta = class_idx * angle_step + ratio * 4 * M_PI;  // 2 spiral turns

    //                 double x = r * std::cos(theta) + noise_dist(gen);
    //                 double y = r * std::sin(theta) + noise_dist(gen);

    //                 data.push_back({x, y, class_idx});
    //             }
    //         }

    //         return data;
    //     }
    // }
}