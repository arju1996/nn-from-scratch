#include "ReLU.h"

ReLU::ReLU() {

}

mynn::Mat ReLU::run(mynn::Mat input) {
    mynn::Mat output(input.size().rows, input.size().cols);
    mynn::Size inputsize = input.size();
    for(int i = 0; i<inputsize.rows; ++i) {
        for(int j = 0; j<inputsize.cols; ++j) {
            output(i, j) = std::max(0.0, input(i, j));
        }
    }
    return output;
}

mynn::Mat ReLU::backward(mynn::Mat dl_da, mynn::Mat input) {
    // logic
    // find dl_dz1
    // dl_dz1 = dl_da1 * da1_dz1
    // da1_dz1 = 1 when z1 is positive else 0


    // code
    mynn::Mat dl_dz = dl_da;
    // mynn::Mat input;
    // assume input has values
    mynn::Size inputsize = input.size();
    for(int i = 0; i<inputsize.rows; ++i) {
        for(int j = 0; j<inputsize.cols; ++j) {
            if(input(i, j) <= 0)
                dl_dz(i, j) = 0;
        }
    }

    // loop in z1, z2, z3 and update dl_dz
    return dl_dz;
}
