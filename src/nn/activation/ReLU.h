#include <vector>
#include <cassert>

#include "Mat.h"


class ReLU {
public:
    ReLU();
    mynn::Mat run(mynn::Mat input);
    mynn::Mat backward(mynn::Mat dl_da, mynn::Mat input);
};