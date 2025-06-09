#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


TEST(Env, shm_utility) {
    const int N = 8;
    float src[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float w[N]   = {1, 1, 1, 1, 1, 1, 1, 1};
    float eps = 1e-5;
    float out[N];

}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
