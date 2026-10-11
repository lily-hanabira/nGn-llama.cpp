// nGn: holds N MiB of device memory until a stop file appears, to emulate a smaller GPU. With the driver's sysmem
// fallback set to "prefer no sysmem fallback", whatever does not fit then fails instead of spilling to shared memory.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: vramhog <MiB> <stopfile>\n"); return 2; }
    const size_t mib = (size_t) atoll(argv[1]);
    std::vector<void *> blocks;
    size_t got = 0;
    for (; got < mib; got += 64) {           // 64 MiB blocks, touched so they are really resident
        void * p = nullptr;
        if (cudaMalloc(&p, 64ull << 20) != cudaSuccess) { fprintf(stderr, "allocation failed at %zu MiB\n", got); break; }
        cudaMemset(p, 0, 64ull << 20);
        blocks.push_back(p);
    }
    cudaDeviceSynchronize();
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    printf("holding %zu MiB, device free %zu MiB of %zu\n", got, fr >> 20, tot >> 20); fflush(stdout);
    while (!std::ifstream(argv[2]).good()) std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (void * p : blocks) cudaFree(p);
    return 0;
}
