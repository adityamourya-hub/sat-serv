/**
 * SGP4 GPU Kernel Interface Header
 *
 * This header defines the CUDA kernel interface for SGP4 propagation.
 * Implementation is in sgp4_gpu.cu (compiled with nvcc)
 *
 * Build: nvcc -c sgp4_gpu.cu -o sgp4_gpu.o
 */

#ifndef SGP4_GPU_H
#define SGP4_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

// CUDA types (for non-CUDA compilation)
#ifndef __CUDACC__
typedef void* cudaStream_t;
#endif

// TLE data structure (GPU-compatible)
typedef struct {
    int sat_num;
    double epoch_jd;
    double inclination;
    double raan;
    double eccentricity;
    double arg_perigee;
    double mean_anomaly;
    double mean_motion;
    double bstar;
    double padding[2];  // Align to 64 bytes
} TLEGPU;

// State output structure
typedef struct {
    double pos[3];  // km
    double vel[3];  // km/s
} StateGPU;

// Kernel launchers (called from Go via cgo)

/**
 * Launch SGP4 propagation kernel
 * @param tle_data: Device pointer to TLE array
 * @param states_out: Device pointer to output states
 * @param N: Number of objects
 * @param target_jd: Target Julian date for propagation
 * @param threads_per_block: CUDA block size
 * @param stream: CUDA stream for async execution
 */
void launch_sgp4_kernel(
    const TLEGPU* tle_data,
    StateGPU* states_out,
    int N,
    double target_jd,
    int threads_per_block,
    cudaStream_t stream
);

/**
 * Launch LBVH construction kernel
 * @param positions: Device pointer to [x,y,z] array
 * @param N: Number of objects
 * @param tree_out: Output LBVH tree pointer
 * @param stream: CUDA stream
 */
void launch_lbvh_build(
    const double* positions,
    int N,
    void** tree_out,
    cudaStream_t stream
);

/**
 * Collision pair structure
 */
typedef struct {
    int object_id1;
    int object_id2;
    double distance;
    double tca_jd;
} CollisionPair;

/**
 * Launch proximity query kernel
 * @param lbvh_tree: LBVH tree from launch_lbvh_build
 * @param threshold: Distance threshold in km
 * @param pairs_out: Output buffer for collision pairs
 * @param pair_count: Atomic counter for number of pairs found
 * @param stream: CUDA stream
 */
void launch_proximity_query(
    void* lbvh_tree,
    double threshold,
    CollisionPair* pairs_out,
    int* pair_count,
    cudaStream_t stream
);

/**
 * Conjunction geometry for Pc calculation
 */
typedef struct {
    double rel_pos[3];
    double rel_vel[3];
    double hbr;
    double primary_cov[6];    // Upper triangle of 3x3 covariance
    double secondary_cov[6];  // Upper triangle of 3x3 covariance
    double padding[2];        // Align to 64 bytes
} ConjunctionGeomGPU;

/**
 * Launch Pc computation kernel (Foster-Alfano method)
 * @param geoms: Device pointer to conjunction array
 * @param pc_out: Output probabilities
 * @param N: Number of conjunctions
 * @param threads_per_block: CUDA block size
 * @param stream: CUDA stream
 */
void launch_pc_kernel(
    const ConjunctionGeomGPU* geoms,
    double* pc_out,
    int N,
    int threads_per_block,
    cudaStream_t stream
);

// CUDA runtime wrappers (for Go)
#define cudaSuccess 0

typedef int cudaError_t;
typedef struct cudaDeviceProp cudaDeviceProp;

cudaError_t cudaSetDevice(int device);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostGetDevicePointer(void** device_ptr, void* host_ptr, unsigned int flags);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);

// Memory flags
#define cudaHostAllocDefault 0x00
#define cudaHostAllocMapped 0x02

#ifdef __cplusplus
}
#endif

#endif // SGP4_GPU_H
