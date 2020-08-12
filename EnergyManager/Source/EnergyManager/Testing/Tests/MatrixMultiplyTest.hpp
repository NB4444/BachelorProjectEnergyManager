#pragma once

#include "EnergyManager/Testing/Tests/Test.hpp"
#include "EnergyManager/Hardware/GPU.hpp"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

/**
 * The testing code was sampled from the `matrixMul` sample in CUDA 10.1.
 */

namespace EnergyManager {
	namespace Testing {
		class TestResults;

		namespace Tests {
			/**
			 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
			 * wA is A's width and wB is B's width
			 */
			template<int BLOCK_SIZE>
			__global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
				// Block index
				int bx = blockIdx.x;
				int by = blockIdx.y;

				// Thread index
				int tx = threadIdx.x;
				int ty = threadIdx.y;

				// Index of the first sub-matrix of A processed by the block
				int aBegin = wA * BLOCK_SIZE * by;

				// Index of the last sub-matrix of A processed by the block
				int aEnd = aBegin + wA - 1;

				// Step size used to iterate through the sub-matrices of A
				int aStep = BLOCK_SIZE;

				// Index of the first sub-matrix of B processed by the block
				int bBegin = BLOCK_SIZE * bx;

				// Step size used to iterate through the sub-matrices of B
				int bStep = BLOCK_SIZE * wB;

				// Csub is used to store the element of the block sub-matrix
				// that is computed by the thread
				float Csub = 0;

				// Loop over all the sub-matrices of A and B
				// required to compute the block sub-matrix
				for(int a = aBegin, b = bBegin;
					a <= aEnd;
					a += aStep, b += bStep) {
					// Declaration of the shared memory array As used to
					// store the sub-matrix of A
					__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

					// Declaration of the shared memory array Bs used to
					// store the sub-matrix of B
					__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

					// Load the matrices from device memory
					// to shared memory; each thread loads
					// one element of each matrix
					As[ty][tx] = A[a + wA * ty + tx];
					Bs[ty][tx] = B[b + wB * ty + tx];

					// Synchronize to make sure the matrices are loaded
					__syncthreads();

					// Multiply the two matrices together;
					// each thread computes one element
					// of the block sub-matrix
#pragma unroll
					for(int k = 0; k < BLOCK_SIZE; ++k) {
						Csub += As[ty][k] * Bs[k][tx];
					}

					// Synchronize to make sure that the preceding
					// computation is done before loading two new
					// sub-matrices of A and B in the next iteration
					__syncthreads();
				}

				// Write the block sub-matrix to device memory;
				// each thread writes one element
				int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
				C[c + wB * ty + tx] = Csub;
			}

			class MatrixMultiplyTest : public Test {
				const Hardware::GPU& gpu_;

				size_t matrixAWidth_;

				size_t matrixAHeight_;

				size_t matrixBWidth_;

				size_t matrixBHeight_;

				static void constantInit(float* data, int size, float val);

				std::map<std::string, std::string> onRun() override;

			public:
				MatrixMultiplyTest(const Hardware::GPU& gpu, const size_t& matrixAWidth, const size_t& matrixAHeight, const size_t& matrixBWidth, const size_t& matrixBHeight);
			};
		}
	}
}