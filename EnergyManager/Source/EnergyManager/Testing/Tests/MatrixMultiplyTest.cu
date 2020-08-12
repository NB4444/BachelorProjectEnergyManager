#include "./MatrixMultiplyTest.hpp"

#include "EnergyManager/Profiling/GPUMonitor.hpp"
#include "EnergyManager/Testing/TestResults.hpp"

namespace EnergyManager {
	namespace Testing {
		namespace Tests {
			void MatrixMultiplyTest::constantInit(float* data, int size, float val) {
				for(int i = 0; i < size; ++i) {
					data[i] = val;
				}
			}

			std::map<std::string, std::string> MatrixMultiplyTest::onRun() {
				printf("[Matrix Multiply Using CUDA] - Starting...\n");

				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaSetDevice(gpu_.getID()));

				int block_size = 32;

				// Set matrix dimensions
				dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
				dimsA.x = matrixAWidth_;
				dimsA.y = matrixAHeight_;
				dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
				dimsB.x = matrixBWidth_;
				dimsB.y = matrixBHeight_;
				if(dimsA.x != dimsB.y) {
					printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
					exit(EXIT_FAILURE);
				}
				if(dimsA.x % 32 != 0 || dimsA.y % 32 != 0 || dimsB.x % 32 != 0 || dimsB.y % 32 != 0) {
					Utility::Logging::logError("Dimensions must be a multiple of 32", __FILE__, __LINE__);
					exit(EXIT_FAILURE);
				}

				printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

				// Allocate host memory for matrices A and B
				unsigned int size_A = dimsA.x * dimsA.y;
				unsigned int mem_size_A = sizeof(float) * size_A;
				auto* h_A = reinterpret_cast<float*>(malloc(mem_size_A));
				unsigned int size_B = dimsB.x * dimsB.y;
				unsigned int mem_size_B = sizeof(float) * size_B;
				auto* h_B = reinterpret_cast<float*>(malloc(mem_size_B));

				// Initialize host memory
				const float valB = 0.01f;
				constantInit(h_A, size_A, 1.0f);
				constantInit(h_B, size_B, valB);

				// Allocate device memory
				float *d_A, *d_B, *d_C;

				// Allocate host matrix C
				dim3 dimsC(dimsB.x, dimsA.y, 1);
				unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
				float* h_C = reinterpret_cast<float*>(malloc(mem_size_C));
				if(h_C == NULL) {
					fprintf(stderr, "Failed to allocate host matrix C!\n");
					exit(EXIT_FAILURE);
				}

				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));

				// copy host memory to device
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

				// Setup execution parameters
				dim3 threads(block_size, block_size);
				dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

				// Create and start timer
				printf("Computing result using CUDA Kernel...\n");

				// Performs warmup operation using matrixMul CUDA kernel
				if(block_size == 16) {
					MatrixMulCUDA<16><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
				} else {
					MatrixMulCUDA<32><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
				}

				printf("done\n");

				cudaDeviceSynchronize();

				// Allocate CUDA events that we'll use for timing
				cudaEvent_t start;
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaEventCreate(&start));
				cudaEvent_t stop;
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaEventCreate(&stop));

				// Record the start event
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaEventRecord(start, NULL));

				// Execute the kernel
				int nIter = 300;

				for(int j = 0; j < nIter; j++) {
					if(block_size == 16) {
						MatrixMulCUDA<16><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
					} else {
						MatrixMulCUDA<32><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
					}
				}

				// Record the stop event
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaEventRecord(stop, NULL));

				// Wait for the stop event to complete
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaEventSynchronize(stop));

				// Calculate elapsed time
				float msecTotal = 0.0f;
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaEventElapsedTime(&msecTotal, start, stop));

				// Compute and print the performance
				float msecPerMatrixMul = msecTotal / nIter;
				double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
				double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
				printf(
					"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
					" WorkgroupSize= %u threads/block\n",
					gigaFlops,
					msecPerMatrixMul,
					flopsPerMatrixMul,
					threads.x * threads.y);

				// Copy result from device to host
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

				printf("Checking computed result for correctness: ");
				bool correct = true;

				// test relative error by the formula
				//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
				double eps = 1.e-6; // machine zero

				for(int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
					double abs_err = fabs(h_C[i] - (dimsA.x * valB));
					double dot_length = dimsA.x;
					double abs_val = fabs(h_C[i]);
					double rel_err = abs_err / abs_val / dot_length;

					if(rel_err > eps) {
						printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
						correct = false;
					}
				}

				printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

				// Clean up memory
				free(h_A);
				free(h_B);
				free(h_C);
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaFree(d_A));
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaFree(d_B));
				ENERGY_MANAGER_HARDWARE_GPU_HANDLE_API_CALL(cudaFree(d_C));

				printf(
					"\nNOTE: The CUDA Samples are not meant for performance"
					"measurements. Results may vary when GPU Boost is enabled.\n");

				int result;
				if(correct) {
					result = EXIT_SUCCESS;
				} else {
					result = EXIT_FAILURE;
				}

				return { { "matrixResult", std::to_string(result) } };
			}

			MatrixMultiplyTest::MatrixMultiplyTest(const Hardware::GPU& gpu, const size_t& matrixAWidth, const size_t& matrixAHeight, const size_t& matrixBWidth, const size_t& matrixBHeight)
				: Test("MatrixMultiplyTest", { { std::shared_ptr<Profiling::Monitor>(new Profiling::GPUMonitor(gpu)), std::chrono::seconds(1) } })
				, gpu_(gpu)
				, matrixAWidth_(matrixAWidth)
				, matrixAHeight_(matrixAHeight)
				, matrixBWidth_(matrixBWidth)
				, matrixBHeight_(matrixBHeight) {
			}
		}
	}
}