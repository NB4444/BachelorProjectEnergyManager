#pragma once

#include "Testing/Tests/Test.hpp"

namespace Testing {
	class TestResults;

	namespace Tests {
		__global__ void vectorAdd(const int* A, const int* B, int* C, int N);

		__global__ void vectorSubtract(const int* A, const int* B, int* C, int N);

		class VectorAddSubtractTest : public Test {
			int computeCount_;

			void doPass(cudaStream_t stream) const;

			TestResults onRun() override;

		public:
			VectorAddSubtractTest(const int& computeCount);
		};
	}
}