#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace Utility {
	namespace Text {
		static std::string join(const std::vector<std::string>& value, const std::string& delimiter) {
			return std::accumulate(value.begin(), value.end(), std::string(), [&](const std::string& left, const std::string& right) -> std::string {
				return left + (left.length() > 0 ? delimiter : "") + right;
			});
		}
	}
}