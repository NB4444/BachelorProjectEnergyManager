#pragma once

#include <chrono>

const auto energySavingInterval = std::chrono::milliseconds(10);

const auto halfingPeriod = 5 * energySavingInterval;

const auto doublingPeriod = 5 * energySavingInterval;