// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/gemm_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::PartialShape> dimensions = {
    {1, 3, 16, 16}
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8()
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GemmTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(dimensions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues)),
    GemmTransformation::getTestCaseName);
}  // namespace




