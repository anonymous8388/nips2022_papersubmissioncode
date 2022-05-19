// Copyright 2016 SoundAI Technology Co., Ltd. (author: Charles Wang)
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_STATISTICSPOOLING_H
#define LAYER_STATISTICSPOOLING_H

#include "layer.h"

namespace ncnn {

class StatisticsPooling : public Layer
{
public:
    StatisticsPooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt);

public:
    // param
    int include_stddev;
};

} // namespace ncnn

#endif // LAYER_STATISTICSPOOLING_H