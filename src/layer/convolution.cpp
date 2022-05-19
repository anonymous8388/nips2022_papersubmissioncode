// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolution.h"

#include "layer_type.h"

#include "fused_activation.h"

#define E 6
#define E_pow_num 64 // 2^6

namespace ncnn {

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);

    record1 = Mat();
    record2 = Mat();
    record3 = Mat();
    record4 = Mat();
    all_select_norms = Mat();
    top_E_indices = Mat();
    top_E_w_vals = Mat();

    exact_compute = true;
    call_count = 0;
    last_time_sparsity = -1;

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    return 0;
}

int Convolution::load_model(const ModelBin& mb)
{
    if (dynamic_weight)
        return 0;

    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

#if NCNN_INT8
    if (int8_scale_term)
    {
        weight_data_int8_scales = mb.load(num_output, 1);
        bottom_blob_int8_scales = mb.load(1, 1);
    }

    if (int8_scale_term > 100)
    {
        top_blob_int8_scales = mb.load(1, 1);
    }
#endif // NCNN_INT8

    return 0;
}

int Convolution::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

#if NCNN_INT8
    // runtime quantize the weight data
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        const int maxk = kernel_w * kernel_h;
        const int num_input = weight_data_size / num_output / maxk;

        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        Mat weight_data_int8;

        Option opt_q = opt;
        opt_q.blob_allocator = weight_data.allocator;
        opt_q.use_packing_layout = false;
        quantize_to_int8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
        if (weight_data_int8.empty())
            return -100;

        weight_data = weight_data_int8.reshape(weight_data_size);
    }
#endif // NCNN_INT8

    return 0;
}


static int mlsys_convolution(const Mat& in_x, Mat& out_y, const Mat& weight_data, const Mat& bias_data,
                       int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                       int activation_type, const Mat& activation_params, const Option& opt, Mat& last_x, Mat& last_y, Mat& w_norm2)
{
    //    fprintf(stderr, "卷卷卷@@raw conv, activation type is %d\n", activation_type);
    const int w = in_x.w;
    const int inch = in_x.c;

    const int outw = out_y.w;
    const int outh = out_y.h;
    const int outch = out_y.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    if (last_x.total() <= 0){
        //        int less_0_count = 0;
        w_norm2.create(outch);

        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
        }

        /**
         * exact compute
         */
        last_x.clone_from(in_x);
        //        last_x.create(out_y.w, out_y.h, kernel_w*kernel_h, in_x.c);
//        fprintf(stderr, "%d ", outch);
        last_y.clone_from(out_y);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outch; k++)
                {
                    float* outptr = out_y.channel(k);
                    outptr += i * outw;

                    float* outptr_last_y = last_y.channel(k);
                    outptr_last_y += i * outw;

                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    // 某层有64个卷积核，kptr即为64个卷积核之一
                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = in_x.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }

                        kptr += maxk;
                    }
                    if (bias_term)
                        outptr_last_y[j] = y_kij - bias_data[k];
                    else
                        outptr_last_y[j] = y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    //                    if (outptr_last_y[j] <= opt.lower)
                    //                        less_0_count += 1;
                }
            }
        }
        //        fprintf(stderr, "less 0 count = %d\n",less_0_count);
    }else{
        float reduced_count=0;
        float total_count = 0;
        float max_reduce_count;
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                /**
                 * compute dx_norm = || x_{ij}^{t} - x_{ij}^{t-1} ||
                 */
                float dx2_sum = 0.0;
                for (int q = 0; q < inch; q++)
                {
                    const Mat& m = in_x.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++)
                    {
                        float val = sptr[space_ofs[w_i]];
                        float val_last_x = sptr_last_x[space_ofs[w_i]];
                        dx2_sum += (val - val_last_x) * (val - val_last_x);
                    }
                }

                float dx_norm = sqrt(dx2_sum);  // 1.2%的开销
                                               //                float dx_norm = 0.0;              // 15773

                for (int k = 0; k < outch; k++)
                {
                    //                    fprintf(stderr, "debug 2\n");
                    float* outptr = out_y.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    /**
                     * get w_norm = || w_k ||
                     * if (\bar{y[ijk]} + dx_norm * w_norm <= - bias_data[k]) // reduce computation
                     * {
                     *      update \bar{y[ijk]} = \bar{y[ijk]} + dx_norm * w_norm
                     * }
                     * else // exact compute
                     */
                    const float* w_norm2_ptr = (const float*)w_norm2.data;
                    float norm_norm = w_norm2_ptr[k] * dx_norm;
                    float* out_bar_ptr = last_y.channel(k);

                    out_bar_ptr += i * outw;

                    total_count+= 1;
//                    total_count += 2*inch*maxk;
                    if (out_bar_ptr[j] + norm_norm <= -y_kij){
                        outptr[j] = 0;
                        reduced_count += 1;
//                        max_reduce_count += 1;
                        out_bar_ptr[j] += norm_norm;
                    }else{
                        out_bar_ptr[j] = -y_kij;
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat& m = in_x.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                            {
                                float val = sptr[space_ofs[w_i]]; // 20.72
                                float wt = kptr[w_i];
                                y_kij += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }

                        out_bar_ptr[j] += y_kij;
                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);

//                        if (y_kij <= 0)
//                            max_reduce_count += 1;
                    }
                }
            }
        }
//        fprintf(stderr, "%.4f/%.4f=%.4f\n", reduced_count, total_count, reduced_count/total_count);
//        fprintf(stderr, "%.4f/%.4f=%.4f <-\n", reduced_count, total_count, max_reduce_count/total_count);
//        fprintf(stderr, "%.1f <-\n", total_count);

//        fprintf(stderr, "%.2f\t",  reduced_count/total_count);
//        fprintf(stderr, "%.2f <-\n",  max_reduce_count/total_count);

        last_x.clone_from(in_x); //没有这行是15899，加了是16058
    }


    return 0;
}

inline void find_top_E(const float* w_arr, float* w_topE_indices_arr, float* w_topE_val_arr, int w_arr_len, float* all_select_norms, float w_full_2){
    /**
     * find top absolute largest E element in arr, indices stored to indices_arr
     * w_arr: len = w_arr_len
     * indices_arr: len = E
     * all_select_norms: len = 2^E
     * w_full_norm: 当前w_k的full元素的norm的      平方
     */
//    fprintf(stderr, "111\t");
    std::vector<std::pair<float, int> > w_ordered;

    w_ordered.reserve(w_arr_len);
    for(int i=0; i<w_arr_len; i++){
        w_ordered.push_back(std::make_pair(abs(w_arr[i]), i));
    }

    std::sort(w_ordered.begin(), w_ordered.end());

    for(int i=0; i<E; i++){
//    for(int i=w_arr_len-1; i>w_arr_len-1-E; i--){
        w_topE_val_arr[i] = w_ordered[w_arr_len-1-i].first;
        w_topE_indices_arr[i] = w_ordered[w_arr_len-1-i].second;
//        w_topE_val_arr[i] = w_ordered[i].first;
//        w_topE_indices_arr[i] = w_ordered[i].second;

//        fprintf(stderr, "%f, %d \t",  w_topE_val_arr[i],  int(w_topE_indices_arr[i]));

        //0.070557, 6 	0.077123, 23 	0.098039, 5 	0.100434, 3 	0.114497, 24 	0.148363, 12
    }
//    fprintf(stderr, "\n");
//    fprintf(stderr, "222\t");

    for(int i=0; i<E_pow_num; i++){
        float tobe_sub = 0.0;
        for (int norm_index=E-1; norm_index>=0; norm_index--){
            if ((i>>norm_index) & 1){   // 当前位=1
                tobe_sub += w_topE_val_arr[E-1-norm_index] * w_topE_val_arr[E-1-norm_index];
            }
        }

        all_select_norms[i] = sqrt(w_full_2 - tobe_sub);
    }
    for (int i=0; i<E; i++){
        w_topE_val_arr[i] =  w_arr[int(w_topE_indices_arr[i])];
    }

//    fprintf(stderr, "333\n");
    w_ordered.clear();
    /**
     * 把相对应位置的, select_norm，转化坐标，存到all_select_norm中
     */
}

static int mlsys_convolution_lower_top_E(const Mat& in_x, Mat& out_y, const Mat& weight_data, const Mat& bias_data,
                             int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                             int activation_type, const Mat& activation_params, const Option& opt, Mat& last_x, Mat& last_y, Mat& w_norm2
                                         ,Mat& all_select_norms, Mat& top_E_indices, Mat& top_E_w_vals, Mat& x_vector_diff)
{
    //    fprintf(stderr, "卷卷卷@@raw conv, activation type is %d\n", activation_type);
    const int w = in_x.w;
    const int inch = in_x.c;

    const int outw = out_y.w;
    const int outh = out_y.h;
    const int outch = out_y.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    float* all_select_norms_ptr = nullptr;
    float* top_E_indices_ptr = nullptr;
    float* top_E_w_vals_ptr = nullptr;
    float* x_vector_diff_ptr = nullptr;

    if (last_x.total() <= 0){
        w_norm2.create(outch);
        all_select_norms.create(outch*E_pow_num);
        top_E_indices.create(outch* E);
        top_E_w_vals.create(outch* E);
        x_vector_diff.create(inch*maxk);

        all_select_norms_ptr = (float*)all_select_norms.data;
        top_E_indices_ptr = (float*)top_E_indices.data;
        top_E_w_vals_ptr = (float*)top_E_w_vals.data;
//        x_vector_diff_ptr = (float*)x_vector_diff.data;

        /**
         * calculate w_norm2
         */

        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
//            fprintf(stderr, "%d\n", k);
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }

            kptr = (const float*)weight_data.data + maxk * inch * k;
            find_top_E(kptr,
                       top_E_indices_ptr + E*k,
                       top_E_w_vals_ptr + E*k,
                       maxk * inch,
                       all_select_norms_ptr+k*E_pow_num,
                       w_norm2_data[k]); // 传进去的是平方
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
        }
        /**
         * exact compute
         */
        last_x.clone_from(in_x);
        //        last_x.create(out_y.w, out_y.h, kernel_w*kernel_h, in_x.c);
        last_y.clone_from(out_y);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outch; k++)
                {
                    float* outptr = out_y.channel(k);
                    outptr += i * outw;

                    float* outptr_last_y = last_y.channel(k);
                    outptr_last_y += i * outw;

                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    // 某层有64个卷积核，kptr即为64个卷积核之一
                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = in_x.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }

                        kptr += maxk;
                    }
                    if (bias_term)
                        outptr_last_y[j] = y_kij - bias_data[k];
                    else
                        outptr_last_y[j] = y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    //                    if (outptr_last_y[j] <= opt.lower)
                    //                        less_0_count += 1;
                }
            }
        }
        //        fprintf(stderr, "less 0 count = %d\n",less_0_count);
    }else{;
        all_select_norms_ptr = (float*)all_select_norms.data;
        top_E_indices_ptr = (float*)top_E_indices.data;
        top_E_w_vals_ptr = (float*)top_E_w_vals.data;
        x_vector_diff_ptr = (float*)x_vector_diff.data;
        //        float reduced_count=0;
        //        float total_count = 0;
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                /**
                 * compute dx_norm = || x_{ij}^{t} - x_{ij}^{t-1} ||
                 */
                float dx2_sum = 0.0;
                for (int q = 0; q < inch; q++)
                {
                    const Mat& m = in_x.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++)
                    {
                        int q_i = q*maxk + w_i;
                        float val = sptr[space_ofs[w_i]];
                        float val_last_x = sptr_last_x[space_ofs[w_i]];
                        x_vector_diff_ptr[q_i] = val - val_last_x;
                        dx2_sum += x_vector_diff_ptr[q_i] * x_vector_diff_ptr[q_i];
                    }
                }

                float dx_norm = sqrt(dx2_sum);  // 1.2%的开销
                                               //                float dx_norm = 0.0;              // 15773

                all_select_norms_ptr = (float*)all_select_norms.data;
                top_E_indices_ptr = (float*)top_E_indices.data;
                top_E_w_vals_ptr = (float*)top_E_w_vals.data;
                for (int k = 0; k < outch; k++)
                {
                    //                    fprintf(stderr, "debug 2\n");
                    float* outptr = out_y.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    /**
                     * get w_norm = || w_k ||
                     * if (\bar{y[ijk]} + dx_norm * w_norm <= - bias_data[k]) // reduce computation
                     * {
                     *      update \bar{y[ijk]} = \bar{y[ijk]} + dx_norm * w_norm
                     * }
                     * else // exact compute
                     */
//                    const float* w_norm2_ptr = (const float*)w_norm2.data;
//                    float norm_norm = w_norm2_ptr[k] * dx_norm;

                    // 比较符号，并且取出相对应的w_norm
                    float diff_sign_sub = 0.0;
                    unsigned int select_norm_index = 0;
                    float temp_ii;
                    for (int ii=0; ii< E; ii++){
                        temp_ii = x_vector_diff_ptr[(int)(top_E_indices_ptr[ii])] * top_E_w_vals[ii];
                        select_norm_index = select_norm_index<<1;
                        if (temp_ii > 0){
                            select_norm_index |= 1;
                        }else if(temp_ii < 0){
//                            diff_sign_sub += temp_ii;
                        }
                    }

                    float* out_bar_ptr = last_y.channel(k);
                    out_bar_ptr += i * outw;


//                    out_bar_ptr[j] += dx_norm * all_select_norms_ptr[select_norm_index] + diff_sign_sub;
                    out_bar_ptr[j] += dx_norm * all_select_norms_ptr[select_norm_index];

//                    if (dx_norm * all_select_norms_ptr[select_norm_index] + diff_sign_sub > norm_norm)
//                        fprintf(stderr, "完了完了%.1f ", (dx_norm * all_select_norms_ptr[select_norm_index] + diff_sign_sub)/norm_norm);

                    //                    total_count += 1;
                    if (out_bar_ptr[j] + y_kij <= 0){
                        outptr[j] = 0;
                        //                        reduced_count += 1;
                        //                        max_reduce_count += 1;
                    }else{
                        out_bar_ptr[j] = -y_kij;
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat& m = in_x.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                            {
                                float val = sptr[space_ofs[w_i]]; // 20.72
                                float wt = kptr[w_i];
                                y_kij += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }

                        out_bar_ptr[j] += y_kij;
                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);

                        //                        if (y_kij <= 0)
                        //                            max_reduce_count += 1;
                    }

                    all_select_norms_ptr += E_pow_num;
                    top_E_indices_ptr += E;
                    top_E_w_vals_ptr += E;
                }
            }
        }
        //        fprintf(stderr, "%.4f/%.4f=%.4f\n", reduced_count, total_count, reduced_count/total_count);
        //        fprintf(stderr, "%.4f/%.4f=%.4f <-\n", max_reduce_count, total_count, max_reduce_count/total_count);

        //        fprintf(stderr, "%.2f\n",  reduced_count/total_count);
        //        fprintf(stderr, "%.2f <-\n",  max_reduce_count/total_count);
//                fprintf(stderr, "\n");

        last_x.clone_from(in_x); //没有这行是15899，加了是16058
    }


    return 0;
}


// 上面，左面，(t-1)全比较
static int temporal_spatial_convolution1(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data,
                                        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                                        int activation_type, const Mat& activation_params, const Option& opt, Mat& last_x, Mat& last_y, Mat& w_norm2,
                                        Mat& last_y_col, Mat& last_y_row)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    float reduce = 0;
    float total = 0;
    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    float norm_norm_row;    // i的norm norm
    float delta_x_row;

    float* last_y_row_ptr = nullptr;
    float* last_y_col_ptr = (float*) last_y_col.data;

    if (last_x.total() <= 0){
//        fprintf(stderr, "enter\n");
        w_norm2.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw);         // outw是h
        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
        }

        /**
         * exact compute
         */
        last_x.clone_from(bottom_blob);
        last_y.clone_from(top_blob);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;

                    float* outptr_last_y = last_y.channel(k);
                    outptr_last_y += i * outw;

                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    // 某层有64个卷积核，kptr即为64个卷积核之一
                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }

                        kptr += maxk;
                    }
                    if (bias_term)
                        outptr_last_y[j] = y_kij - bias_data[k];
                    else
                        outptr_last_y[j] = y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    //                    if (outptr_last_y[j] <= opt.lower)
                    //                        less_0_count += 1;
                }
            }
        }
        //        fprintf(stderr, "less 0 count = %d\n",less_0_count);
    }else{
        //        fprintf(stderr, "啊啊啊啊啊\n");
        float reduced_count = 0.0;
        float total_count = 0.0;
        float max_reduce_count = 0.0;
        float our_count = 0.0;
        float mlsys_count = 0.0;
        for (int i = 0; i < outh; i++)
        {
            last_y_row_ptr = (float*)last_y_row.data;
            for (int j = 0; j < outw; j++)
            {
//                delta_x_col = 0.0;
//                delta_x_row = 0.0;
//                if (j!=0){
//                /** 上一列的基准
//                 * calculate ||x(i, j) - x(i, j-1)||
//                 */
//                    for (int q = 0; q < inch; q++)
//                    {
//                        const Mat m = bottom_blob.channel(q);
//                        const float* sptr = m.row(i * stride_h) + j * stride_w;
//
//                        const float* prev_sptr = sptr - stride_w;
//
//                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
//                        {
//                            float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
//                            delta_x_col += delta * delta;
//                        }
//                    }
//                    delta_x_col = sqrt(delta_x_col);
//                }
//
//                if (i!=0){
//                /** 上一行的基准
//                 * calculate ||x(i, j) - x(i-1, j)||
//                 */
//                    for (int q = 0; q < inch; q++)
//                    {
//                        const Mat m = bottom_blob.channel(q);
//                        const float* sptr = m.row(i * stride_h) + j * stride_w;
//
//                        const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;
//
//                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
//                        {
//                            float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
//                            delta_x_row += delta * delta;
//                        }
//                    }
//                    delta_x_row = sqrt(delta_x_row);
//                }
//
//
//                /**
//                 * compute dx_norm = || x_{ij}^{t} - x_{ij}^{t-1} ||
//                 */
//
//                float dx2_sum = 0.0;
//                for (int q = 0; q < inch; q++)
//                {
//                    const Mat& m = bottom_blob.channel(q);
//                    const float* sptr = m.row(i * stride_h) + j * stride_w;
//
//                    const Mat& m_last_x = last_x.channel(q);
//                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;
//
//                    for (int w_i = 0; w_i < maxk; w_i++)
//                    {
//                        float val = sptr[space_ofs[w_i]];
//                        float val_last_x = sptr_last_x[space_ofs[w_i]];
//                        dx2_sum += (val - val_last_x) * (val - val_last_x);
//                    }
//                }
//                float dx_norm = sqrt(dx2_sum);

                delta_x_col = 0.0;
                delta_x_row = 0.0;
                float dx2_sum = 0.0;
                const float* prev_i_sptr = nullptr;
                const float* prev_j_sptr = nullptr;
                for (int q = 0; q < inch; q++)
                {
                    const Mat& m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;
//                    auto stride_delta_x = (unsigned char*)m.data + (size_t)m.w * i * stride_h * m.elemsize + j * stride_w;
//                    const float* sptr = (const float*)(stride_delta_x);

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;

                    if (i!=0){
                        prev_i_sptr = m.row((i-1) * stride_h) + j * stride_w;
//                        prev_i_sptr = (const float*)(stride_delta_x - (size_t)m.w * stride_h * m.elemsize);
                    }

                    if (j!=0){
                        prev_j_sptr = sptr - stride_w;
                    }

                    for (int w_i = 0; w_i < maxk; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = sptr_last_x[w_i_offset] - base;
                        dx2_sum += temporal_diff * temporal_diff;

                        if (i!=0){
                            float spatial_i_diff = prev_i_sptr[w_i_offset] - base;
                            delta_x_row += spatial_i_diff * spatial_i_diff;
                        }

                        if (j!=0){
                            float spatial_j_diff = prev_j_sptr[w_i_offset] - base;
                            delta_x_col += spatial_j_diff * spatial_j_diff;
                        }
                    }
                }
                if (i!=0){
                    delta_x_row = sqrt(delta_x_row);
                }

                if (j!=0){
                    delta_x_col = sqrt(delta_x_col);
                }
                float dx_norm = sqrt(dx2_sum);

                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;


                    const float* w_norm2_ptr = (const float*)w_norm2.data;
                    float* out_bar_ptr = last_y.channel(k);
                    out_bar_ptr += i * outw;

                    /**
                     * 注意our_bar_ptr是j，而last_y_col_ptr是k
                     */
                    float norm_norm = out_bar_ptr[j] + w_norm2_ptr[k] * dx_norm;
                    min_norm_norm = norm_norm;

                    if (j!=0){
                        norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2_ptr[k];
                        min_norm_norm = std::min(min_norm_norm, norm_norm_col);
                    }

                    if (i!=0){
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2_ptr[k];
                        min_norm_norm = std::min(min_norm_norm, norm_norm_row);
                    }

//                    total_count+=1;
                    if (min_norm_norm + y_kij <= 0){
//                        if (i!=0 && norm_norm_row==min_norm_norm)
//                            our_count += 1;
//                        else if (j!=0 && norm_norm_col==min_norm_norm)
//                            our_count += 1;
//                        else
//                            mlsys_count += 1;
                        last_y_col_ptr[k] = min_norm_norm;
                        last_y_row_ptr[k] = min_norm_norm;
                        out_bar_ptr[j] = min_norm_norm;
                        outptr[j] = 0;
//                        reduced_count += 1;
//                        max_reduce_count += 1;
                    }else{
//                        out_bar_ptr[j] = -y_kij;
//                        last_y_col_ptr[k] = -y_kij;
//                        last_y_row_ptr[k] = -y_kij;
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat& m = bottom_blob.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                            {
                                float val = sptr[space_ofs[w_i]]; // 20.72
                                float wt = kptr[w_i];
                                y_kij += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }

                        if (bias_term){
                            out_bar_ptr[j] = y_kij - bias_data[k];
                            last_y_col_ptr[k] = y_kij - bias_data[k];;
                            last_y_row_ptr[k] = y_kij - bias_data[k];;
                        }else{
                            out_bar_ptr[j] = y_kij;
                            last_y_col_ptr[k] = y_kij;
                            last_y_row_ptr[k] = y_kij;
                        }

                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);

//                        if (y_kij <= 0)
//                            max_reduce_count += 1;
                    }
                }
                last_y_row_ptr += outch;
            }
        }
        //        fprintf(stderr, "%.4f/%.4f=%.4f\n", reduced_count, total_count, reduced_count/total_count);
        //        fprintf(stderr, "%.4f/%.4f=%.4f <-\n", max_reduce_count, total_count, max_reduce_count/total_count);

//                fprintf(stderr, "%.2f\n",  reduced_count/total_count);
        //        fprintf(stderr, "%.2f <-\n",  max_reduce_count/total_count);

//        fprintf(stderr, "进入原有的:%.2f \t 进入我们的:%.2f\n",  mlsys_count/reduced_count, our_count/reduced_count);

        last_x.clone_from(bottom_blob); //没有这行是15899，加了是16058
    }


    return 0;
}



// 如果\delta x过大，就采用我们的方法
static int change_temporal_spatial_convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data,
                                         int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                                         int activation_type, const Mat& activation_params, const Option& opt, Mat& last_x, Mat& last_y, Mat& w_norm2,
                                         Mat& last_y_col, Mat& last_y_row, float& last_x_sparsity)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    float reduce = 0;
    float total = 0;

    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    float norm_norm_row;    // i的norm norm
    float delta_x_row;

    if (w_norm2.total() <= 0)
    {
        //        fprintf(stderr, "enter\n");
        w_norm2.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw); // outw是h
        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*)w_norm2.data;
        for (int k = 0; k < outch; k++)
        {
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++)
            {
                for (int w_i = 0; w_i < maxk; w_i++)
                {
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
        }
    }

    float* last_y_row_ptr = nullptr;
    float* last_y_col_ptr = (float*) last_y_col.data;
    if (false){
        /**
         * exact compute
        */
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    // 某层有64个卷积核，kptr即为64个卷积核之一

                    const float* kptr = (const float*)weight_data + maxk * inch * k;
                    total += 1;
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45
                        }

                        kptr += maxk;
                    }
                    if (y_kij < 0)
                        reduce += 1;

                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                }
            }
        }
    }
    else{

        for (int i = 0; i < outh; i++)
        {
            last_y_row_ptr = (float*)last_y_row.data;
            for (int j = 0; j < outw; j++)
            {
                delta_x_col = 0.0;
                delta_x_row = 0.0;
                if (j!=0){
            /** 上一列的基准
             * calculate ||x(i, j) - x(i, j-1)||
             */
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        const float* prev_sptr = sptr - stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                            delta_x_col += delta * delta;
                        }
                    }
                    delta_x_col = sqrt(delta_x_col);
                }

                if (i!=0){
            /** 上一行的基准
             * calculate ||x(i, j) - x(i-1, j)||
             */
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                            delta_x_row += delta * delta;
                        }
                    }
                    delta_x_row = sqrt(delta_x_row);
                }

                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;

                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    total += 1;

                    if (j!=0){
                        norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2[k];
                        min_norm_norm = norm_norm_col;
                    }

                    if (i!=0){
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2[k];
                        if(j!=0)
                            min_norm_norm = std::min(norm_norm_row, min_norm_norm);
                        else
                            min_norm_norm = norm_norm_row;
                    }


                    if ((i!=0||j!=0) && min_norm_norm + y_kij <= 0){
                        last_y_col_ptr[k] = min_norm_norm;
                        last_y_row_ptr[k] = min_norm_norm;
                        outptr[j] = 0;


                    }else{
                        last_y_col_ptr[k] = -y_kij;
                        last_y_row_ptr[k] = -y_kij;
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat m = bottom_blob.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++)
                            {
                                float val = sptr[space_ofs[w_i]];
                                float wt = kptr[w_i];
                                y_kij += val * wt;

                            }
                            kptr += maxk;
                        }


                        if (y_kij < 0)
                            reduce += 1;

                        last_y_col_ptr[k] += y_kij;
                        last_y_row_ptr[k] += y_kij;
                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    }
                }
                last_y_row_ptr += outch;
            }
        }
    }

    last_x_sparsity = reduce / total;

    return 0;
}


// 只比较左边和(t-1)
static int temporal_spatial_convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data,
                                        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                                        int activation_type, const Mat& activation_params, const Option& opt, Mat& last_x, Mat& last_y, Mat& w_norm2,
                                        Mat& last_y_col, Mat& last_y_row)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    float reduce = 0;
    float total = 0;
    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    float norm_norm_row;    // i的norm norm
    float delta_x_row;

    if (last_x.total() <= 0){
        //        fprintf(stderr, "enter\n");
        w_norm2.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw);         // outw是h
        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
        }

        /**
         * exact compute
         */
        last_x.clone_from(bottom_blob);
        last_y.clone_from(top_blob);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;

                    float* outptr_last_y = last_y.channel(k);
                    outptr_last_y += i * outw;

                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    // 某层有64个卷积核，kptr即为64个卷积核之一
                    const float* kptr = (const float*)weight_data + maxk * inch * k;

                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }

                        kptr += maxk;
                    }
                    if (bias_term)
                        outptr_last_y[j] = y_kij - bias_data[k];
                    else
                        outptr_last_y[j] = y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                }
            }
        }
        //        fprintf(stderr, "less 0 count = %d\n",less_0_count);
    }else{
        float* last_y_row_ptr = nullptr;
        float* last_y_col_ptr = (float*) last_y_col.data;

        float reduced_count = 0.0;
        float total_count = 0.0;
//        float max_reduce_count = 0.0;
//        float our_count = 0.0;
//        float mlsys_count = 0.0;
        float dx2_sum = 0.0;
        const float* prev_i_sptr = nullptr;
        const float* prev_j_sptr = nullptr;
        for (int i = 0; i < outh; i++)
        {
            last_y_row_ptr = (float*)last_y_row.data;
            for (int j = 0; j < outw; j++)
            {
                delta_x_col = 0.0;
                delta_x_row = 0.0;
                dx2_sum = 0.0;
                for (int q = 0; q < inch; q++)
                {
                    const Mat& m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;

                    if (i!=0){
                        prev_i_sptr = m.row((i-1) * stride_h) + j * stride_w;
                    }

                    if (j!=0){
                        prev_j_sptr = sptr - stride_w;
                    }

                    for (int w_i = 0; w_i < maxk; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = sptr_last_x[w_i_offset] - base;
                        dx2_sum += temporal_diff * temporal_diff;

                        if (i!=0){
                            float spatial_i_diff = prev_i_sptr[w_i_offset] - base;
                            delta_x_row += spatial_i_diff * spatial_i_diff;
                        }

                        if (j!=0){
                            float spatial_j_diff = prev_j_sptr[w_i_offset] - base;
                            delta_x_col += spatial_j_diff * spatial_j_diff;
                        }
                    }
                }
                if (i!=0){
                    delta_x_row = sqrt(delta_x_row);
                }

                if (j!=0){
                    delta_x_col = sqrt(delta_x_col);
                }
                float dx_norm = sqrt(dx2_sum);

                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;


                    const float* w_norm2_ptr = (const float*)w_norm2.data;
                    float* out_bar_ptr = last_y.channel(k);
                    out_bar_ptr += i * outw;

                    /**
                     * 注意our_bar_ptr是j，而last_y_col_ptr是k
                     */
                    float norm_norm = out_bar_ptr[j] + w_norm2_ptr[k] * dx_norm;
                    min_norm_norm = norm_norm;

                    if (j!=0){
                        norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2_ptr[k];
                        min_norm_norm = std::min(min_norm_norm, norm_norm_col);
                    }

                    if (i!=0){
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2_ptr[k];
                        min_norm_norm = std::min(min_norm_norm, norm_norm_row);
                    }

                    total_count += 1;
                    if (min_norm_norm + y_kij <= 0){
                        last_y_col_ptr[k] = min_norm_norm;
                        last_y_row_ptr[k] = min_norm_norm;
                        out_bar_ptr[j] = min_norm_norm;
                        outptr[j] = 0;

                        reduced_count += 1;
                    }else{
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat& m = bottom_blob.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                            {
                                float val = sptr[space_ofs[w_i]]; // 20.72
                                float wt = kptr[w_i];
                                y_kij += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }

                        if (bias_term){
                            out_bar_ptr[j] = y_kij - bias_data[k];
                            last_y_col_ptr[k] = y_kij - bias_data[k];;
                            last_y_row_ptr[k] = y_kij - bias_data[k];
                        }else{
                            out_bar_ptr[j] = y_kij;
                            last_y_col_ptr[k] = y_kij;
                            last_y_row_ptr[k] = y_kij;
                        }

                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    }
                }
                last_y_row_ptr += outch;
            }
        }
        //        fprintf(stderr, "%.4f/%.4f=%.4f\n", reduced_count, total_count, reduced_count/total_count);
        //        fprintf(stderr, "%.4f/%.4f=%.4f <-\n", max_reduce_count, total_count, max_reduce_count/total_count);

//        fprintf(stderr, "%.2f\t",  reduced_count/total_count);
        //        fprintf(stderr, "%.2f <-\n",  max_reduce_count/total_count);

        //        fprintf(stderr, "进入原有的:%.2f \t 进入我们的:%.2f\n",  mlsys_count/reduced_count, our_count/reduced_count);

        last_x.clone_from(bottom_blob); //没有这行是15899，加了是16058
    }


    return 0;
}

// 只比较左边和(t-1), 并且用了select-norm
static int temporal_spatial_convolution_lower_bound(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data,
                                        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                                        int activation_type, const Mat& activation_params, const Option& opt, Mat& last_x, Mat& last_y, Mat& w_norm2,
                                        Mat& last_y_col, Mat& last_y_row, Mat& w_norm2_lower)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    float reduce = 0;
    float total = 0;
    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    //    float norm_norm_row;    // i的norm norm
    //    float delta_x_row;

    float* last_y_row_ptr = nullptr;
    float* last_y_col_ptr = (float*) last_y_col.data;

    if (last_x.total() <= 0){
        //        fprintf(stderr, "enter\n");
        w_norm2.create(outch);
        w_norm2_lower.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw);         // outw是h
        /**
         * calculate w_norm2
         */
        float* w_norm2_data_ptr = (float*) w_norm2.data;
        float* w_norm2_data_lower_ptr = (float*) w_norm2_lower.data;    // 去掉第0个元素
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data_ptr[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data_ptr[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }
            w_norm2_data_lower_ptr[k] = sqrt(w_norm2_data_ptr[k] -
                                             ((const float*)weight_data.data + maxk * inch * k)[0]
                                             * ((const float*)weight_data.data + maxk * inch * k)[0]);
            w_norm2_data_ptr[k] = sqrt(w_norm2_data_ptr[k]);
        }

        /**
         * exact compute
         */
        last_x.clone_from(bottom_blob);
        last_y.clone_from(top_blob);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;

                    float* outptr_last_y = last_y.channel(k);
                    outptr_last_y += i * outw;

                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    // 某层有64个卷积核，kptr即为64个卷积核之一
                    const float* kptr = (const float*)weight_data + maxk * inch * k;

//                    total += 1;
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }

                        kptr += maxk;
                    }
                    if (bias_term)
                        outptr_last_y[j] = y_kij - bias_data[k];
                    else
                        outptr_last_y[j] = y_kij;
//                    if (y_kij <= 0)
//                        reduce += 1;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                }
            }
        }
//        fprintf(stderr, "%.0f %.0f\n", reduce, total);
    }else{
        float dx2_sum = 0.0;
        //        const float* prev_i_sptr = nullptr;
        const float* prev_j_sptr = nullptr;
        for (int i = 0; i < outh; i++)
        {
            //            last_y_row_ptr = (float*)last_y_row.data;
            for (int j = 0; j < 1; j++)
            {
                float record_xij_0 = 0;
                bool first = true;
                delta_x_col = 0.0;
                //                delta_x_row = 0.0;
                dx2_sum = 0.0;
                for (int q = 0; q < 1; q++)
                {
                    const Mat& m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;


                    for (int w_i = 0; w_i < 1; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = base - sptr_last_x[w_i_offset];

                        record_xij_0 = temporal_diff;

                        dx2_sum += temporal_diff * temporal_diff;

                        //                        if (i!=0){
                        //                            float spatial_i_diff = prev_i_sptr[w_i_offset] - base;
                        //                            delta_x_row += spatial_i_diff * spatial_i_diff;
                        //                        }

                    }

                    for (int w_i = 1; w_i < maxk; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = base - sptr_last_x[w_i_offset];
                        dx2_sum += temporal_diff * temporal_diff;


                    }
                }
                for (int q = 1; q < inch; q++)
                {
                    const Mat& m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;


                    for (int w_i = 0; w_i < maxk; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = base - sptr_last_x[w_i_offset];

                        dx2_sum += temporal_diff * temporal_diff;

                    }
                }

                float dx_norm = sqrt(dx2_sum);

                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;


                    const float* w_norm2_ptr = (const float*)w_norm2.data;
                    float* out_bar_ptr = last_y.channel(k);
                    out_bar_ptr += i * outw;

                    /**
                     * 注意our_bar_ptr是j，而last_y_col_ptr是k
                     */
                    float norm_norm;
                    if (record_xij_0 * kptr[0] <= 0)
                        norm_norm = out_bar_ptr[j] + w_norm2_lower[k] * dx_norm + record_xij_0 * kptr[0];
                    else
                        norm_norm = out_bar_ptr[j] + w_norm2_ptr[k] * dx_norm;

                    min_norm_norm = norm_norm;

                    if (min_norm_norm + y_kij <= 0){
                        last_y_col_ptr[k] = min_norm_norm;
                        out_bar_ptr[j] = min_norm_norm;
                        outptr[j] = 0;
                    }else{
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat& m = bottom_blob.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                            {
                                float val = sptr[space_ofs[w_i]]; // 20.72
                                float wt = kptr[w_i];
                                y_kij += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }

                        if (bias_term){
                            out_bar_ptr[j] = y_kij - bias_data[k];
                            last_y_col_ptr[k] = y_kij - bias_data[k];;
                            //                            last_y_row_ptr[k] = y_kij - bias_data[k];;
                        }else{
                            out_bar_ptr[j] = y_kij;
                            last_y_col_ptr[k] = y_kij;
                            //                            last_y_row_ptr[k] = y_kij;
                        }

                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    }
                }
                //                last_y_row_ptr += outch;
            }
            for (int j = 1; j < outw; j++)
            {
                float record_xij_0 = 0;
                bool first = true;
                delta_x_col = 0.0;
                //                delta_x_row = 0.0;
                dx2_sum = 0.0;
                for (int q = 0; q < 1; q++)
                {
                    const Mat& m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;

                    //                    if (i!=0){
                    //                        prev_i_sptr = m.row((i-1) * stride_h) + j * stride_w;
                    //                    }

                    prev_j_sptr = sptr - stride_w;


                    for (int w_i = 0; w_i < 1; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = base - sptr_last_x[w_i_offset];

                        record_xij_0 = temporal_diff;

                        dx2_sum += temporal_diff * temporal_diff;


                        float spatial_j_diff = prev_j_sptr[w_i_offset] - base;
                        delta_x_col += spatial_j_diff * spatial_j_diff;

                    }

                    for (int w_i = 1; w_i < maxk; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = base - sptr_last_x[w_i_offset];
                        dx2_sum += temporal_diff * temporal_diff;

                        float spatial_j_diff = prev_j_sptr[w_i_offset] - base;
                        delta_x_col += spatial_j_diff * spatial_j_diff;
                    }
                }
                for (int q = 1; q < inch; q++)
                {
                    const Mat& m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const Mat& m_last_x = last_x.channel(q);
                    const float* sptr_last_x = m_last_x.row(i * stride_h) + j * stride_w;

                    prev_j_sptr = sptr - stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++)
                    {
                        auto w_i_offset = space_ofs[w_i];
                        float base = sptr[w_i_offset];
                        float temporal_diff = base - sptr_last_x[w_i_offset];

                        dx2_sum += temporal_diff * temporal_diff;

                        float spatial_j_diff = prev_j_sptr[w_i_offset] - base;
                        delta_x_col += spatial_j_diff * spatial_j_diff;

                    }
                }


                delta_x_col = sqrt(delta_x_col);

                float dx_norm = sqrt(dx2_sum);

                for (int k = 0; k < outch; k++)
                {
                    float* outptr = top_blob.channel(k);
                    outptr += i * outw;
                    float y_kij = 0.f;

                    if (bias_term)
                        y_kij = bias_data[k];

                    const float* kptr = (const float*)weight_data + maxk * inch * k;


                    const float* w_norm2_ptr = (const float*)w_norm2.data;
                    float* out_bar_ptr = last_y.channel(k);
                    out_bar_ptr += i * outw;

                    /**
                     * 注意our_bar_ptr是j，而last_y_col_ptr是k
                     */
                    float norm_norm;
                    if (kptr[0] <=0 || record_xij_0 * kptr[0] > 0)
                        norm_norm = out_bar_ptr[j] + w_norm2_lower[k] * dx_norm;
                    else
                        norm_norm = out_bar_ptr[j] + w_norm2_ptr[k] * dx_norm;

                    min_norm_norm = norm_norm;

                    norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2_ptr[k];
                    min_norm_norm = std::min(min_norm_norm, norm_norm_col);


                    if (min_norm_norm + y_kij <= 0){
                        last_y_col_ptr[k] = min_norm_norm;
                        //                        last_y_row_ptr[k] = min_norm_norm;
                        out_bar_ptr[j] = min_norm_norm;
                        outptr[j] = 0;
                    }else{
                        for (int q = 0; q < inch; q++)
                        {
                            const Mat& m = bottom_blob.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                            {
                                float val = sptr[space_ofs[w_i]]; // 20.72
                                float wt = kptr[w_i];
                                y_kij += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }

                        if (bias_term){
                            out_bar_ptr[j] = y_kij - bias_data[k];
                            last_y_col_ptr[k] = y_kij - bias_data[k];;
                            //                            last_y_row_ptr[k] = y_kij - bias_data[k];;
                        }else{
                            out_bar_ptr[j] = y_kij;
                            last_y_col_ptr[k] = y_kij;
                            //                            last_y_row_ptr[k] = y_kij;
                        }

                        outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                    }
                }
                //                last_y_row_ptr += outch;
            }
        }
        //        fprintf(stderr, "%.4f/%.4f=%.4f\n", reduced_count, total_count, reduced_count/total_count);
        //        fprintf(stderr, "%.4f/%.4f=%.4f <-\n", max_reduce_count, total_count, max_reduce_count/total_count);

        //                fprintf(stderr, "%.2f\n",  reduced_count/total_count);
        //        fprintf(stderr, "%.2f <-\n",  max_reduce_count/total_count);

        //        fprintf(stderr, "进入原有的:%.2f \t 进入我们的:%.2f\n",  mlsys_count/reduced_count, our_count/reduced_count);

        last_x.clone_from(bottom_blob); //没有这行是15899，加了是16058
    }
    return 0;
}


float find_max(const float* arr, int size){
    float* bs = (float*)malloc(size *sizeof(float));
    for (int i=0; i< size; i++){
        bs[i] = arr[i];
    }

    for(int i = 0;i<=size-1;i++){
        for(int j = size-1;j>i;j--){
            if(bs[j]<bs[j-1])
            {
                int cup = bs[j-1];
                bs[j-1] = bs[j];
                bs[j] = cup;
            }
        }
    }

    float max_2=0;
    for (int i=size-3; i<size; i++){
        max_2 += arr[i] * arr[i];
    }

    free(bs);

    return max_2;
}

// 比较左边和上面
static int spatial_convolution_lower_bound_first_one(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h,
                               int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt,
                               Mat& w_norm2, Mat& last_y_col, Mat& last_y_row, Mat& w_norm2_lower)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    //    Mat w_norm2 = Mat();
    //    Mat last_y_col = Mat();
    if (w_norm2.total() <= 0){
        w_norm2.create(outch);
        w_norm2_lower.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw);         // outw是h
        float* w_norm2_data_lower_ptr = (float*) w_norm2_lower.data;    // 去掉第0个元素
        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data_lower_ptr[k] = kptr[0] * kptr[0];
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }

//            float abs_max_2 = find_max((const float*)weight_data.data + maxk * inch * k, maxk*inch);

            w_norm2_data_lower_ptr[k] = sqrt(w_norm2_data[k] -
                                             w_norm2_data_lower_ptr[k]);
//            w_norm2_data_lower_ptr[k] = sqrt(w_norm2_data[k] - abs_max_2);
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
//            fprintf(stderr, "%f\n", w_norm2_data_lower_ptr[k]/ w_norm2_data[k]);
        }
//        fprintf(stderr, "\n");
    }


    float reduce = 0;
    float total = 0;

    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    float norm_norm_row;    // i的norm norm
    float delta_x_row;

    float* last_y_row_ptr = nullptr;
    float* last_y_col_ptr = (float*) last_y_col.data;

    //    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outh; i++)
    {
        last_y_row_ptr = (float*)last_y_row.data;
        for (int j = 0; j < outw; j++)
        {
            float record_delta_xij_0 = 0; //    record i delta
            delta_x_col = 0.0;
            delta_x_row = 0.0;
            if (j!=0){
                /** 上一列的基准
             * calculate ||x(i, j) - x(i, j-1)||
             */
                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = sptr - stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_col += delta * delta;
                    }
                }
                delta_x_col = sqrt(delta_x_col);
            }

            if (i!=0){
                /** 上一行的基准
             * calculate ||x(i, j) - x(i-1, j)||
             */
                for (int q = 0; q < 1; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;
                    for (int w_i = 0; w_i < 1; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        record_delta_xij_0 = delta;
                        delta_x_row += delta * delta;
                    }

                    for (int w_i = 1; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_row += delta * delta;
                    }
                }
                for (int q = 1; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_row += delta * delta;
                    }
                }
                delta_x_row = sqrt(delta_x_row);
            }

            for (int k = 0; k < outch; k++)
            {
                float* outptr = top_blob.channel(k);
                outptr += i * outw;

                float y_kij = 0.f;

                if (bias_term)
                    y_kij = bias_data[k];

                const float* kptr = (const float*)weight_data + maxk * inch * k;

                total += 1;

                if (j!=0){
                    norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2[k];
                    min_norm_norm = norm_norm_col;
                }

                if (i!=0){
                    float temp = record_delta_xij_0 * kptr[0];
                    if (temp <= 0)
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2_lower[k] + temp;
                    else
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2[k];

                    if(j!=0)
                        min_norm_norm = std::min(norm_norm_row, min_norm_norm);
                    else
                        min_norm_norm = norm_norm_row;
                }

                if ((i!=0||j!=0) && min_norm_norm + y_kij <= 0){
                    last_y_col_ptr[k] = min_norm_norm;
                    last_y_row_ptr[k] = min_norm_norm;
                    outptr[j] = 0;
                    reduce += 1;
                }else{
                    last_y_col_ptr[k] = -y_kij;
                    last_y_row_ptr[k] = -y_kij;
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }
                        kptr += maxk;
                    }

                    last_y_col_ptr[k] += y_kij;
                    last_y_row_ptr[k] += y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                }
            }
            last_y_row_ptr += outch;
            //            last_y_row_ptr = (float*)((unsigned char*)last_y_row_ptr + (size_t)w * last_y_row.elemsize);
        }
    }
    //    fprintf(stderr, "%f/%f=%.2f\n", reduce, total, reduce/total);
    return 0;
}

// 比较左边和上面
static int spatial_convolution_lower_bound_first_E(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h,
                                                     int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt,
                                                     Mat& w_norm2, Mat& last_y_col, Mat& last_y_row, Mat& w_norm2_lower,
                                                   Mat& all_select_norms, Mat& top_E_indices, Mat& top_E_w_vals)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    //    Mat w_norm2 = Mat();
    //    Mat last_y_col = Mat();
    if (w_norm2.total() <= 0){
        w_norm2.create(outch);
        w_norm2_lower.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw);         // outw是h
        float* w_norm2_data_lower_ptr = (float*) w_norm2_lower.data;    // 去掉第0个元素
        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data_lower_ptr[k] = kptr[0] * kptr[0];
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }

            //            float abs_max_2 = find_max((const float*)weight_data.data + maxk * inch * k, maxk*inch);

            w_norm2_data_lower_ptr[k] = sqrt(w_norm2_data[k] -
                                             w_norm2_data_lower_ptr[k]);
            //            w_norm2_data_lower_ptr[k] = sqrt(w_norm2_data[k] - abs_max_2);
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
            //            fprintf(stderr, "%f\n", w_norm2_data_lower_ptr[k]/ w_norm2_data[k]);
        }
        //        fprintf(stderr, "\n");
    }


    float reduce = 0;
    float total = 0;

    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    float norm_norm_row;    // i的norm norm
    float delta_x_row;

    float* last_y_row_ptr = nullptr;
    float* last_y_col_ptr = (float*) last_y_col.data;

    //    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outh; i++)
    {
        last_y_row_ptr = (float*)last_y_row.data;
        for (int j = 0; j < outw; j++)
        {
            float record_delta_xij_0 = 0; //    record i delta
            delta_x_col = 0.0;
            delta_x_row = 0.0;
            if (j!=0){
                /** 上一列的基准
             * calculate ||x(i, j) - x(i, j-1)||
             */
                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = sptr - stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_col += delta * delta;
                    }
                }
                delta_x_col = sqrt(delta_x_col);
            }

            if (i!=0){
                /** 上一行的基准
             * calculate ||x(i, j) - x(i-1, j)||
             */
                for (int q = 0; q < 1; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;
                    for (int w_i = 0; w_i < 1; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        record_delta_xij_0 = delta;
                        delta_x_row += delta * delta;
                    }

                    for (int w_i = 1; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_row += delta * delta;
                    }
                }
                for (int q = 1; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_row += delta * delta;
                    }
                }
                delta_x_row = sqrt(delta_x_row);
            }

            for (int k = 0; k < outch; k++)
            {
                float* outptr = top_blob.channel(k);
                outptr += i * outw;

                float y_kij = 0.f;

                if (bias_term)
                    y_kij = bias_data[k];

                const float* kptr = (const float*)weight_data + maxk * inch * k;

                total += 1;

                if (j!=0){
                    norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2[k];
                    min_norm_norm = norm_norm_col;
                }

                if (i!=0){
                    float temp = record_delta_xij_0 * kptr[0];
                    if (temp <= 0)
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2_lower[k] + temp;
                    else
                        norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2[k];

                    if(j!=0)
                        min_norm_norm = std::min(norm_norm_row, min_norm_norm);
                    else
                        min_norm_norm = norm_norm_row;
                }

                if ((i!=0||j!=0) && min_norm_norm + y_kij <= 0){
                    last_y_col_ptr[k] = min_norm_norm;
                    last_y_row_ptr[k] = min_norm_norm;
                    outptr[j] = 0;
                    reduce += 1;
                }else{
                    last_y_col_ptr[k] = -y_kij;
                    last_y_row_ptr[k] = -y_kij;
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }
                        kptr += maxk;
                    }

                    last_y_col_ptr[k] += y_kij;
                    last_y_row_ptr[k] += y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                }
            }
            last_y_row_ptr += outch;
            //            last_y_row_ptr = (float*)((unsigned char*)last_y_row_ptr + (size_t)w * last_y_row.elemsize);
        }
    }
    //    fprintf(stderr, "%f/%f=%.2f\n", reduce, total, reduce/total);
    return 0;
}



// 比较左边和上面
static int spatial_convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h,
                               int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt,
                               Mat& w_norm2, Mat& last_y_col, Mat& last_y_row, float& last_sparsity, float& call_time)
{
    call_time += 1;
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

//    Mat w_norm2 = Mat();
//    Mat last_y_col = Mat();
    if (w_norm2.total() <= 0){
        w_norm2.create(outch);
        last_y_col.create(outch);
        last_y_row.create(outch, outw);         // outw是h
        /**
         * calculate w_norm2
         */
        float* w_norm2_data = (float*) w_norm2.data;
        for (int k=0; k<outch; k++){
            const float* kptr = (const float*)weight_data.data + maxk * inch * k;
            w_norm2_data[k] = 0.0;
            for (int q = 0; q < inch; q++){
                for (int w_i = 0; w_i < maxk; w_i++){
                    w_norm2_data[k] += kptr[w_i] * kptr[w_i];
                }
                kptr += maxk;
            }
            w_norm2_data[k] = sqrt(w_norm2_data[k]);
        }
    }


    float reduce = 0;
    float total = 0;

    float min_norm_norm;

    float norm_norm_col;
    float delta_x_col;

    float norm_norm_row;    // i的norm norm
    float delta_x_row;

    float* last_y_row_ptr = nullptr;
    float* last_y_col_ptr = (float*) last_y_col.data;

        //    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outh; i++)
    {
        last_y_row_ptr = (float*)last_y_row.data;
        for (int j = 0; j < outw; j++)
        {
            delta_x_col = 0.0;
            delta_x_row = 0.0;
            if (j!=0){
            /** 上一列的基准
             * calculate ||x(i, j) - x(i, j-1)||
             */
                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = sptr - stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_col += delta * delta;
                    }
                }
                delta_x_col = sqrt(delta_x_col);
            }

            if (i!=0){
            /** 上一行的基准
             * calculate ||x(i, j) - x(i-1, j)||
             */
            for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    const float* prev_sptr = m.row((i-1) * stride_h) + j * stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float delta = sptr[space_ofs[w_i]] - prev_sptr[space_ofs[w_i]];
                        delta_x_row += delta * delta;
                    }
                }
                delta_x_row = sqrt(delta_x_row);
            }

            for (int k = 0; k < outch; k++)
            {
                float* outptr = top_blob.channel(k);
                outptr += i * outw;

                float y_kij = 0.f;

                if (bias_term)
                    y_kij = bias_data[k];

                const float* kptr = (const float*)weight_data + maxk * inch * k;

                total += 1;

                if (j!=0){
                    norm_norm_col = last_y_col_ptr[k] + delta_x_col * w_norm2[k];
                    min_norm_norm = norm_norm_col;
                }

                if (i!=0){
                    norm_norm_row = last_y_row_ptr[k] + delta_x_row * w_norm2[k];
                    if(j!=0)
                        min_norm_norm = std::min(norm_norm_row, min_norm_norm);
                    else
                        min_norm_norm = norm_norm_row;
                }

                if ((i!=0||j!=0) && min_norm_norm + y_kij <= 0){
                    last_y_col_ptr[k] = min_norm_norm;
                    last_y_row_ptr[k] = min_norm_norm;
                    outptr[j] = 0;
//                    reduce += 2 * inch * maxk;
                }else{
                    last_y_col_ptr[k] = -y_kij;
                    last_y_row_ptr[k] = -y_kij;
                    for (int q = 0; q < inch; q++)
                    {
                        const Mat m = bottom_blob.channel(q);
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                        {
                            float val = sptr[space_ofs[w_i]]; // 20.72
                            float wt = kptr[w_i];
                            y_kij += val * wt; // 41.45

                        }
                        kptr += maxk;
                    }

                    last_y_col_ptr[k] += y_kij;
                    last_y_row_ptr[k] += y_kij;
                    outptr[j] = activation_ss(y_kij, activation_type, activation_params);
                }
            }
            last_y_row_ptr += outch;
//            last_y_row_ptr = (float*)((unsigned char*)last_y_row_ptr + (size_t)w * last_y_row.elemsize);
        }
    }
//    if (last_sparsity < 0)
//        last_sparsity = reduce;
//    else
//        last_sparsity = ((call_time - 1) * last_sparsity + reduce)/call_time;
//    fprintf(stderr, "%.0f ", reduce);
    return 0;
}


// 保留原来的convolution
static int raw_convolution(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    //    #pragma omp parallel for num_threads(opt.num_threads)
    double sparsity = 0.0;
    double total = 0.0;
    double flops = 0.0;
    for (int i = 0; i < outh; i++)
    {
        for (int j = 0; j < outw; j++)
        {
            for (int k = 0; k < outch; k++)
            {
                float* outptr = top_blob.channel(k);
                outptr += i * outw;

                float y_kij = 0.f;

                if (bias_term)
                    y_kij = bias_data[k];

                const float* kptr = (const float*)weight_data + maxk * inch * k;

                flops += inch*maxk;
                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int w_i = 0; w_i < maxk; w_i++) // 29.23
                    {
                        float val = sptr[space_ofs[w_i]]; // 20.72
                        float wt = kptr[w_i];
                        y_kij += val * wt; // 41.45
                    }

                    kptr += maxk;
                }

                if (y_kij <= 0){
                    sparsity += 1;
                }
                total += 1;

                outptr[j] = activation_ss(y_kij, activation_type, activation_params);
            }
        }
    }
//    fprintf(stderr, "%f %f\n", flops, sparsity/total);

    return 0;
}

int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output);
            pd.set(1, bias_term);
            pd.set(2, weight_data_size);
            pd.set(8, int8_scale_term);
            pd.set(9, activation_type);
            pd.set(10, activation_params);

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;

#if NCNN_INT8
            if (int8_scale_term)
            {
                weights[2] = weight_data_int8_scales;
                weights[3] = bottom_blob_int8_scales;
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));

            op->create_pipeline(opt);

            // forward
            op->forward(bottom_blob, top_blob, opt);

            op->destroy_pipeline(opt);

            delete op;

            return 0;
        }
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret;
    if (opt.use_reserved_0){
//    if (false){
//        ret = our_convolution(bottom_blob_bordered, top_blob,
//                              weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                              last_x, theta_thres, w_unit, w_norm, exact_compute);

        ret = mlsys_convolution(bottom_blob_bordered, top_blob,
                                weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
                            record1, record2, record3);

//        ret = mlsys_convolution_lower_top_E(bottom_blob_bordered, top_blob,
//                                weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                            record1, record2, record3, all_select_norms, top_E_indices, top_E_w_vals, record4);

//    ret = change_temporal_spatial_convolution(bottom_blob_bordered, top_blob,
//                                    weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                                    record1, record2, record3, record4, record5, last_time_sparsity);

//        ret = spatial_convolution(bottom_blob_bordered, top_blob,
//                                  weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                          record1, record2, record3, last_time_sparsity, call_count);

//        ret = spatial_convolution_lower_bound_first_one(bottom_blob_bordered, top_blob,
//                                                weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                                                record1, record2, record3, record4);

//        ret = spatial_convolution_lower_bound_first_E(bottom_blob_bordered, top_blob,
//                                                      weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                                                      record1, record2, record3, record4, all_select_norms, top_E_indices, top_E_w_vals);

//        ret = temporal_spatial_convolution(bottom_blob_bordered, top_blob,
//                                  weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                                  record1, record2, record3, record4, all_select_norms);

//        ret = temporal_spatial_convolution_lower_bound(bottom_blob_bordered, top_blob,
//                                  weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                                  record1, record2, record3, record4, all_select_norms, top_E_indices);
    }
//    else if(opt.use_reserved_4){
//        ret = temporal_spatial_convolution(bottom_blob_bordered, top_blob,
//                                           weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt,
//                                           record1, record2, record3, record4, all_select_norms);
//    }
    else{
        ret = raw_convolution(bottom_blob_bordered, top_blob,
                              weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
//        fprintf(stderr, "raw conv\n");
    }
    if (ret != 0)
        return ret;

    return 0;
}

int Convolution::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, _kernel_w, _kernel_h, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, _num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = raw_convolution(bottom_blob_bordered, top_blob, weight_data_flattened, bias_data_flattened, _kernel_w, _kernel_h,
                              stride_w, stride_h, dilation_w, dilation_h, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, opt);
}

void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int _kernel_w, int _kernel_h, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

int Convolution::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Convolution input %d x %d  ksize=%d %d  stride=%d %d", w, h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scales, opt_g);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // int8
    bool use_int8_requantize = int8_scale_term > 100;
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

// num_output
#pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        signed char* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                int sum = 0;

                const signed char* kptr = (const signed char*)weight_data + maxk * channels * p;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        int val = sptr[space_ofs[k]];
                        int wt = kptr[k];
                        sum += val * wt;
                    }

                    kptr += maxk;
                }

                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                float sumfp32 = sum * scale_in;

                if (bias_term)
                    sumfp32 += bias_data[p];

                sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                if (use_int8_requantize)
                {
                    // requantize
                    float scale_out = top_blob_int8_scales[0];
                    signed char sums8 = float2int8(sumfp32 * scale_out);
                    outptr[0] = sums8;
                    outptr += 1;
                }
                else
                {
                    // dequantize
                    ((float*)outptr)[0] = sumfp32;
                    outptr += 4;
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
