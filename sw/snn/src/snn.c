static __attribute__((noinline)) void __snn_inl_c_sm32_dm_bstatic(
    double *    s_vals,
    uint32_t *  s_idcs,
    uint32_t *  s_ptrs,
    uint32_t    s_rows,
    double *    w_vals,
    uint32_t    w_maj_stride,   // e.g. CSR x DR -> DR: 1
    uint32_t    w_min_shift,    // e.g. CSR x DR -> DR: log2_cols_b
    double *    v,
    uint32_t    v_maj_stride, // Co for HWC
    uint32_t    v_min_stride, // 1
    uint32_t    v_maj_len,    // W_COLS
    int    *    of_map,
    uint32_t    of_map_min_stride, // 1, as we are storing dense data in HWC format
    uint32_t    of_map_maj_stride  // Co, as we are storing dense data in HWC format
) { 
    double alpha = 0.2;
    double vth   = 2.54;
    
    for (int r = 0; r < s_rows; ++r) 
    { // change a row
        uint32_t s_elems;
        s_elems = s_ptrs[r+1] - s_ptrs[r];
        if(s_elems != 0){
            for(int k = 0; k < v_maj_len; k++)
            { // change b col
                double acc = 0.0;
                for (int e = s_ptrs[r]; e < s_ptrs[r+1]; ++e){ 
                    acc += s_vals[e]*w_vals[s_idcs[e] << w_min_shift];        
                }
                acc += *v;

                //-- LIF dynamics
                if(acc >= vth)
                    *of_map = 1;
                else
                    *of_map = 0;

                *v = acc * alpha;
                w_vals += 1; 
                v      += 1;
                of_map += of_map_maj_stride;              
           }
        }else{
            for(int k = 0; k < v_maj_len; k++) // change b col (res_maj_len = 4)
            {   
                *v *= 0.2;
                v  += 1;
            }
            w_vals += v_maj_len;
            of_map += of_map_maj_stride * v_maj_len; 
        }
        w_vals -= v_maj_len;
    }
}


static __attribute__((noinline)) void __snn_inl_issr_sm32_dm_bstatic_no_val(
    uint32_t *  s_idcs,
    uint32_t *  s_ptrs,
    uint32_t    s_rows,
    double *    w_vals,
    uint32_t    w_maj_stride,   // e.g. CSR x DR -> DR: 1
    uint32_t    w_min_shift,    // e.g. CSR x DR -> DR: log2_cols_b
    double *    v,
    uint32_t    v_maj_stride, // Co for HWC
    uint32_t    v_min_stride, // 1
    uint32_t    v_maj_len,    // W_COLS
    int    *    of_map,
    uint32_t    of_map_min_stride, // 1, as we are storing dense data in HWC format
    uint32_t    of_map_maj_stride  // Co, as we are storing dense data in HWC format
) { 
   double res_val;
   int spk_out;
    __rt_sssr_cfg_write(__RT_SSSR_IDX_CFG(2,w_min_shift,0), 1, __RT_SSSR_REG_IDX_CFG);
    
    for (int r = 0; r < s_rows; ++r) // change a row
    {   
        uint32_t s_elems;
        s_elems = s_ptrs[r+1] - s_ptrs[r];
        if(s_elems != 0){
             
            for(int k = 0; k < v_maj_len; k++) // change b col (res_maj_len = 4)
            {   
                __rt_sssr_cfg_write(s_elems-1, 1, __RT_SSSR_REG_BOUND_0);
                __rt_sssr_cfg_write_ptr(w_vals, 1, __RT_SSSR_REG_IDX_BASE);
                __rt_sssr_cfg_write_ptr(s_idcs, 1, __RT_SSSR_REG_RPTR_INDIR);
                __RT_SSSR_BLOCK_BEGIN // __sla_inl_issr_sv8_dv_dotp
                asm volatile (
                    "fcvt.d.w   ft3, zero               \n"
                    // Use 4 reduction registers
                    "frep.i     %[sldc], 1, 2, 0b0001   \n"
                    "fmv.d      ft4, ft3                \n"
                    "frep.o     %[aldc], 1, 3, 0b0101   \n"
                    "fadd.d     ft3, ft1, ft3           \n"
                    SLA_REDUCE_FT3_4(%[acc]) 
                    //-- LIF Dynamics
                    "fld        ft5, 0(%[v])            \n"
                    "fadd.d     %[acc], ft5, %[acc]     \n"
                    //"fle.d      t3, %[vth], %[acc]      \n" // compare vth with v 
                    //"fmul.d     %[acc], %[acc], %[alpha]\n" // compute v(t)
                    "fle.d      %[sout], %[vth], %[acc] \n" // compare vth with v and put the result in s_out variable
                    "fmul.d     %[acc], %[acc], %[alpha]\n" // compute v(t)
                    : [acc]"+&f"(res_val), [ofmap]"+&r"(of_map), [sout]"+&r"(spk_out)
                    : [v]"r"(v), [aidc]"r"(s_idcs), [aldc]"r"(s_elems-1),
                      [icfg]"r"(2), [bstr]"r"(8), [sldc]"r"(2), [sostr]"r"(of_map_maj_stride << 2), [alpha]"f"(0.2), [vth]"f"(2.54)
                    : "memory", "ft3", "ft4", "ft5", "ft6", "fs0"
                );
                __RT_SSSR_BLOCK_END
                *of_map = spk_out; 
                of_map += 1;
                *v      = res_val;
                v      += 1;
                w_vals += 1; 
            }
        }else{
            for(int k = 0; k < v_maj_len; k++) // change b col (res_maj_len = 4)
            {   
                *v *= 0.2;
                v      += 1;
            } 
            w_vals += v_maj_len;
            of_map += of_map_maj_stride * v_maj_len; 
        }
        s_idcs += s_elems;
        w_vals -= v_maj_len;
    }
}

static __attribute__((noinline)) void __snn_inl_ssr_dm32_dm(
    double *    a_vals,
    uint32_t    a_maj_stride,
    uint32_t    a_rows,
    double *    w_vals,
    uint32_t    w_maj_stride,
    uint32_t    bound,
    uint32_t    w_min_shift,        // e.g. CSR x DR -> DR: log2_cols_b
    double *    v,
    uint32_t    v_maj_len,          // W_COLS
    int    *    of_map,
    uint32_t    of_map_min_stride, // 1, as we are storing dense data in HWC format
    uint32_t    of_map_maj_stride  // Co, as we are storing dense data in HWC format
) { 
    double res_val;
    int    spk_out;
    int iter = 0;
    __rt_sssr_cfg_write(8, 0, __RT_SSSR_REG_STRIDE_0);
    __rt_sssr_cfg_write(16, 1, __RT_SSSR_REG_STRIDE_0); 
    __rt_sssr_cfg_write(bound, 0, __RT_SSSR_REG_BOUND_0);
    __rt_sssr_cfg_write(bound, 1, __RT_SSSR_REG_BOUND_0);

    for(int i = 0; i < a_rows; i++){
        double * a_base = a_vals + i * a_maj_stride;
        for(int j = 0; j < v_maj_len; j++){
            __rt_sssr_cfg_write_ptr(w_vals, 1, __RT_SSSR_REG_RPTR_0); // + j * w_maj_stride
            __rt_sssr_cfg_write_ptr(a_base, 0, __RT_SSSR_REG_RPTR_0);
            __RT_SSSR_BLOCK_BEGIN
            asm volatile (
                "fcvt.d.w   ft3, zero                   \n"   
                // Use 4 reduction registers
                "frep.i     %[sldc], 1, 2, 0b0001       \n"
                "fmv.d      ft4, ft3                    \n"
                "frep.o     %[bnd], 1, 3, 0b1001        \n"
                "fmadd.d    ft3, ft1, ft0, ft3          \n"
                SLA_REDUCE_FT3_4(%[acc])
                //-- LIF Dynamics
                "fld        ft5, 0(%[v])                \n"
                "fadd.d     %[acc], ft5, %[acc]         \n"
                "fle.d      %[sout], %[vth], %[acc]     \n" // compare vth with v and put the result in s_out variable
                "fmul.d     %[acc], %[acc], %[alpha]    \n" // compute v(t)
                : [acc]"+&f"(res_val), [sout]"+&r"(spk_out)
                : [v]"r"(v), [aval]"r"(a_vals), [bnd]"r"(bound),
                  [bval]"r"(w_vals), [sldc]"r"(2), [sostr]"r"(of_map_maj_stride << 2), [alpha]"f"(0.2), [vth]"f"(2.54)
                : "memory", "ft3", "ft4", "ft5", "ft6", "fs0"
            );
            __RT_SSSR_BLOCK_END
           *of_map = spk_out; 
            of_map += 1;
           *v      = res_val;
            v      += 1;
            w_vals += 1; 
        }   
        w_vals -= v_maj_len;
    }
}

static __attribute__((noinline)) void __snn_inl_ssr_dm32_dm(
    double *    a_vals,
    uint32_t    a_maj_stride,
    uint32_t    a_rows,
    double *    w_vals,
    uint32_t    w_maj_stride,
    uint32_t    bound,
    uint32_t    w_min_shift,    // e.g. CSR x DR -> DR: log2_cols_b
    double *    v,
    uint32_t    v_maj_len,    // W_COLS
    int    *    of_map,
    uint32_t    of_map_min_stride, // 1, as we are storing dense data in HWC format
    uint32_t    of_map_maj_stride  // Co, as we are storing dense data in HWC format
) { 
    double res_val;
    int    spk_out;
    int iter = 0;
    __rt_sssr_cfg_write(8, 0, __RT_SSSR_REG_STRIDE_0);
    __rt_sssr_cfg_write(16, 1, __RT_SSSR_REG_STRIDE_0); 
    __rt_sssr_cfg_write(bound, 0, __RT_SSSR_REG_BOUND_0);
    __rt_sssr_cfg_write(bound, 1, __RT_SSSR_REG_BOUND_0);

    for(int i = 0; i < a_rows; i++){
        double * a_base = a_vals + i * a_maj_stride;
        for(int j = 0; j < v_maj_len; j++){
            __rt_sssr_cfg_write_ptr(w_vals, 1, __RT_SSSR_REG_RPTR_0); // + j * w_maj_stride
            __rt_sssr_cfg_write_ptr(a_base, 0, __RT_SSSR_REG_RPTR_0);
            __RT_SSSR_BLOCK_BEGIN
            asm volatile (
                "fcvt.d.w   ft3, zero                   \n"   
                // Use 4 reduction registers
                "frep.i     %[sldc], 1, 2, 0b0001       \n"
                "fmv.d      ft4, ft3                    \n"
                "frep.o     %[bnd], 1, 3, 0b1001        \n"
                "fmadd.d    ft3, ft1, ft0, ft3          \n"
                SLA_REDUCE_FT3_4(%[acc])
                //-- LIF Dynamics
                "fld        ft5, 0(%[v])                \n"
                "fadd.d     %[acc], ft5, %[acc]         \n"
                "fle.d      %[sout], %[vth], %[acc]     \n" // compare vth with v and put the result in s_out variable
                "fmul.d     %[acc], %[acc], %[alpha]    \n" // compute v(t)
                : [acc]"+&f"(res_val), [sout]"+&r"(spk_out)
                : [v]"r"(v), [aval]"r"(a_vals), [bnd]"r"(bound),
                  [bval]"r"(w_vals), [sldc]"r"(2), [sostr]"r"(of_map_maj_stride << 2), [alpha]"f"(0.2), [vth]"f"(2.54)
                : "memory", "ft3", "ft4", "ft5", "ft6", "fs0"
            );
            __RT_SSSR_BLOCK_END
           *of_map = spk_out; 
            of_map += 1;
           *v      = res_val;
            v      += 1;
            w_vals += 1; 
        }    
        w_vals -= v_maj_len;
    }
}

void dense_matmul_lif_c(const double* image, int height_a, int width_a, const double* filter, int width_b, double* v, int* s_out) {
    // Iterate over each output element
    int image_index, filter_index;
    double sum;
    double alpha = 0.2;
    double vth   = 2.54;
    int s_out_idx = 0;
    int v_idx = 0;
    double v_val = 0;
    
    for (int i = 0; i < height_a; i++) { // Cahnge row a
        for (int j = 0; j < width_b; j++) { // change b col
            sum = 0.0f;
            // Perform dot product of the corresponding row in image and column in filter
            for(int k = 0; k < width_a; k++){
                image_index = i * width_a + k;
                filter_index = j + k * 2; //k * width + j
                sum += image[image_index] * filter[filter_index];
            }
            //-- LIF dynamics
            v_val = v[v_idx];
            v_val += sum;
            if(v_val >= vth)
                s_out[++s_out_idx] = 1;
            else 
                s_out[++s_out_idx] = 0;
            
            v[v_idx] = v_val * alpha;
            v_idx++;
        }
    }
}
