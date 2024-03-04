#include <math.h>
#include "snn.h"
#include "data.h"

#define K 3

uint32_t __attribute__((section(".l1"))) time_im2row, time_csr_comp, time_snn_tile, time_snn_c_bstatic, time_snn_tile_bstatic, time_lif_act, time_dense, time_dense_ssr, time_dense_ssr_2d, time_spike_to_csr;
int32_t ilog2(uint32_t x) {return 31-__builtin_clz(x);}

double const EPS = 1e-6;

int smain(uint32_t coreid, uint32_t num_cores) {
    if (coreid != 0) while (1) asm volatile ("wfi");
    uint32_t start_time, stop_time;

    int Hi = A_H;
    int Wi = A_W;
    int Ci = A_C;
    int Ho = Hi - K + 1;
    int Wo = Wi - K + 1;
    int Co = OF_CHAN;
    int Wr = Ci * K * K;
    int Hr = Ho * Wo;

    int v_error = Ho * Wo * Co;
    
    //int    row[Hr * Wr];       // im2row output
    double row[Hr * Wr];
    double empty_vals[Hi * Wi * Ci]; // we still need to feed the spikes array of the SSSRs
    int    s_out_bstatic_c[Ho * Wo * Co];
    int    s_out[Ho * Wo * Co];
    int    s_out_dense[Ho * Wo * Co];
    int    s_out_dense_ssr[Ho * Wo * Co];
    double row_double[Hr * Wr];
    int __attribute__ ((aligned (4))) s_out_bstatic[Ho * Wo * Co];// spikes out 

    mini_csr_t csr_nl[Wo * Ho]; //-- THe correct dimension would be Ho(l+1) * Wo(l+1)
    mini_csr_p_t old_csr;

    //-- reset initial target of direct im2col counters
    for(int i = 0; i < Wo * Ho; i++)
        csr_nl[i].counter = 0;
                      
    CSRMatrix csr, csr_dir;
                          
    time_spike_to_csr = 0;
    int time_for = 0;
    int time_init = 0;
    int iterations = 0;
    int time_init_acc = 0; 
    int time_for_acc = 0; 
    int time_im2csr;
    int errors_csr_idx_dir = CSR_IDX_GOLD_LEN;
            
    //---------------------------------------------------------------------------- Direct Conversion --------------------------------//
        // Probalby is the mini csr part
        int old_x_val = - 1;
        int old_y_val = - 1;
        int z_ax_iter =   0;
        for(int i = 0; i < Hi; i++)
            for(int j = 0; j < Wi; j++)
                for(int m = 0; m < Ci; m++){
                    if(spikes[m + j * Ci + i * Wi * Ci] != 0.0){
                        start_time = snrt_mcycle();
                        if(old_x_val == j && old_y_val == i){
                            coo_to_im2row_csr_3d_z_axis(
                                spikes[m + j * Ci + i * Wi], 
                                m,
                                K,
                                Ci,
                                &old_csr,
                                z_ax_iter
                            );
                        z_ax_iter++;
                        }else{
                            z_ax_iter = 0;
                            coo_to_im2row_csr_3d_z_opt(
                                csr_nl,
                                Hi,
                                Wi,
                                Ci,
                                K,
                                S, 
                                j,  //-- x_p
                                i,  //-- y_p
                                m,  //-- z_p
                                spikes[m + j * Ci + i * Wi],
                                &old_csr
                            );
                            old_x_val = j;
                            old_y_val = i; 
                            z_ax_iter = 0;
                        }
                        stop_time = snrt_mcycle();
                        time_spike_to_csr += (stop_time - start_time);
                    }
                }

    //-- Check direct im2csr result
    int counter_csr_ver = 0;
    for (int i = 0; i < Wi; i++){
        for (int j = 0; j < csr_nl[i].counter; j++){
            if(csr_nl[i].col_idcs[j] != row_csr_idx_gold[counter_csr_ver + j])
                printf(" error on: csr[%d].col_idcs[%d]: %d row_csr_idx_gold[%d]: %d\n", i, j, csr_nl[i].col_idcs[j], counter_csr_ver + j,row_csr_idx_gold[counter_csr_ver + j]);
        }
        counter_csr_ver += csr_nl[i].counter;
    }
    printf("Corrrect direct spike to CSR transformation, time: %d iter: %d\n", time_spike_to_csr, iterations);                               
    //----------------------------------------------------------------------------
    //-------------------------------------- im2csr ------------------------------//
    // im2col and csr compression merged in one function
    start_time = snrt_mcycle();
    im2csr_hwc(
        spikes,
        Hi,
        Wi,
        Ci,
        K,
        &csr_dir
    );
    stop_time = snrt_mcycle();
    time_im2csr = stop_time - start_time;
    
    //-- Check im2csr transformation
    for (int i=0; i<CSR_IDX_GOLD_LEN; ++i) {
        if (csr_dir.col_ind[i] == row_csr_idx_gold[i]) --errors_csr_idx_dir;
    }

    if(errors_csr_idx_dir == 0)
        printf("correct im2csr transformation, cycles: %d\n", time_im2csr);
    else{
        printf("Error on im2csr\n");
        return errors_csr_idx_dir;
    }

//--------------------------------------------------- Dot product kernel implemented using various techniques: Dense, Dense w/ SSRs, Sparse-Dense, Sparse-Dense w/ISSRs ---------//
    //-- snn sparse-dense C baseline
    start_time = snrt_mcycle();
    __snn_inl_c_sm32_dm_bstatic(csr_dir.values, csr_dir.col_ind, csr_dir.row_ptr, Ho * Wo, w_vals, 1, ilog2(W_WIDTH), v_vals_c_bstatic, 1, W_WIDTH, W_WIDTH, s_out_bstatic_c, 1, 1);
    stop_time  = snrt_mcycle();
    __rt_fpu_fence_full();
    time_snn_c_bstatic = stop_time - start_time;   
    printf("time SNN sparse-dense baseline: %d\n", time_snn_c_bstatic);

    //-- snn sparse-dense issr baseline 
    start_time = snrt_mcycle();
    __snn_inl_issr_sm32_dm_bstatic_no_val(csr_dir.col_ind, csr_dir.row_ptr, Ho * Wo, w_vals, 1, ilog2(W_WIDTH), v_vals_bstatic, 1, W_WIDTH, W_WIDTH, s_out_bstatic, 1, 1);    
    stop_time = snrt_mcycle();
    __rt_fpu_fence_full();
    time_snn_tile_bstatic = stop_time - start_time;
    printf("time SNN sparse-dense issr: %d\n", time_snn_tile_bstatic);

    //-- dense ssr 
    start_time = snrt_mcycle();
    __snn_inl_ssr_dm32_dm(row, K * K * Ci, Ho * Wo, w_vals, K * K * Ci, K * K * Ci - 1, ilog2(W_WIDTH), v_vals_dense_ssr, W_WIDTH, s_out_dense_ssr, 1, 1);
    stop_time = snrt_mcycle();
    time_dense_ssr = stop_time - start_time;
    printf("time dense ssr: %d\n", time_dense_ssr);
    
    //-- dense baseline
    start_time = snrt_mcycle();
    dense_matmul_lif_c(row, Ho * Wo, K * K * Ci, w_vals, Co, v_vals_dense, s_out_dense);
    stop_time  = snrt_mcycle();
    __rt_fpu_fence_full();
    time_dense = stop_time - start_time;
    printf("time dense baseline: %d\n", time_dense);

    //----------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------- Print/Check Results on v/s_out -----------------------------------------------------------------// 
     
    for(int i = 0; i < Ho * Wo * Co; i++){
        if((fabs(v_vals[i] - v_vals_bstatic[i]) < EPS) && (s_out[i] == s_out_bstatic[i]) && (fabs(v_vals[i] - v_vals_c_bstatic[i]) && (fabs(v_vals_dense[i] - v_vals_dense_ssr[i]) < EPS))) v_error--;
    }
    return v_error;
}