#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

void tanh_layer(
		dma_data* output_core,
		dma_data* weight,
		int M, int N, int C, int K,
		ap_uint<32> Base_addr2,
		ap_uint<32> Base_addr3) {
    dma_data tmp, tmp1;
    int tm, tc, tr;
    for (tm = 0; tm < M; tm += 2) 
        for(tc = 0; tc < C; tc++)
            for(tr = 0; tr < C; tr++) {
                #pragma HLS PIPELINE
                tmp = output_core[(tm)/2*C*C + (tc)*C + tr + Base_addr3/dataw];
                tmp1 = weight[M*N*K*K/2 + tm/2 + Base_addr2/dataw];
                tmp.data.data0 = tanh(tmp.data.data0 + tmp1.data.data0);
                tmp.data.data1 = tanh(tmp.data.data1 + tmp1.data.data1);
                output_core[(tm)/2*C*C + (tc)*C + tr + Base_addr3/dataw] = tmp;
            }
}


void tanh_layer_fc(
		dma_data* output_core,
		dma_data* weight,
		int M,
		int N,
		ap_uint<32> Base_addr2,
		ap_uint<32> Base_addr3) {
    dma_data tmp, tmp1;
    int tm;
    for (tm = 0; tm < M; tm += 2)
    {
        #pragma HLS PIPELINE
        tmp = output_core[(tm)/2 + Base_addr3/dataw];
        tmp1 = weight[M*N/2 + tm/2 + Base_addr2/dataw];
        tmp.data.data0=tanhf(tmp.data.data0 + tmp1.data.data0);
        tmp.data.data1=tanhf(tmp.data.data1 + tmp1.data.data1);
        output_core[(tm)/2 + Base_addr3/dataw]=tmp;
    }
}


void max_pool(
		dma_data* input_next,
		dma_data* output,
		int M,
		int C_next,
		int C,
		ap_uint<32> Base_addr2,
		ap_uint<32> Base_addr3) {
    dma_data tmp1, tmp2, tmp3, tmp4;
    dma_data max_tmp;
    int tm,tc,tr;
    for (tm=0; tm<M;tm+=2)
        for(tc=0;tc<C;tc+=2)
            for(tr=0; tr<C;tr+=2) {
                #pragma HLS PIPELINE
                tmp1=output[(tm)/2*C*C + (tc)*C + tr + Base_addr3/dataw];
                tmp2=output[(tm)/2*C*C + (tc+1)*C + tr + Base_addr3/dataw];
                tmp3=output[(tm)/2*C*C + (tc)*C + tr+1 + Base_addr3/dataw];
                tmp4=output[(tm)/2*C*C + (tc+1)*C + tr+1 + Base_addr3/dataw];
                max_tmp.data.data0=fmax(fmax(tmp1.data.data0,tmp2.data.data0), fmax(tmp3.data.data0, tmp4.data.data0));
                max_tmp.data.data1=fmax(fmax(tmp1.data.data1,tmp2.data.data1), fmax(tmp3.data.data1, tmp4.data.data1));
                input_next[(tm)/2*C_next*C_next + (tc/2)*C_next + tr/2 + Base_addr2/dataw] = max_tmp;
            }
}

