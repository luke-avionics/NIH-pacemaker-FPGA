#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

/*-----------------------------------------------------------------------------
| Function: read_ifmap_conv2d
| 
| Purpose:
|
| Parameters:
|   data_type feature_temp[Tn][Tr][Tc] -- 
|   dma_data* feature -- 
|   int tr -- 
|   int ti -- 
|   int tc --
|   int H --
|   int C -- 
|   int K --
|   ap_uint<32> Base_addr2 -- 
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int Tr, int Tc, int Tn>
void read_ifmap_conv2d(
		data_type feature_temp[Tn][Tr][Tc],
		dma_data* feature,
		int tr, int ti, int tc,
		int H, int C, int K,
		ap_uint<32> Base_addr2) {
    #pragma HLS INLINE off
	dma_data tmp;
    int dim_start = (int)(K/2);
    int dim_end = (int)(C + K/2-1);
	int trr, tcc, tii;
	for (tii = 0; tii < Tn; tii+=2) 
		for (trr = 0; trr < Tr; trr++)
			for (tcc = 0; tcc < Tc; tcc++) {
				#pragma HLS PIPELINE
                if ((tr+trr) < dim_start || (tr+trr) > dim_end || (tc+tcc) < dim_start || (tc+tcc) > dim_end) {
                    feature_temp[tii][trr][tcc]=0;
                    feature_temp[tii+1][trr][tcc]=0;
                }
                else {
                    tmp=feature[(tii+ti)/2*C*C + (tr+trr-dim_start)*C + (tc+tcc-dim_start) + Base_addr2/dataw];
                    feature_temp[tii][trr][tcc] = tmp.data.data0;
                    feature_temp[tii+1][trr][tcc] = tmp.data.data1;
                }
			}
}


// template < int Tn>
// void read_ifmap_conv_fc(data_type feature_temp[Tn][1][1], dma_data* feature,int ti,int N, int C,ap_uint<32> Base_addr2) {
// #pragma HLS INLINE off
	// dma_data tmp;
	// int tii, input_channel, row, col;
	// for (tii = 0; tii < Tn; tii+=2) {
		// #pragma HLS PIPELINE
        // input_channel=(int)((tii+ti/2)/(C*C));
        // row=(int)((tii+ti-input_channel*C*C)/C);
        // width=(int)(tii+ti-input_channel*C*C-row*C);
        // tmp=feature[(tii+ti)/2 +Base_addr2/dataw];
        // feature_temp[tii][0][0] = tmp.data.data0;
        // feature_temp[tii+1][0][0] = tmp.data.data1;
	// }
// }


/*-----------------------------------------------------------------------------
| Function: read_wek
|
| Purpose: Read in k weightsRead in weight buffer (specifically w/ kernel size >1)

| Parameters:
|   data_type weight_temp[Tm][Tn][Tk][Tk] -- 
|   dma_data* weight --
|   int to -- 
|   int ti --
|   int K --
|   int N --
|   ap_uint<32> Base_addr1 --
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int Tk, int Tm, int Tn>
void read_wek(
		data_type weight_temp[Tm][Tn][Tk][Tk],
		dma_data* weight,
		int to, int ti,
		int K, int N,
		ap_uint<32> Base_addr1) {
    #pragma HLS INLINE
	int too,tii, tkk1,tkk2;
	dma_data tmp;
	for (too = 0; too < Tm; too += 2)
        for (tii = 0; tii < Tn; tii++)
            for(tkk1 = 0; tkk1 < K; tkk1++)
                for(tkk2 = 0; tkk2 < K; tkk2++) {
                    #pragma HLS PIPELINE
                    tmp= weight[(too+to)/2*N*K*K + (tii+ti)*K*K + tkk1*K + tkk2 + Base_addr1/dataw];
                    weight_temp[too][tii][tkk1][tkk2] = tmp.data.data0;
                    weight_temp[too+1][tii][tkk1][tkk2] = tmp.data.data1;
                }
}


/*-----------------------------------------------------------------------------
| Function: comp_engine_conv_2d
|
| Purpose: Compute engine used for 2D convolution
|
| Parameters:
|   data_type weight_temp[TmW][TnW][Tk][Tk] --
|   data_type feature_temp[TnBuff][Tri][Tci] --
|   data_type output_core_temp[TmBuff][Tr][Tc] --
|   int K -- kernel size
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int Tr, int Tc, int TmBuff, int TnBuff, int Tm, int Tn, int TmW, int TnW, int Tri, int Tci,int Tk>
void comp_engine_conv_2d(
		data_type weight_temp[TmW][TnW][Tk][Tk],
		data_type feature_temp[TnBuff][Tri][Tci],
		data_type output_core_temp[TmBuff][Tr][Tc],
		int K) {
    #pragma HLS INLINE off
	int too, tcc, tii, trr,tkk1,tkk2,tncomp,tmcomp;
    data_type tmp0,tmp1;
	//TODO: balanced unrolling input channel and output channel
    for(tncomp = 0; tncomp < TnBuff; tncomp += Tn)
        for(tmcomp = 0 ; tmcomp < TmBuff; tmcomp += Tm)
            for (tkk1 = 0; tkk1 < K; tkk1++)
                for(tkk2 = 0; tkk2 < K; tkk2++)
                    for (tcc = 0; tcc < Tc; ++tcc)
                        for (trr = 0; trr < Tr; ++trr) {
                        #pragma HLS PIPELINE
                            for (tii = 0; tii < Tn; ++tii) {
                                #pragma HLS UNROLL
                                #pragma HLS DEPENDENCE variable=feature_temp inter false
                                tmp1 = feature_temp[tncomp+tii][trr+tkk1][tcc+tkk2];
                                for (too = 0; too < Tm; ++too) {
                                    #pragma HLS DEPENDENCE variable=output_core_temp inter false
                                    #pragma HLS UNROLL
                                    output_core_temp[tmcomp+too][trr][tcc] += tmp1*weight_temp[tmcomp+too][tncomp+tii][tkk1][tkk2];

                                }
                            }
                        }
}


/*-----------------------------------------------------------------------------
| Function: single_conv_k
|
| Purpose:
|
| Parameters:
|   dma_data* weight --
|   dma_data* feature --
|   dma_data* output_core --
|   int con -- 
|   ap_uint<32> Base_addr1 -- 
|   ap_uint<32> Base_addr2 -- 
|   ap_uint<32> Base_addr3 -- 
|   int M -- number of output channels
|   int N -- number of input channels
|   int H -- input dimension (width/height), equivilant to (C*S+K-1)
|   int C -- output dimension (width/height)
|   int K -- kernel/filter dimension (width/height)
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int TmBuff, int TnBuff, int Tr, int Tc, int Tm, int Tn, int TmW, int TnW, int Tk, int Tri, int Tci>
void single_conv_k(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
		int con,
		ap_uint<32> Base_addr1,
		ap_uint<32> Base_addr2,
		ap_uint<32> Base_addr3,
		int M, int N, int H, int C, int K) {
    dma_data tmp;
    int tr, tc;
    int to, ti, trr, tcc, too, tii;
    int tc_r, tr_r, to_r, ti_r;
    int lr_i = 0;

    data_type output_core_temp[TmBuff][Tr][Tc] = { 0 };
    #pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=1
    //#pragma HLS ARRAY_PARTITION variable=output_core_temp block factor=Tr/2 dim=2
    //#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=3
    #pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

    data_type weight_temp[TmW][TnW][Tk][Tk] = { 0 }, feature_temp[TnBuff][Tri][Tci] = { 0 };
    #pragma HLS RESOURCE variable=feature_temp core=RAM_2P_BRAM
    //#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_BRAM
    #pragma HLS RESOURCE variable=weight_temp core=RAM_2P_LUTRAM

    #pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=1
    //#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=3
    //#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=4

    data_type feature_temp1[TnBuff][Tri][Tci] = { 0 };
    #pragma HLS RESOURCE variable=feature_temp1 core=RAM_2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=1
    //#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=3

    if (con == 0x00000001) {
        //Expand
        #pragma HLS allocation instances=single_conv_k limit=1 function

        //TODO: buffer initialization
        read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature, 0, 0, 0, H, C, K, Base_addr2);
        for (tc = 0; tc < C; tc += Tc) {
            for (tr = 0; tr < C; tr += Tr) {
                for (to = 0; to < M; to += TmBuff) {
                    for (ti = 0; ti < N; ti += TnBuff) {
                        read_wek<Tk, TmW, TnW>(weight_temp, weight, to, ti, K, N, Base_addr1);
                        //read_ifmap_conv2d<Tri, Tci, TnBuff>(feature_temp, feature, tr, ti, tc, H, C, K, Base_addr2);
                        //comp_engine_conv_2d<Tr, Tc, TmBuff, TnBuff, Tm, Tn, TmW, TnW, Tri, Tci, Tk>(weight_temp, feature_temp, output_core_temp, K);

                        if (lr_i==0) {
                            //ping pong logic for index shifting
                            //ti_r=ti;
                            to_r=to;
                            tc_r=tc;
                            tr_r=tr;
                            ti_r=ti+TnBuff;
                            if (ti_r == N) {
                                ti_r = 0;
                                if(to == M - TmBuff ) {
                                    tr_r = tr + Tr;
                                    if(tr_r == C) {
                                        tr_r = 0;
                                        tc_r = tc_r + Tc;
                                        if(tc_r == C) {
                                            tc_r = 0;
                                        }
                                    }
                                }
                            }
                            //TODO: controlling port to switch
                            read_ifmap_conv2d<Tri, Tci, TnBuff>(feature_temp1, feature, tr_r, ti_r, tc_r, H, C, K, Base_addr2);
                            comp_engine_conv_2d<Tr, Tc, TmBuff, TnBuff, Tm, Tn, TmW, TnW, Tri, Tci, Tk>(weight_temp, feature_temp, output_core_temp, K);
                            lr_i = 1 - lr_i;
                        }
                        else {
                            //ping pong logic for index shifting
                            //ti_r=ti;
                            to_r = to;
                            tc_r = tc;
                            tr_r = tr;
                            ti_r = ti + TnBuff;
                            if (ti_r == N) {
                                ti_r = 0;
                                if(to == M - TmBuff ) {
                                    tr_r = tr + Tr;
                                    if(tr_r == C) {
                                        tr_r = 0;
                                        tc_r = tc_r + Tc;
                                        if(tc_r == C) {
                                            tc_r = 0;
                                        }
                                    }
                                }
                            }
                            //TODO: controlling port to switch
                            read_ifmap_conv2d<Tri, Tci, TnBuff>(feature_temp, feature, tr_r, ti_r, tc_r, H, C, K, Base_addr2);
                            comp_engine_conv_2d<Tr, Tc, TmBuff, TnBuff, Tm, Tn, TmW, TnW, Tri, Tci, Tk>(weight_temp, feature_temp1, output_core_temp, K);
                            lr_i = 1 - lr_i;
                        }
                    }

                    for (too = 0; too < TmBuff; too += 2)
                        for (trr = 0; trr < Tr; trr++)
                            for (tcc = 0; tcc < Tc; tcc++) {
                                #pragma HLS PIPELINE
                                tmp.data.data0 = output_core_temp[too][trr][tcc];
                                tmp.data.data1 = output_core_temp[too+1][trr][tcc];
                                output_core[(too+to)/2*C*C + (tr+trr)*C + tc + tcc + Base_addr3/dataw]=tmp;
                            }
              
                    for (trr = 0; trr < Tr; ++trr)
                        for (tcc = 0; tcc < Tc; ++tcc)
                            #pragma HLS PIPELINE
                            for (too = 0; too < TmBuff; ++too) {
                                #pragma HLS UNROLL
                                output_core_temp[too][trr][tcc] = 0;
                            }
                }
            }
        }
    }
}
