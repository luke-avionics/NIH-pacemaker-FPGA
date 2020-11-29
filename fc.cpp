#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

/*-----------------------------------------------------------------------------
| Function: read_ifmap_fc
|
| Purpose: Buffer reading function for FC layers. Reads input buffers dedicated for input features.
|
| Parameters:
|   data_type weight_temp[Tm][Tn][1][1] -- the buffer location (BRAM, not reflected here)
|   dma_data* weight -- source of the features (DRAM)
|   int ti -- index for where to start reading data from in memory
|   ap_uint<32> Base_addr2 -- TBD, currently placeholder
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int Tn>
void read_ifmap_fc(
		data_type feature_temp[Tn][1][1],
		dma_data* feature,
		int ti,
		ap_uint<32> Base_addr2) {
	#pragma HLS INLINE off
	dma_data tmp;
	int tii;
	for (tii = 0; tii < Tn; tii+=2) {
		#pragma HLS PIPELINE
        tmp = feature[(tii+ti)/2 + Base_addr2/dataw];
        feature_temp[tii][0][0] = tmp.data.data0;
        feature_temp[tii+1][0][0] = tmp.data.data1;
	}
}


/*-----------------------------------------------------------------------------
| Function: read_we1
|
| Purpose: Read in weight buffer (specifically w/ kernel size 1). Seperated 
|		   from greater kernels due to pipelining.
|
| Parameters:
|   data_type weight_temp[Tm][Tn][1][1] -- the buffer location (BRAM, not reflected here)
|   dma_data* weight -- source of the features (DRAM)
|   int to -- tell relative address of elements in weight kernel (output)
|   int ti -- tell relative address of elements in weight kernel (input)
|   int N -- number of input channels
|   ap_uint<32> Base_addr1 -- TBD, currently placeholder
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int Tm, int Tn>
void read_we1(
		data_type weight_temp[Tm][Tn][1][1],
		dma_data* weight,
		int to, int ti, int N,
		ap_uint<32> Base_addr1) {
	#pragma HLS INLINE
    int too, tii;
    dma_data tmp;
    for (too = 0; too < Tm; too += 2) {
        for (tii = 0; tii < Tn; tii++) {
			#pragma HLS PIPELINE
            tmp = weight[(too+to) / 2 * N + (tii+ti) + Base_addr1 / dataw];
            weight_temp[too][tii][0][0] = tmp.data.data0;
            weight_temp[too+1][tii][0][0] = tmp.data.data1;

        }
    }
}


/*-----------------------------------------------------------------------------
| Function: comp_engine_fc
|
| Purpose: Computation engine design for FC layers, compute data in buffer. Iterates
|		   through buffer loading input.
|
| Parameters:
|   data_type weight_temp[TmW][TnW][Tk][Tk] --
|   data_type feature_temp[TnBuff][Tri][Tci] --
|   data_type output_core_temp[TmBuff][Tr][Tc] --
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int TmBuff, int TnBuff, int Tm, int Tn, int TmW, int TnW>
void comp_engine_fc(
		data_type weight_temp[TmW][TnW][1][1],
		data_type feature_temp[TnBuff][1][1],
		data_type output_core_temp[TmBuff][1][1]) {
    #pragma HLS INLINE off
	int too, tii, tncomp, tmcomp;
    data_type tmp1;
	//TODO: balanced unrolling input channel and output channel
    for(tncomp = 0; tncomp < TnBuff; tncomp += Tn) // iterate through indexes of the buffer
        for(tmcomp = 0; tmcomp < TmBuff; tmcomp += Tm) {    
            #pragma HLS PIPELINE
            for (tii = 0; tii < Tn; ++tii) {
                #pragma HLS UNROLL
                #pragma HLS DEPENDENCE variable=feature_temp inter false
                tmp1=feature_temp[tncomp+tii][0][0];
                for (too = 0; too < Tm; ++too) {
                    #pragma HLS DEPENDENCE variable=output_core_temp inter false
                    #pragma HLS UNROLL
                    output_core_temp[tmcomp+too][0][0] += tmp1*weight_temp[tmcomp+too][tncomp+tii][0][0];
                }
            }
        }
}


/*-----------------------------------------------------------------------------
| Function: single_fc
|
| Purpose: Utilizes two feature buffers so one can be written to while the other executes 
|
| Parameters:
|   dma_data* weight --
|   dma_data* feature --
|   dma_data* output_core -- where in the dram we will output to
|	int con --
|   ap_uint<32> Base_addr1 --
|   ap_uint<32> Base_addr2 --
|   ap_uint<32> Base_addr3 --
|   int M -- number of output channels
|   int N -- number of input channels
|
| Returns: void
-----------------------------------------------------------------------------*/
template <int TmBuff, int TnBuff, int Tm, int Tn, int TmW, int TnW>
void single_fc(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
		int con,
		ap_uint<32> Base_addr1,
		ap_uint<32> Base_addr2,
		ap_uint<32> Base_addr3,
		int M, int N) {
	dma_data tmp;
	int to, ti, too, tii;
	int ti_r;
	int lr_i = 0;

	// Define buffers
	data_type output_core_temp[TmBuff][1][1] = { 0 };
	#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=1
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp block factor=Tr/2 dim=2
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=3
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

	data_type weight_temp[TmW][TnW][1][1] = { 0 }, feature_temp[TnBuff][1][1] = { 0 };
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

	data_type feature_temp1[TnBuff][1][1] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp1 core=RAM_2P_BRAM
	#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=1
	//#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=2
	//#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=3

	if (con == 0x00000001) {
		//Expand
		#pragma HLS allocation instances=dw_comp_engine2_111 limit=1 function

		//TODO: buffer initialization
		read_ifmap_fc<TnBuff>(feature_temp, feature, 0, Base_addr2);
		for (to = 0; to < M; to += TmBuff) {
			for (ti = 0; ti < N; ti += TnBuff) {
				read_we1<TmW, TnW>(weight_temp, weight, to, ti, N, Base_addr1);
				//read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp,feature,tr,ti,tc,H,C,K,Base_addr2);
				//comp_engine_conv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K);

				if (lr_i == 0) {
					//ping pong logic for index shifting
					//ti_r = ti;
					ti_r = ti + TnBuff;
					if (ti_r == N) {
						ti_r = 0;
					}
					//TODO: controlling port to switch
					read_ifmap_fc<TnBuff>(feature_temp1, feature, ti_r, Base_addr2);
					comp_engine_fc<TmBuff, TnBuff, Tm, Tn, TmW, TnW>(weight_temp, feature_temp, output_core_temp);
					lr_i = 1 - lr_i;
				}
				else {
					//ping pong logic for index shifting
					//ti_r = ti;
					ti_r = ti + TnBuff;
					if (ti_r == N) {
						ti_r = 0;
					}
					//TODO: controlling port to switch
					read_ifmap_fc<TnBuff>(feature_temp, feature, ti_r, Base_addr2);
					comp_engine_fc<TmBuff, TnBuff, Tm, Tn, TmW, TnW>(weight_temp, feature_temp1, output_core_temp);
					lr_i = 1 - lr_i;
				}
			}

			for (too = 0; too < TmBuff; too += 2) {
				#pragma HLS PIPELINE
				tmp.data.data0 = output_core_temp[too][0][0];
				tmp.data.data1 = output_core_temp[too + 1][0][0];
				output_core[(too+to) / 2 + Base_addr3 / dataw] = tmp;

			}

			for (too = 0; too < TmBuff; ++too) {
				#pragma HLS UNROLL
				output_core_temp[too][0][0] = 0;
			}
		}
	}
}
