#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

void batchnorm(dma_data* input, dma_data* weight, dma_data* bias,dma_data* output, data_type eps,
				int input_dim1, int input_dim2, int num_features)
{
	// We expect the data to be packed by channels, and we expect the input to be
	// size [num_channels x input_dim1 x input_dim2]
    data_type mean0, mean1, variance0, variance1, normalized;
    dma_data *batch_layers, *out_layers;
    int i, j ,k;
    // Run each batch. Since data is packed, we can calculate two batches at once (for 2 channels)
    for (i = 0; i < num_features; i+= 2) {
		//#pragma HLS unroll (with unrolling utilization nearly doubles, for an 0.1ms latency improvement)
        mean0 = mean1 = 0;
        variance0 = variance1 = 0;
        // Pointers to current batch and output dma packs
        batch_layers = input + (i/2 * input_dim1 * input_dim2);
        out_layers = output + (i/2 * input_dim1 * input_dim2);
        // Calculate mean
        for (j = 0; j < input_dim1; j++) {
            for (k = 0; k < input_dim2; k++) {
				#pragma HLS pipeline
                mean0 += batch_layers[j * input_dim2 + k].data.data0;
                mean1 += batch_layers[j * input_dim2 + k].data.data1;
            }
        }
        mean0 /= input_dim1 * input_dim2;
        mean1 /= input_dim1 * input_dim2;
        // Calculate variance
        for (j = 0; j < input_dim1; j++) {
            for (k = 0; k < input_dim2; k++) {
				#pragma HLS pipeline
                variance0 += pow(batch_layers[j * input_dim2 + k].data.data0 - mean0, 2);
                variance1 += pow(batch_layers[j * input_dim2 + k].data.data1 - mean1, 2);
            }
        }
        variance0 /= input_dim1 * input_dim2;
        variance1 /= input_dim1 * input_dim2;
        // Calculate output using learned weight and bias.
        for (j = 0; j < input_dim1; j++) {
            for (k = 0; k < input_dim2; k++) {
				#pragma HLS pipeline
                normalized =
                  (batch_layers[j * input_dim2 + k].data.data0 - mean0) / pow(variance0 + eps, 0.5);
                out_layers[j * input_dim2 + k].data.data0 =
                  weight[i/2].data.data0 * normalized + bias[i/2].data.data0;
                normalized =
                  (batch_layers[j * input_dim2 + k].data.data1 - mean1) / pow(variance1 + eps, 0.5);
                out_layers[j * input_dim2 + k].data.data1 =
                  weight[i/2].data.data1 * normalized + bias[i/2].data.data1;
            }
        }
    }
}


/* bilinear interpolation, scale_factor = 2, align_corners = False */
void upsample(dma_data* input, dma_data* output, int N, int C)
{
    data_type x_in, y_in;
    int x_west, y_north, x_east, y_south;
    data_type dx_west, dx_east, dy_north, dy_south;
    data_type a0, b0, c0, d0;
    data_type a1, b1, c1, d1;
    data_type pixel0, pixel1;

    int ch = N;
    int h_in = C;
    int w_in = C;
    int factor = 2;

    int i, j, k;

    // loop over output feature pixels
    for (i = 0; i < h_in*factor; i++)
        for (j = 0; j < w_in*factor; j++)
        {
			#pragma HLS PIPELINE
            // map output coord (i,j) to input space (y_in, x_in)
            //  align_corners = False
            x_in = ((data_type)j + 0.5)/factor - 0.5;
            y_in = ((data_type)i + 0.5)/factor - 0.5;

            /** align_corners = True
            x_in = ((data_type)(w_in - 1)/(w_in*factor - 1)) * j;
            y_in = ((data_type)(h_in - 1)/(h_in*factor - 1)) * i;
            */

            // get indices of nearest neighbors
            x_west  = floor(x_in);      // j (col) coord of west neighbors
            x_east  = x_west + 1;       // j (col) coord of east neighbors
            y_north = floor(y_in);      // i (row) coord of north neighbors
            y_south = y_north + 1;      // i (row) coord of south neighbors

            // check boundaries
            x_west  = (x_west < 0)? 0 : x_west;
            x_east  = (x_east < w_in)? x_east : w_in - 1;
            y_north = (y_north < 0)? 0 : y_north;
            y_south = (y_south < h_in)? y_south : h_in - 1;

            // calculate distances relative to neighbors
            dx_west  = x_in - x_west;
            dx_east  = 1 - dx_west;
            dy_north = y_in - y_north;
            dy_south = 1 - dy_north;

            // loop over output channels (two at once for packed data)
            for(k = 0; k < ch; k += 2)
            {
			#pragma HLS UNROLL
                // get neighbor values
                a0 = input[k/2*C1*C1 + y_north*C1 + x_west].data.data0;    // NW
                b0 = input[k/2*C1*C1 + y_north*C1 + x_east].data.data0;    // NE
                c0 = input[k/2*C1*C1 + y_south*C1 + x_west].data.data0;    // SW
                d0 = input[k/2*C1*C1 + y_south*C1 + x_east].data.data0;    // SE

                a1 = input[k/2*C1*C1 + y_north*C1 + x_west].data.data1;    // NW
                b1 = input[k/2*C1*C1 + y_north*C1 + x_east].data.data1;    // NE
                c1 = input[k/2*C1*C1 + y_south*C1 + x_west].data.data1;    // SW
                d1 = input[k/2*C1*C1 + y_south*C1 + x_east].data.data1;    // SE

                // calculate output (interpolating over both dimensions)
                pixel0 = dy_south*(a0*dx_east + b0*dx_west) +
                         dy_north*(c0*dx_east + d0*dx_west);
                pixel1 = dy_south*(a1*dx_east + b1*dx_west) +
                         dy_north*(c1*dx_east + d1*dx_west);
                output[k/2*C1*factor*C1*factor + i*C1*factor + j].data.data0 = pixel0;
                output[k/2*C1*factor*C1*factor + i*C1*factor + j].data.data1 = pixel1;
            }
        }

}


template <int Tr, int Tc, int Tn>
void read_ifmap_deconv2d(data_type feature_temp[Tn][Tr][Tc], dma_data* feature, int tr, int ti, int tc, int H, int C, int K, ap_uint<32> Base_addr2){
#pragma HLS INLINE off
	dma_data tmp;
    int dim_start = (int)(K/2);
    int dim_end = (int)(C+K/2-1);
	int trr, tcc, tii;
	for (tii = 0; tii < Tn; tii+=2) {
		for (trr = 0; trr < Tr; trr++) {
			for (tcc = 0; tcc < Tc; tcc++) {
				#pragma HLS PIPELINE
                if ((tr+trr) < dim_start || (tr+trr) > dim_end || (tc+tcc) < dim_start || (tc+tcc) > dim_end){
                    feature_temp[tii][trr][tcc]=0;
                    feature_temp[tii+1][trr][tcc]=0;
                }
                else{
                    tmp=feature[(tii+ti)/2*C*C + (tr+trr-dim_start)*C +(tc+ tcc-dim_start)+Base_addr2/dataw];
                    feature_temp[tii][trr][tcc] = tmp.data.data0;
                    feature_temp[tii+1][trr][tcc] = tmp.data.data1;
                }
			}
		}
	}
}


template <int Tr, int Tc, int Tn, int N, int C, int S ,int K>
void padding(data_type frame[N][Tr][Tc], data_type fpo[N][C*S+K-1][C*S+K-1],int tnn, int H,
				 int Cout, int pr, int pc)
{
	int row, col, depth;
	int temp;

	for (depth = 0; depth < Tn; depth++)
	{
		//#pragma HLS piepline
		for (row = 0; row < Cout+pr; row++)
		{
			//#pragma HLS UNROLL
			for (col = 0; col < Cout+pc; col++)
			{
				#pragma HLS PIPELINE
				if (((row>=pr/2) && (row<=(Cout-1+pr/2))) && ((col>=pc/2) && (col<=(Cout-1+pc/2))) )
				//if (((row>=2) && (row<=3)) && ((col>=2) && (col<=3)) )
				{
					fpo[tnn+depth][row][col]=frame[tnn+depth][row-pr/2][col-pc/2];
				}
				else
				{
					fpo[tnn+depth][row][col]=0;
				}
			}
		}
	}
}


template <int Tr, int Tc, int Tn,int t_C, int t_S,int t_K,int t_N>
void read_padfm_conv2d(data_type padded_fm_temp[Tn][Tr][Tc], data_type data_padded_fm[t_N][t_C*t_S+t_K-1][t_C*t_S+t_K-1], int tr, int ti, int tc, int H, int C, int K){
#pragma HLS INLINE off
	int Row, Col, Depth;

	for (Depth = 0; Depth < Tn ; Depth++)
	{
		for (Row = 0; Row < Tr+K-1; Row++)
		{
			for (Col = 0; Col < Tc+K-1; Col++)
			{
#pragma HLS PIPELINE
				padded_fm_temp[Depth][Row][Col] = data_padded_fm[ti+Depth][tr+Row][tc+Col];
			}
		}
	}
}


template < int Tk, int Tm,int Tn>
void read_wek1(data_type weight_temp[Tm][Tn][Tk][Tk],dma_data* weight, int to, int ti,
             int K, int N, ap_uint<32> Base_addr1){
#pragma HLS INLINE
	int too, tii, tkk1, tkk2;
	dma_data tmp;
	for (too = 0; too < Tm; too+=2) {
        for (tii = 0; tii < Tn; tii++) {
            for(tkk1 = 0; tkk1 < K; tkk1++){
                for(tkk2 = 0; tkk2 < K; tkk2++){
                    #pragma HLS PIPELINE
                    tmp= weight[(too+to)/2*N*K*K + (tii+ti)*K*K + tkk1*K + tkk2 + Base_addr1/dataw];
                    weight_temp[too][tii][tkk1][tkk2] = tmp.data.data0;
                    weight_temp[too+1][tii][tkk1][tkk2] = tmp.data.data1;
                }
            }
        }
	}
}


template <int Tr, int Tc, int TmBuff, int TnBuff,int Tm, int Tn,int TmW, int TnW, int Tri, int Tci,int Tk>
void comp_engine_deconv_2d(
				 data_type weight_temp[TmW][TnW][Tk][Tk], data_type feature_temp[TnBuff][Tri][Tci],data_type output_core_temp[TmBuff][Tr][Tc],
                 int K , int S
                 ){
	#pragma HLS INLINE off
	int too, tcc, tii, trr,tkk1,tkk2,tncomp,tmcomp;
    data_type tmp0,tmp1;
	//TODO: balanced unrolling input channel and output channel
    for(tncomp=0;tncomp <TnBuff;tncomp+=Tn){
        for(tmcomp=0;tmcomp <TmBuff;tmcomp+=Tm){
            for (tkk1=0; tkk1<K; tkk1++){
                for(tkk2=0; tkk2<K; tkk2++){
                    for (tcc = 0; tcc < Tc; ++tcc) {
                        for (trr = 0; trr < Tr; ++trr) {
                #pragma HLS PIPELINE
                            for (tii = 0; tii < Tn; ++tii) {
                #pragma HLS UNROLL
                                #pragma HLS DEPENDENCE variable=feature_temp inter false
                                tmp1=feature_temp[tncomp+tii][trr*S+tkk1][tcc*S+tkk2];
                                for (too = 0; too < Tm; ++too) {
                                    #pragma HLS DEPENDENCE variable=output_core_temp inter false
                #pragma HLS UNROLL
                                    output_core_temp[tmcomp+too][trr][tcc]+=
                                    tmp1*weight_temp[tmcomp+too][tncomp+tii][tkk1][tkk2];

                                }
                            }

                        }
                    }
                }
            }
        }
    }
}



template <
int TmBuff, int TnBuff, int Tr, int Tc, int Tm, int Tn,int TmW,int TnW, int Tk,int Tri,int Tci,
int t_N,int t_C, int t_S, int t_K>
void deconv(
                        dma_data* weight,
                        dma_data* feature,
                        dma_data* output_core,
                        ap_uint<32> Base_addr1,
                        ap_uint<32>  Base_addr2,
                        ap_uint<32>  Base_addr3,
						data_type feature_temp_p[Tn][t_C][t_C],
						data_type padded_fm[t_N][t_C*t_S+t_K-1][t_C*t_S+t_K-1],
                       data_type output_core_temp[TmBuff][Tr][Tc],data_type weight_temp[TmW][TnW][Tk][Tk],
                       data_type feature_temp[TnBuff][Tri][Tci] , data_type feature_temp1[TnBuff][Tri][Tci],
                       int M, int H,
					   int C,  int K , int S,
					   int CO,
					   int N,
                       dma_data tmp,
                        int tr, int tc, int to, int ti, int trr,
                        int tcc, int too, int tii,
                        int tc_r, int tr_r, int to_r, int ti_r,
                        int lr_i
                       )
{
	int padr = C*S-CO+K-1;  //CO+padr= C*S-CO+K-1 +CO = C*S+K-1
	int padc = C*S-CO+K-1;



    for (trr = 0; trr < C; trr += C){

  		for (tcc = 0; tcc < C; tcc += C){
			for (ti = 0; ti < N; ti += Tn){
  				read_ifmap_deconv2d<t_C,t_C,Tn>(feature_temp_p,feature,trr,ti,tcc,H,C,K,Base_addr2);
  				padding<t_C,t_C,Tn,t_N,t_C,t_S,t_K>(feature_temp_p, padded_fm, ti, H,CO, padr, padc);
  			}
  		}
  	}

    //TODO: buffer initialization
    read_padfm_conv2d<Tri,Tci,TnBuff,t_C,t_S,t_K,t_N>(feature_temp, padded_fm,0,0,0, H,C,K);
    for (tc=0; tc<C; tc+=Tc){
        for (tr=0; tr <C; tr+=Tr){
            for (to = 0; to < M; to += TmBuff) {
                for (ti = 0; ti < N; ti += TnBuff) {
                    read_wek1<Tk,TmW,TnW>(weight_temp,weight,to,ti,K,N,Base_addr1);
                    //read_padfm_conv2d<Tri,Tci,TnBuff>(feature_temp,feature,tr,ti,tc,H,C,K,Base_addr2);
                    //comp_engine_conv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);

                    if (lr_i==0){

                        //ping pong logic for index shifting
                        //ti_r=ti;
                        to_r=to;
                        tc_r=tc;
                        tr_r=tr;
                        ti_r=ti+TnBuff;
                        if (ti_r==N){
                            ti_r=0;
                            if(to == M-TmBuff ){
                                tr_r=tr+Tr;
                                if(tr_r==C){
                                    tr_r=0;
                                    tc_r=tc_r+Tc;
                                    if(tc_r==C){
                                        tc_r=0;
                                    }
                                }
                            }
                        }
                        //TODO: controlling port to switch
                        read_padfm_conv2d<Tri,Tci,TnBuff,t_C,t_S,t_K,t_N>(feature_temp1,padded_fm,tr_r,ti_r,tc_r,H,C,K);
                        comp_engine_deconv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);
                        lr_i=1-lr_i;
                    }
                    else{


                        //ping pong logic for index shifting
                        //ti_r=ti;
                        to_r=to;
                        tc_r=tc;
                        tr_r=tr;
                        ti_r=ti+TnBuff;
                        if (ti_r==N){
                            ti_r=0;
                            if(to == M-TmBuff ){
                                tr_r=tr+Tr;
                                if(tr_r==C){
                                    tr_r=0;
                                    tc_r=tc_r+Tc;
                                    if(tc_r==C){
                                        tc_r=0;
                                    }
                                }
                            }
                        }
                        //TODO: controlling port to switch
                        read_padfm_conv2d<Tri,Tci,TnBuff,t_C,t_S,t_K,t_N>(feature_temp, padded_fm,tr_r,ti_r,tc_r,H,C,K);
                        comp_engine_deconv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp1,output_core_temp,K,S);
                        lr_i=1-lr_i;
                    }

                }

            for (too = 0; too < TmBuff; too+=2) {
                for (trr = 0; trr < Tr; trr++) {
                    for (tcc = 0; tcc < Tc; tcc++) {
                        #pragma HLS PIPELINE
                        tmp.data.data0=output_core_temp[too][trr][tcc];
                        tmp.data.data1=output_core_temp[too+1][trr][tcc];
                        output_core[(too + to)/2*C*C + (tr+trr)*C +tc+ tcc+Base_addr3/dataw]=tmp;
                    }
                }
            }


            for (trr = 0; trr < Tr; ++trr) {
                for (tcc = 0; tcc < Tc; ++tcc) {
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


}


template <int N,int C, int S, int K,int TmBuff, int TnBuff,  int Tr, int Tc, int Tm, int Tn, int TmW, int TnW, int Tk,int Tri,int Tci>
void deconv_k_wrapper(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
	int con,
	ap_uint<32> Base_addr1,
	ap_uint<32>  Base_addr2,
	ap_uint<32>  Base_addr3,
    int temp_M,int temp_H, int temp_C, int temp_K , int temp_N,int temp_S, int temp_CO) {
	dma_data tmp;
	int tr,tc;
	int to, ti, trr, tcc, too, tii;
    int tc_r, tr_r, to_r, ti_r;
	int lr_i=0;

	data_type output_core_temp[TmBuff][Tr][Tc] = { 0 };
	#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=1
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp block factor=Tr/2 dim=2
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=3
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

	data_type weight_temp[TmW][TnW][Tk][Tk] = { 0}, feature_temp[TnBuff][Tri][Tci] = { 0 };
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

	data_type feature_temp_p[Tn][C][C];

	data_type padded_fm[N][C*S+K-1][C*S+K-1];

    if(con==0x00000001){
        //Expand
        #pragma HLS allocation instances=deconv limit=1 function

     deconv<TmBuff,TnBuff,Tr, Tc, Tm,Tn,TmW,TnW, Tk,Tri,Tci,N,C,S,K>
     	 	 	 	 (       weight,
                             feature,
                             output_core,
                             Base_addr1,
                             Base_addr2,
                             Base_addr3,
							 feature_temp_p,
							 padded_fm,
                             output_core_temp,weight_temp,
                             feature_temp, feature_temp1,
                             temp_M,temp_H,
							 temp_C, temp_K , temp_S,
							 temp_CO,
							 temp_N,
                             tmp,
                             tr,tc,
                             to, ti, trr, tcc, too, tii,
                             tc_r, tr_r, to_r, ti_r,
                             lr_i
                       );
    }
}
