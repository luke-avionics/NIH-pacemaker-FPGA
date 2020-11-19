#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;


//TmBuff TnBuff are used to determine the size of the buffer
//      which is standing alone from the parallelism 
//For simplicity now, just use TmBuff=Tm ....

template <int Tr, int Tc, int Tn>
void read_ifmap_conv2d(data_type feature_temp[Tn][Tr][Tc], dma_data* feature, int tr, int ti, int tc, int H, int C, int K, ap_uint<32> Base_addr2){
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




// template < int Tn>
// void read_ifmap_conv_fc(data_type feature_temp[Tn][1][1], dma_data* feature,int ti,int N, int C,ap_uint<32> Base_addr2){
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


template < int Tn>
void read_ifmap_fc(data_type feature_temp[Tn][1][1], dma_data* feature,int ti, ap_uint<32> Base_addr2){
#pragma HLS INLINE off
	dma_data tmp;
	int tii;
	for (tii = 0; tii < Tn; tii+=2) {
		#pragma HLS PIPELINE
        tmp=feature[(tii+ti)/2 +Base_addr2/dataw];
        feature_temp[tii][0][0] = tmp.data.data0;
        feature_temp[tii+1][0][0] = tmp.data.data1;

	}
}

template <int Tm,int Tn>
void read_we1(data_type weight_temp[Tm][Tn][1][1],dma_data* weight, int to, int ti,
               int N, ap_uint<32> Base_addr1){
#pragma HLS INLINE
	int too,tii;
	dma_data tmp;

	for (too = 0; too < Tm; too+=2) {
        for (tii = 0; tii < Tn; tii++) {
            #pragma HLS PIPELINE
            tmp= weight[(too + to)/2*N +(tii+ti) +Base_addr1/dataw];
            weight_temp[too][tii][0][0] = tmp.data.data0;
            weight_temp[too+1][tii][0][0] = tmp.data.data1;

        }
	}
}


template < int Tk, int Tm,int Tn>
void read_wek(data_type weight_temp[Tm][Tn][Tk][Tk],dma_data* weight, int to, int ti,
             int K, int N, ap_uint<32> Base_addr1){
#pragma HLS INLINE
	int too,tii, tkk1,tkk2;
	dma_data tmp;

	for (too = 0; too < Tm; too+=2) {
        for (tii = 0; tii < Tn; tii++) {
            for(tkk1 =0; tkk1<K; tkk1++){
                for(tkk2 =0; tkk2<K; tkk2++){
                    #pragma HLS PIPELINE
                    tmp= weight[(too + to)/2*N*K*K + (tii+ti)*K*K+tkk1*K +tkk2 +Base_addr1/dataw];
                    weight_temp[too][tii][tkk1][tkk2] = tmp.data.data0;
                    weight_temp[too+1][tii][tkk1][tkk2] = tmp.data.data1;
                }
            }
        }
	}
}


template <int Tr, int Tc, int TmBuff, int TnBuff,int Tm, int Tn,int TmW, int TnW, int Tri, int Tci,int Tk>
void comp_engine_conv_2d(
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



template <int TmBuff, int TnBuff,int Tm, int Tn,int TmW, int TnW>
void comp_engine_fc(
				 data_type weight_temp[TmW][TnW][1][1], data_type feature_temp[TnBuff][1][1],data_type output_core_temp[TmBuff][1][1],
                 int S
                 ){
#pragma HLS INLINE off
	int too,tii,tncomp,tmcomp;
    data_type tmp1;
	//TODO: balanced unrolling input channel and output channel
    for(tncomp=0;tncomp <TnBuff;tncomp+=Tn){
        for(tmcomp=0;tmcomp <TmBuff;tmcomp+=Tm){    
            #pragma HLS PIPELINE
            for (tii = 0; tii < Tn; ++tii) {
                #pragma HLS UNROLL
                #pragma HLS DEPENDENCE variable=feature_temp inter false
                tmp1=feature_temp[tncomp+tii][0][0];
                for (too = 0; too < Tm; ++too) {
                    #pragma HLS DEPENDENCE variable=output_core_temp inter false
                    #pragma HLS UNROLL
                    output_core_temp[tmcomp+too][0][0]+=
                    tmp1*weight_temp[tmcomp+too][tncomp+tii][0][0];
                                
                }
            }
                            

        }
    }
}

template <
int TmBuff, int TnBuff, int Tr, int Tc, int Tm, int Tn,int TmW,int TnW, int Tk,int Tri,int Tci
>
void single_conv_k(
                        dma_data* weight,
                        dma_data* feature,
                        dma_data* output_core,
                        ap_uint<32> Base_addr1,
                        ap_uint<32>  Base_addr2,
                        ap_uint<32>  Base_addr3,
                       data_type output_core_temp[TmBuff][Tr][Tc],data_type weight_temp[TmW][TnW][Tk][Tk],
                       data_type feature_temp[TnBuff][Tri][Tci] , data_type feature_temp1[TnBuff][Tri][Tci],
                       int M, int N,int H, int C,  int K , int S,
                       dma_data tmp,
                        int tr, int tc, int to, int ti, int trr,
                        int tcc, int too, int tii,
                        int tc_r, int tr_r, int to_r, int ti_r,
                        int lr_i
                       )
{
    
    
    //TODO: buffer initialization
    read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature,0,0,0, H,C,K,Base_addr2);
    for (tc=0; tc<C; tc+=Tc){
        for (tr=0; tr <C; tr+=Tr){
            for (to = 0; to < M; to += TmBuff) {
                for (ti = 0; ti < N; ti += TnBuff) {
                    read_wek<Tk,TmW,TnW>(weight_temp,weight,to,ti,K,N,Base_addr1);
                    //read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp,feature,tr,ti,tc,H,C,K,Base_addr2);
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
                        read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp1,feature,tr_r,ti_r,tc_r,H,C,K,Base_addr2);
                        comp_engine_conv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);
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
                        read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp, feature,tr_r,ti_r,tc_r,H,C,K,Base_addr2);
                        comp_engine_conv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp1,output_core_temp,K,S);
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



template <
int TmBuff, int TnBuff, int Tm, int Tn,int TmW,int TnW
>
void single_fc(
                        dma_data* weight,
                        dma_data* feature,
                        dma_data* output_core,
                        ap_uint<32> Base_addr1,
                        ap_uint<32>  Base_addr2,
                        ap_uint<32>  Base_addr3,
                       data_type output_core_temp[TmBuff][1][1],data_type weight_temp[TmW][TnW][1][1],
                       data_type feature_temp[TnBuff][1][1] , data_type feature_temp1[TnBuff][1][1],
                       int M, int N,int S,
                       dma_data tmp,
                        int to, int ti, 
                        int too, int tii,
                        int ti_r,
                        int lr_i
                       )
{
    
    
    //TODO: buffer initialization
	read_ifmap_fc<TnBuff>(feature_temp, feature,0,Base_addr2);
	for (to = 0; to < M; to += TmBuff) {
		for (ti = 0; ti < N; ti += TnBuff) {
			read_we1<TmW,TnW>(weight_temp,weight,to,ti,N,Base_addr1);
			//read_ifmap_conv2d<Tri,Tci,TnBuff>(feature_temp,feature,tr,ti,tc,H,C,K,Base_addr2);
			//comp_engine_conv_2d<Tr,Tc,TmBuff,TnBuff,Tm,Tn,TmW,TnW,Tri,Tci,Tk>(weight_temp,feature_temp,output_core_temp,K,S);

			if (lr_i==0){

				//ping pong logic for index shifting
				//ti_r=ti;
				ti_r=ti+TnBuff;
				if (ti_r==N){
					ti_r=0;
				}
				//TODO: controlling port to switch
				read_ifmap_fc<TnBuff>(feature_temp1,feature,ti_r,Base_addr2);
				comp_engine_fc<TmBuff,TnBuff,Tm,Tn,TmW,TnW>(weight_temp,feature_temp,output_core_temp,S);
				lr_i=1-lr_i;
			}
			else{


				//ping pong logic for index shifting
				//ti_r=ti;
				ti_r=ti+TnBuff;
				if (ti_r==N){
					ti_r=0;
				}
				//TODO: controlling port to switch
				read_ifmap_fc<TnBuff>(feature_temp,feature,ti_r,Base_addr2);
				comp_engine_fc<TmBuff,TnBuff,Tm,Tn,TmW,TnW>(weight_temp,feature_temp1,output_core_temp,S);
				lr_i=1-lr_i;
			}

		}

	for (too = 0; too < TmBuff; too+=2) {
		#pragma HLS PIPELINE
		tmp.data.data0=output_core_temp[too][0][0];
		tmp.data.data1=output_core_temp[too+1][0][0];
		output_core[(too + to)/2+Base_addr3/dataw]=tmp;

	}


	for (too = 0; too < TmBuff; ++too) {
		#pragma HLS UNROLL
			output_core_temp[too][0][0] = 0;
	}



	}

   
                       
}


template <
int TmBuff, int TnBuff,  int Tm, int Tn, int TmW, int TnW
>
void fc_wrapper(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
	int con,
	ap_uint<32> Base_addr1,
	ap_uint<32>  Base_addr2,
	ap_uint<32>  Base_addr3,
    int temp_M,int temp_N , int temp_S) {
	dma_data tmp;
	int to, ti, too, tii;
    int ti_r;
	int lr_i=0;

	data_type output_core_temp[TmBuff][1][1] = { 0 };
	#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=1
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp block factor=Tr/2 dim=2
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=3
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

	data_type weight_temp[TmW][TnW][1][1] = { 0}, feature_temp[TnBuff][1][1] = { 0 };
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
    

    if(con==0x00000001){
        //Expand
        #pragma HLS allocation instances=dw_comp_engine2_111 limit=1 function
        
     single_fc<TmBuff,TnBuff,Tm,Tn,TmW,TnW>
                     (       weight,
                             feature,
                             output_core,
                             Base_addr1,
                             Base_addr2,
                             Base_addr3,
                             output_core_temp,weight_temp,
                             feature_temp, feature_temp1,
                             temp_M,temp_N, temp_S,
                             tmp,
                             to, ti, too, tii,
                             ti_r,
                             lr_i
                       );
    }
}



template <
int TmBuff, int TnBuff,  int Tr, int Tc, int Tm, int Tn, int TmW, int TnW, int Tk,int Tri,int Tci
>
void conv_k_wrapper(
		dma_data* weight,
		dma_data* feature,
		dma_data* output_core,
	int con,
	ap_uint<32> Base_addr1,
	ap_uint<32>  Base_addr2,
	ap_uint<32>  Base_addr3,
    int temp_M,int temp_N, int temp_H, int temp_C, int temp_K , int temp_S) {
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
    

    if(con==0x00000001){
        //Expand
        #pragma HLS allocation instances=single_conv_k limit=1 function
        
     single_conv_k<TmBuff,TnBuff,Tr, Tc, Tm,Tn,TmW,TnW, Tk,Tri,Tci>
                     (       weight,
                             feature,
                             output_core,
                             Base_addr1,
                             Base_addr2,
                             Base_addr3,
                             output_core_temp,weight_temp,
                             feature_temp, feature_temp1,
                             temp_M,temp_N, temp_H, temp_C, temp_K , temp_S,
                             tmp,
                             tr,tc,
                             to, ti, trr, tcc, too, tii,
                             tc_r, tr_r, to_r, ti_r,
                             lr_i
                       );
    }
}


void tanh_layer(dma_data* output_core,dma_data* weight,
                int M, int N,int C,int K,
                ap_uint<32> Base_addr2,ap_uint<32> Base_addr3){
    dma_data tmp,tmp1;
    int tm,tc,tr;
    for (tm=0; tm<M;tm+=2)
        for(tc=0;tc<C;tc++)
            for(tr=0; tr<C;tr++){
                #pragma HLS PIPELINE
                tmp=output_core[(tm)/2*C*C + (tc)*C +tr+Base_addr3/dataw];
                tmp1=weight[M*N*K*K/2+tm/2+Base_addr2/dataw];
                tmp.data.data0=tanh(tmp.data.data0+tmp1.data.data0);
                tmp.data.data1=tanh(tmp.data.data1+tmp1.data.data1);
                output_core[(tm)/2*C*C + (tc)*C +tr+Base_addr3/dataw]=tmp;
                }

}

void tanh_layer_fc(dma_data* output_core,dma_data* weight,
                int M, int N,
                ap_uint<32> Base_addr2,ap_uint<32> Base_addr3){
    dma_data tmp,tmp1;
    int tm;
    for (tm=0; tm<M;tm+=2)
    {
        #pragma HLS PIPELINE
        tmp=output_core[(tm)/2+Base_addr3/dataw];
        tmp1=weight[M*N/2+tm/2+Base_addr2/dataw];
        tmp.data.data0=tanhf(tmp.data.data0+tmp1.data.data0);
        tmp.data.data1=tanhf(tmp.data.data1+tmp1.data.data1);
        output_core[(tm)/2+Base_addr3/dataw]=tmp;
    }

}

void max_pool(dma_data* input_next,dma_data* output,
                int M, int C_next,int C,
                ap_uint<32> Base_addr2,ap_uint<32> Base_addr3){

    dma_data tmp1,tmp2,tmp3,tmp4;
    dma_data max_tmp;
    int tm,tc,tr;

    for (tm=0; tm<M;tm+=2)
        for(tc=0;tc<C;tc+=2)
            for(tr=0; tr<C;tr+=2){
                #pragma HLS PIPELINE
                tmp1=output[(tm)/2*C*C + (tc)*C +tr+Base_addr3/dataw];
                tmp2=output[(tm)/2*C*C + (tc+1)*C +tr+Base_addr3/dataw];
                tmp3=output[(tm)/2*C*C + (tc)*C +tr+1+Base_addr3/dataw];
                tmp4=output[(tm)/2*C*C + (tc+1)*C +tr+1+Base_addr3/dataw];
                max_tmp.data.data0=fmax(fmax(tmp1.data.data0,tmp2.data.data0),fmax(tmp3.data.data0,tmp4.data.data0));
                max_tmp.data.data1=fmax(fmax(tmp1.data.data1,tmp2.data.data1),fmax(tmp3.data.data1,tmp4.data.data1));
                input_next[(tm)/2*C_next*C_next + (tc/2)*C_next +tr/2+Base_addr2/dataw]=max_tmp;
            }
}


/* bilinear interpolation, scale_factor = 2, align_corners = False */
void upsample(dma_data* input, dma_data* output)
{
    data_type x_in, y_in;
    int x_west, y_north, x_east, y_south;
    data_type dx_west, dx_east, dy_north, dy_south;
    data_type a0, b0, c0, d0;
    data_type a1, b1, c1, d1;
    data_type pixel0, pixel1;

    int ch = N1;
    int h_in = C1;
    int w_in = C1;
    int factor = 2;

    int i, j, k;

    // loop over output feature pixels
    for (i = 0; i < h_in*factor; i++)
        for (j = 0; j < w_in*factor; j++)
        {
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
            x_east  = ceil(x_in);       // j (col) coord of east neighbors
            y_north = floor(y_in);      // i (row) coord of north neighbors
            y_south = ceil(y_in);       // i (row) coord of south neighbors

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
#pragma HLS PIPELINE
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


void unet_top (
dma_data* weight1,
dma_data* feature1,
dma_data* output_core1,
dma_data* weight2,
dma_data* feature2,
dma_data* output_core2,
dma_data* weight3,
dma_data* feature3,
dma_data* output_core3,
 dma_data* weight4,
// dma_data* feature4,
 dma_data* output_core4,
// dma_data* weight5,
// dma_data* feature5,
// dma_data* output_core5,
// dma_data* weight6,
// dma_data* feature6,
// dma_data* output_core6,
// dma_data* weight7,
// dma_data* feature7,
// dma_data* output_core7,
// dma_data* weight8,
// dma_data* feature8,
// dma_data* output_core8,
//dma_data* weight9,
//dma_data* feature9,
//dma_data* output_core9,
//dma_data* weight10,
//dma_data* feature10,
//dma_data* output_core10,
//dma_data* weight11,
//dma_data* feature11,
//dma_data* output_core11,
//dma_data* weight12,
//dma_data* feature12,
//dma_data* output_core12,
//dma_data* weight13,
//dma_data* feature13,
//dma_data* output_core13,
//dma_data* weight14,
//dma_data* feature14,
//dma_data* output_core14,
//dma_data* weight15,
//dma_data* feature15,
//dma_data* output_core15,
//dma_data* weight16,
//dma_data* feature16,
//dma_data* output_core16,
//dma_data* weight17,
//dma_data* feature17,
//dma_data* output_core17,
int con,
// ap_uint<32> Base_addr1,
// ap_uint<32>  Base_addr2,
// ap_uint<32>  Base_addr3,
// ap_uint<32> Base_addr4,
// ap_uint<32>  Base_addr5,
// ap_uint<32>  Base_addr6,
// ap_uint<32> Base_addr7,
// ap_uint<32>  Base_addr8,
// ap_uint<32>  Base_addr9,
// ap_uint<32> Base_addr10,
// ap_uint<32>  Base_addr11,
// ap_uint<32>  Base_addr12,
//ap_uint<32> Base_addr13,
//ap_uint<32>  Base_addr14,
//ap_uint<32>  Base_addr15,
//ap_uint<32> Base_addr16,
//ap_uint<32>  Base_addr17,
//ap_uint<32>  Base_addr18,
//ap_uint<32> Base_addr19,
//ap_uint<32>  Base_addr20,
//ap_uint<32>  Base_addr21,
//ap_uint<32> Base_addr22,
//ap_uint<32>  Base_addr23,
//ap_uint<32>  Base_addr24,
//ap_uint<32> Base_addr25,
//ap_uint<32>  Base_addr26,
//ap_uint<32>  Base_addr27,
  ap_uint<32> Base_addr28,
 // ap_uint<32>  Base_addr29,
  ap_uint<32>  Base_addr30,
ap_uint<32> Base_addr31,
ap_uint<32>  Base_addr32,
ap_uint<32>  Base_addr33,
ap_uint<32> Base_addr34,
ap_uint<32>  Base_addr35,
ap_uint<32>  Base_addr36,
ap_uint<32> Base_addr37,
ap_uint<32>  Base_addr38,
ap_uint<32>  Base_addr39
//ap_uint<32> Base_addr40,
//ap_uint<32>  Base_addr41,
//ap_uint<32>  Base_addr42,
//ap_uint<32> Base_addr43,
//ap_uint<32>  Base_addr44,
//ap_uint<32>  Base_addr45,
//ap_uint<32> Base_addr46,
//ap_uint<32>  Base_addr47,
//ap_uint<32>  Base_addr48,
//ap_uint<32> Base_addr49,
//ap_uint<32>  Base_addr50,
//ap_uint<32>  Base_addr51
){
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=con bundle=CRTL_BUS

#pragma HLS INTERFACE s_axilite port=Base_addr37 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr38 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr39 bundle=CRTL_BUS
#pragma HLS INTERFACE m_axi depth=M1*N1*K1*K1/2 port=weight1
#pragma HLS INTERFACE m_axi depth=N1*H1*H1/2 port=feature1
#pragma HLS INTERFACE m_axi depth=M1*C1*C1/2 port=output_core1
#pragma HLS data_pack variable=weight1
#pragma HLS data_pack variable=feature1
#pragma HLS data_pack variable=output_core1


#pragma HLS INTERFACE s_axilite port=Base_addr34 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr35 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr36 bundle=CRTL_BUS
#pragma HLS INTERFACE m_axi depth=M2*N2*K2*K2/2 port=weight2
#pragma HLS INTERFACE m_axi depth=M2*H2*H2/2 port=feature2
#pragma HLS INTERFACE m_axi depth=M2*C2*C2/2 port=output_core2
#pragma HLS data_pack variable=weight2
#pragma HLS data_pack variable=feature2
#pragma HLS data_pack variable=output_core2

#pragma HLS INTERFACE s_axilite port=Base_addr31 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr32 bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=Base_addr33 bundle=CRTL_BUS
#pragma HLS INTERFACE m_axi depth=M3*N3/2 port=weight3
#pragma HLS INTERFACE m_axi depth=N3/2 port=feature3
#pragma HLS INTERFACE m_axi depth=M3/2 port=output_core3
#pragma HLS data_pack variable=weight3
#pragma HLS data_pack variable=feature3
#pragma HLS data_pack variable=output_core3


 #pragma HLS INTERFACE s_axilite port=Base_addr28 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr29 bundle=CRTL_BUS
 #pragma HLS INTERFACE s_axilite port=Base_addr30 bundle=CRTL_BUS
 #pragma HLS INTERFACE m_axi depth=M4*N4/2 port=weight4
// #pragma HLS INTERFACE m_axi depth=N4/2 port=feature4
 #pragma HLS INTERFACE m_axi depth=M4/2 port=output_core4
 #pragma HLS data_pack variable=weight4
// #pragma HLS data_pack variable=feature4
 #pragma HLS data_pack variable=output_core4


// #pragma HLS INTERFACE s_axilite port=Base_addr1 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr2 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr3 bundle=CRTL_BUS
// #pragma HLS INTERFACE m_axi depth=M4*N4*K4*K4/4 port=weight5
// #pragma HLS INTERFACE m_axi depth=N4*H4*H4/4 port=feature5
// #pragma HLS INTERFACE m_axi depth=M4*C4*C4/4 port=output_core5
// #pragma HLS data_pack variable=weight5
// #pragma HLS data_pack variable=feature5
// #pragma HLS data_pack variable=output_core5

// #pragma HLS INTERFACE s_axilite port=Base_addr4 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr5 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr6 bundle=CRTL_BUS
// #pragma HLS INTERFACE m_axi depth=M5*N5*K5*K5/4 port=weight6
// #pragma HLS INTERFACE m_axi depth=N5*H5*H5/4 port=feature6
// #pragma HLS INTERFACE m_axi depth=M5*C5*C5/4 port=output_core6
// #pragma HLS data_pack variable=weight6
// #pragma HLS data_pack variable=feature6
// #pragma HLS data_pack variable=output_core6

// #pragma HLS INTERFACE s_axilite port=Base_addr7 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr8 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr9 bundle=CRTL_BUS
// #pragma HLS INTERFACE m_axi depth=M6*N6*K6*K6/4 port=weight7
// #pragma HLS INTERFACE m_axi depth=N6*H6*H6/4 port=feature7
// #pragma HLS INTERFACE m_axi depth=M6*C6*C6/4 port=output_core7
// #pragma HLS data_pack variable=weight7
// #pragma HLS data_pack variable=feature7
// #pragma HLS data_pack variable=output_core7

// #pragma HLS INTERFACE s_axilite port=Base_addr10 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr11 bundle=CRTL_BUS
// #pragma HLS INTERFACE s_axilite port=Base_addr12 bundle=CRTL_BUS
// #pragma HLS INTERFACE m_axi depth=M7*N7*K7*K7/4 port=weight8
// #pragma HLS INTERFACE m_axi depth=N7*H7*H7/4 port=feature8
// #pragma HLS INTERFACE m_axi depth=M7*C7*C7/4 port=output_core8
// #pragma HLS data_pack variable=weight8
// #pragma HLS data_pack variable=feature8
// #pragma HLS data_pack variable=output_core8

//
//#pragma HLS INTERFACE s_axilite port=Base_addr13 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr14 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr15 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M8*N8*K8*K8/4 port=weight9
//#pragma HLS INTERFACE m_axi depth=N8*H8*H8/4 port=feature9
//#pragma HLS INTERFACE m_axi depth=M8*C8*C8/4 port=output_core9
//#pragma HLS data_pack variable=weight9
//#pragma HLS data_pack variable=feature9
//#pragma HLS data_pack variable=output_core9
//
//#pragma HLS INTERFACE s_axilite port=Base_addr16 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr17 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr18 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M9*N9*K9*K9/4 port=weight10
//#pragma HLS INTERFACE m_axi depth=N9*H9*H9/4 port=feature10
//#pragma HLS INTERFACE m_axi depth=M9*C9*C9/4 port=output_core10
//#pragma HLS data_pack variable=weight10
//#pragma HLS data_pack variable=feature10
//#pragma HLS data_pack variable=output_core10
//
//#pragma HLS INTERFACE s_axilite port=Base_addr19 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr20 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr21 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M10*N10*K10*K10/4 port=weight11
//#pragma HLS INTERFACE m_axi depth=N10*H10*H10/4 port=feature11
//#pragma HLS INTERFACE m_axi depth=M10*C10*C10/4 port=output_core11
//#pragma HLS data_pack variable=weight11
//#pragma HLS data_pack variable=feature11
//#pragma HLS data_pack variable=output_core11
//
//#pragma HLS INTERFACE s_axilite port=Base_addr22 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr23 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr24 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M11*N11*K11*K11/4 port=weight12
//#pragma HLS INTERFACE m_axi depth=N11*H11*H11/4 port=feature12
//#pragma HLS INTERFACE m_axi depth=M11*C11*C11/4 port=output_core12
//#pragma HLS data_pack variable=weight12
//#pragma HLS data_pack variable=feature12
//#pragma HLS data_pack variable=output_core12
//
//#pragma HLS INTERFACE s_axilite port=Base_addr25 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr26 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr27 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M12*N12*K12*K12/4 port=weight13
//#pragma HLS INTERFACE m_axi depth=N12*H12*H12/4 port=feature13
//#pragma HLS INTERFACE m_axi depth=M12*C12*C12/4 port=output_core13
//#pragma HLS data_pack variable=weight13
//#pragma HLS data_pack variable=feature13
//#pragma HLS data_pack variable=output_core13
//
//
//#pragma HLS INTERFACE s_axilite port=Base_addr40 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr41 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr42 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M13*N13*K13*K13/4 port=weight14
//#pragma HLS INTERFACE m_axi depth=N13*H13*H13/4 port=feature14
//#pragma HLS INTERFACE m_axi depth=M13*C13*C13/4 port=output_core14
//#pragma HLS data_pack variable=weight14
//#pragma HLS data_pack variable=feature14
//#pragma HLS data_pack variable=output_core14
//
//#pragma HLS INTERFACE s_axilite port=Base_addr43 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr44 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr45 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M14*N14*K14*K14/4 port=weight15
//#pragma HLS INTERFACE m_axi depth=N14*H14*H14/4 port=feature15
//#pragma HLS INTERFACE m_axi depth=M14*C14*C14/4 port=output_core15
//#pragma HLS data_pack variable=weight15
//#pragma HLS data_pack variable=feature15
//#pragma HLS data_pack variable=output_core15
//
//#pragma HLS INTERFACE s_axilite port=Base_addr46 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr47 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr48 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M15*N15*K15*K15/4 port=weight16
//#pragma HLS INTERFACE m_axi depth=N15*H15*H15/4 port=feature16
//#pragma HLS INTERFACE m_axi depth=M15*C15*C15/4 port=output_core16
//#pragma HLS data_pack variable=weight16
//#pragma HLS data_pack variable=feature16
//#pragma HLS data_pack variable=output_core16
//
//#pragma HLS INTERFACE s_axilite port=Base_addr49 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr50 bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Base_addr51 bundle=CRTL_BUS
//#pragma HLS INTERFACE m_axi depth=M16*N16*K16*K16/4 port=weight17
//#pragma HLS INTERFACE m_axi depth=N16*H16*H16/4 port=feature17
//#pragma HLS INTERFACE m_axi depth=M16*C16*C16/4 port=output_core17
//#pragma HLS data_pack variable=weight17
//#pragma HLS data_pack variable=feature17
//#pragma HLS data_pack variable=output_core17

    //Be sure to check the address order when going for implementation
//	conv1:conv1_1_1<M1,N1,SE1,OM1,H1,C1,HFD1,CFD1,K1,S1,Tr1,Tc1,Tm1,Tn1,Tm1,Tm1>(weight1,feature1,output_core1,con,Base_addr37,Base_addr38,Base_addr39);
//	conv2:conv1_1_1<M2,N2,SE2,OM2,H2,C2,HFD2,CFD2,K2,S2,Tr2,Tc2,Tm2,Tn2,Tm2,Tm2>(weight3,feature3,output_core3,con,Base_addr31,Base_addr32,Base_addr33);
//	conv3:repeat2_conv1_1_1<M3,N3,SE3,OM3,H3,C3,HFD3,CFD3,K3,S3,Tr3,Tc3,Tm3,Tn3,Tm3,Tm3>(weight4,feature4,output_core4,con,Base_addr28,Base_addr29,Base_addr30);
//	conv4:conv1_1_1<M5,N5,SE5,OM5,H5,C5,HFD5,CFD5,K5,S5,Tr5,Tc5,Tm5,Tn5,Tm5,Tm5>(weight5,feature5,output_core5,con,Base_addr1,Base_addr2,Base_addr3);
//	conv5:conv1_1_1<M6,N6,SE6,OM6,H6,C6,HFD6,CFD6,K6,S6,Tr6,Tc6,Tm6,Tn6,Tm6,Tm6>(weight6,feature6,output_core6,con,Base_addr4,Base_addr5,Base_addr6);
//	conv6:repeat2_conv1_1_1<M7,N7,SE7,OM7,H7,C7,HFD7,CFD7,K7,S7,Tr7,Tc7,Tm7,Tn7,Tm7,Tm7>(weight7,feature7,output_core7,con,Base_addr7,Base_addr8,Base_addr9);
//	conv7:conv1_1_1<M9,N9,SE9,OM9,H9,C9,HFD9,CFD9,K9,S9,Tr9,Tc9,Tm9,Tn9,Tm9,Tm9>(weight8,feature8,output_core8,con,Base_addr10,Base_addr11,Base_addr12);
//	conv8:conv1_1_1<M10,N10,SE10,OM10,H10,C10,HFD10,CFD10,K10,S10,Tr10,Tc10,Tm10,Tn10,Tm10,Tm10>(weight9,feature9,output_core9,con,Base_addr13,Base_addr14,Base_addr15);
//	conv9:repeat2_conv1_1_1<M11,N11,SE11,OM11,H11,C11,HFD11,CFD11,K11,S11,Tr11,Tc11,Tm11,Tn11,Tm11,Tm11>(weight10,feature10,output_core10,con,Base_addr16,Base_addr17,Base_addr18);
//	conv10:conv1_1_1<M13,N13,SE13,OM13,H13,C13,HFD13,CFD13,K13,S13,Tr13,Tc13,Tm13,Tn13,Tm13,Tm13>(weight11,feature11,output_core11,con,Base_addr19,Base_addr20,Base_addr21);
//	conv11:repeat2_conv1_1_1<M14,N14,SE14,OM14,H14,C14,HFD14,CFD14,K14,S14,Tr14,Tc14,Tm14,Tn14,Tm14,Tm14>(weight12,feature12,output_core12,con,Base_addr22,Base_addr23,Base_addr24);
//	conv12:conv1_1_1<M16,N16,SE16,OM16,H16,C16,HFD16,CFD16,K16,S16,Tr16,Tc16,Tm16,Tn16,Tm16,Tm16>(weight13,feature13,output_core13,con,Base_addr25,Base_addr26,Base_addr27);
//	conv13:conv1_1_1<M17,N17,SE17,OM17,H17,C17,HFD17,CFD17,K17,S17,Tr17,Tc17,Tm17,Tn17,Tm17,Tm17>(weight14,feature14,output_core14,con,Base_addr40,Base_addr41,Base_addr42);
//	conv14:repeat3_conv1_1_1<M18,N18,SE18,OM18,H18,C18,HFD18,CFD18,K18,S18,Tr18,Tc18,Tm18,Tn18,Tm18,Tm18>(weight15,feature15,output_core15,con,Base_addr43,Base_addr44,Base_addr45);
//	conv15:conv1_1_1<M21,N21,SE21,OM21,H21,C21,HFD21,CFD21,K21,S21,Tr21,Tc21,Tm21,Tn21,Tm21,Tm21>(weight16,feature16,output_core16,con,Base_addr46,Base_addr47,Base_addr48);
	// conv16:conv1_1_1<M16,N16,SE16,OM16,H16,C16,HFD16,CFD16,K16,S16,Tr16,Tc16,Tm16,Tn16,Tm16,Tm16>(weight17,feature17,output_core17,con,Base_addr49,Base_addr50,Base_addr51);



      conv1: conv_k_wrapper<
                             TmBuff1, TnBuff1,Tr1,Tc1,Tm1,Tn1, TmBuff1,TnBuff1,Tk1,Tri1,Tci1>
      (weight1,feature1,output_core1,con,Base_addr37,Base_addr38,Base_addr39,
      M1,N1,H1,C1,K1,S1);


      act1: tanh_layer(output_core1,weight1, M1,N1,C1,K1,Base_addr37,Base_addr39);

      max1: max_pool(feature2,output_core1,M1,C2,C1,Base_addr35,Base_addr39);


      upsample1: upsample(output_core1, feature2);

    
    
//      conv2: conv_k_wrapper<
//                             TmBuff2, TnBuff2,Tr2,Tc2,Tm2,Tn2, TmBuff2,TnBuff2,Tk2,Tri2,Tci2>
//      (weight2,feature2,output_core2,con,Base_addr34,Base_addr35,Base_addr36,
//      M2,N2,H2,C2,K2,S2);
//
//      act2: tanh_layer(output_core2,weight2, M2,N2,C2,K2,Base_addr34,Base_addr36);
//
//      max2: max_pool(feature3,output_core2,M2,C2/2,C2,Base_addr32,Base_addr36);
      
//      fc1: fc_wrapper<
//                      TmBuff3, TnBuff3,Tm3,Tn3,TmBuff3,TnBuff3>
//      (weight3,feature3,output_core3,con,Base_addr31,Base_addr32,Base_addr33,
//      M3,N3,S3);
//      act3: tanh_layer_fc(output_core3,weight3, M3,N3,Base_addr31,Base_addr33);


//      fc2: fc_wrapper<
//                      TmBuff4, TnBuff4,Tm4,Tn4,TmBuff4,TnBuff4>
//      (weight4,output_core3,output_core4,con,Base_addr28,Base_addr33,Base_addr30,
//      M4,N4,S4);
//      act4: tanh_layer_fc(output_core4,weight4, M4,N4,Base_addr28,Base_addr30);

      // conv2: conv_k_wrapper<
                             // TmBuff2, TnBuff2,Tr2,Tc2,Tm2,Tn2, TmBuff2,TnBuff2,Tk2,Tri2,Tci2>
      // (weight2,feature2,output_core2,con,Base_addr34,Base_addr35,Base_addr36,
      // M2,N2,H2,C2,K2,S2);
      // conv3: conv_k_wrapper<
                             // TmBuff3, TnBuff3,Tr3,Tc3,Tm3,Tn3, TmBuff3,TnBuff3,Tk3,Tri3,Tci3>
      // (weight3,feature3,output_core3,con,Base_addr31,Base_addr32,Base_addr33,
      // M3,N3,H3,C3,K3,S3);
      // conv4: conv_k_wrapper<
                             // TmBuff4, TnBuff4,Tr4,Tc4,Tm4,Tn4, TmBuff4,TnBuff4,Tk4,Tri4,Tci4>
      // (weight4,feature4,output_core4,con,Base_addr28,Base_addr29,Base_addr30,
      // M4,N4,H4,C4,K4,S4);
      // conv5: conv_k_wrapper<
                             // TmBuff5, TnBuff5,Tr5,Tc5,Tm5,Tn5, TmBuff5,TnBuff5,Tk5,Tri5,Tci5>
      // (weight5,feature5,output_core5,con,Base_addr1,Base_addr2,Base_addr3,
      // M5,N5,H5,C5,K5,S5);
      // conv6: conv_k_wrapper<
                             // TmBuff6, TnBuff6,Tr6,Tc6,Tm6,Tn6, TmBuff6,TnBuff6,Tk6,Tri6,Tci6>
      // (weight6,feature6,output_core6,con,Base_addr4,Base_addr5,Base_addr6,
      // M6,N6,H6,C6,K6,S6);
      // conv7: conv_k_wrapper<
                             // TmBuff7, TnBuff7,Tr7,Tc7,Tm7,Tn7, TmBuff7,TnBuff7,Tk7,Tri7,Tci7>
      // (weight7,feature7,output_core7,con,Base_addr7,Base_addr8,Base_addr9,
      // M7,N7,H7,C7,K7,S7);
      // conv8: conv_k_wrapper<
                             // TmBuff8, TnBuff8,Tr8,Tc8,Tm8,Tn8, TmBuff8,TnBuff8,Tk8,Tri8,Tci8>
      // (weight8,feature8,output_core8,con,Base_addr10,Base_addr11,Base_addr12,
      // M8,N8,H8,C8,K8,S8);
    
}
