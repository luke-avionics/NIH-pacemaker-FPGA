#ifndef unet
#define unet
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
//#include <ap_cint.h>

#define dataw 32
#define FLOAT 1

#if FLOAT==1
	typedef float data_type;
#else
	typedef ap_fixed<8,3> data_type;
#endif

struct data_pack{
	data_type data0;
	data_type data1;
};

struct dma_data{
	data_pack data;
};
const int TmBuff1=4, TnBuff1=2,Tr1=16,Tc1=16,Tm1=4,Tn1=2,Tk1=7,Tri1=22,Tci1=22;
const int M1 = 48, N1=2,C1=16,H1=22, K1=7, S1=1;

const int TmBuff2=2, TnBuff2=2,Tr2=4,Tc2=4,Tm2=2,Tn2=2,Tk2=5,Tri2=8,Tci2=8;
const int M2 = 96, N2=48,C2=8,H2=12, K2=5, S2=1;

const int TmBuff3=16, TnBuff3=16,Tm3=16,Tn3=16;
const int M3 = 60*4*4, N3=96*4*4, S3=1;

const int TmBuff4=2, TnBuff4=2,Tm4=2,Tn4=2;
const int M4 = 40*4*4, N4=60*4*4, S4=1;


void unet_top (
dma_data* weight1,
dma_data* feature1,
dma_data* output_core1,
dma_data* batchnorm_weight1,
dma_data* batchnorm_bias1,
data_type batchnorm_eps,
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
);

#endif
