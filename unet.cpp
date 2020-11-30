#include "unet.h"
#include "conv.cpp"
#include "deconv.cpp"
#include "fc.cpp"
#include "tanh.cpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;


//TmBuff TnBuff are used to determine the size of the buffer
//      which is standing alone from the parallelism 
//For simplicity now, just use TmBuff=Tm ....


void unet_top(
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
    dma_data* output_core4, /*
    dma_data* weight5,
    dma_data* feature5,
    dma_data* output_core5,
    dma_data* weight6,
    dma_data* feature6,
    dma_data* output_core6,
    dma_data* weight7,
    dma_data* feature7,
    dma_data* output_core7,
    dma_data* weight8,
    dma_data* feature8,
    dma_data* output_core8,
    dma_data* weight9,
    dma_data* feature9,
    dma_data* output_core9,
    dma_data* weight10,
    dma_data* feature10,
    dma_data* output_core10,
    dma_data* weight11,
    dma_data* feature11,
    dma_data* output_core11,
    dma_data* weight12,
    dma_data* feature12,
    dma_data* output_core12,
    dma_data* weight13,
    dma_data* feature13,
    dma_data* output_core13,
    dma_data* weight14,
    dma_data* feature14,
    dma_data* output_core14,
    dma_data* weight15,
    dma_data* feature15,
    dma_data* output_core15,
    dma_data* weight16,
    dma_data* feature16,
    dma_data* output_core16,
    dma_data* weight17,
    dma_data* feature17,
    dma_data* output_core17,*/
    int con, /*
    ap_uint<32> Base_addr1,
    ap_uint<32>  Base_addr2,
    ap_uint<32>  Base_addr3,
    ap_uint<32> Base_addr4,
    ap_uint<32>  Base_addr5,
    ap_uint<32>  Base_addr6,
    ap_uint<32> Base_addr7,
    ap_uint<32>  Base_addr8,
    ap_uint<32>  Base_addr9,
    ap_uint<32> Base_addr10,
    ap_uint<32>  Base_addr11,
    ap_uint<32>  Base_addr12,
    ap_uint<32> Base_addr13,
    ap_uint<32>  Base_addr14,
    ap_uint<32>  Base_addr15,
    ap_uint<32> Base_addr16,
    ap_uint<32>  Base_addr17,
    ap_uint<32>  Base_addr18,
    ap_uint<32> Base_addr19,
    ap_uint<32>  Base_addr20,
    ap_uint<32>  Base_addr21,
    ap_uint<32> Base_addr22,
    ap_uint<32>  Base_addr23,
    ap_uint<32>  Base_addr24,
    ap_uint<32> Base_addr25,
    ap_uint<32>  Base_addr26,
    ap_uint<32>  Base_addr27, */
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
    ap_uint<32>  Base_addr39 /*
    ap_uint<32> Base_addr40,
    ap_uint<32>  Base_addr41,
    ap_uint<32>  Base_addr42,
    ap_uint<32> Base_addr43,
    ap_uint<32>  Base_addr44,
    ap_uint<32>  Base_addr45,
    ap_uint<32> Base_addr46,
    ap_uint<32>  Base_addr47,
    ap_uint<32>  Base_addr48,
    ap_uint<32> Base_addr49,
    ap_uint<32>  Base_addr50,
    ap_uint<32>  Base_addr51 */
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
	// Pragmas for batchnorm input data
	#pragma HLS INTERFACE m_axi depth=M1/2 port=batchnorm_weight1
	#pragma HLS INTERFACE m_axi depth=M1/2 port=batchnorm_bias1
	#pragma HLS data_pack variable=batchnorm_weight1
	#pragma HLS data_pack variable=batchnorm_bias1


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


	#pragma HLS INTERFACE s_axilite port=Base_addr1 bundle=CRTL_BUS
	/*#pragma HLS INTERFACE s_axilite port=Base_addr2 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr3 bundle=CRTL_BUS

	#pragma HLS INTERFACE m_axi depth=N4*H4*H4/4 port=feature5
	#pragma HLS INTERFACE m_axi depth=M4*C4*C4/4 port=output_core5
	#pragma HLS data_pack variable=weight5
	#pragma HLS data_pack variable=feature5
	#pragma HLS data_pack variable=output_core5*/

	#pragma HLS INTERFACE s_axilite port=Base_addr4 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr5 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr6 bundle=CRTL_BUS
	/* #pragma HLS INTERFACE m_axi depth=M5*N5*K5*K5/4 port=weight6
	#pragma HLS INTERFACE m_axi depth=N5*H5*H5/4 port=feature6
	#pragma HLS INTERFACE m_axi depth=M5*C5*C5/4 port=output_core6
	#pragma HLS data_pack variable=weight6
	#pragma HLS data_pack variable=feature6
	#pragma HLS data_pack variable=output_core6 */

	#pragma HLS INTERFACE s_axilite port=Base_addr7 bundle=CRTL_BUS
	// #pragma HLS INTERFACE s_axilite port=Base_addr8 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr9 bundle=CRTL_BUS
	/* #pragma HLS INTERFACE m_axi depth=M6*N6*K6*K6/4 port=weight7
	#pragma HLS INTERFACE m_axi depth=N6*H6*H6/4 port=feature7
	#pragma HLS INTERFACE m_axi depth=M6*C6*C6/4 port=output_core7
	#pragma HLS data_pack variable=weight7
	#pragma HLS data_pack variable=feature7
	#pragma HLS data_pack variable=output_core7 */

	#pragma HLS INTERFACE s_axilite port=Base_addr10 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr11 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr12 bundle=CRTL_BUS
	/* #pragma HLS INTERFACE m_axi depth=M7*N7*K7*K7/4 port=weight8
	#pragma HLS INTERFACE m_axi depth=N7*H7*H7/4 port=feature8
	#pragma HLS INTERFACE m_axi depth=M7*C7*C7/4 port=output_core8
	#pragma HLS data_pack variable=weight8
	#pragma HLS data_pack variable=feature8
	#pragma HLS data_pack variable=output_core8 */

	#pragma HLS INTERFACE s_axilite port=Base_addr13 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr14 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr15 bundle=CRTL_BUS
	/*#pragma HLS INTERFACE m_axi depth=M8*N8*K8*K8/4 port=weight9
	#pragma HLS INTERFACE m_axi depth=N8*H8*H8/4 port=feature9
	#pragma HLS INTERFACE m_axi depth=M8*C8*C8/4 port=output_core9
	#pragma HLS data_pack variable=weight9
	#pragma HLS data_pack variable=feature9
	#pragma HLS data_pack variable=output_core9 */

	#pragma HLS INTERFACE s_axilite port=Base_addr16 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr17 bundle=CRTL_BUS
	// #pragma HLS INTERFACE s_axilite port=Base_addr18 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M9*N9*K9*K9/4 port=weight10
	#pragma HLS INTERFACE m_axi depth=N9*H9*H9/4 port=feature10
	#pragma HLS INTERFACE m_axi depth=M9*C9*C9/4 port=output_core10
	#pragma HLS data_pack variable=weight10
	#pragma HLS data_pack variable=feature10
	#pragma HLS data_pack variable=output_core10

	/* #pragma HLS INTERFACE s_axilite port=Base_addr19 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr20 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr21 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M10*N10*K10*K10/4 port=weight11
	#pragma HLS INTERFACE m_axi depth=N10*H10*H10/4 port=feature11 */
	#pragma HLS INTERFACE m_axi depth=M10*C10*C10/4 port=output_core11
	/* #pragma HLS data_pack variable=weight11
	#pragma HLS data_pack variable=feature11
	#pragma HLS data_pack variable=output_core11 */

	/*#pragma HLS INTERFACE s_axilite port=Base_addr22 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr23 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr24 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M11*N11*K11*K11/4 port=weight12
	#pragma HLS INTERFACE m_axi depth=N11*H11*H11/4 port=feature12
	#pragma HLS INTERFACE m_axi depth=M11*C11*C11/4 port=output_core12
	#pragma HLS data_pack variable=weight12
	#pragma HLS data_pack variable=feature12
	#pragma HLS data_pack variable=output_core12

	#pragma HLS INTERFACE s_axilite port=Base_addr25 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr26 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr27 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M12*N12*K12*K12/4 port=weight13
	#pragma HLS INTERFACE m_axi depth=N12*H12*H12/4 port=feature13
	#pragma HLS INTERFACE m_axi depth=M12*C12*C12/4 port=output_core13
	#pragma HLS data_pack variable=weight13
	#pragma HLS data_pack variable=feature13
	#pragma HLS data_pack variable=output_core13


	#pragma HLS INTERFACE s_axilite port=Base_addr40 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr41 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr42 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M13*N13*K13*K13/4 port=weight14
	#pragma HLS INTERFACE m_axi depth=N13*H13*H13/4 port=feature14
	#pragma HLS INTERFACE m_axi depth=M13*C13*C13/4 port=output_core14
	#pragma HLS data_pack variable=weight14
	#pragma HLS data_pack variable=feature14
	#pragma HLS data_pack variable=output_core14

	#pragma HLS INTERFACE s_axilite port=Base_addr43 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr44 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr45 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M14*N14*K14*K14/4 port=weight15
	#pragma HLS INTERFACE m_axi depth=N14*H14*H14/4 port=feature15
	#pragma HLS INTERFACE m_axi depth=M14*C14*C14/4 port=output_core15
	#pragma HLS data_pack variable=weight15
	#pragma HLS data_pack variable=feature15
	#pragma HLS data_pack variable=output_core15

	#pragma HLS INTERFACE s_axilite port=Base_addr46 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr47 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr48 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M15*N15*K15*K15/4 port=weight16
	#pragma HLS INTERFACE m_axi depth=N15*H15*H15/4 port=feature16
	#pragma HLS INTERFACE m_axi depth=M15*C15*C15/4 port=output_core16
	#pragma HLS data_pack variable=weight16
	#pragma HLS data_pack variable=feature16
	#pragma HLS data_pack variable=output_core16

	#pragma HLS INTERFACE s_axilite port=Base_addr49 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr50 bundle=CRTL_BUS
	#pragma HLS INTERFACE s_axilite port=Base_addr51 bundle=CRTL_BUS
	#pragma HLS INTERFACE m_axi depth=M16*N16*K16*K16/4 port=weight17
	#pragma HLS INTERFACE m_axi depth=N16*H16*H16/4 port=feature17
	#pragma HLS INTERFACE m_axi depth=M16*C16*C16/4 port=output_core17
	#pragma HLS data_pack variable=weight17
	#pragma HLS data_pack variable=feature17
	#pragma HLS data_pack variable=output_core17*/


/***** Part 1: Convolution and Max Pooling with Batch Normalization *****/

	/*
    conv1: single_conv_k<
                            TmBuff1, TnBuff1, Tr1, Tc1, Tm1, Tn1, TmBuff1, TnBuff1, Tk1, Tri1, Tci1>
	(weight1, feature1, output_core1, con, Base_addr37, Base_addr38, Base_addr39,
	M1, N1, H1, C1, K1);

    act1: tanh_layer(output_core1, weight1, M1, N1, C1, K1, Base_addr37, Base_addr39);

	// Here, there are M1 features (since the conv layer will put out this many outputs)
	// Output should be [M1 x C1 x C1] (assuming same conv)

	// batchnorm never actually reuses input values, so use the same memory for input and output
	batchnorm1: batchnorm(output_core1, batchnorm_weight1, batchnorm_bias1, output_core1, batchnorm_eps, C1, C1, M1);

	max1: max_pool(feature2,output_core1,M1,C2,C1,Base_addr35,Base_addr39);

    conv2: single_conv_k<
							TmBuff2, TnBuff2, Tr2, Tc2, Tm2, Tn2, TmBuff2, TnBuff2, Tk2, Tri2, Tci2>
	(weight2, feature2, output_core2, con, Base_addr34, Base_addr35, Base_addr36,
	M2, N2, H2, C2, K2);

	// conv2 synthesized with conv1 parameters (Chunk 1)
    conv2: single_conv_k<
							TmBuff1, TnBuff1, Tr1, Tc1, Tm1, Tn1, TmBuff1, TnBuff1, Tk1, Tri1, Tci1>
	(weight2, feature2, output_core2, con, Base_addr34, Base_addr35, Base_addr36,
	M2, N2, H2, C2, K2);

    act2: tanh_layer(output_core2, weight2, M2, N2, C2, K2, Base_addr34, Base_addr36);

	// batchnorm2 would go here

	// upsample1 for synthesis testing only, not valid location in NN
	upsample1: upsample(output_core1, feature2, M1, C2);	// M1xC2xC2 (48x8x8) back up to M1xC1xC1 (48x16x16)

    max2: max_pool(feature3, output_core2, M2, C2/2, C2, Base_addr32, Base_addr36);
    */


/***** Part 2: FC LAYERS 1-3 *****/

	/*
    fc1: single_fc<
					TmBuff3, TnBuff3, Tm3, Tn3, TmBuff3, TnBuff3>
	(weight3, feature3, output_core3, con, Base_addr31, Base_addr32, Base_addr33,
	M3, N3);

    act3: tanh_layer_fc(output_core3, weight3, M3, N3, Base_addr31, Base_addr33);

    fc2: single_fc<
                    TmBuff4, TnBuff4, Tm4, Tn4, TmBuff4, TnBuff4>
	(weight4, output_core3, output_core4, con, Base_addr28, Base_addr33, Base_addr30,
	M4, N4);

	// fc2 synthesized with fc1 parameters (Chunk 2)
    fc2: single_fc<
                    TmBuff3, TnBuff3, Tm3, Tn3, TmBuff3, TnBuff3>
	(weight4, output_core3, output_core4, con, Base_addr28, Base_addr33, Base_addr30,
	M4, N4);

    act4: tanh_layer_fc(output_core4, weight4, M4, N4, Base_addr28, Base_addr30);

	// fc3 synthesized with fc1 parameters (Chunk 2) and fc2 inputs
    fc3: single_fc<
                    TmBuff3, TnBuff3, Tm3, Tn3, TmBuff3, TnBuff3>
	(weight4, output_core3, output_core4, con, Base_addr28, Base_addr33, Base_addr30,
	M5, N5);
	*/

/***** Part 3: FC LAYERS 4-6 *****/

	// fc4 synthesized with fc1 parameters (Chunk 3) and fc2 inputs
    fc4: single_fc<
					TmBuff3, TnBuff3, Tm3, Tn3, TmBuff3, TnBuff3>
	(weight4, output_core3, output_core4, con, Base_addr28, Base_addr33, Base_addr30,
	M6, N6);

	// fc5 synthesized with fc1 parameters (Chunk 3) and fc2 inputs
	fc5: single_fc<
					TmBuff3, TnBuff3, Tm3, Tn3, TmBuff3, TnBuff3>
	(weight4, output_core3, output_core4, con, Base_addr28, Base_addr33, Base_addr30,
	M7, N7);

	// fc6 synthesized with fc1 parameters (Chunk 3) and fc2 inputs
	fc6: single_fc<
					TmBuff3, TnBuff3, Tm3, Tn3, TmBuff3, TnBuff3>
	(weight4, output_core3, output_core4, con, Base_addr28, Base_addr33, Base_addr30,
	M8, N8);


/** Does not currently synthesize (but should be final implementation)
	fc3: single_fc<
					TmBuff5, TnBuff5,Tm5,Tn5,TmBuff5,TnBuff5>
	(weight5, output_core4, output_core5, con, Base_addr1, Base_addr30, Base_addr3,
	M5, N5);

	act5: tanh_layer_fc(output_core5, weight5, M5, N5, Base_addr1, Base_addr3);

	fc4: single_fc<
					TmBuff6, TnBuff6, Tm6, Tn6, TmBuff6, TnBuff6>
	(weight6, output_core5, output_core6, con, Base_addr4, Base_addr3, Base_addr6,
	M6, N6);

	act6: tanh_layer_fc(output_core6, weight6, M6, N6, Base_addr4, Base_addr6);

	fc5: single_fc<
					TmBuff7, TnBuff7,Tm7,Tn7,TmBuff7,TnBuff7>
	(weight7, output_core6, output_core7, con, Base_addr7, Base_addr6, Base_addr9,
	M7, N7);

	act7: tanh_layer_fc(output_core7, weight7, M7, N7, Base_addr7, Base_addr9);

	fc6: single_fc<
					TmBuff8, TnBuff8, Tm8, Tn8, TmBuff8, TnBuff8>
	(weight8, output_core7, output_core8, con, Base_addr9, Base_addr10, Base_addr12,
	M8, N8);

	act8: tanh_layer_fc(feature9, weight8, M8, N8, Base_addr10, Base_addr12);
 	*/


/***** Part 4: Deconvolution and Upsampling with Batch Normalization *****/

	/*
	deconv1: deconv_k_wrapper<N9, C9, K9, S9,
						 TmBuff9, TnBuff9, Tr9, Tc9, Tm9, Tn9, TmBuff9, TnBuff9, Tk9, Tri9, Tci9>
	(weight9, feature9, output_core9, con, Base_addr13, Base_addr14, Base_addr15,
	M9, H9, C9, K9, N9, S9, 6);

	act9: tanh_layer(output_core9, weight9, M9, N9, C9, K9, Base_addr13, Base_addr15);

	// batchnorm3 would go here

	// upsample1 would go here

	deconv2: deconv_k_wrapper<N10, C10, K10, S10,
						  TmBuff10, TnBuff10, Tr10, Tc10, Tm10, Tn10, TmBuff10, TnBuff10, Tk10, Tri10, Tci10>
	(weight10, output_core10, output_core10, con, Base_addr16, Base_addr17, Base_addr18,
	M10, H10, C10, K10, N10, S10, 6);

	act10: tanh_layer(output_core10, weight10, M10, N10, C10, K10, Base_addr16, Base_addr18);

	// batchnorm4 would go here

	// upsample2 would go here

	// fc7 would go here
	*/
}
