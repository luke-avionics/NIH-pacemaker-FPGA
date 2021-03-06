#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream> // To use ifstream
#include "unet.h"
#include <ap_int.h>
using namespace std;

//
//#define dataw 32
//#define FLOAT 1
//
//#if FLOAT==1
//	typedef float data_type;
//#else
//	typedef ap_fixed<8,3> data_type;
//#endif
//
//struct data_pack{
//	data_type data0;
//	data_type data1;
//};
//
//struct dma_data{
//	data_pack data;
//};
//const int TmBuff1=2, TnBuff1=2,Tr1=4,Tc1=4,Tm1=2,Tn1=2,Tk1=7,Tri1=10,Tci1=10;
//const int M1 = 48, N1=2,C1=16,H1=22, K1=7, S1=1;




int main(){
    ifstream inputFile;
    data_type * flatten_weight=(data_type *)malloc((M1*N1*K1*K1+M1)*sizeof(data_type));
    data_type * flatten_input=(data_type *)malloc(N1*C1*C1*sizeof(data_type));
    //data_type * flatten_output=(data_type *)malloc(M1*C1*C1*sizeof(data_type));
    data_type * flatten_weight2=(data_type *)malloc((M2*N2*K2*K2+M2)*sizeof(data_type));
    //data_type * flatten_input2=(data_type *)malloc(N2*C2*C2*sizeof(data_type));
    //data_type * flatten_output2=(data_type *)malloc(M2*C2*C2*sizeof(data_type));
    
    data_type * flatten_weight3=(data_type *)malloc((M3*N3+M3)*sizeof(data_type));
    //data_type * flatten_input3=(data_type *)malloc(N3*sizeof(data_type));
    data_type * flatten_weight4=(data_type *)malloc((M4*N4+M4)*sizeof(data_type));
    
    
    //modification anchor 
    data_type * golden_output=(data_type *)malloc(M4*1*1*sizeof(data_type));
    inputFile.open("test_output.inc");
    for(unsigned int i=0; i<M2; i++)
	   for(unsigned int j=0; j<C2/2; j++)
		   for(unsigned int k=0; k<C2/2; k++){
			   inputFile>>golden_output[i];
		   } 
    inputFile.close();
    
    
    inputFile.open("test_weight.inc");

   for(unsigned int i=0; i<M1;i++)
       for(unsigned int j=0; j<N1;j++)
           for(unsigned int k=0; k<K1; k++)
               for(unsigned int l=0; l<K1; l++)
               {
            	   inputFile>>flatten_weight[i*N1*K1*K1+j*K1*K1+k*K1+l];
               }
    for(unsigned int i=0; i<M1;i++)
        inputFile>>flatten_weight[M1*N1*K1*K1+i];
    inputFile.close();
    
    inputFile.open("test_weight2.inc");
    for(unsigned int i=0; i<M2;i++)
       for(unsigned int j=0; j<N2;j++)
           for(unsigned int k=0; k<K2; k++)
               for(unsigned int l=0; l<K2; l++)
               {
            	   inputFile>>flatten_weight2[i*N2*K2*K2+j*K2*K2+k*K2+l];
               }
    for(unsigned int i=0; i<M2;i++){
        inputFile>>flatten_weight2[M2*N2*K2*K2+i];
    }
    inputFile.close();
    
    inputFile.open("test_weight3.inc");
    for(unsigned int i=0; i<M3;i++)
       for(unsigned int j=0; j<N3;j++)
           for(unsigned int k=0; k<1; k++)
               for(unsigned int l=0; l<1; l++)
               {
            	   inputFile>>flatten_weight3[i*N3*1*1+j*1*1+k*1+l];
               }
    for(unsigned int i=0; i<M3;i++){
        inputFile>>flatten_weight3[M3*N3*1*1+i];
    }
    inputFile.close();


    inputFile.open("test_weight4.inc");
    for(unsigned int i=0; i<M4;i++)
       for(unsigned int j=0; j<N4;j++)
           for(unsigned int k=0; k<1; k++)
               for(unsigned int l=0; l<1; l++)
               {
            	   inputFile>>flatten_weight4[i*N4*1*1+j*1*1+k*1+l];
               }
    for(unsigned int i=0; i<M4;i++){
        inputFile>>flatten_weight4[M4*N4*1*1+i];
    }
    inputFile.close();
    // cout<<M2*N2*K2*K2<<endl;
    // cout<<flatten_weight[0]<<',';
    // cout<<flatten_weight2[1]<<',';
    // cout<<flatten_weight2[100]<<',';
    // cout<<flatten_weight2[40*5*5]<<',';
    // cout<<flatten_weight2[115295]<<',';
    // cout<<flatten_weight2[115294]<<',';
    
    inputFile.open("test_x.inc");
   
           
   for(unsigned int i=0; i<N1; i++)
	   for(unsigned int j=0; j<C1; j++)
		   for(unsigned int k=0; k<C1; k++){
			   inputFile>>flatten_input[i*C1*C1+j*C1+k];
		   }
    inputFile.close();
    
    
    //define packed data       
    dma_data * packed_weight=(dma_data *)malloc((M1*N1*K1*K1+M1)/2*sizeof(dma_data));
    dma_data * packed_input=(dma_data *)malloc(N1*C1*C1/2*sizeof(dma_data));
    dma_data * packed_output=(dma_data *)malloc(M1*C1*C1/2*sizeof(dma_data));
    dma_data * packed_weight2=(dma_data *)malloc((M2*N2*K2*K2+M2)/2*sizeof(dma_data));
    dma_data * packed_input2=(dma_data *)malloc(N2*C2*C2/2*sizeof(dma_data));
    dma_data * packed_output2=(dma_data *)malloc(M2*C2*C2/2*sizeof(dma_data));
    dma_data * packed_weight3=(dma_data *)malloc((M3*N3+M3)/2*sizeof(dma_data));
    dma_data * packed_input3=(dma_data *)malloc(N3/2*sizeof(dma_data));
    dma_data * packed_output3=(dma_data *)malloc(M3/2*sizeof(dma_data));
    dma_data * packed_weight4=(dma_data *)malloc((M4*N4+M4)/2*sizeof(dma_data));
    dma_data * packed_output4=(dma_data *)malloc(M4/2*sizeof(dma_data));
    //packing weight
    for(unsigned int i=0; i<M1;i+=2)
       for(unsigned int j=0; j<N1;j++)
           for(unsigned int k=0; k<K1; k++)
               for(unsigned int l=0; l<K1; l++)
               {
                   packed_weight[i/2*N1*K1*K1+j*K1*K1+k*K1+l].data.data0=flatten_weight[i*N1*K1*K1+j*K1*K1+k*K1+l];
                   packed_weight[i/2*N1*K1*K1+j*K1*K1+k*K1+l].data.data1=flatten_weight[(i+1)*N1*K1*K1+j*K1*K1+k*K1+l];
               }
    for(unsigned int i=0; i<M1;i+=2){
        packed_weight[M1*N1*K1*K1/2+i/2].data.data0=flatten_weight[M1*N1*K1*K1+i];
        packed_weight[M1*N1*K1*K1/2+i/2].data.data1=flatten_weight[M1*N1*K1*K1+i+1];
    }

    for(unsigned int i=0; i<M2;i+=2)
       for(unsigned int j=0; j<N2;j++)
           for(unsigned int k=0; k<K2; k++)
               for(unsigned int l=0; l<K2; l++)
               {
                   packed_weight2[i/2*N2*K2*K2+j*K2*K2+k*K2+l].data.data0=flatten_weight2[i*N2*K2*K2+j*K2*K2+k*K2+l];
                   packed_weight2[i/2*N2*K2*K2+j*K2*K2+k*K2+l].data.data1=flatten_weight2[(i+1)*N2*K2*K2+j*K2*K2+k*K2+l];
               }
    for(unsigned int i=0; i<M2;i+=2){
        packed_weight2[M2*N2*K2*K2/2+i/2].data.data0=flatten_weight2[M2*N2*K2*K2+i];
        packed_weight2[M2*N2*K2*K2/2+i/2].data.data1=flatten_weight2[M2*N2*K2*K2+i+1];
    }
    
    for(unsigned int i=0; i<M3;i+=2)
       for(unsigned int j=0; j<N3;j++)
           for(unsigned int k=0; k<1; k++)
               for(unsigned int l=0; l<1; l++)
               {
                   packed_weight3[i/2*N3+j].data.data0=flatten_weight3[i*N3+j];
                   packed_weight3[i/2*N3+j].data.data1=flatten_weight3[(i+1)*N3+j];
               }
    for(unsigned int i=0; i<M3;i+=2){
        packed_weight3[M3*N3/2+i/2].data.data0=flatten_weight3[M3*N3+i];
        packed_weight3[M3*N3/2+i/2].data.data1=flatten_weight3[M3*N3+i+1];
    }
    
    for(unsigned int i=0; i<M4;i+=2)
       for(unsigned int j=0; j<N4;j++)
           for(unsigned int k=0; k<1; k++)
               for(unsigned int l=0; l<1; l++)
               {
                   packed_weight4[i/2*N4+j].data.data0=flatten_weight4[i*N4+j];
                   packed_weight4[i/2*N4+j].data.data1=flatten_weight4[(i+1)*N4+j];
               }
    for(unsigned int i=0; i<M4;i+=2){
        packed_weight4[M4*N4/2+i/2].data.data0=flatten_weight4[M4*N4+i];
        packed_weight4[M4*N4/2+i/2].data.data1=flatten_weight4[M4*N4+i+1];
    }
    
    //packing input
    for(unsigned int i=0; i<N1; i+=2)
	   for(unsigned int j=0; j<C1; j++)
		   for(unsigned int k=0; k<C1; k++){
			   packed_input[i/2*C1*C1+j*C1+k].data.data0=flatten_input[i*C1*C1+j*C1+k];
               packed_input[i/2*C1*C1+j*C1+k].data.data1=flatten_input[(i+1)*C1*C1+j*C1+k];
		   }
    
    
    
    
    
    unet_top (
        packed_weight,
        packed_input,
        packed_output,
        packed_weight2,
        packed_input2,
        packed_output2,
        packed_weight3,
        packed_input3,
        packed_output3,
		packed_weight4,
		packed_output4,
        // dma_data* weight4,
        // dma_data* feature4,
        // dma_data* output_core4,
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
        1,
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
         // ap_uint<32> Base_addr28,
         // ap_uint<32>  Base_addr29,
         // ap_uint<32>  Base_addr30,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000,
        0x0000
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
    
    dma_data tmp;
    for(unsigned int i=0; i<M2;i+=2)
       for(unsigned int j=0; j<N2;j++)
           for(unsigned int k=0; k<K2; k++)
               for(unsigned int l=0; l<K2; l++)
               {
                tmp=packed_weight2[i/2*N2*K2*K2+j*K2*K2+k*K2+l];
                if (tmp.data.data0!= flatten_weight2[i*N2*K2*K2+j*K2*K2+k*K2+l] || tmp.data.data1!= flatten_weight2[(i+1)*N2*K2*K2+j*K2*K2+k*K2+l]){
                    
                    cout<<"failed"<<' ';
                    cout<<tmp.data.data0<<' ';
                    cout<<tmp.data.data1<<endl;
                    break;
                }
               }
               
    for(unsigned int i=0; i<M2;i+=2){
        tmp=packed_weight2[M2*N2*K2*K2/2+i/2];
        
        if (tmp.data.data0!= flatten_weight2[M2*N2*K2*K2+i] || tmp.data.data1!= flatten_weight2[M2*N2*K2*K2+i+1]){
            
            cout<<"failed"<<' ';
            cout<<tmp.data.data0<<' ';
            cout<<tmp.data.data1<<endl;
            break;
        }
    }
    
    for(unsigned int i=0; i<M3;i+=2)
       for(unsigned int j=0; j<N3;j++)
           for(unsigned int k=0; k<1; k++)
               for(unsigned int l=0; l<1; l++)
               {
                tmp=packed_weight3[i/2*N3+j];
                if (tmp.data.data0!= flatten_weight3[i*N3+j] || tmp.data.data1!= flatten_weight3[(i+1)*N3+j]){
                    
                    cout<<"weight3 failed"<<' ';
                    cout<<tmp.data.data0<<' ';
                    cout<<tmp.data.data1<<endl;
                    break;
                }
               }
    
    for(unsigned int i=0; i<M3;i+=2){
        tmp=packed_weight3[M3*N3/2+i/2];
        
        if (tmp.data.data0!= flatten_weight3[M3*N3+i] || tmp.data.data1!= flatten_weight3[M3*N3+i+1]){
            
            cout<<"weight3 bias failed"<<' ';
            cout<<tmp.data.data0<<' ';
            cout<<tmp.data.data1<<endl;
            break;
        }
    }
    // dma_data tmp;
    // for(unsigned int i=0; i<N1; i+=2)
	   // for(unsigned int j=0; j<C1; j++)
		   // for(unsigned int k=0; k<C1; k++){
                // tmp=packed_input[i/2*C1*C1+j*C1+k];
                // if (tmp.data.data0!= flatten_input[i*C1*C1+j*C1+k] || tmp.data.data1!= flatten_input[(i+1)*C1*C1+j*C1+k]){
                    
                    // cout<<"failed"<<' ';
                    // cout<<tmp.data.data0<<' ';
                    // cout<<tmp.data.data1<<endl;
                    // break;
                // }
               // }   

    //modification anchor
    for(unsigned int i=0; i<M2; i+=2)
	   for(unsigned int j=0; j<C2/2; j++)
		   for(unsigned int k=0; k<C2/2; k++){
                tmp=packed_input3[i/2*C2*C2/4+j*C2/2+k];
                if (abs(tmp.data.data0-golden_output[i*C2*C2/4+j*C2/2+k])/golden_output[i*C2*C2/4+j*C2/2+k] > 0.1
                    || abs(tmp.data.data1-golden_output[(i+1)*C2*C2/4+j*C2/2+k])/golden_output[(i+1)*C2*C2/4+j*C2/2+k]>0.1){
                    
                    cout<<"failed"<<' ';
                    cout<<tmp.data.data0<<',';
                    cout<<golden_output[i*C2*C2/4+j*C2/2+k]<<endl;
                    cout<<tmp.data.data1<<',';
                    cout<<golden_output[(i+1)*C2*C2/4+j*C2/2+k]<<endl;
                    cout<<i<<',';
                    cout<<j<<',';
                    cout<<k<<endl;
                    break;
                }
               }   

//
//     for(unsigned int i=0; i<M2; i+=2){
//	    for(unsigned int j=0; j<C2; j++){
//                cout<<'\n'<<' ';
//		    for(unsigned int k=0; k<C2; k++){
//			    cout<<packed_output2[i/2*C2*C2+j*C2+k].data.data0<<',';
//		    }
//        }
//	    for(unsigned int j=0; j<C2; j++){
//                cout<<'\n'<<' ';
//		    for(unsigned int k=0; k<C2; k++){
//			    cout<<packed_output2[i/2*C2*C2+j*C2+k].data.data1<<',';
//		    }
//        }
//
//     }
    // cout<<"\n=========================================================================\n"<<' ';
    // cout<<"\n=========================================================================\n"<<' ';
    // cout<<"\n=========================================================================\n"<<' ';
    // cout<<"\n=========================================================================\n"<<' ';
    // cout<<"\n=========================================================================\n"<<' ';
    // for(unsigned int i=0; i<M1; i+=2){
	   // for(unsigned int j=0; j<C2; j++){
               // cout<<'\n'<<' ';
		   // for(unsigned int k=0; k<C2; k++){
			   // cout<<packed_input2[i/2*C2*C2+j*C2+k].data.data0<<',';
		   // }
       // }
	   // for(unsigned int j=0; j<C2; j++){
               // cout<<'\n'<<' ';
		   // for(unsigned int k=0; k<C2; k++){
			   // cout<<packed_input2[i/2*C2*C2+j*C2+k].data.data1<<',';
		   // }
       // }
                      
    // } 
    return 0;

}
