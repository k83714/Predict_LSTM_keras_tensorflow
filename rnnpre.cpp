#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//test: input data
#define win 10
#define dim 3120
#define testnum 2

//layer 1 lstm
// weights:
//  in_dim, out_dim*4
//  out_dim, out_dim*4
//  out_dim*4
#define lstm1_win 10  //window size
#define lstm1_in 3120
#define lstm1_out 20

//layer 2 lstm
#define lstm2_win 10
#define lstm2_in 20
#define lstm2_out 50

// layer 3 dense
// weights:
//  in_dim, out_dim
//  out_dim, out_dim

#define dense_in 50
#define dense_out 1


struct LSTM1{
	int windowsize = lstm1_win;
	int input_dim = lstm1_in;
	int output_dim = lstm1_out;
    float weight0[lstm1_in][lstm1_out * 4];
    float weight1[lstm1_out][lstm1_out * 4];
    float weight2[lstm1_out * 4];
};

struct LSTM2{
	int windowsize = lstm2_win;
	int input_dim = lstm2_in;
	int output_dim = lstm2_out;
    float weight0[lstm2_in][lstm2_out * 4];
    float weight1[lstm2_out][lstm2_out * 4];
    float weight2[lstm2_out * 4];
};

struct DENSE{
	int input_dim = dense_in;
	int output_dim = dense_out;
	float weight0[dense_in][dense_out];
	float weight1[dense_out];
};


float sigmoid(float x);
void lstm1_predict(
		float ioutput[win][lstm1_out],
		float itest[win][lstm1_in],
		float weight0[lstm1_in][lstm1_out*4], 
		float weight1[lstm1_out][lstm1_out*4], 
		float weight2[lstm1_out*4]);

void lstm2_predict(
		float ioutput[1][lstm2_out],
		float itest[win][lstm2_in],
		float weight0[lstm2_in][lstm2_out*4], 
		float weight1[lstm2_out][lstm2_out*4], 
		float weight2[lstm2_out*4]);

void dense_predict(
		float ioutput[dense_out],
		float itest[1][dense_in],
		float weight0[dense_in][dense_out], 
		float weight1[dense_out] );

int main(int argc,char *argv[])
{
    std::cout << "hello, world" << std::endl;

    FILE *fp = fopen("flatweights.txt", "r");
    
	LSTM1 lstm1;
	LSTM2 lstm2;
	DENSE dense;

    int i,j, k;

	
    if(fp == NULL)
        printf("file open failed!");
    else{
		int lstm1_total = lstm1_out*4;

	    for(i=0; i < lstm1_in; i++)
		    for(j=0; j < lstm1_total; j++){
				fscanf(fp, "%e", &lstm1.weight0[i][j]);
			}

	    for(i=0; i < lstm1_out; i++)
		    for(j=0; j < lstm1_total; j++){
				fscanf(fp, "%e", &lstm1.weight1[i][j]);
			}

	    for(i=0; i < lstm1_total; i++)
			fscanf(fp, "%e", &lstm1.weight2[i]);


		int lstm2_total = lstm2_out*4;

	    for(i=0; i < lstm2_in; i++)
		    for(j=0; j < lstm2_total; j++){
				fscanf(fp, "%e", &lstm2.weight0[i][j]);
			}
/*
		int p =0;
		for ( int p=0; p < 2; p++){
		    printf("%f %f %f %f \n",lstm2.weight0[p][0] ,lstm2.weight0[p][1] ,lstm2.weight0[p][2],lstm2.weight0[p][3]);
		}
*/

	    for(i=0; i < lstm2_out; i++)
		    for(j=0; j < lstm2_total; j++){
				fscanf(fp, "%e", &lstm2.weight1[i][j]);
			}

	    for(i=0; i < lstm2_total; i++)
			fscanf(fp, "%e", &lstm2.weight2[i]);


	    for(i=0; i < dense_in; i++)
		    for(j=0; j < dense_out; j++){
				fscanf(fp, "%e", &dense.weight0[i][j]);
			}

	    for(i=0; i < dense_out; i++)
			fscanf(fp, "%e", &dense.weight1[i]);

	}

	
// get test input
    fclose(fp);
    fp = fopen("flattest.txt", "r");

	float test[testnum][win][dim];
	
	if(fp == NULL) {
        printf("file open failed!");
		return 0;
	}
    
	for(i = 0; i < testnum; i++)
		for ( j = 0; j < win; j++)
			for ( k = 0; k < dim; k++)
				fscanf(fp, "%e", &test[i][j][k]);
    fclose(fp);
//	printf("pos1\n");
    float output_lstm1[testnum][win][lstm1_out];
    float output_lstm2[testnum][1][lstm2_out];
	float output_dense[testnum][1];
	for( i = 0; i < testnum; i++){
        lstm1_predict(output_lstm1[i], test[i], lstm1.weight0, lstm1.weight1, lstm1.weight2);
		/*
		for(j=0; j < 2; j++){
			for ( k=0; k < 20; k++){
				printf( " %f ", output_lstm1[0][j][k]);
			}
			printf("\n");
		}
		*/

        lstm2_predict(output_lstm2[i], output_lstm1[i], lstm2.weight0, lstm2.weight1, lstm2.weight2);

		dense_predict(output_dense[i], output_lstm2[i], dense.weight0, dense.weight1);

	}
    
//	printf("pos2\n");





	return(0);
}

float sigmoid ( float x) {
	return 1.0 / (1.0 + exp(-x));
}

void lstm1_predict(
		float ioutput[win][lstm1_out],
		float itest[win][lstm1_in],
		float weight0 [lstm1_in][lstm1_out*4], 
		float weight1[lstm1_out][lstm1_out*4], 
		float weight2[lstm1_out*4]){
	int i, j;

	float c_tm[1][lstm1_out], h_tm[1][lstm1_out];
	for(j=0; j < lstm1_out; j++){
		c_tm[0][j]=0;
		h_tm[0][j]=0;
	}
    float h_t[1][lstm1_out], c_t[1][lstm1_out];
	int k,t,p;

    for(i = 0; i < win; i++){
		float ii[1][lstm1_out];
		float ff[1][lstm1_out];
		float cc[1][lstm1_out];
		float oo[1][lstm1_out];
		float s[1][lstm1_out*4];
	    
		for( p=0; p < lstm1_out*4; p++) s[0][p]=0;	
		
		for( j = 0; j < lstm1_in; j++)
			for( k=0; k < lstm1_out*4; k++)
				s[0][k] += itest[i][j] * weight0[j][k];
			
			
		for( j = 0; j < lstm1_out; j++)
			for( k=0; k < lstm1_out*4; k++)
			    s[0][k] += h_tm[0][j] * weight1[j][k];
		for( k=0; k < lstm1_out*4; k++)
				s[0][k] += weight2[k];
			
		

		int offset2=lstm1_out*2;
		int offset3=lstm1_out*3;
		for(p=0; p < lstm1_out; p++){
			ii[0][p] = sigmoid(s[0][p]);
			ff[0][p] = sigmoid(s[0][p + lstm1_out]);
			cc[0][p] = tanh(s[0][p + offset2]);
			oo[0][p] = sigmoid(s[0][p + offset3]);
		}

        for ( p=0; p<lstm1_out; p++){
			c_t[0][p] = ii[0][p] * cc[0][p] + ff[0][p] * c_tm[0][p];
			h_t[0][p] = oo[0][p] * tanh(c_t[0][p]);
				        
		}

		for ( p =0 ; p < lstm1_out; p++){
		c_tm[0][p] = c_t[0][p];
		h_tm[0][p] = h_t[0][p];
		ioutput[i][p] = h_t[0][p];
		}
	}
}




void lstm2_predict(
		float ioutput[1][lstm2_out],
		float itest[win][lstm2_in],
		float weight0 [lstm2_in][lstm2_out*4], 
		float weight1[lstm2_out][lstm2_out*4], 
		float weight2[lstm2_out*4]){
	int i, j;

	float c_tm[1][lstm2_out], h_tm[1][lstm2_out];
	for(j=0; j < lstm2_out; j++){
		c_tm[0][j]=0;
		h_tm[0][j]=0;
	}
    float h_t[1][lstm2_out], c_t[1][lstm2_out];
	int k,t,p;

    for(i = 0; i < win; i++){
//		printf("Round %d --------------\n\n\n", i);
		float ii[1][lstm2_out];
		float ff[1][lstm2_out];
		float cc[1][lstm2_out];
		float oo[1][lstm2_out];
		float s[1][lstm2_out*4];
	    
		for( p=0; p < lstm2_out*4; p++) s[0][p]=0;	
		
		for( j = 0; j < lstm2_in; j++)
			for( k=0; k < lstm2_out*4; k++)
				s[0][k] += itest[i][j] * weight0[j][k];
			
			
		for( j = 0; j < lstm2_out; j++)
			for( k=0; k < lstm2_out*4; k++)
			    s[0][k] += h_tm[0][j] * weight1[j][k];
/*
	    for (p=0; p < 5; p++)	printf("%e ", itest[i][p]);
	    for (p=0; p < 5; p++)	printf("%e ", weight0[0][p]);
	    for (p=0; p < 5; p++)	printf("%e ", h_tm[0][p]);
	    for (p=0; p < 5; p++)	printf("%e ", weight1[0][p]);
	    for (p=0; p < 5; p++)	printf("%e ", weight2[p]);
    	printf("\n");
*/
		for( k=0; k < lstm2_out*4; k++)
				s[0][k] += weight2[k];
/*
	    for( p = 0; p < 10; p++)printf("%f ", s[0][p]);
		printf("\n");
*/
			
		

	//printf("pos4\n");
		int offset2=lstm2_out*2;
		int offset3=lstm2_out*3;
		for(p=0; p < lstm2_out; p++){
			ii[0][p] = sigmoid(s[0][p]);
			ff[0][p] = sigmoid(s[0][p + lstm2_out]);
			cc[0][p] = tanh(s[0][p + offset2]);
			oo[0][p] = sigmoid(s[0][p + offset3]);
		}
/*
	    for (p=0; p < 5; p++)	printf("%f ", ii[0][p]);
	    for (p=0; p < 5; p++)	printf("%f ", ff[0][p]);
	    for (p=0; p < 5; p++)	printf("%f ", cc[0][p]);
	    for (p=0; p < 5; p++)	printf("%f ", oo[0][p]);
    	printf("pos5\n");
*/
        for ( p=0; p<lstm2_out; p++){
			// c_t = ii * cc + ff * c_tm;
			c_t[0][p] = ii[0][p] * cc[0][p] + ff[0][p] * c_tm[0][p];
			h_t[0][p] = oo[0][p] * tanh(c_t[0][p]);
				        
		}

		for ( p =0 ; p < lstm2_out; p++){
		c_tm[0][p] = c_t[0][p];
		h_tm[0][p] = h_t[0][p];
		//ioutput[i][p] = h_t[0][p];
		}
/*
	    for( p=0; p < 5; p++){
		    printf("%f ", h_t[0][p]);
	    }
		printf("\n\n");
		*/
	}
	for ( p =0 ; p < lstm2_out; p++){
		ioutput[0][p] = h_t[0][p];
	}
/*   
	printf("output---\n");
	for( i=0; i < 5; i++){
		printf("%f ", h_t[0][i]);
	}
	printf("output---end\n");
*/	

}



void dense_predict(
		float ioutput[dense_out],
		float itest[1][dense_in],
		float weight0[dense_in][dense_out], 
		float weight1[dense_out] ){
	int i, j;

	for ( i = 0; i < dense_out; i++){
		ioutput[i] = weight1[i];
		for ( j =0; j < dense_in; j++){
			ioutput[i] += itest[0][j] * weight0[j][i];
		}
	}

	printf("%f \n", ioutput[0]);

}

