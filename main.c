#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <omp.h>

typedef struct 
{
    int rank;
    int * shape;
    float * data;
} tensor;





int tensor_num_elems(tensor * t) 
{
    int num_elems = 1;
    for (int i = 0; i < t->rank; i++) { num_elems *= t->shape[i]; }
    return num_elems;
}
tensor * tensor_create(int rank, int * shape) 
{
    tensor * t = (tensor*) malloc(sizeof(tensor));

    // manage the shape information for the tensor
    t->rank = rank;
    t->shape = (int*) malloc(sizeof(int) * rank);
    for (int i = 0; i < rank; i++) { t->shape[i] = shape[i]; }

    // allocate memory for the tensor's data
    int num_elems = tensor_num_elems(t);
    t->data = (float*) malloc(sizeof(float) * num_elems);

    return t;
}
tensor * tensor_copy(tensor * t) {
    tensor * new_t = tensor_create(t->rank, t->shape);

    // allocate memory for the tensor's data
    int num_elems = tensor_num_elems(t);
    t->data = (float*) malloc(sizeof(float) * num_elems);

    // write t's data to new_t's data
    for (int i = 0; i < num_elems; i++) {
        new_t->data[i] = t->data[i];
    }

    return new_t;
}
/**
 * @brief Tensor Broadcast Add
 * 
 * Add the elements of t1 to t0, elementwise.
 * 
 * after the function: 
 * 
 * t0 = t0 + t1
 * t1 = t1
 * 
 * @param t0 
 * the tensor being added to 
 * @param t1 
 * the tensor being added to t0
 */
void tensor_broadcast_add(tensor * t0, tensor * t1)
{
    int num_elems = tensor_num_elems(t0);

    for (int i = 0; i < num_elems; i ++) {
        t0->data[i] += t1->data[i];
    }
}
void free_tensor(tensor * t) 
{
    free(t->shape);
    free(t->data);
    free(t);
}
float get_tensor_element(tensor* t, int* indices) 
{
    int idx = 0;
    for (int i = 0; i < t->rank; i++) {
        idx = idx * t->shape[i] + indices[i];
    }
    return t->data[idx];
}
void tensor_print(tensor * t) {

    printf("\n");

    if (t->rank == 1) {
        printf("[");
        for (int i = 0; i < t->shape[0]; i++) {
            int idx[1] = {i};
            float elem = get_tensor_element(t, idx);
            printf(" %f ", elem);
        }
        printf("]\n");
    } else {

        int R = t->shape[0];
        int C = t->shape[1];
        for (int i = 0; i < R; i++) {
            printf("[ ");
            for (int j = 0; j < C; j++) {
                int idx[2] = {i, j};
                float elem = get_tensor_element(t, idx);
                printf(" %f ",elem);
            }
            printf("]\n");
        }
    }
    printf("shape: (");
    for (int i = 0; i < t->rank; i++) {
        printf(" %d", t->shape[i]);
        if (i != t->rank-1) {
            printf(",");
        } else {
            printf(" ");
        }
    }
    printf(")\n");
}


int tensor_vector_transformation(tensor * v, tensor * m, tensor * out)
{
    // make sure that the tensors are valid
    if (v->rank != 1) { fprintf(stderr, "tensor_vector_transformation: v should be of rank 1"); return 1; }
    if (m->rank != 2) { fprintf(stderr, "tensor_vector_transformation: m should be of rank 2"); return 1; }
    if (v->shape[0] != m->shape[1]) { fprintf(stderr, "tensor_vector_transformation: v and m have incompatible shapes"); return 1; }

    int out_dim = out->shape[0];

    // for each row in the matrix
    for (int i = 0; i < m->shape[0]; i++) {
        float dot_product = 0.0f;
        int row_idx_offset = i * out_dim;
        // for each column in the matrix, and element of out
        for (int j = 0; j < out_dim; j++) {
            dot_product += out->data[j] * m->data[row_idx_offset + j];
        }
        out->data[i] = dot_product;
    }

    return 0;
}


typedef struct 
{   
    int hidden_dim;

    tensor * w_1;
    tensor * b_1;

    tensor * w_2;
    tensor * b_2;

    tensor * hidden_cache;
    tensor * output_cache;

} feedforward_node;
feedforward_node * feedforward_node_create(int dim, int hidden_dim) 
{
    feedforward_node * ff = (feedforward_node*) malloc(sizeof(feedforward_node));

    int w_1_shape[2] = { hidden_dim, dim };
    ff->w_1 = tensor_create(2, w_1_shape);
    int b_1_shape[1] = { hidden_dim };
    ff->b_1 = tensor_create(1, b_1_shape);

    int w_2_shape[2] = { dim, hidden_dim };
    ff->w_2 = tensor_create(2, w_2_shape);
    int b_2_shape[1] = { dim };
    ff->b_2 = tensor_create(1, b_2_shape);

    return ff;
}

/**
 * @brief Attention Node
 * 
 * 
 **/
typedef struct 
{
    tensor * w_key;
    tensor * b_key;

    tensor * w_query;
    tensor * b_query;

    tensor * w_value;
    tensor * w_value; 
    
    // the dimensionality of all the weight vectors
    int rank;

    tensor * attention_weight_matrix;
    tensor * output_cache;

} attention_node; 

/**
 * @brief Block Node 
 * 
 * This structure contains the node elements of a certain transformer node 
 * at a specific block
 * 
 */
typedef struct 
{
    attention_node * attention;
    feedforward_node * feedforward;

    tensor * output_cache;
} block_node;

typedef struct 
{    
    // the blocks of the given node
    int layer_c;
    struct block_node** layers;
} transformer_node;

typedef struct 
{
    // the number of nodes that the transformer has
    int node_c;
    struct transformer_node** nodes;
}  transformer;

/**
 * @brief Tensor Head Scaled Dot Product
 * 
 * Compute the attention of the two vectors t1 and t0. 
 * While that normal transformers will split the heads of 
 * the vectors, creating confusing tensor operations, we 
 * just take dot products of the different sections of the 
 * tensors. Here is an example:
 * 
 * Let t0 be [a0,b0,c0,d0,e0,f0,g0,h0],
 * 
 * Let t1 be [a1,b1,c1,d1,e1,f1,g1,h1],
 * 
 * let head_c be 2,
 * 
 * Then this function will return:
 * 
 * [ a0*a1 + b0*b1 + c0*c1 + d0*d1, e0*e1 + f0*f1 + g0*g1, h0*h1 ]
 * 
 * @param t0 
 * first vector
 * @param t1 
 * second vector
 * @param head_c
 * number of heads 
 * @param out
 * a section of memory of size: sizeof(float)*head_c, to write the output attention weights to 
 * @return 
 * an integer that is 1 or 0, representing if the function finished successfully
 */
int tensor_head_scaled_dot_product(tensor * t0, tensor * t1, int head_c, float * out)
{   
    // check to make sure that the tensors are valid
    if (t0->rank != 1 || t1->rank != 1) { fprintf(stderr, "tensor_head_scaled_dot_product: t0 and t1 should both be of rank 1"); return 1; }
    int dim = t0->shape[0];
    if (dim != t1->shape[0]) { fprintf(stderr, "tensor_head_scaled_dot_product: the t0 and t1 have different dimensionalities!"); return 1; }
    if (dim % head_c != 0) {fprintf(stderr, "tensor_head_scaled_dot_product: the dimensionality of t0 and t1 must be divisible by the number of heads"); return 1;}


    int head_dim = dim / head_c;

    float scale = sqrt((float) dim);


    // the actual computation may need to be changed in the future, 
    // if there is a need for parrellization

    // we use k as a counter to minimize the number of multiplication
    // operations that the function uses as without k, it would
    // be nescessary to use multiplications to address parts of t0 and t1
    int k = 0;
    for (int i = 0; i < head_c; i++ ){
        float dot_product = 0.0;
        for (int j = 0; j < head_dim; j++) {
            dot_product += t0->data[k] * t1->data[k];
            k++;
        }
        out[i] = dot_product / scale;
    }

    return 0;
}
int tensor_columnwise_softmax(tensor * t)
{
    if (t->rank != 2) { fprintf(stderr, "tensor_columnwise_softmax: t should be a matrix (it should have rank 2)"); return 1; }
    
    int num_elems = tensor_num_elems(t);

    // turn each element of t into an exponent of e
    for (int i = 0; i < num_elems; i++) {
        t->data[i] = exp(t->data[i]);
    }

    // get the summations of each of the matrix's columns
    
    int num_columns = t->shape[1];
    float* column_sums = (float*) malloc(sizeof(float) * num_columns);
    // set each column sum to 0
    for (int i = 0; i < num_columns; i++) { column_sums[i] = 0.0; }
    int k = 0;
    for (int i = 0; i < num_elems; i++) {
        column_sums[k] += t->data[i];
        k++;
        if (k >= num_columns) {
            k = 0;
        }
    }
    k = 0;

    // divide each element by its column's summation
    for (int i = 0; i < num_elems; i ++) {
        t->data[i] /= column_sums[k];
        k++;
        if (k >= num_columns) {
            k = 0;
        }
    }

    return 0;

}
void attention_node_forward(attention_node * node, int node_idx, int head_c, block_node** input_blocks, int block_c) 
{
    // NOTE: Do not be confused and think that this is the
    // attention between every node and every other node, 
    // this is the attention between the given node and every 
    // other node. Since the heads are all computed at once here,
    // this is a matix of shape [ other nodes being attended to, number of heads]
    int attention_weight_matrix_shape[2] = { block_c, head_c};
    node->attention_weight_matrix = tensor_create(2, attention_weight_matrix_shape);

    // put the scaled dot products into the attention weight matrix
    for (int block_idx = 0; block_idx < block_c; block_idx++) {
        int weight_mat_idx = block_idx * head_c;
        float * weight_mat_ptr = &node->attention_weight_matrix->data[weight_mat_idx];
        tensor_head_scaled_dot_product(input_blocks[block_idx]->output_cache, input_blocks[node_idx]->output_cache, head_c, weight_mat_ptr);
    }

    // compute softmax
    tensor_columnwise_softmax(node->attention_weight_matrix);

    // create the attention output
    node->output_cache = tensor_create(1, input_blocks[node_idx]->output_cache->shape);
    
    
    // apply the attention weights to the output vector of the node
    int dim = tensor_num_elems(node->output_cache);
    int head_dim = dim / head_c;

    // make sure that the attention output is all 0s
    for (int i = 0; i < dim; i++) { node->output_cache->data[i] = 0.0; }
    
    // for each block
    for (int block_idx = 0; block_idx < block_c; block_idx++) {

        int k = 0;

        int block_idx_offset = 0;

        // for each head section of the dim
        for (int head = 0; head < head_c; head++) {

            float weight = node->attention_weight_matrix->data[block_idx_offset + head];

            // for each scalar in the dim
            for (int i = 0; i < head_dim; i++) {

                node->output_cache->data[k] += weight * input_blocks[block_idx]->output_cache->data[k];

                k++;
            }   
        }

        block_idx_offset += head_c;
    }
    

    printf("attention weights: \n");
    tensor_print(node->attention_weight_matrix);

    printf("attention output: \n");
    tensor_print(node->output_cache);
}

/**
 * @brief Tensor Zeros
 * 
 * sets every element of t to 0
 * 
 * @param t 
 * the tensor being turned set to zeros
 */
void tensor_zeros(tensor * t) 
{
    int num_elems = tensor_num_elems(t);

    //#pragma omp parallel for 
    for (int i = 0; i < num_elems; i++) {
        t->data[i] = 0;
    }
}

int main ()
{
    int shape0[1] = {16};

    tensor * t0 = tensor_create(1, shape0);
    tensor_zeros(t0);

    tensor_print(t0);


    

   

    return 0;
}

int test1()
{
    int shape0[1] = {16};

    tensor * t0 = tensor_create(1, shape0);
    for (int i = 0; i < t0->shape[0]; i++) {t0->data[i] = 1.5;};

    tensor * t1 = tensor_create(1, shape0);
    for (int i = 0; i < t1->shape[0]; i++) {t1->data[i] = 1.0;}

    block_node * b0 = (block_node*) malloc(sizeof(block_node));
    b0->output_cache = t0;

    block_node * b1 = (block_node*) malloc(sizeof(block_node));
    b1->output_cache = t1;
    block_node * input_blocks[2] = {b0,b1};

    attention_node * node = (attention_node*) malloc(sizeof(attention_node));

    attention_node_forward(node, 0, 4, input_blocks, 2);
    return 0;
} 


int test0() 
{
    int shape0[1] = {8};
    tensor * t0 = tensor_create(1, shape0);
    for (int i = 0; i < t0->shape[0]; i++) {t0->data[i] = 1.0;}
    tensor_print(t0);

    tensor * t1 = tensor_create(1, shape0);
    for (int i = 0; i < t1->shape[0]; i++) {t1->data[i] = 1.0;}
    tensor_print(t1);

    float * out = (float*) malloc(sizeof(float) * 2);
    tensor_head_scaled_dot_product(t0, t1, 2, out);
    
    int shape3[1] = {2};
    tensor * t3 = tensor_create(1, shape3);
    free(t3->data);
    t3->data = out;
    tensor_print(t3);

   // return 0;
}