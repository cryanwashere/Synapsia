#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
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

typedef struct {

} feedforward_node;

/**
 * @brief Attention Node
 * 
 * 
 **/
typedef struct {

    // for full clarity, these are NOT matrices!
    // these is just the weights for one particular node,
    // and therefore they are just one dimensional vectors.
    struct tensor * w_key;
    struct tensor * w_query;
    struct tensor * w_value; 

    // the dimensionality of all the weight vectors
    int rank;

} attention_node; 

/**
 * @brief Block Node 
 * 
 * This structure contains the node elements of a certain transformer node 
 * at a specific block
 * 
 */
typedef struct {

    struct attention_node * attention;

    struct feedforward_node * feedforward;

} block_node;

typedef struct {    

    // the blocks of the given node
    int layer_c;
    struct block_node** layers;


} transformer_node;


typedef struct {

    // the number of nodes that the transformer has
    int node_c;
    struct transformer_node** nodes;

}   transformer;



int main ()
{
    
    
    return 0;
}