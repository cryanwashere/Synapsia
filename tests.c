int test_transform()
{
    int shape0[1] = {16};

    tensor * t0 = tensor_create(1, shape0);
    tensor_ones(t0);

    int shape1[2] = {4, 16};
    tensor * t1 = tensor_create(2, shape1);
    tensor_ones(t1);

    int shape2[1] = {4};
    tensor * t2 = tensor_create(1, shape2);


    tensor_vector_transformation(t0, t1, t2);

    tensor_print(t0);
    tensor_print(t1);

    tensor_print(t2);
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

    return 0;
}