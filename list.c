#include <stdlib.h>
#include <stdio.h>

typedef struct list_node
{   
    void * data; 
    struct list_node * next;
} list_node;

typedef struct
{
    list_node * head;
} list;

list_node * list_node_create(void * data)
{
    list_node * new_node = (list_node *) malloc(sizeof(list_node));
    new_node->data = data;
    new_node->next = NULL;
    return new_node;
}

list * list_create()
{
    list * new_list = (list*) malloc(sizeof(list));
    return new_list;
}

void list_append(list * ls, void * data)
{
    list_node * new_node = list_node_create(data);

    if (ls->head == NULL) {
        ls->head = new_node;
    } else {
        list_node * current = ls->head;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_node;
    }
}

void
list_print (list * ls, void ( * print_fn)(void *))
{
    list_node * current = ls->head;

    printf("[ ");
    while (current != NULL) {
        print_fn(current->data);
        current = current->next;
    }
    printf(" ]\n");
}

void
print_int (void * data)
{
    int * val = (int *) data;
    printf(" %d ", *val);
}


int main()
{

    list * ls = list_create();

    int a = 1;
    int b = 2;
    int c = 3;

    list_append(ls, &a);
    list_append(ls, &b);
    list_append(ls, &c);

    list_print(ls, print_int);

}