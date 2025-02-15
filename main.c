// main.c
#include "neural.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* Linear congruential generator for reproducible randomness */
uint32_t P = 2147483647;
uint32_t A = 16807;
uint32_t current = 1;

#define TOTAL_ITERATIONS 10000000
#define CHECKPOINT_INTERVAL 100000

double Rand() {
    current = (uint64_t)current * A % P;
    return (double)current / P;
}

/* Print network parameters */
void print_network(const Network* network) {
    printf("Weights (Input -> Hidden):\n");
    for (uint32_t i = 0; i < network->n_inputs; i++) {
        for (uint32_t j = 0; j < network->n_hidden; j++) {
            printf("%9.6f ", network->weights_hidden[i * network->n_hidden + j]);
        }
        printf("\n");
    }
    printf("Biases (Hidden):\n");
    for (uint32_t j = 0; j < network->n_hidden; j++) {
        printf("%9.6f ", network->biases_hidden[j]);
    }
    printf("\n");

    printf("Weights (Hidden -> Output):\n");
    for (uint32_t j = 0; j < network->n_hidden; j++) {
        for (uint32_t k = 0; k < network->n_outputs; k++) {
            printf("%9.6f ", network->weights_output[j * network->n_outputs + k]);
        }
        printf("\n");
    }
    printf("Biases (Output):\n");
    for (uint32_t k = 0; k < network->n_outputs; k++) {
        printf("%9.6f ", network->biases_output[k]);
    }
    printf("\n");
}

/* Logical operation functions */
static uint32_t xor_op(uint32_t i, uint32_t j)   { return i ^ j; }
static uint32_t xnor_op(uint32_t i, uint32_t j)  { return 1 - (i ^ j); }
static uint32_t or_op(uint32_t i, uint32_t j)    { return i | j; }
static uint32_t and_op(uint32_t i, uint32_t j)   { return i & j; }
static uint32_t nor_op(uint32_t i, uint32_t j)   { return 1 - (i | j); }
static uint32_t nand_op(uint32_t i, uint32_t j)  { return 1 - (i & j); }

#define ITERS 40000
#define ITERS2 (ITERS + 4960000)

int main() {
    /* Create a network with 2 inputs, 10 hidden neurons, and 6 outputs
       (one output per logical function: XOR, XNOR, OR, AND, NOR, NAND) */
    // Before training starts
    Network network = {0};
    network_init(&network, 2, 10, 6, Rand);

    /* Training parameters */
    double learning_rate = 0.1;
    double momentum      = 0.9;
    int checkpoint = network_load(&network, "checkpoint.dat");
    // Try to load a checkpoint
    if (checkpoint == 0) {
        printf("Resumed from checkpoint.\n");
    } else {
        printf("No checkpoint found, starting fresh training.\n");
    }

    Trainer trainer = {0};
    trainer_init(&trainer, &network);
    
    /* Training data: four possible inputs */
    double inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    /* Expected outputs for each of the 6 logic operations:
       Order: XOR, XNOR, OR, AND, NOR, NAND */
    double outputs[4][6] = {
        { (double)xor_op(0, 0), (double)xnor_op(0, 0), (double)or_op(0, 0),
          (double)and_op(0, 0), (double)nor_op(0, 0), (double)nand_op(0, 0) },
        { (double)xor_op(0, 1), (double)xnor_op(0, 1), (double)or_op(0, 1),
          (double)and_op(0, 1), (double)nor_op(0, 1), (double)nand_op(0, 1) },
        { (double)xor_op(1, 0), (double)xnor_op(1, 0), (double)or_op(1, 0),
          (double)and_op(1, 0), (double)nor_op(1, 0), (double)nand_op(1, 0) },
        { (double)xor_op(1, 1), (double)xnor_op(1, 1), (double)or_op(1, 1),
          (double)and_op(1, 1), (double)nor_op(1, 1), (double)nand_op(1, 1) }
    };


    
    printf("Initial results:\n Input -> (XOR, XNOR, OR, AND, NOR, NAND)\n");
    
    for (int i = 0; i < 4; i++) {
        if(checkpoint != 0) {
            network_predict(&network, inputs[i]);
            printf("%.0f, %.0f = ", inputs[i][0], inputs[i][1]);
            for (int k = 0; k < network.n_outputs; k++) {
                printf("%.3f ", network.output[k]);
            }
            
            
        } else {
            double* input = inputs[i % 4];
            network_predict(&network, input);
            printf(
                "%.0f,%.0f = %.3f %.3f %.3f %.3f %.3f %.3f\n",
                input[0],
                input[1],
                network.output[0],
                network.output[1],
                network.output[2],
                network.output[3],
                network.output[4],
                network.output[5]);
        }

        
        printf("\n");
    }


    /* First training phase */
    for (int i = 0; i < ITERS; i++) {
        int index = i % 4;
        trainer_train(&trainer, &network, inputs[index], outputs[index],
                      learning_rate, momentum);
        if (i % CHECKPOINT_INTERVAL == 0) {
            network_save(&network, "checkpoint.dat");
        }
    }
    printf("\nResults after %d iterations:\n Input -> (XOR, XNOR, OR, AND, NOR, NAND)\n", ITERS);
    for (int i = 0; i < 4; i++) {
        network_predict(&network, inputs[i]);
        printf("%.0f, %.0f = ", inputs[i][0], inputs[i][1]);
        for (uint32_t k = 0; k < network.n_outputs; k++) {
            printf("%.3f ", network.output[k]);
        }
        printf("\n");
    }

    /* Second training phase */
    for (int i = 0; i < ITERS2; i++) {
        int index = i % 4;
        trainer_train(&trainer, &network, inputs[index], outputs[index],
                      learning_rate, momentum);
        if (i % CHECKPOINT_INTERVAL == 0) {
            network_save(&network, "checkpoint.dat");
        }
    }
    printf("\nResults after %d iterations:\n Input -> (XOR, XNOR, OR, AND, NOR, NAND)\n", ITERS2);
    for (int i = 0; i < 4; i++) {
        network_predict(&network, inputs[i]);
        printf("%.0f, %.0f = ", inputs[i][0], inputs[i][1]);
        for (uint32_t k = 0; k < network.n_outputs; k++) {
            printf("%.3f ", network.output[k]);
        }
        printf("\n");
    }


    /* Print final network parameters */
    print_network(&network);

    trainer_free(&trainer);
    network_free(&network);
    return 0;
}
