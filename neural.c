// neural.c
#include "neural.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* Sigmoid activation and its derivative */
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_prim(double f) {
    return f * (1.0 - f);
}

/* Initialize network with Xavier initialization */
Network* network_init(Network* network,
                      uint32_t n_inputs,
                      uint32_t n_hidden,
                      uint32_t n_outputs,
                      RandFcn rand) {
    network->n_inputs  = n_inputs;
    network->n_hidden  = n_hidden;
    network->n_outputs = n_outputs;

    network->weights_hidden = calloc(n_inputs * n_hidden, sizeof(*network->weights_hidden));
    network->biases_hidden  = calloc(n_hidden, sizeof(*network->biases_hidden));
    network->weights_output = calloc(n_hidden * n_outputs, sizeof(*network->weights_output));
    network->biases_output  = calloc(n_outputs, sizeof(*network->biases_output));
    network->hidden         = calloc(n_hidden, sizeof(*network->hidden));
    network->output         = calloc(n_outputs, sizeof(*network->output));

    if (!network->weights_hidden || !network->biases_hidden ||
        !network->weights_output || !network->biases_output ||
        !network->hidden || !network->output) {
        fprintf(stderr, "Memory allocation failed in network_init\n");
        exit(EXIT_FAILURE);
    }

    /* Xavier initialization for weights (for sigmoid activation) */
    double limit_hidden = sqrt(6.0 / (n_inputs + n_hidden));
    for (size_t i = 0; i < n_inputs * n_hidden; i++) {
        network->weights_hidden[i] = (rand() * 2.0 - 1.0) * limit_hidden;
    }
    /* biases are already zeroed by calloc */

    double limit_output = sqrt(6.0 / (n_hidden + n_outputs));
    for (size_t i = 0; i < n_hidden * n_outputs; i++) {
        network->weights_output[i] = (rand() * 2.0 - 1.0) * limit_output;
    }

    return network;
}

void network_free(Network* network) {
    free(network->weights_hidden);
    free(network->biases_hidden);
    free(network->weights_output);
    free(network->biases_output);
    free(network->hidden);
    free(network->output);
}

/* Feed-forward prediction */
void network_predict(Network* network, const double* input) {
    // Hidden layer activations
    for (uint32_t j = 0; j < network->n_hidden; j++) {
        double sum = 0.0;
        for (uint32_t i = 0; i < network->n_inputs; i++) {
            sum += input[i] * network->weights_hidden[i * network->n_hidden + j];
        }
        sum += network->biases_hidden[j];
        network->hidden[j] = sigmoid(sum);
    }
    // Output layer activations
    for (uint32_t k = 0; k < network->n_outputs; k++) {
        double sum = 0.0;
        for (uint32_t j = 0; j < network->n_hidden; j++) {
            sum += network->hidden[j] * network->weights_output[j * network->n_outputs + k];
        }
        sum += network->biases_output[k];
        network->output[k] = sigmoid(sum);
    }
}

/* Trainer initialization with momentum arrays */
Trainer* trainer_init(Trainer* trainer, Network* network) {
    trainer->grad_hidden = calloc(network->n_hidden, sizeof(*trainer->grad_hidden));
    trainer->grad_output = calloc(network->n_outputs, sizeof(*trainer->grad_output));
    trainer->velocity_hidden = calloc(network->n_inputs * network->n_hidden, sizeof(*trainer->velocity_hidden));
    trainer->velocity_output = calloc(network->n_hidden * network->n_outputs, sizeof(*trainer->velocity_output));

    if (!trainer->grad_hidden || !trainer->grad_output ||
        !trainer->velocity_hidden || !trainer->velocity_output) {
        fprintf(stderr, "Memory allocation failed in trainer_init\n");
        exit(EXIT_FAILURE);
    }
    return trainer;
}

/* Train network on one training example using backpropagation with momentum */
void trainer_train(Trainer* trainer,
                   Network* network,
                   const double* input,
                   const double* y,
                   double lr,
                   double momentum) {
    // Forward pass
    network_predict(network, input);

    // Compute output layer gradient
    for (uint32_t k = 0; k < network->n_outputs; k++) {
        trainer->grad_output[k] = (network->output[k] - y[k]) * sigmoid_prim(network->output[k]);
    }

    // Compute hidden layer gradient
    for (uint32_t j = 0; j < network->n_hidden; j++) {
        double sum = 0.0;
        for (uint32_t k = 0; k < network->n_outputs; k++) {
            sum += trainer->grad_output[k] * network->weights_output[j * network->n_outputs + k];
        }
        trainer->grad_hidden[j] = sum * sigmoid_prim(network->hidden[j]);
    }

    // Update output weights with momentum
    for (uint32_t j = 0; j < network->n_hidden; j++) {
        for (uint32_t k = 0; k < network->n_outputs; k++) {
            size_t idx = j * network->n_outputs + k;
            double delta = lr * trainer->grad_output[k] * network->hidden[j];
            trainer->velocity_output[idx] = momentum * trainer->velocity_output[idx] + delta;
            network->weights_output[idx] -= trainer->velocity_output[idx];
        }
    }
    // Update output biases
    for (uint32_t k = 0; k < network->n_outputs; k++) {
        network->biases_output[k] -= lr * trainer->grad_output[k];
    }

    // Update hidden weights with momentum
    for (uint32_t i = 0; i < network->n_inputs; i++) {
        for (uint32_t j = 0; j < network->n_hidden; j++) {
            size_t idx = i * network->n_hidden + j;
            double delta = lr * trainer->grad_hidden[j] * input[i];
            trainer->velocity_hidden[idx] = momentum * trainer->velocity_hidden[idx] + delta;
            network->weights_hidden[idx] -= trainer->velocity_hidden[idx];
        }
    }
    // Update hidden biases
    for (uint32_t j = 0; j < network->n_hidden; j++) {
        network->biases_hidden[j] -= lr * trainer->grad_hidden[j];
    }
}

void trainer_free(Trainer* trainer) {
    free(trainer->grad_hidden);
    free(trainer->grad_output);
    free(trainer->velocity_hidden);
    free(trainer->velocity_output);
}

/* Save network parameters to a binary file */
void network_save(const Network* network, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("fopen");
        return;
    }
    fwrite(&network->n_inputs, sizeof(uint32_t), 1, file);
    fwrite(&network->n_hidden, sizeof(uint32_t), 1, file);
    fwrite(&network->n_outputs, sizeof(uint32_t), 1, file);
    fwrite(network->weights_hidden, sizeof(double), network->n_inputs * network->n_hidden, file);
    fwrite(network->biases_hidden, sizeof(double), network->n_hidden, file);
    fwrite(network->weights_output, sizeof(double), network->n_hidden * network->n_outputs, file);
    fwrite(network->biases_output, sizeof(double), network->n_outputs, file);
    fclose(file);
}

/* Load network parameters from a binary file */
int network_load(Network* network, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("fopen");
        return -1;
    }
    uint32_t n_inputs, n_hidden, n_outputs;
    fread(&n_inputs, sizeof(uint32_t), 1, file);
    fread(&n_hidden, sizeof(uint32_t), 1, file);
    fread(&n_outputs, sizeof(uint32_t), 1, file);
    if (n_inputs != network->n_inputs || n_hidden != network->n_hidden ||
        n_outputs != network->n_outputs) {
        fprintf(stderr, "Network dimensions mismatch!\n");
        fclose(file);
        return -1;
    }
    fread(network->weights_hidden, sizeof(double), network->n_inputs * network->n_hidden, file);
    fread(network->biases_hidden, sizeof(double), network->n_hidden, file);
    fread(network->weights_output, sizeof(double), network->n_hidden * network->n_outputs, file);
    fread(network->biases_output, sizeof(double), network->n_outputs, file);
    fclose(file);
    return 0;
}
