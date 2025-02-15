// neural.h
#ifndef NEURAL_H
#define NEURAL_H

#include <stdint.h>

typedef double (*RandFcn)(void);

typedef struct {
    uint32_t n_inputs;
    uint32_t n_hidden;
    uint32_t n_outputs;
    double* weights_hidden;
    double* biases_hidden;
    double* weights_output;
    double* biases_output;
    double* hidden;
    double* output;
} Network;

typedef struct {
    double* grad_hidden;
    double* grad_output;
    double* velocity_hidden;
    double* velocity_output;
} Trainer;

Network* network_init(Network* network, uint32_t n_inputs, uint32_t n_hidden, uint32_t n_outputs, RandFcn rand);
void network_free(Network* network);
void network_predict(Network* network, const double* input);
Trainer* trainer_init(Trainer* trainer, Network* network);
void trainer_train(Trainer* trainer, Network* network, const double* input, const double* y, double lr, double momentum);
void trainer_free(Trainer* trainer);
void network_save(const Network* network, const char* filename);
int network_load(Network* network, const char* filename);

#endif  // NEURAL_H

