# Neural Network in C for Learning Logic Functions

This project implements a simple neural network in C that learns basic logic functions (XOR, XNOR, OR, AND, NOR, NAND). 
It uses a single hidden layer with backpropagation and momentum-based updates, along with Xavier initialization for the weights. 
The network also supports checkpointing so that training can be resumed from a saved state.

## Features

- **Logic Functions:** Learns 6 logic operations (XOR, XNOR, OR, AND, NOR, NAND) from 2 inputs.
- **Network Architecture:** Configurable with one hidden layer (default: 10 neurons) and 6 output neurons.
- **Xavier Initialization:** Uses Xavier (Glorot) initialization to set initial weights for improved training.
- **Momentum-Based Training:** Implements momentum in the weight updates for smoother and faster convergence.
- **Checkpointing:** Saves and loads network parameters from a file (`checkpoint.dat`), allowing you to resume training.
- **Reproducible Randomness:** Uses a linear congruential generator for deterministic random number generation.
- **Configurable Training:** Adjust the learning rate, momentum, total iterations, and checkpoint intervals directly in the code.

## Project Structure

- **`main.c`**  
  Contains the `main()` function, sets up the network and trainer, defines training data (inputs and expected outputs), and handles the training loop along with checkpointing.

- **`neural.c`**  
  Implements the neural network functions including initialization (`network_init`), feed-forward prediction (`network_predict`), training (`trainer_train`), checkpoint saving (`network_save`), and loading (`network_load`).

- **`neural.h`**  
  The header file that declares the network and trainer structures and prototypes for all functions.

## Requirements

- GCC (or any standard C compiler)
- Make

## Compilation and Execution

1. **Build the Project:**  
   Run the following command in the project root directory:
   ```bash
   make
   ```
This will compile the source files and produce the executable (default output: ./build/cmain).

### Run the Executable:
Execute the compiled program:
  ```bash
  ./build/cmain
  ```
The program will attempt to load a checkpoint from checkpoint.dat. If none is found, it starts with fresh random initialization.
Training Parameters
The training behavior is controlled by constants defined in main.c:

CHECKPOINT_INTERVAL – Interval (in iterations) at which the network state is saved to checkpoint.dat.
learning_rate – The learning rate used in training.
momentum – The momentum factor used to update weights.
You can modify these parameters to suit your training needs.

### Checkpointing
The network periodically saves its parameters to checkpoint.dat.
On subsequent runs, the program will load these parameters to resume training from the last checkpoint.
Example Output
The program prints:

Initial predictions (before training or after resuming from a checkpoint).
Results after a fixed number of iterations (e.g., after 40,000 iterations and again after 5,000,000 iterations).
Final network parameters: Displays the weights and biases for both the hidden and output layers.


### Customization
Network Architecture:
Change the number of inputs, hidden neurons, or outputs in the call to network_init() in main.c.

### Training Data:
Modify or extend the training data (inputs and expected outputs) as needed.

### Parameters:
Adjust the learning rate, momentum, total iterations, and checkpoint interval to experiment with training performance.

Based on Daniel Lidstrom code for neural networks.
