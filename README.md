## Federated Learning for Computer Networks Simulation

# Project Description

This project is a Python-based simulation of a Federated Learning (FL) environment. Instead of a central server collecting all data (which creates privacy risks), this simulation models a system where multiple clients (e.g., devices or hospitals) train a model on their local data.

The clients then send only the model updates (weights) back to a central server, not their private data. The server aggregates these updates to create an improved global model, which is then sent back to the clients.

This project focuses on:

Client-Server Simulation: Modeling the interactions between multiple clients and a central server.

Local Training: Each client independently trains a copy of the model on its own dataset.

Privacy-Preserving Aggregation: Implementing a central server that aggregates model weights (e.g., using an algorithm like Federated Averaging or FedAvg) without ever seeing the raw data.

This project was built to explore privacy-preserving AI and the practical challenges of training models in a distributed network.

# Technologies Used

Python

PyTorch: For building, training, and updating the neural network models on both the client and server side.

NumPy: For numerical operations and data handling.

# How to Run

Clone the repository:

git clone [https://github.com/theairo/federated-learning-computer-networks.git](https://github.com/theairo/federated-learning-computer-networks.git)
cd federated-learning-computer-networks

pip install torch numpy

3.  Run the simulation. This requires starting the server and one or more clients in separate terminals.

 *(Note: You may need to change `server.py` and `client.py` to match your actual file names.)*

 **Terminal 1 (Run the Server):**
 ```bash
 python server.py
 ```

 **Terminal 2 (Run a Client):**
 ```bash
 python client.py
 ```

 **Terminal 3 (Run another Client):**
 ```bash
 python client.py
 ```

## Project Status

In-development student project to explore advanced machine learning concepts, distributed systems, and privacy in AI.
