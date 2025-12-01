# Import python libraries
import socket
import threading
import time
import torch
import torch.nn as nn
import ssl
import traceback

# Import from custom files
from data_utils import get_partitions
from fl_utils import federated_average,test_global
from network_utils import receive_data, send_data
from model import mnistNet

# Network configuration
HOST = '127.0.0.1'  # Bind to all interfaces
PORT = 7878
N_clients = 2
num_rounds = 3 # Must be the same in server and client

# Synchronization primitives
start_event = threading.Event()
end_event = threading.Event()
lock = threading.Lock()

list_of_state_dicts = []
model_global = mnistNet()

# Handles communication with a single client for the duration of training
def client_handler(client_socket, addr, training_data, model_dict, context):
    try:
        # Secure connection
        client_socket = context.wrap_socket(client_socket, server_side=True)

        # Sending number of rounds
        send_data(client_socket, num_rounds, lock)

        # Sending the training data to clients
        send_data(client_socket, training_data, lock)

        # Training loop
        for round in range(num_rounds):
            # Wait until all N_clients are connected
            start_event.wait()

            # Sending the model to clients
            send_data(client_socket, model_dict, lock)

            # Receive weights from clients
            state_dict = receive_data(client_socket)
            with lock:
                list_of_state_dicts.append(state_dict)

            # Wait until the next round (until other threads are done)
            end_event.wait()

        # End of the training
        print(f"{addr} worked good")

    # Client was disconnected
    except ConnectionResetError:
        print(f"{addr} disconnected")

    # SSL error
    except ssl.SSLError as e:
        print(f"SSL Error: {e}")
        server_socket.close()
        print("Server shut down due to the error")
        return

    # Other issues
    except Exception as e:
        print(f"{addr} Error")

    finally:
        client_socket.close()

def main():
    client_sockets = 0

    # Creating server socket and allow to reuse port immediately
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Security
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile='cert.pem',keyfile='key.pem')

    # Binding
    try:
        server_socket.bind((HOST, PORT))

    # Cannot bind, usually if same port is already in use
    except OSError as e:
        print("Can't bind the socket.")
        return

    # Get data for training and test
    partitions = get_partitions(N_clients)
    test_data = partitions[-1]

    # Listening
    server_socket.listen(N_clients)
    print("Server is listening...")

    # Connecting to the clients
    try:
        while client_sockets < N_clients:
            # Accepting the client
            client_sock, addr = server_socket.accept()
            print(f"Accepted connection {client_sockets} from {addr}. "
                  f"Total needed {N_clients}, {N_clients - client_sockets - 1} left")

            # Creating and running a new thread
            thread = threading.Thread(target=client_handler,
                                      args=(client_sock, addr, partitions[client_sockets], model_global.state_dict(), context))
            thread.start()
            client_sockets += 1

    # Manual server interruption
    except KeyboardInterrupt:
        print("Server is down")

    # Other issues
    except Exception as e:
        print(f"Error accepting connections: {e}")

    # Training loop
    try:
        for round in range(num_rounds):
            print(f"Round {round + 1}")
            print("Starting training")

            # Delete weights from previous rounds
            list_of_state_dicts.clear()

            # Start threads (training)
            start_event.set()
            end_event.clear()

            # Wait until all threads complete training
            while True:
                if len(list_of_state_dicts) == N_clients:
                    break
                time.sleep(0.1)
            start_event.clear()

            # Average weights
            avg_state_dict = federated_average(list_of_state_dicts)
            model_global.load_state_dict(avg_state_dict)

            # Compute loss per sample and accuracy
            avg_loss, accuracy=test_global(model_global,test_data)

            # Print results: loss and accuracy
            print(f"Average Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

            end_event.set()

    # Other issues
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {type(e).__name__} - {e}")

        # Detailed error explanation
        traceback.print_exc()

    finally:
        server_socket.close()
        print("Server shut down")

if __name__ == "__main__":
    main()