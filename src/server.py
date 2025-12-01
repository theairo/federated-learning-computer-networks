# Import python libraries
import socket
import threading
import time
import torch
import torch.nn as nn
import ssl
import traceback
import select

# Import from custom files
from utils.data_utils import get_partitions
from utils.fl_utils import federated_average,test_global
from utils.network_utils import receive_data, send_data
from model import mnistNet

from config import KEY_PATH, CERT_PATH, OUTPUT_DIR

# Network configuration
HOST = '77.47.196.66'
PORT = 7878
N_clients = 2
num_rounds = 3 # Must be the same in server and client

# Synchronization primitives
start_event = threading.Event()
end_event = threading.Event()
lock = threading.Lock()

client_sockets = 0
sockets_miss = []
list_of_state_dicts = []
model_global = mnistNet()

class ClientDisconnected(Exception):
    """Raised when the client disconnects gracefully or forcibly"""
    pass

# Handles communication with a single client for the duration of training
def client_handler(client_socket, client_number, addr, training_data, model_dict, context):
    global client_sockets
    try:
        # Secure connection
        client_socket = context.wrap_socket(client_socket, server_side=True)

        # Sending number of rounds
        send_data(client_socket, num_rounds, lock)

        # Sending the training data to clients
        send_data(client_socket, training_data, lock)

        # Training loop
        for round in range(num_rounds):
            # Checking whether the client has disconnected and waiting until all clients are ready
            while not start_event.is_set():
                start_event.wait(timeout = 1)
                readable, _, _ = select.select([client_socket], [], [], 0)
                if readable:
                    raise ClientDiconnected("Client sent signal during wait phase")

            # Sending the model to clients
            send_data(client_socket, model_dict, lock)

            # Receive weights from clients
            state_dict = receive_data(client_socket)
            if state_dict == None:
                raise ClientDisconnected("Client sent empty data block")
            with lock:
                list_of_state_dicts.append(state_dict)

            # Wait until the next round (until other threads are done)
            end_event.wait()

        # End of the training
        print(f"{addr} worked good")

    # SSL error
    except ssl.SSLError as e:
        print(f"SSL Error: {e} on the client {addr}")
        return

    # Client has disconnected
    except (OSError, ClientDisconnected) as e:
        print(f"Client {client_number} ({addr}) disconnected: {e}")
        with lock:
            sockets_miss.append(client_number)
            client_sockets -= 1

    # Other issues
    except Exception as e:
        print(f"CAPTURED EXCEPTION TYPE: {type(e).__name__}")
        print(f"{addr} Error: {e}")

    finally:
        client_socket.close()

def main():
    global client_sockets

    # Creating server socket and allow to reuse port immediately
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Security
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=CERT_PATH,keyfile=KEY_PATH)

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
            if (len(sockets_miss) != 0):
                partition_number = sockets_miss.pop()
            else:
                partition_number = client_sockets
            thread = threading.Thread(target=client_handler,
                                      args=(client_sock, partition_number, addr, partitions[partition_number], model_global.state_dict(), context))
            thread.start()
            client_sockets += 1

    # Manual server interruption
    except KeyboardInterrupt:
        print("Server is down")

    # Other issues
    except Exception as e:
        print(f"Error accepting connections: {e}")
        server_socket.close()
        print("Server shut down due to the error")
        return

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
                if len(list_of_state_dicts) >= client_sockets and client_sockets > 0:
                    break
                if client_sockets == 0:
                    print("All clients disconnected")
                    return

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
        
        # Save the global model
        print("Saving the final model...")
        torch.save(model_global.state_dict(), OUTPUT_DIR / 'server_model.pth')
        print("Model saved to server_model.pth")

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