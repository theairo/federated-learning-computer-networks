import socket
import threading
import pickle
import time
import data_utils
from fl_utils import federated_average, test_global
import struct
import model
import torch

HOST = '77.47.196.66'
PORT = 7878
N_clients = 2
num_rounds = 10

start_event = threading.Event()
end_event = threading.Event()

client_sockets = 0
lock = threading.Lock()
list_of_state_dicts = []

model_global = model.mnistNet()

def client_handler(client_socket, addr, training_data, model):
    try:
        # Sending the training data to clients
        serialized_training_data = pickle.dumps(training_data)
        training_data_length = struct.pack("Q", len(serialized_training_data))
        client_socket.sendall(training_data_length)
        client_socket.sendall(serialized_training_data)

        # Training loop
        for round in range(num_rounds):
            # Wait until all N_clients are connected
            start_event.wait()

            # Sending the model to clients
            with lock:
                serialized_model = pickle.dumps(model)
            model_length = struct.pack("Q", len(serialized_model))
            client_socket.sendall(model_length)
            client_socket.sendall(serialized_model)

            # Receive weights from clients
            model_length = struct.unpack("Q", client_socket.recv(8))[0]
            model_bytes = client_socket.recv(model_length)
            model = pickle.loads(model_bytes)
            with lock:
                list_of_state_dicts.append(model.state_dict())

            # Wait
            end_event.wait()

    except ConnectionResetError:
        print(f"{addr} disconnected")
    except Exception as e:
        print(f"{addr} Error")
    finally:
        print(f"{addr} worked good")
        client_socket.close()

def main():
    client_sockets = 0
    # Creating server socket and allow to reuse port immediately
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Binding
    try:
        server_socket.bind((HOST, PORT))
    except OSError as e:
        print("FATALITY! Can't bind")
        return

    # Get data for training and test
    partitions = data_utils.get_partitions_test(N_clients)
    test_data = partitions[-1]

    # Listening
    server_socket.listen(N_clients)
    print("Server is listening")

    # Connecting to the clients
    try:
        while client_sockets < N_clients:
            # Accepting the client
            client_sock, addr = server_socket.accept()
            client_sockets+=1
            print(f"Accepted connection {client_sockets} from {addr}. "
                  f"Total needed {N_clients}, {N_clients - client_sockets} left")

            # Creating and running a new thread
            thread = threading.Thread(target=client_handler, args=(client_sock, addr, partitions[client_sockets], model_global))
            thread.start()
    except KeyboardInterrupt:
        print("Server is down")
    except Exception as e:
        print(f"Error accepting connections: {e}")

    # Training loop
    try:
        for round in range(num_rounds):
            print(f"Round {round + 1}")
            print("Starting training")
            # Start threads
            start_event.set()
            end_event.clear()

            # Wait until all threads are done
            while True:
                if len(list_of_state_dicts) == N_clients:
                    break
                time.sleep(0.1)

            # Average weights
            start_event.clear()
            avg_state_dict = federated_average(list_of_state_dicts)
            model_global.load_state_dict(avg_state_dict)

            # Test new model
            accuracy=test_global(model_global, test_data)
            print(f"New accuracy: {accuracy}")

            # Continuing
            end_event.set()

    except Exception as e:
        print(f"{e}")
        return
    finally:
        server_socket.close()
        print("Server shut down")

main()