import socket
import threading
import time
import threading
import data_utils
from fl_utils import federated_average
from network_utils import receive_data, send_data
import model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

HOST = '127.0.0.1' # Bind to all interfeces
PORT = 7878
N_clients = 2
num_rounds = 5

start_event = threading.Event()
end_event = threading.Event()

lock = threading.Lock()
list_of_state_dicts = []

model_global = model.mnistNet()

def client_handler(client_socket, addr, training_data, model):
    try:
        # Sending the training data to clients
        send_data(client_socket, training_data, lock)

        # Training loop
        for round in range(num_rounds):
            # Wait until all N_clients are connected
            start_event.wait()

            # Sending the model to clients
            send_data(client_socket, model, lock)

            # Receive weights from clients
            state_dict = receive_data(client_socket)
            with lock:
                list_of_state_dicts.append(state_dict)

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
        print("Can't bind the socket.")
        return

    # Get data for training and test
    partitions = data_utils.get_partitions_test(N_clients)
    test_data = partitions[-1]

    print(test_data)

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
            thread = threading.Thread(target=client_handler, args=(client_sock, addr, partitions[client_sockets], model_global))
            client_sockets+=1 # !
            thread.start()
    except KeyboardInterrupt:
        print("Server is down")
    except Exception as e:
        print(f"Error accepting connections: {e}")

    lossFun = nn.CrossEntropyLoss()

    # Training loop
    try:
        for round in range(num_rounds):
            print(f"Round {round + 1}")
            print("Starting training")

            list_of_state_dicts.clear() # !!!
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
            
            TEST_BATCH_SIZE = 128
            test_loader = DataLoader(
                test_data, 
                batch_size=TEST_BATCH_SIZE, 
                shuffle=False 
            )

            total_loss = 0
            correct = 0
            total_samples = 0

            for X, y in test_loader:
                
                model_global.eval() 
                with torch.no_grad():
                    yHat = model_global(X) 
                    loss = lossFun(yHat, y)
                    
                total_loss += loss.item() * X.size(0)

                _, predicted = torch.max(yHat.data, 1)
                total_samples += y.size(0)
                correct += (predicted == y).sum().item()

            avg_loss = total_loss / total_samples

            accuracy = 100 * correct / total_samples

            print(f"Round {round + 1} - Average Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
            
            end_event.set()

    except Exception as e:
        print(f"{e}")
        return
    finally:
        server_socket.close()
        print("Server shut down")

if __name__ == "__main__":
    main()