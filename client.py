import socket
import pickle
import struct
import time
from fl_utils import train_local

HOST = '77.47.196.66'
PORT = 7878

def main():
    # Training parameters
    num_epochs = 5
    learning_rate = 0.1
    num_rounds = 10

    # Creating client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connecting to the server
    try:
        client_socket.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")
    except ConnectionRefusedError:
        print("Refused connection")
        return
    except Exception as e:
        print(f"Exception {e}")
        return
    try:
        # Receiving training data
        print("Receiving training data")
        training_data_length = struct.unpack("Q", client_socket.recv(8))[0]
        print("Received data length")
        training_data = pickle.loads(client_socket.recv(training_data_length))
        print("Received training data")

        # Communication loop
        for round in range(num_rounds):
            print(f"Round {round + 1}")
            # Receiving model
            print("Receiving the model")
            model_length = struct.unpack("Q", client_socket.recv(8))[0]
            model_bytes = client_socket.recv(model_length)
            model = pickle.loads(model_bytes)
            print("Model received")

            print("Training...")
            # Training
            train_local(model, training_data, num_epochs, learning_rate)
            print("Training complete")

            # Sending the model
            serialized_model = pickle.dumps(model)
            model_length = struct.pack("Q", len(serialized_model))
            client_socket.sendall(model_length)
            client_socket.sendall(serialized_model)
            print("Model send")

    except KeyboardInterrupt:
        print(f"Client shutting down")
    except Exception as e:
        print(f"Client Error")
    finally:
        client_socket.close()

if __name__ == "__main__":
    main()