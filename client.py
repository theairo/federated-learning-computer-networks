# Import python libraries
import socket
import threading
import traceback

# Import from custom files
from fl_utils import train_local
from network_utils import send_data, receive_data
from model import mnistNet

# Network configuration
HOST = '127.0.0.1'
PORT = 7878
num_rounds = 3 # Must be the same in server and client

# Synchronisation primitives
lock = threading.Lock()

def main():
    # Training parameters
    num_epochs = 5
    learning_rate = 0.1

    # Model
    model = mnistNet()

    # Creating client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connecting to the server
    try:
        client_socket.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")

    # If server is not running or port is wrong
    except ConnectionRefusedError:
        print("Refused connection")
        return

    # Timeout and other issues
    except Exception as e:
        print(f"Exception {e}")
        return

    try:

        training_data = receive_data(client_socket)

        # Checking whether training data is not empty
        if training_data is None:
            print("Error. Training data is None.")
            return

        # Communication loop
        for round in range(num_rounds):
            print(f"Round {round + 1}")

            # Receiving model
            print("Waiting to receive the model")
            model_dict = receive_data(client_socket)
            model.load_state_dict(model_dict)
            print("Model received")

            # Training
            print("Training...")
            train_local(model, training_data, num_epochs, learning_rate)
            print("Training complete")

            # Sending the model
            send_data(client_socket, model.state_dict(), lock)
            print("Model sent")

        # End of the training
        print('The client successfully completed the work')

    # Manual client interruption
    except KeyboardInterrupt:
        print(f"Client was manually interrupted")

    # Other issues
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {type(e).__name__} - {e}")

        # Detailed error explanation
        traceback.print_exc()

    finally:
        client_socket.close()

if __name__ == "__main__":
    main()