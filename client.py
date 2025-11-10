import socket
import pickle
import struct
import time

HOST = '77.47.196.66'
PORT = 7878

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        print(f"Connecting to {HOST}:{PORT}")
        client_socket.connect((HOST, PORT))
        print("Connected")
    except ConnectionRefusedError:
        print("Refused connection")
    except Exception as e:
        print(f"Exception {e}")
    
    try:
        welcome_msg = f"Welcome Client"
        client_socket.send(welcome_msg.encode('utf-8'))

        data = client_socket.recv(1024)

        if data:
            print(f"Client Received {data.decode('utf-8')}")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"Client shutting down")

    except Exception as e:
        print(f"Client Error")

    finally:
        client_socket.close()

if __name__ == "__main__":
    main()