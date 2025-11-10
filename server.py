import socket
import threading
import pickle
import struct
import time

HOST = '77.47.196.66'
PORT = 7878
N_clients = 2

client_threads = []
client_sockets = {}
lock = threading.Lock()

def client_handler(client_socket, addr):
    """ Worker function """

    print(f"Thread for {addr}")

    try:
        welcome_msg = f"Welcome {addr}"
        client_socket.send(welcome_msg.encode('utf-8'))

        data = client_socket.recv(1024)

        if data:
            print(f"{addr} Received {data.decode('utf-8')}")

        while True:
            time.sleep(1)

    except ConnectionResetError:
        print(f"{addr} disconnected")

    except Exception as e:
        print(f"{addr} Error")

    finally:
        print(f"{addr} worked good")
        with lock:
            if addr in client_sockets:
                del client_sockets[addr]
        client_socket.close()

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
    except OSError as e:
        print("FATALITY! Can't bind")
        return
    
    server_socket.listen(N_clients)
    print("Server is listening")

    try:
        while len(client_sockets) < N_clients:
            client_sock, addr = server_socket.accept()

            with lock:
                client_sockets[addr] = client_sock

            print(f"Accepted connection from {addr}")

            thread = threading.Thread(target=client_handler, args=(client_sock, addr))

            thread.daemon = True

            thread.start()

            client_threads.append(thread)

            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("Server is down")
    except Exception as e:
        print(f"Error accepting connecitons: {e}")

    # Orchestration Phase
    if len(client_sockets) == N_clients:
        print("Clients connected")
    
    server_socket.close()
    print("Server shut down")

if __name__ == "__main__":
    main()

    



