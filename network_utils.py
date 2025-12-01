import struct
import pickle

def send_data(conn, data, lock):
    with lock:
        serialized_data = pickle.dumps(data)
    header = struct.pack("Q", len(serialized_data)) 
    conn.sendall(header + serialized_data)

def receive_data(conn):
    header = conn.recv(8)
    if not header:
        return None
    data_length = struct.unpack("Q", header)[0]

    data = bytearray()
    while len(data) < data_length:
        remaining = data_length - len(data)
        packet = conn.recv(remaining) 
        if not packet:
            raise ConnectionResetError("Connection closed while receiving data.")
        data.extend(packet) 
    return pickle.loads(data)