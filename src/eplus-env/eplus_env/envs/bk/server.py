import socket               # Import socket module

s = socket.socket()         # Create a socket object
s.bind(('0.0.0.0', 11334))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
print (':'.join(str(e) for e in s.getsockname()))
while True:
	c, addr = s.accept()     # Establish connection with client.
	print ('Got connection from', addr)
	while True:
		recv = c.recv(1024).decode()
		print (recv)
		if recv == 'get':
			c.sendall(bytearray(str([1,2,3,4]), encoding = 'utf-8'));
		else:		
			c.sendall(b'I don not understand')
