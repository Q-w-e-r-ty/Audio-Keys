from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
from Crypto.Random import get_random_bytes

def encrypt_file(file_path, key):
    # Read data from file
    with open(file_path, 'rb') as file:
        data = file.read()

    # Initialize AES cipher with key
    cipher = AES.new(key, AES.MODE_CBC)

    # Encrypt the data
    encrypted_data = cipher.encrypt(pad(data, AES.block_size))

    # Write encrypted data to file
    with open(file_path + ".enc", 'wb') as file:
        file.write(cipher.iv)
        file.write(encrypted_data)

def decrypt_file(file_path, key):
    # Read IV and encrypted data from file
    with open(file_path, 'rb') as file:
        iv = file.read(16)  # IV is always 16 bytes
        encrypted_data = file.read()

    # Initialize AES cipher with key and IV
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt the data
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    # Write decrypted data to file
    with open(os.path.splitext(file_path)[0], 'wb') as file:
        file.write(decrypted_data)

# Example usage:
# Replace 'file_path' and 'key' with appropriate values
# encrypt_file('plaintext.txt', b'this_is_a_16_or_32_byte_key')
# decrypt_file('plaintext.txt.enc', b'this_is_a_16_or_32_byte_key')

#import os
#file_path = r'C:/Users/Tanmay/Downloads/sample.txt'
#key = os.urandom(16)

# Encrypt the file
#encrypt_file(file_path, key)
#print("File encrypted successfully.")

#Decrypt the encrypted file
#decrypt_file(file_path + ".enc", key)
#print("File decrypted successfully.")