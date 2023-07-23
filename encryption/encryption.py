import os
import random
import string


def generate_random_key(length):
    """Generate a random encryption key of a specified length."""
    letters = string.ascii_letters + string.digits
    key = ''.join(random.choice(letters) for _ in range(length))
    print(f"Here is the encryption key save to decrypt the file later: {key}")
    return key


def encrypt_files(directory, key):
    """ Encrypt files in the specified directory using a given encryption key."""
    key_bytes = key.encode()  # Convert the key to bytes
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.txt'):
                with open(file_path, 'rb') as f:
                    data = f.read()

                encrypted_data = bytes(
                    byte ^ key_byte for byte, key_byte in zip(data, key_bytes))

                with open(file_path, 'wb') as f:
                    f.write(encrypted_data)


def decrypt_file(directory, key):
    """Decrypt files in the specified directory using a given key"""
    encrypt_files(directory, key)


def main():
    target_directory = input("Enter the target directory path: ")
    key_length = int(input("Enter the key length: "))
    encryption_key = generate_random_key(key_length)

    print("Encrypting files...")
    encrypt_files(target_directory, encryption_key)
    print("Files encrypted successfully.")

    choice = input("Do you want to decrypt the files? (y/n): ")
    if choice.upper() == "Y":
        decryption_key = input("Enter the decryption key: ")
        print("Decrypting files...")
        decrypt_file(target_directory, decryption_key)
        print("Files decrypted successfully.")


if __name__ == "__main__":
    main()
