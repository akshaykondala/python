import re

def is_strong_password(password):
    # At least 8 characters in length
    if len(password) < 8:
        return False

    # Contains at least one uppercase letter
    if not any(char.isupper() for char in password):
        return False

    # Contains at least one lowercase letter
    if not any(char.islower() for char in password):
        return False

    # Contains at least one digit
    if not any(char.isdigit() for char in password):
        return False

    # Contains at least one special character
    if not re.search(r'[!@#\$%^&*()_+{}\[\]:;<>,.?~\\/-]', password):
        return False

    return True

# Get a password from the user
password = input("Enter a password: ")

# Check if the password is strong
if is_strong_password(password):
    print("Strong password. Good job!")
else:
    print("Weak password. Please make it stronger.")
