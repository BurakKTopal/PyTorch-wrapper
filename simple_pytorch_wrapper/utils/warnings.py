WARNING_ON = True

def suppress_warnings():
    """
    Suppress warning messages globally.
    """
    global WARNING_ON  # Declare WARNING_ON as a global variable
    WARNING_ON = False  # Set the flag to False to disable warnings
    return

def enable_warnings():
    """
    Enables the display of warning messages.
    """
    global WARNING_ON
    WARNING_ON = True
    return 

def display_warning(keyword, message):
    """
    Display a warning message to the user.
    """
    if WARNING_ON:
        print(f"\033[91mwarning\033[0m @{keyword}: \033[93m{message}\033[0m \n")
    return