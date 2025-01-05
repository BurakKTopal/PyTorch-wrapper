def display_warning(keyword, message):
    """
    Display a warning message to the user.
    """
    print(f"\033[91mwarning\033[0m @{keyword}: \033[93m{message}\033[0m \n")
    return