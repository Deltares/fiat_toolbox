import re

def _compile_pattern(pattern):
    """
    Compile a pattern with placeholders into a regex pattern.
    Args:
        pattern (str): The pattern containing placeholders in the format '{var}'.
    Returns:
        tuple: A tuple containing the compiled regex pattern and a list of placeholders.
    """
    # Escape special characters in pattern except for '{var}'
    escaped_pattern = re.escape(pattern)
    # Find all placeholders in the pattern
    placeholders = re.findall(r'\\{(.*?)\\}', escaped_pattern)
    # Replace placeholders with regex groups
    for placeholder in placeholders:
        escaped_pattern = escaped_pattern.replace(f'\\{{{placeholder}\\}}', f'(?P<{placeholder}>.*?)')
    # Compile the regex pattern
    regex = re.compile(f"^{escaped_pattern}$")
    return regex, placeholders

def matches_pattern(string, pattern):
    """
    Check if a string matches a pattern with placeholders.
    Args:
        string (str): The input string to be checked.
        pattern (str): The pattern containing placeholders in the format '{var}'.
    Returns:
        bool: True if the string matches the pattern, False otherwise.
    """
    regex, _ = _compile_pattern(pattern)
    return bool(regex.match(string))

def replace_pattern(string, pattern, replacement):
    """
    Replace placeholders in a string based on a pattern with a replacement string.
    Args:
        string (str): The input string to be processed.
        pattern (str): The pattern containing placeholders in the format '{var}'.
        replacement (str): The replacement string where placeholders will be replaced with corresponding values from the input string.
    Returns:
        str: The processed string with placeholders replaced by corresponding values from the input string. 
             If the pattern does not match the input string, the original string is returned.
    """
    regex, placeholders = _compile_pattern(pattern)
    
    # Find the match
    match = regex.match(string)
    if match:
        # Replace placeholders in the replacement with the captured groups
        for placeholder in placeholders:
            replacement = replacement.replace(f'{{{placeholder}}}', match.group(placeholder))
        return replacement
    return string