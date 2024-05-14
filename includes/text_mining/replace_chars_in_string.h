#pragma once
#include<string>

/*source: https://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string

this function replace all chars "from" in a string to "to" */

std::string replace_chars_in_string(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}