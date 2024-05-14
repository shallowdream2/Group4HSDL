#pragma once


/* remember to use C++17 (not C17) for #include <filesystem>
https://stackoverflow.com/questions/50668814/vs2017-e0135-namespace-std-has-no-member-filesystem

https://stackoverflow.com/questions/53612757/where-to-define-silence-cxx17-allocator-void-deprecation-warning-or-silence-al
*/
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

std::vector<std::pair<std::string, bool>> list_all_files_in_a_directory(std::string path) {

    /*in pair<string, bool>, if bool is true, then this string is a directory name, otherwise it is a file name*/

    std::vector<std::pair<std::string, bool>> file_names;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        bool is_directory_type = false;
        if (entry.is_directory()) {
            is_directory_type = true;
        }
        std::string path_string{ entry.path().u8string() }; // https://stackoverflow.com/questions/45401822/how-to-convert-filesystem-path-to-string
        file_names.push_back({ path_string.substr(path.size() + 1, path_string.size() - path.size()) , is_directory_type });
    }

    return file_names;
}





/*
------------
#include <text_mining/list_all_files_in_a_directory.h>
int main()
{
    example_list_all_files_in_a_directory();
}
------------------
*/

void example_list_all_files_in_a_directory() {

    std::string path = "C:/Users";
    std::vector<std::pair<std::string, bool>> files = list_all_files_in_a_directory(path);

    for (int i = 0; i < files.size() && i < 3e2; i++) {
        if (files[i].second) {
            std::cout << "item: |" << files[i].first << "| (directory)" << std::endl;
        }
        else {
            std::cout << "item: |" << files[i].first << "| (file)" << std::endl;
        }

    }
}