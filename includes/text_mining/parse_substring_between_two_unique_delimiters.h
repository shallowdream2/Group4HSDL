#pragma once


#include <text_mining/parse_string.h> 

std::string parse_substring_between_two_unique_delimiters
(std::string& parse_target, std::string delimiter1, std::string delimiter2) {



	std::vector<std::string> Parsed_content1 = parse_string(parse_target, delimiter2);
	std::vector<std::string> Parsed_content2 = parse_string(Parsed_content1[0], delimiter1);

	return Parsed_content2[Parsed_content2.size() - 1];

}