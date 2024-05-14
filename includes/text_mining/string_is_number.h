#pragma once
#include<string>

bool string_is_number(const std::string& s) {

	/*this works for double or int*/

	char* end = 0;
	double val = strtod(s.c_str(), &end);
	return end != s.c_str() && val != HUGE_VAL;
}