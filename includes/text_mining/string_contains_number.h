#pragma once
#include<string>

bool string_contains_number(const std::string& c) {

	if (c.find('0') != std::string::npos ||
		c.find('1') != std::string::npos ||
		c.find('2') != std::string::npos ||
		c.find('3') != std::string::npos ||
		c.find('4') != std::string::npos ||
		c.find('5') != std::string::npos ||
		c.find('6') != std::string::npos ||
		c.find('7') != std::string::npos ||
		c.find('8') != std::string::npos ||
		c.find('9') != std::string::npos)
	{
		return true;
	}

	return false;

}