#pragma once
#include <string>
#include <cctype>

bool compareChar_StringCompare_caseInSensitive(char & c1, char & c2)
{
	if (c1 == c2)
		return true;
	else if (std::toupper(c1) == std::toupper(c2))
		return true;
	return false;
}

/*
 * Case Insensitive String Comparision
 */
bool StringCompare_caseInSensitive(std::string & str1, std::string &str2)
{
	return ((str1.size() == str2.size()) &&
		std::equal(str1.begin(), str1.end(), str2.begin(), &compareChar_StringCompare_caseInSensitive));
}