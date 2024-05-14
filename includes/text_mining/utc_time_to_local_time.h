#pragma once
#include <string>
#include <ctime>

#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
/*https://stackoverflow.com/questions/19321804/this-function-or-variable-may-be-unsafe-visual-studio/36521657*/

std::string utc_time_to_local_time(int utc_time) {

	/*https://www.linuxquestions.org/questions/programming-9/convert-a-given-date-to-epoch-time-and-vice-versa-854390/*/

	static char timestamp[64] = "";
	time_t tt = 0;
	memset(timestamp, '\0', 64);
	tt = utc_time;
	strftime(timestamp, 64, "%Y-%m-%d:%H:%M:%S", localtime(&tt));

	return timestamp;

}
