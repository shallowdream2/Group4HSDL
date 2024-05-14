#pragma once


#ifdef _WIN32
#define NOMINMAX
#include "windows.h"
#include "psapi.h"

#else
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

int parseLine(char* line) {
	// This assumes that a digit will be found and the line ends in " Kb".
	int i = strlen(line);
	const char* p = line;
	while (*p < '0' || *p > '9') p++;
	line[i - 3] = '\0';
	i = atoi(p);
	return i;
}

///* Virtual Memory currently used by current process https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process */
//int Current_Memory_Consumption_of_This_Process_linux() { //Note: this value is in KB!
//	FILE* file = fopen("/proc/self/status", "r");
//	int result = -1;
//	char line[128];
//
//	while (fgets(line, 128, file) != NULL) {
//		if (strncmp(line, "VmSize:", 7) == 0) {
//			result = parseLine(line);
//			break;
//		}
//	}
//	fclose(file);
//	return result;
//}

/* Physical Memory currently used by current process: */
int Current_Memory_Consumption_of_This_Process_linux() { //Note: this value is in KB!
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmRSS:", 6) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}
#endif


double Current_Memory_Consumption_of_This_Process() {

	/*return unit: MB*/

#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS_EX pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
	return (double) pmc.PrivateUsage / 1024 / 1024;
#else
	return (double) Current_Memory_Consumption_of_This_Process_linux() / 1024;
#endif


}


/*an example main file:

-------------------
#include <tool_functions/Current_Memory_Consumption_of_This_Process.h>
//using namespace std;" must be below #include <tool_functions/Current_Memory_Consumption_of_This_Process.h>! Or there is a conflict of the name byte!

#include<iostream>

int main()
{
	std::cout << Current_Memory_Consumption_of_This_Process() << "MB";
}
----------------------------------------
*/

