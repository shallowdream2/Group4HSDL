#pragma once
#include <fstream>
#include <string>
#include <iostream>


void read_file_total_line_number(std::string file_name) {

	std::string line_content;
	std::ifstream myfile(file_name); // open the file
	if (myfile.is_open()) // if the file is opened successfully
	{
		long long int count = 0;
		while (getline(myfile, line_content)) // read file line by line
		{
			count++;
		}
		myfile.close(); //close the file

		std::cout << "Total Line Num: " << count << std::endl;
	}
	else
	{
		std::cout << "Unable to open file " << file_name << std::endl << "Please check the file location or file name." << std::endl; // throw an error message
		getchar(); // keep the console window
		exit(1); // end the program
	}

}