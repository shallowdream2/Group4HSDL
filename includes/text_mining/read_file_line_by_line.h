#pragma once
#include <fstream>
#include <string>
#include <iostream>

void read_file_line_by_line(std::string file_name, int line_num_between_getchar) {

	std::string line_content;
	std::ifstream myfile(file_name); // open the file
	if (myfile.is_open()) // if the file is opened successfully
	{
		int count = 0;
		while (getline(myfile, line_content)) // read file line by line
		{
			count++;
			std::cout << line_content << std::endl;
			if (count%line_num_between_getchar == 0) {
				getchar();
			}
		}
		myfile.close(); //close the file

		std::cout << "Total Line Num: " << count << std::endl;
	}
	else
	{
		std::cout << "Unable to open file " << file_name << std::endl 
			<< "Please check the file location or file name." << std::endl; // throw an error message
		getchar(); // keep the console window
		exit(1); // end the program
	}

}