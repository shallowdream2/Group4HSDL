#pragma once
#include<vector>
#include<string>
#include<fstream>
#include<iostream>
#include<list>

std::vector<std::vector<std::string>> read_csv(std::string file_name) {

	/*assume that there is no , in the content*/

	std::vector<std::vector<std::string>> input;
	std::vector<std::string> null_string_vector;

	int line_num = 0;

	std::string line_content;
	std::ifstream myfile(file_name); // open the file
	if (myfile.is_open()) // if the file is opened successfully
	{
		while (getline(myfile, line_content)) // read file line by line
		{
			input.insert(input.end(), null_string_vector); // add a std::vector<string>

														   // parse the sting£ºline_content
			std::list<std::string> Parsed_content;
			std::string delimiter = ","; // the delimiter
			size_t pos = 0;
			std::string token;
			while ((pos = line_content.find(delimiter)) != std::string::npos) {
				// find(const string& str, size_t pos = 0) function returns the position of the first occurrence of str in the string, or npos if the string is not found.
				token = line_content.substr(0, pos);
				// The substr(size_t pos = 0, size_t n = npos) function returns a substring of the object, starting at position pos and of length npos
				Parsed_content.push_back(token); // store the subtr to the list
				line_content.erase(0, pos + delimiter.length()); // remove the front substr and the first delimiter
			}
			Parsed_content.push_back(line_content); // store the subtr to the list

			//cout << "Parsed_content.size():" << Parsed_content.size() << endl;
			while (Parsed_content.size() > 0) {

				if (Parsed_content.front().compare("")) { // it is not empty 
					input[line_num].insert(input[line_num].end(), Parsed_content.front().c_str());
				}

				Parsed_content.pop_front();
			}

			line_num++;

		}

		myfile.close(); //close the file
		return input;
	}
	else
	{
		std::cout << "Unable to open file " << file_name << std::endl
			<< "Please check the file location or file name." << std::endl; // throw an error message
		getchar(); // keep the console window
		exit(1); // end the program
	}

}