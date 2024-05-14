#pragma once
#include<limits.h>


template<class output_iterator>
void convert_number_to_array_of_binary(const unsigned number,
	output_iterator first, output_iterator last)
{
	const unsigned number_bits = CHAR_BIT * sizeof(int);
	//extract bits one at a time
	for (unsigned i = 0; i < number_bits && first != last; ++i) {
		const unsigned shift_amount = number_bits - i - 1;
		const unsigned this_bit = (number >> i) & 1;
		*first = this_bit;
		++first;
	}
	//pad the rest with zeros
	while (first != last) {
		*first = 0;
		++first;
	}
}


/*Example:
-------------------------
#include <text_mining/convert_number_to_array_of_binary.h>
#include<vector>
#include<iostream>
int main()
{
	std::vector<int> count_binary(3);
	convert_number_to_array_of_binary(3, std::begin(count_binary), std::end(count_binary));
	std::cout << count_binary[2] << count_binary[1] << count_binary[0] << std::endl;
}
-----------------------------

*/