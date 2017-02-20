

#include <iostream>
#include <stdlib.h>

#include "opencl_adapter.hpp"

namespace clipp {

	// void OpenCLAdapter::ASSERT(const bool cond, const std::string &msg)
	// {
	// 	if(!cond) {
	// 		std::cout << msg << std::endl;
	// 		abort();
	// 	}
	// }

	OpenCLAdapter::OpenCLAdapter()
	{
		global_size[0] = 1;
		global_size[1] = 1;
		global_size[2] = 1;
	}

	void OpenCLAdapter::set_global_size(const int i, const int size)
	{
		global_size[i] = size;
	}

	int OpenCLAdapter::get_global_id(const int i) const
	{
		return 0;
	}

	int OpenCLAdapter::get_global_size(const int i) const
	{
		return global_size[i];
	}

	int OpenCLAdapter::get_group_id(const int i) const
	{
		return 0;
	}

	int OpenCLAdapter::get_local_size(const int i) const
	{
		return 1;
	}

	int OpenCLAdapter::get_local_id(const int i) const
	{
		return 0;
	}
}
