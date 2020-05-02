#include <iostream>
#include <array>


void stl_main()
{
  std::array<int,5> myarray = { 2, 16, 77, 34, 50 };
  std::cout << "size of myarray: " << myarray.size() << std::endl;
  std::cout << "sizeof(myarray): " << sizeof(myarray) << std::endl;
  std::cout << "myarray contains:";
  for ( auto it = myarray.begin(); it != myarray.end(); ++it )
    std::cout << ' ' << *it;
  std::cout << '\n';
  std::cout << "myarray contains:";
  for( auto& i : myarray )
  {
    std::cout << ' ' << i;
  }
  std::cout << std::endl;
  std::cout << "myarray contains:";
  for( unsigned i=0; i<myarray.size(); i++ )
  {
    std::cout << ' ' << myarray[i];
  }
  std::cout << std::endl;
  std::cout << "myarray contains:";
  for( unsigned i=0; i<myarray.size(); i++ )
  {
    std::cout << ' ' << myarray.at(i);
  }
  std::cout << std::endl;
  std::cout << "myarray " << (myarray.empty() ? "is empty" : "is not empty") << '\n';
  std::cout << "front is: " << myarray.front() << std::endl;
  std::cout << "back is: " << myarray.back() << std::endl;

}