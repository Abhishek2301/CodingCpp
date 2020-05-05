#include <iostream>
#include <array>
#include <vector>
#include <deque>

void stl_main()
{

  /************************************std::array*****************************/
  std::array<int,5> array1 = { 2, 16, 77, 34, 50 };
  std::cout << "size of array1: " << array1.size() << std::endl;
  std::cout << "sizeof(array1): " << sizeof(array1) << std::endl;
  std::cout << "array1 contains:";
  for ( auto it = array1.begin(); it != array1.end(); ++it )
    std::cout << ' ' << *it;
  std::cout << '\n';
  std::cout << "array1 contains:";
  for( auto& i : array1 )
  {
    std::cout << ' ' << i;
  }
  std::cout << std::endl;
  std::cout << "array1 contains:";
  for( unsigned i=0; i<array1.size(); i++ )
  {
    std::cout << ' ' << array1[i];
  }
  std::cout << std::endl;
  std::cout << "array1 contains:";
  for( unsigned i=0; i<array1.size(); i++ )
  {
    std::cout << ' ' << array1.at(i);
  }
  std::cout << std::endl;
  std::cout << "array1 " << (array1.empty() ? "is empty" : "is not empty") << '\n';
  std::cout << "array1 front is: " << array1.front() << std::endl;
  std::cout << "array1 back is: " << array1.back() << std::endl;
  /***************************************************************************/

  /************************************std::vector****************************/
  std::vector<int> vector1 = { 1, 2, 3, 4, 5};
  std::vector<int> vector2;                                // empty vector of ints
  std::vector<int> vector3 (4,100);                       // four ints with value 100
  vector1[0] = 1; vector1[1] = 2;
  std::cout << "size of vector1: " << vector1.size() << std::endl;
  std::cout << "sizeof(vector1): " << sizeof(vector1) << std::endl;
  std::cout << "vector1 contains:";
  for ( auto it = vector1.begin(); it != vector1.end(); ++it )
    std::cout << ' ' << *it;
  std::cout << std::endl;
  std::cout << "vector3 contains:";
  for( auto& i : vector3 )
  {
    std::cout << ' ' << i;
  }
  std::cout << std::endl;
  std::cout << "vector1 contains:";
  for( unsigned i=0; i<vector1.size(); i++ )
  {
    std::cout << ' ' << vector1[i];
  }
  std::cout << std::endl;
  std::cout << "vector1 contains:";
  for( unsigned i=0; i<vector1.size(); i++ )
  {
    std::cout << ' ' << vector1.at(i);
  }
  std::cout << std::endl;
  std::cout << "vector2 " << (vector2.empty() ? "is empty" : "is not empty") << '\n';
  std::cout << "vector1 front is: " <<vector1.front() << std::endl;
  std::cout << "vector1 back is: " << vector1.back() << std::endl;
  vector2.push_back(10); vector2.push_back(20); vector2.push_back(30);
  while (!vector2.empty())
  {
    vector2.pop_back();
  }
  auto itr = vector2.begin();
  itr = vector2.insert( itr, 100 );
  itr = vector2.insert ( itr, 2, 200);
  std::cout << "vector2 contains:";
  for( auto& i : vector2 )
  {
    std::cout << ' ' << i;
  }
  std::cout << std::endl;
  // erase the 2nd element
  vector1.erase ( vector1.begin()+1 );
  // erase the first 3 elements:
  vector1.erase ( vector1.begin(), vector1.begin()+3 );
  /***************************************************************************/
  
  /************************************std::deque*****************************/
  std::deque<int> deque1;                                // empty deque of ints
  std::deque<int> deque2 (4,100);                       // four ints with value 100
  std::deque<int> deque3 (deque2.begin(),deque2.end());  // iterating through second
  std::deque<int> deque4 (deque3);                       // a copy of third
  std::deque<int> deque5 = { 1, 2, 3, 4, 5};
  std::cout << "size of deque5: " << deque5.size() << std::endl;
  std::cout << "sizeof(deque5): " << sizeof(deque5) << std::endl;
  std::cout << "deque5 contains:";
  for ( auto it = deque5.begin(); it != deque5.end(); ++it )
    std::cout << ' ' << *it;
  std::cout << std::endl;

}