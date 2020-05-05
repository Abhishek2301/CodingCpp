#include <iostream>
#include <vector>
#include <map>
using namespace std;

//void tree_apps_main();
//void array_apps_main();
void stl_main();

/*
Find a cutting index to separate the number of elements equal to X (# of green)
on the left half of the cutting line to be equal to the number of elements not-equal to 
X (# of red) on the right half of the cutting line
*/
int findCut( int X, vector<int>& A)
{
  unsigned red = 0;
  unsigned  green = 0;
  unsigned red_ptr = A.size();
  for( unsigned i=0; i<red_ptr; i++) {
    if(green == red) {
      if(A[i] == X) {
        green++;
      }
    }
    while( red != green || ( A[red_ptr-1]==X && red_ptr > i+1 ) ) {
      if(A[red_ptr-1] != X) {
        red++;
      }
      red_ptr--;
    }
  }
  return red_ptr;
}


int main() {
  int int_arr[] = {5, 5, 4, 4, 4, 4, 4};
  vector<int> A(int_arr, int_arr + ( sizeof(int_arr)/sizeof(int_arr[0]) ) );
  //cout << findCut( 5, A ) << endl;
  
  //tree_apps_main();
  //array_apps_main();
  stl_main();
}