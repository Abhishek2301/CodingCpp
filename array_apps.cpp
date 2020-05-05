#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <climits>
#include <algorithm>
#include <queue>
using namespace std;

int max(int a, int b)
{
  return a > b ? a : b;
}

int min(int a, int b)
{
  return a < b ? a : b;
}

float median2(int a, int b)
{
  return (a+b)/2.0;
}

float median3(int a, int b, int c)
{
  return a + b + c + - max(a, max(b, c)) - min(a, min(b, c));
}

float median4(int a, int b, int c, int d)
{
  int max_4 = max(a, max(b, max(c, d)));
  int min_4 = min(a, min(b, min(c, d)));
  return (a + b + c + d - max_4 - min_4)/2.0;
}

float findMedianUtil(int a[ ], int N, int b[ ], int M)
{
  if(N == 1)
    {
      if(M == 1)
	{
	  return median2(a[0], b[0]);
	}
      if(M & 1)
	{
	  return median2(b[M/2], median3(b[M/2-1], b[M/2+1], a[0]));
	}
      return median3(b[M/2-1], b[M/2], a[0]);
    }
  if(N == 2)
    {
      if(M == 2)
	{
	  return median4(a[0], a[1], b[0], b[1]);
	}
      if(M & 1)
	{
	  return median3(b[M/2], max(a[0], b[M/2-1]), min(a[1], b[M/2+1]));
	}
      return median4(b[M/2], b[M/2-1], max(a[0], b[M/2-2]), min(a[1], b[M/2+1])); 
    }
  int idxA = (N-1)/2;
  int idxB = (M-1)/2;
  if(a[idxA] <= b[idxB])
    {
      return findMedianUtil(a+idxA, N/2+1, b, M-idxA);
    }
  return findMedianUtil(a, N/2+1, b+idxA, M-idxA);
}

float findMedian(int a[ ], int N, int b[ ], int M)
{
  if(N > M)
    {
      return findMedianUtil(b, M, a, N);
    }
  return findMedianUtil(a, N, b, M);
}

// Time Complexity: O(m+n)
template<size_t m_size, size_t n_size>
float findMedianSortedArrays( std::array<int, m_size>& arr1, std::array<int, n_size>& arr2)
{
  unsigned i = 0, j = 0; // i curr idx for arr1, j curr idx for arr2
  unsigned count = 0; // loop through half of the elements in arr1+arr2
  int m1=-1, m2=-1;
  unsigned m = arr1.size(); unsigned n = arr2.size();
  if( (m+n) % 2 == 1 ) // m+n is odd then the median is m+n/2
  {
    for( count=0; count<=(m+n)/2; count++ )
    {
      if( i != m && j != n) // arr1 and arr2 have elements left
      {
        m1 = ( arr1[i] > arr2[j] ) ? arr2[j++] : arr1[i++];
      }
      else if( i < m ) // arr2 has no more elements left
      {
        m1 = arr1[i++];
      }
      else // arr1 has no more elements left
      {
        m1 = arr2[j++];
      }
    }
    return m1;
  }
  else // m+n is even then median is the avg of (m+n)/2-1 and (m+n)/2
  {
    for( count=0; count<=(m+n)/2; count++ )
    {
      m2 = m1;
      if( i != m && j != n) // arr1 and arr2 have elements left
      {
        m1 = ( arr1[i] > arr2[j] ) ? arr2[j++] : arr1[i++];
      }
      else if( i < n ) // arr2 has no more elements left
      {
        m1 = arr1[i++];
      }
      else // arr1 has no more elements left
      {
        m1 = arr2[j++];
      }
    }
    return (m1+m2)/2;
  }
}

// Ascending ordered sorted array; Time Complexity: O(log(min(m,n)))
template<size_t m_size, size_t n_size>
float findMedianSortedArrays1( std::array<int, m_size>& arr1, std::array<int, n_size>& arr2)
{
  unsigned m = arr1.size(); unsigned n = arr2.size();
  int min_index=0, max_index=m, i, j, median;
  while( min_index <= max_index)
  {
    // partition arr1 and arr2 such that left-halves of arr1+arr2 have the same number of elements
    // with right-halves of arr1+arr2 for a valid partition
    i = ( min_index + max_index ) / 2; // intialize partition arr1 with a default value of arr1 mid-point
    // i is the first element of right-half of arr1
    j = ( m + n + 1 ) / 2 - i; // initialize partition arr2 to make equal number left-half-right half valid partition
    // j is the first element of right-hal of arr2
    // find the correct partition i,j such that all elements of left-halves arr1+arr2 is less than
    // all elements of right-halves arr1+arr2
    if( j < 0 ) // partition with i elements from arr1[] not possible
    {
      max_index = i-1; // reduce the range to the left half
      continue;
    }
    if( i<m && j>0 && arr2[j-1]>arr1[i]) // i<m->thare are elements on right half of arr1
    // j>0->there are elements on left half of arr2, arr2[j-1]>arr1[i] -> left half element of arr2
    // is more than right half element of arr1 which is invalid partition
    {
      min_index = i+1; // right-shift arr2 and left-shift arr1 by i elements to binary search the partition i,j
    }
    else if( i>0 && j<n && arr2[j]<arr1[i-1]) // invalid parition since left-half arr1 element arr1[i-1] is greater than
    // right-half arr2 element arr2[j]
    {
      max_index = i-1; // left-shift arr1 partition and right-shift arr2 partition by i o binary search
    }
    else // we have found the desired value
    {
      if( i==0 ) //left-half arr1 is empty
      {
        median = arr2[j-1]; // return the last element of left-half arr2
      }
      else if( j==0 ) // left-half arr2 is empty
      {
        median = arr1[i-1]; // return the last element of left-half arr1
      }
      else
      {
        median = max( arr1[i-1], arr2[j-1]); // assumption of acending ordered arr1, arr2
      }
      break;
    }
  }
  if( ( n+m ) % 2 == 1 ) // arr1+arr2 is odd
    return (float)median; // there is only 1 median
  if( i==m ) // right-half arr1 is empty
  {
    return ( median+arr2[j] )/2.0; // return avg of 2 medians
  }
  else if( j==n ) // right-half arr2 is empty
  {
    return ( median+arr1[i] )/2.0; // return avg of 2 medians
  }
  else
  {
    return median+min(arr1[i],arr2[j]/2.0); // assumption of ascending ordered arr1 and arr2
  }
}

bool hasPairWithSum( const std::vector<int> & data, int sum )
{
  std::unordered_set<int> comp;
  for( int value : data )
  {
    if( comp.find( value ) != comp.end() )
      return true;
    comp.insert( sum-value );
  }
  return false;
}

int findPivot( int arr[], int low, int high )
{
  if( high<low )
    return -1;
  if( high==low )
    return low;
  int mid = ( low+high )/2;
  if( mid<high && arr[mid]>arr[mid+1] ) // 3, 4, 5, 1, 2
    return mid;
  if( mid>low && arr[mid]<arr[mid-1] ) // 4, 5, 1, 2, 3
    return mid-1;
  if( arr[low]>=arr[mid] ) // 5, 1, 2, 3, 4
    return findPivot( arr, low, mid-1 );
  else
    return findPivot( arr, mid+1, high ); // 2, 3, 4, 5, 1
}

int binarySearch( int arr[], int low, int high, int key )
{
  if( high<low )
    return -1;
  int mid=(low+high)/2;
  if( key==arr[mid] )
    return mid;
  if( arr[mid]<key )
    return binarySearch( arr, mid+1, high, key );
  else
    return binarySearch( arr, low, mid-1, key );
}

int pivotedBinarySearch( int arr[], int n, int key )
{
  int pivot = findPivot( arr, 0, n-1 );
  if( pivot == -1 )
    return binarySearch( arr, 0, n-1, key );
  if( arr[pivot] == key )
    return pivot;
  if( arr[0] <= key )
    return binarySearch( arr, 0, pivot-1, key );
  else
    return binarySearch( arr, pivot+1, n-1, key );
}

int pivotedBinarySearch1( int arr[], int low, int high, int key )
{
  if( low > high )
    return -1;
  int mid = ( low+high )/2;
  if( arr[mid]==key )
    return mid;
  if( arr[low]<=arr[mid] ) // arr[low,mid] is sorted
  {
    if( key>=arr[low] && key<=arr[mid] ) //recurse left half first
      return pivotedBinarySearch1( arr, low, mid-1, key );
    else // otherwise recurse right half
      return pivotedBinarySearch1( arr, mid+1, high, key );
  }
  else // if arr[low,mid] is not sorted then arr[mid,high] is sorted
  {
    if( key>=arr[mid] && key<=arr[high] ) //recurse right half first
      return pivotedBinarySearch1( arr, mid+1, high, key );
    else //otherwise recurse left half
      return pivotedBinarySearch1( arr, low, mid-1, key );
  }
}

bool hasZeroSumSubArray( int arr[], int n )
{
  unordered_set<int> sumSet;
  int sum = 0;
  for( int i=0; i<n; i++ )
  {
    sum += arr[i];
    if( sum == 0 || sumSet.find( sum ) != sumSet.end() )
      return true;
    sumSet.insert( sum );
  }
  return false;
}

bool isSubsetSum( int set[], int n, int sum )
{
  if( sum == 0 )
    return true;
  if( n == 0 && sum != 0 )
    return false;
  if( set[n-1] > sum )
    return isSubsetSum( set, n-1, sum );
  return isSubsetSum( set, n-1, sum ) || isSubsetSum( set, n-1, sum-set[n-1] ) ;
}

void swap( int * a, int * b )
{
  int temp =*a;
  *a = *b;
  *b = temp;
}

int partition( int arr[], int l, int r )
{
  int pivot = arr[r], i = l;
  for( unsigned j=l; j<=r-1; j++ )
  {
    if( arr[j] <= pivot )
    {
      swap( &arr[i], &arr[j] );
      i++;
    }
  }
  swap( &arr[i], &arr[r] );
  return i;
}

/* The idea is, not to do complete quicksort, but stop at the point where pivot itself is k’th smallest element. Also, not to recur for both left and right sides of pivot, but recur for one of them according to the position of pivot. The worst case time complexity of this method is O(n^2), but it works in O(n) on average. */
int findKthSmallest( int arr[], int l, int r, int k )
{
  if( k>0 && k<=r-l+1 )
  {
    /* Picks a random pivot element between l and r and partitions arr[l..r] arount the randomly picked element. The worst case time complexity of the above solution is still O(n^2). In worst case, the randomized function may always pick a corner element. The expected time complexity of above randomized QuickSelect is O(n) */
    // int pivot = rand() % (r-l+1);
    // swap( &arr[l+pivot], &arr[r] );
    int pos = partition( arr, l, r );
    if( pos-l == k-1 )
      return arr[pos];
    if( pos-l > k-1 )
      return findKthSmallest( arr, l, pos-1, k );
    return findKthSmallest( arr, pos+1, r, k-pos+l-1 );
  }
  return INT_MAX;
}

/* Time complexity of this solution is O(k + (n-k)*Logk) */
int findKthSmallest( int arr[], int n, int k )
{
  priority_queue<int> maxHeap;
  for( int i=0; i<k; i++ )
    maxHeap.push( arr[i] );
  for( int i=k; i<n; i++ )
  {
    if( arr[i] < maxHeap.top() )
    {
      maxHeap.pop();
      maxHeap.push( arr[i] );
    }
  }
  return maxHeap.top();
}

/* Given an array and a number k where k is smaller than size of array, we need to find the k’th smallest element in the given array. It is given that all array elements are distinct.*/
/* The idea in this new method is similar to quickSelect(), we get worst case linear time by selecting a pivot that divides array in a balanced way (there are not very few elements on one side and many on other side). 
 * After the array is divided in a balanced way, we apply the same steps as used in quickSelect() to decide whether to go left or right of pivot. */
int findMedian( int arr[], int n )
{
  sort( arr, arr+n );
  return arr[n/2];
}

int findKthSmallest1( int arr[], int l, int r, int k )
{
  if( k>0 && k<=r-l+1 )
  {
    int n = r-l+1; // No of elements in the array
    int i, median[(n+4)/5]; // There will be floor((n=4)/5) groups
    for( i=0; i<n/5; i++ )
      median[i] = findMedian( arr+l+i*5, 5 );
    if( i*5 < n ) // for the last group
    {
      median[i] = findMedian( arr+l+i*5, n%5 );
      i++;
    }
    // find median of medians recursively. if median has onyl 1 element then no need
    int medOfMeds = ( i==1 ) ? median[i-1] : findKthSmallest1( median, 0, i-1, i/2 );
    unsigned j;
    for( j=l; j<r; j++ )
    {
      if( arr[j] == medOfMeds )
        break;
    }
    swap( &arr[j], &arr[r] ); // medofMeds to act as pivot by swapping to the last pos r as required by partition(..)
    int pos = partition( arr, l, r );
    if( pos-l == k-1 )
      return arr[pos]; // if pivot pos is k then return
    if( pos-l > k-1 )
      return findKthSmallest1( arr, l, pos-1, k ); // recurse on the left subarray since pivot is larger than k
    return findKthSmallest1( arr, pos+1, r, k-pos+l-1 ); // recurse on the right subarray since pivot is smaller than k
  }
  return INT_MAX;
}

/* Given an unsorted array of nonnegative integers, find a continous subarray which adds to a given number.*/
bool subArraySum( unsigned arr[], unsigned n, unsigned sum )
{
  unsigned curr_sum = arr[0], start = 0, i;
  for( i=1; i<n; i++ )
  {
    while( curr_sum > sum && start < i-1 )
    {
      curr_sum -= arr[start];
      start++;
    }
    if( curr_sum == sum )
    {
      cout << "Sum found between indexes " << start << " and " << i-1 << endl;
      return true;
    }
    if( i<n )
      curr_sum += arr[i];
  }
  return false;
}

/* Given an unsorted array of integers, find a subarray which adds to a given number. If there are more than one subarrays with sum as the given number, print any of them.*/
bool subArraySum( int arr[], unsigned n, int sum )
{
  unordered_map<int, unsigned> map;
  int curr_sum = 0;
  for( unsigned i=0; i<n; i++ )
  {
    curr_sum += arr[i];
    if( curr_sum == sum ) // subarray starting at index 0 with sum
    {
      cout << "Sum found between indexes 0 to " << i << endl;
      return true;
    }
    if( map.find( curr_sum-sum ) != map.end() ) // subarray with target sum at nonzero starting index
    {
      cout << "Sum found between indexes " << map[ curr_sum-sum ]+1 << " to " << i << endl;
      return true;
    }
    map[curr_sum] = i;
  }
  return false;
}

/* Given an array containing only 0s and 1s, find the largest subarray which contain equal no of 0s and 1s. Expected time complexity is O(n). */
unsigned getMaxSumSubArray( int arr[], int n )
{
  for( unsigned i=0; i<n; i++ )
    arr[i] = (arr[i]==0) ? -1 : 1;
  int sum=0; unsigned max_len=0; int ending_idx=-1;
  unordered_map<int,unsigned> hM;
  for( unsigned i=0; i<n; i++ )
  {
    sum += arr[i];
    if( sum == 0 )
    {
      max_len = i+1;
      ending_idx = i;
    }
    if( hM.find( sum ) != hM.end() )
    {
      if( max_len < i-hM[sum] )
      {
        max_len = i-hM[sum];
	ending_idx = i;
      }
    }
    else
      hM[sum] = i;
  }
  for( unsigned i=0; i<n; i++ )
    arr[i] = (arr[i]==-1) ? 0 : 1;
  cout << ending_idx-max_len+1 << " to " << ending_idx<< endl;
  return max_len;
}

/* Given an array of integers, find length of the largest subarray with sum equals to 0 */
unsigned getMaxLengthZeroSumSubArray( int arr[], int n )
{
  unordered_map<int,unsigned> presum;
  int sum=0; unsigned max_len=0;
  for( unsigned i=0; i<n; i++ )
  {
    sum += arr[i];
    if( arr[i]==0 && max_len==0 )
      max_len=1;
    if( sum==0 ) // matching zero sum sub-array starting from idx 0
      max_len = i+1;
    if( presum.find(sum) != presum.end() )
      max_len = max( max_len, i-presum[sum] );
    else
      presum[sum] = i;
  }
  return max_len;
}

/* Given an array of positive and negative numbers, arrange them such that all negative integers appear before all the positive integers in the array without using any additional data structure like hash table, arrays, etc. The order of appearance should be maintained. Time: O(NlogN) Space:O(logN )*/
/****************************************************************************************************************************************************************************************************************/
void reverse( int arr[], int l, int r )
{
  if( l < r )
  {
    swap( &arr[l], &arr[r] );
    reverse( arr, ++l, --r );
  }
}
void merge( int arr[], int l, int m, int r ) // first sub-array is [l..m] and the second is [m+1..r]
{
  int i=l;
  int j=m+1;
  while( i<=m && arr[i]<0 )
    i++; // arr[i..m] is +ve
  while( j<=r && arr[j]<0 )
    j++; // arr[j..r] is +ve
  reverse( arr, i, m ); // reverse +ve part of left subarray [Lp] -> [Lp']
  reverse( arr, m+1, j-1 ); // reverse -ve part of right subarray [Rn] -> [Rn']
  // [Ln Lp Rn Rp] -> [Ln Lp’ Rn’ Rp]
  reverse( arr, i, j-1 ); // [Lp’ Rn’] -> [Rn Lp]. Thus: [Ln Lp’ Rn’ Rp] -> [Ln Rn Lp Rp]
}
void printArray( int arr[], unsigned size )
{
  for( unsigned i=0; i<size; i++ )
    cout << arr[i] << " ";
  cout << endl;
}
void rearrangePosNegPreservingOrder( int arr[], int l, int r )
{
  if( l < r )
  {
    int m = l + (r - l) / 2;
    rearrangePosNegPreservingOrder( arr, l, m );
    rearrangePosNegPreservingOrder( arr, m+1, r );
    merge( arr, l, m, r );
  }  
}
/****************************************************************************************************************************************************************************************************************/

/* Rearrange positive and negative numbers in O(n) time and O(1) extra space. Rearrange the array elements so that positive and negative numbers are placed alternatively. */
void rearrangePosNeg( int arr[], int n )
{
  // The following few lines are similar to partition process
  // of QuickSort.  The idea is to consider 0 as pivot and
  // divide the array around it.
  int i=-1;
  for( int j=0; j<n; j++ )
  {
    if( arr[j]<0 )
    {
      i++;
      swap( &arr[i], &arr[j] );
    }
  }
  // Now all positive numbers are at end and negative numbers
  // at the beginning of array. Initialize indexes for starting
  // point of positive and negative numbers to be swapped
  int pos=i+1, neg=0;
  // Increment the negative index by 2 and positive index by 1,
  // i.e., swap every alternate negative number with next 
  // positive number
  while( pos<n && neg<pos && arr[neg]<0 )
  {
    swap( &arr[neg], &arr[pos] );
    pos++;
    neg+=2;
  }
}

/* Given two arrays: arr1[0..m-1] and arr2[0..n-1]. Find whether arr2[] is a subset of arr1[] or not. Both the arrays are not in sorted order. It may be assumed that elements in both array are distinct. */
bool isSubset( int arr1[], int arr2[], int m, int n )
{
  int i=0, j=0;
  if( m < n )
    return false;
  sort( arr1, arr1+m );
  sort( arr2, arr2+n );
  while( i<n && j<m )
  {
    if( arr1[j] < arr2[i] )
      j++;
    else if( arr1[j] == arr2[i] )
    {
      i++; j++;
    }
    else if( arr1[j] > arr2[i] )
      return false;
  }
  return ( i < n ) ? false : true;
}

/* Given k sorted lists of integers of size n each, find the smallest range that includes at least element from each of the k lists. If more than one smallest ranges are found, print any one of them. Time: O(n*K*k) total no array elements is n*k and k for finding the min/max of the range-strip*/
void findSmallestRange( int arr[][5], int n, int k )
{
  int ptr[k];
  for( int i=0; i<k; i++ )
    ptr[i] = 0; // initialize every element of ptr[0..k] to 0
  int min_range=INT_MAX, min_idx=0, curr_min=0, curr_max=0, total_min=0, total_max=0;
  bool flag=false;
  while( 1 )
  {
    min_idx = -1, curr_min=INT_MAX, curr_max=INT_MIN, flag=false;
    for( int i=0; i<k; i++ )
    {
      if( ptr[i] == n ) // Repeat the following steps until atleast one list exhausts
      {
        flag=true;
        break; // list[i] elements are done, so break 
      }
      if( ptr[i]<n && arr[i][ptr[i]]<curr_min ) // find min val of the strip ptr[]
      {
        min_idx = i; // store the pointer to the current min element of the strip for incrementing in the next iteration
        curr_min = arr[i][ptr[i]];
      }
      if( ptr[i]<n && arr[i][ptr[i]]>curr_max ) // find max val of the strip ptr[]
        curr_max = arr[i][ptr[i]];
    }
    if( flag )
      break;
    ptr[min_idx]++; // move the ptr of the strip with the min element for the next iteration
    if( (curr_max-curr_min) < min_range ) // update the min strip ranges
    {
      total_min = curr_min;
      total_max = curr_max;
      min_range = curr_max-curr_min;
    }
  }
  cout << "The smallest range is: [" << total_min << " , " << total_max << "]" << endl;
}

typedef struct heap
{
  int element;
  int list_idx;
  int next_idx;
} minHeapNode;

class MinHeapNodeComparator
{
  public:
    int operator() ( const minHeapNode & h1, const minHeapNode & h2 )
    {
      return h1.element > h2.element;
    }
};

void findSmallestRange1( int arr[][5], int n, int k )
{
  std::priority_queue< minHeapNode, vector<minHeapNode>, MinHeapNodeComparator > minHeap;
  int min_range=INT_MAX, curr_min=0, curr_max=0, total_min=0, total_max=0;
  for( int i=0; i<k; i++ )
  {
    minHeapNode m;
    m.element = arr[i][0];
    m.list_idx = i;
    m.next_idx = 1;
    if( m.element > curr_max )
      curr_max = m.element;
    minHeap.push( m );
  }
  while(1)
  {
    minHeapNode root = minHeap.top();
    curr_min = root.element;
    if( (curr_max-curr_min) < min_range ) // update the min strip ranges
    {
      total_min = curr_min;
      total_max = curr_max;
      min_range = curr_max-curr_min;
    }
    if( root.next_idx < n )
    {
      minHeapNode m;
      m.element = arr[root.list_idx][root.next_idx];
      m.list_idx = root.list_idx;
      m.next_idx = root.next_idx + 1;
      if( root.element > curr_max )
        curr_max = root.element;
      minHeap.pop();
      minHeap.push( m );
    }
    else
      break;
  }
  cout << "The smallest range is: [" << total_min << " , " << total_max << "]" << endl;
}

/* Write an efficient C program to find the sum of contiguous subarray within a one-dimensional array of numbers which has the largest sum. Kadane's algorithm O(n). */
int maxSubArraySum( int a[], int size )
{
  int max_so_far=0, max_ending_here=0;
  for( int i=0; i<size; i++ )
  {
    max_ending_here += a[i];
    if( max_ending_here < 0 )
      max_ending_here = 0;
    else if( max_so_far < max_ending_here )
      max_so_far = max_ending_here;
  }
  return max_so_far;
}

/* In a daily share trading, a buyer buys shares in the morning and sells it on same day. If the trader is allowed to make at most 2 transactions in a day, where as second transaction can only start after first one is complete (Sell->buy->sell->buy). Given stock prices throughout day, find out maximum profit that a share trader could have made. (1) Create a table profit[0..n-1] and initialize all values in it 0. (2) Traverse price[] from right to left and update profit[i] such that profit[i] stores maximum profit achievable from one transaction in subarray price[i..n-1] (3) Traverse price[] from left to right and update profit[i] such that profit[i] stores maximum profit such that profit[i] contains maximum achievable profit from two transactions in subarray price[0..i]. (4) Return profit[n-1]. To do step 1, we need to keep track of maximum price from right to left side and to do step 2, we need to keep track of minimum price from left to right. Why we traverse in reverse directions? The idea is to save space, in second step, we use same array for both purposes, maximum with 1 transaction and maximum with 2 transactions. After an iteration i, the array profit[0..i] contains maximum profit with 2 transactions and profit[i+1..n-1] contains profit with two transactions. */
int maxProfit( int price[], int n )
{
  int profit[n];
  for( int i=0; i<n; i++ )
    profit[i]=0;
  // Get the maximum profit with only one transaction allowed. After this loop, profit[i] contains maximum profit from price[i..n-1] using at most one trans.
  int max_price = price[n-1];
  for( int i=n-2; i>=0; i-- )
  {
    if( price[i]>max_price )
      max_price = price[i];
    // we can get profit[i] by taking maximum of: (a) previous maximum, i.e., profit[i+1] (b) profit by buying at price[i] and selling at max_price
    profit[i] = max( profit[i+1], max_price-price[i] );
  }
  // Get the maximum profit with two transactions allowed. After this loop, profit[n-1] contains the result
  int min_price = price[0];
  for( int i=1; i<n; i++ )
  {
    if( price[i]<min_price )
      min_price = price[i];
    // Maximum profit is maximum of: (a) previous maximum, i.e., profit[i-1] (b) (Buy, Sell) at (min_price, price[i]) and add profit of other trans. stored in profit[i+1] to avoid conflict of buying/selling in the same day.
    profit[i] = max( profit[i-1], profit[i+1]+(price[i]-min_price) );
  }
  return profit[n-1];
}

void findZeroSumTriplets( int arr[], int n )
{
  bool found = false;
  for( int i=0; i<n-1; i++ )
  {
    unordered_set<int> s;
    for( int j=i+1; j<n; j++ )
    {
      int x = -( arr[i]+arr[j] );
      if( s.find(x) != s.end() )
      {
        cout << x << " " << arr[i] << " " << arr[j] << endl;
        found = true;
      }
      else
        s.insert( arr[j] );
    }
  }
  if( !found )
    cout << "No Triplet with zero sum found!" << endl;
}

void findZeroSumTriplets1( int arr[], int n )
{
  bool found = false;
  sort( arr, arr+n );
  for( int i=0; i<n-1; i++ )
  {
    int l = i+1;
    int r = n-1;
    int x = arr[i];
    while( l<r )
    {
      if( x+arr[l]+arr[r] == 0 )
      {
        cout << x << " " << arr[l] << " " << arr[r] << endl;
        l++;
        r--;
        found = true;
      }
      else if( x+arr[l]+arr[r] < 0 )
        l++;
      else
        r--;
    }
  }
  if( !found )
    cout << "No Triplet with zero sum found!" << endl;
}

void findSumQuadruples( int arr[], int n, int k )
{
  unordered_map< int, pair<int,int> > pairMap;
  for( int i=0; i<n-1; i++ )
    for( int j=i+1; j<n; j++ )
      pairMap[arr[i]+arr[j]] = {i,j};
  for( int i=0; i<n-1; i++ )
  {
    for( int j=i+1; j<n; j++ )
    {
      int sum = arr[i] + arr[j];
      if( pairMap.find(k-sum) != pairMap.end() )
      {
        pair<int,int> p = pairMap[k-sum];
        if( p.first!=i && p.first!=j && p.second!=i && p.second!= j ) // ensure each element of the 2 pairs are distinct
        {
          cout << arr[i] << ", " << arr[j] << ", " << arr[p.first] << ", " << arr[p.second] << endl;
          return;
        }
      }
    }
  }
}

void findMedianStreamingUtil( double x, double & median )
{
  priority_queue<int> max_heap_left;
  priority_queue<int,vector<int>,greater<int>> min_heap_right;
  if( max_heap_left.size() > min_heap_right.size() )
  {
    if( x < median )
    {
      min_heap_right.push( max_heap_left.top() );
      max_heap_left.pop();
      max_heap_left.push( x );
    }
    else
      min_heap_right.push( x );
    median = ((double)max_heap_left.top()+(double)min_heap_right.top())/2.0;
  }
  else if( max_heap_left.size() == min_heap_right.size() )
  {
    if( x < median )
    {
      max_heap_left.push( x );
      median = max_heap_left.top();
    }
    else
    {
      min_heap_right.push( x );
      median = min_heap_right.top();
    }
  }
  else
  {
    if( x < median )
    {
      max_heap_left.push( min_heap_right.top() );
      min_heap_right.pop();
      min_heap_right.push( x );
    }
    else
      max_heap_left.push( x );
    median = ((double)max_heap_left.top()+(double)min_heap_right.top())/2.0;
  }
}

double findMedianStreaming( int arr[], int n )
{
  double median = 0;
  for( int i=0; i<n; i++ )
  {
    findMedianStreamingUtil( arr[i], median );
    cout << median << endl;
  }
}

void array_apps_main()
{
  //int a[ ] = {900};
  //int b[ ] = {5, 8, 10, 20};
  //int N = sizeof(a)/sizeof(a[0]);
  //int M = sizeof(b)/sizeof(b[0]);
  //printf("findMedian: %f\n", findMedian(a, N, b, M));
  std::array<int,5> arr1={3,5,10,11,17}; std::array<int,6> arr2={9,13,20,21,23,27};
  cout << "findMedanSortedArrays: " << findMedianSortedArrays1(arr1, arr2) << endl;
  //int inArrHasPairWithSum[] = { 2, 4, 2, 7, 3 };
  //std::vector<int> inVecHasPairWithSum( inArrHasPairWithSum, inArrHasPairWithSum+sizeof(inArrHasPairWithSum)/sizeof(int) );
  //cout << "hasPairWithSum: " << hasPairWithSum( inVecHasPairWithSum, 8 ) << endl;
  //int b1[] = {5, 6, 7, 8, 9, 10, 1, 2, 3};
  //cout << "pivotedBinarySearch: " << pivotedBinarySearch( b1, sizeof(b1)/sizeof(b1[0]), 3 ) << endl;
  //cout << "pivotedBinarySearch1: " << pivotedBinarySearch1( b1, 0, (sizeof(b1)/sizeof(b1[0]))-1, 3 ) << endl;
  //int inHasZeroSumSubArray[]={1,4,-2,-7,5,-4,3}; cout << "hasZeroSumSubArray: " << hasZeroSumSubArray( inHasZeroSumSubArray, sizeof(inHasZeroSumSubArray)/sizeof(int) ) << endl;
  //int inIsSubsetSum[]={3,34,4,12,5,2}; cout << "isSubsetSum: " << isSubsetSum( inIsSubsetSum, sizeof(inIsSubsetSum)/sizeof(int), 9 ) << endl;
  //int inFindKthSmallest[]={12,3,5,7,4,19,26}; cout << "findKthSmallest: " << findKthSmallest( inFindKthSmallest, 0, ( sizeof(inFindKthSmallest)/sizeof(int) ) - 1, 3 ) << endl;
  //int inFindKthSmallest[]={12,3,5,7,4,19,26}; cout << "findKthSmallest: " << findKthSmallest( inFindKthSmallest, sizeof(inFindKthSmallest)/sizeof(inFindKthSmallest[0]), 3 ) << endl;
  //unsigned inSubArraySum[]={15,2,4,8,9,5,10,23}; cout << "subArraySum: " << subArraySum( inSubArraySum, sizeof(inSubArraySum)/sizeof(unsigned), 24 ) << endl;
  //int inSubArraySum[]={10,2,-2,-20,10}; cout << "subArraySum: " << subArraySum( inSubArraySum, sizeof(inSubArraySum)/sizeof(int), -10 ) << endl;
  //int inGetMaxSumSubArray[]={1,0,1,1,1,0,0}; cout << "getMaxSumSubArray: " << getMaxSumSubArray( inGetMaxSumSubArray, sizeof(inGetMaxSumSubArray)/sizeof(inGetMaxSumSubArray[0]) ) << endl;
  //int arr[]={15,-2,2,-8,1,7,10,23}; cout << "getMaxLengthZeroSumSubArray: " << getMaxLengthZeroSumSubArray( arr, sizeof(arr)/sizeof(arr[0]) ) << endl;
  //int arr[]={-12,11,-13,-5,6,-7,5,-3,-6};unsigned size=sizeof(arr)/sizeof(arr[0]);rearrangePosNegPreservingOrder(arr,0,size-1);printArray(arr,size);
  //int arr1[]={11,1,13,21,3,7};int arr2[]={11,3,7,1,9};cout<<"isSubset: "<<isSubset(arr1,arr2,sizeof(arr1)/sizeof(arr1[0]),sizeof(arr2)/sizeof(arr2[0]))<<endl;
  //int arr[][5]={{4,7,9,12,15},{0,8,10,14,20},{6,12,16,30,50}}; findSmallestRange1(arr,5,sizeof(arr)/sizeof(arr[0]));
  //int price[]={2,30,15,10,8,25,80};cout<<"Maximum Profit: "<<maxProfit(price,sizeof(price)/sizeof(price[0]))<<endl;
  //int arr[]={0,-1,2,-3,1};findZeroSumTriplets1(arr, sizeof(arr)/sizeof(arr[0]));
  //int arr[]={10,20,30,40,1,2};cout<<"findSumQuadruples: ";findSumQuadruples(arr,sizeof(arr)/sizeof(arr[0]),91);
  //int arr[]={5,15,10,20,3};findMedianStreaming(arr,sizeof(arr)/sizeof(arr[0]));
  
}