#include <iostream>
#include <algorithm>

int read_input() {
  int N;
  std::cout << "Please give me a number [1..10]: ";
  std::cin >> N;

  return std::min(10,std::max(1,N));
}

int main() {

  int N = read_input();

  int sum=0, prod=1;
#pragma omp parallel
{
#pragma omp sections
 {
#pragma omp section
  for (int i=1; i<=N; i++) sum += i;
#pragma omp section
  for (int i=1; i<=N; i++) prod *= i;
#pragma omp section
  std::cout << "sum=" << sum << "  prod=" << prod << std::endl;
 }
} 
  return 0;
}
