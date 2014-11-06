/*
USER: pdp24u81 -- bugos
LANG: C++
TASK: minpali
*/
#include <stdio.h>

int N;

int input() {
	scanf("%d\n", &N);

	char *p = new char[N]; //todo-change
	for (int i = 0; i < N; i++)
		fscanf(in, "%c", &p[i]);
}


//brute force to find the first valid center
int solve() {
	int center = (N/2)+1;
	bool center_valid;
	for (;center<N; center++)
	{
		if (p[center] == p[center + 1]) {
			//even palindromes
			center_valid=true;
			for(int j=1; j<N-center+1-1; j++)
			{
				if (p[center-j-1] != p[center+j-1+1]) {
					center_valid=false; 
					break;
				}
			}
			if (center_valid) 
				return 2*(center+1);
		} 
		else 
			//odd palindromes
			center_valid=true;
			for(int j=1; j<N-center+1; j++)
			{
				if (p[center-j-1] != p[center+j-1]) {
					center_valid=false; 
					break;
				}
			}
		}
		if (center_valid)
			return 2*(center+1) - 1;
	}
}


int main() 
{
	freopen("minpali.in" , "r", stdin );
	freopen("minpali.out", "w", stdout);

	input();
	
	printf("%d\n", solve());

	return 0;
}
