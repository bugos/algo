/*
did not pass the test cause it didnt consider baab a palindrome with center aa.
Written by bugos. 2012
*/
///////////////////////
/*
USER: pdp24u81
LANG: C++
TASK: minpali
*/
#include <stdio.h>

int main() 
{
	FILE *in  = fopen("minpali.in" , "r");
	FILE *out = fopen("minpali.out", "w");
	int N;
	fscanf(in, "%d\n", &N);
	
	char *p = new char[N];
	for (int i = 0; i < N; i++)
		fscanf(in, "%c", &p[i]);
	
	int center = N/2+1;
	for (;center<N; center++)
	{
		int flag=1;
		for(int j=1; j <= N-center; j++)
		{
			if (p[center-j-1] != p[center+j-1])
				{flag=0; break;}
		}
		if (flag) break;
	}	
	//printf("center:%d\n", center);
	//printf("answer:%d\n", 2*center - 1);
	fprintf(out,"%d\n", 2*center - 1);

	fclose(in);
	fclose(out);
	return 0;
}
