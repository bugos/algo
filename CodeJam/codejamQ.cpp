#include <stdio.h>
#include <math.h>
#define problem_n problemB

int problemA()
{
    FILE *in = fopen("coded.txt", "r"), *out = fopen("decoded.txt", "w");
    //key is an array that has the alphabet encoded. a encoded is y etc.
    char key[] = {'y', 'h', 'e', 's', 'o', 'c', 'v', 'x', 'd', 'u', 'i', 'g', 'l', 'b', 'k', 'r', 'z', 't', 'n', 'w', 'j', 'p', 'f', 'm', 'a', 'q'};
    int T;
    fscanf(in, "%d\n", &T);
    for(int t=0;t<T;t++)
    {
        fprintf(out, "Case #%d: ", t+1);
        while(!feof(in))//while not end of file
        {
            char c;
            fscanf(in, "%c", &c);
            if (c=='\n') break;//end of Case!
            if (c==' ') fprintf(out," ");
            else fprintf(out, "%c", key[c-97]);
        }
        fprintf(out, "\n");
    }
    return 0;
}

int problemB()
{
    FILE *in = fopen("in.txt", "r"), *out = fopen("out.txt", "w");
    int T;
    fscanf(in, "%d", &T);
    for (int t=1; t<T; t++)
    {
        int N,S,p, o=0; //o=output number
        fscanf(in, "%d %d %d", &N, &S, &p);

        for (int n=0; n<N; n++)
        {
            int total;
            fscanf(in, "%d", &total);
            if ((total>=3*p-4) && total)
            {
                if (total>=3*p-2) o++;
                else if (S) {S--;o++;}//if we have surprising scores to use..
            }
        }
        fprintf(out, "Case #%d: %d\n", t, o);
    }
    return 0;
}
int power(int y)
    {
        int r=10;
        for (int i=1;i<y;i++) r*=10;
        return r;
    }

int problemC()
{

    FILE *in = fopen("in.txt", "r"), *out = fopen("out.txt", "w");
    //return power(3);
    int T;
    fscanf(in,"%d", &T);
    for(int t=1;t<=T;t++)
    {
    int a,b,c=0;
    fscanf(in,"%d %d", &a, &b);
    int ten=1; while(pow(10,ten)<=a) ten++;ten--;
    for (;a<b;a++)
    {
        for(int i=1;i<=ten;i++)
        {
            int rr=ten-i+1;
            int q = power(i),
                r = power(rr);
            int p=(a%q) * r + a/q;
            if ((p>a) && (p<=b)) {c++;}//printf("%d %d\n", a,p);}
        }
    }
    fprintf(out,"Case #%d: %d\n",t ,c);
    }
    return 0;
}


int main() { return problem_n(); }
