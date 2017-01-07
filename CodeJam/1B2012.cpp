#include <stdio.h>

FILE *in=fopen("in.txt", "r"), *out=fopen("out.txt","w");

int main()
{
    int cases;
    fscanf(in, "%d", &cases);
    for (int C=0; C<cases; C++)
    {
        int n;
        double max=100, max2=100, sum=0;
        fscanf(in, "%d", &n);
        double ji[1000];
        for (int i=0;i<n;i++)
        {
            fscanf(in, "%Lf", &ji[i]);
            printf("%Lf\n", ji[i]);
            if (ji[i]< max) {max2=max; max = ji[i];}
            else if (ji[i]< max2) {max2 = ji[i];}
            sum+=ji[i];
        }
        printf ("sum is %Lf\n", sum);
        printf("max is %Lf, max 2 is %Lf\n", max, max2);

        //printf("Case #%d: ", C);
        for (int i=0;i<n;i++)
        {
            double tmax = max;
            if (ji[i]==max) tmax=max2;
            double A= ( (tmax-ji[i])/sum ) *50 + 50;
            printf ("max:%Lf sum:%Lf ji:%Lf > %Lf \n", tmax, sum, ji[i], A);
        }
        printf("\n\n");
    }
    return 0;
}
