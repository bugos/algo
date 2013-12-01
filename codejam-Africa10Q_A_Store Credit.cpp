#include <stdio.h>

int main()
{
    FILE *in  = fopen("A-large.in" , "r"),
         *out = fopen("A-large.out", "w");

    // Read number of cases
    int n;
    fscanf(in, "%4d", &n);

    // Case #i
    for(int i = 0; i < n; i++)
    {
        // Read credits and number of items
        int credit, items;
        fscanf(in, "%4d %4d", &credit, &items);

        // Read prices
        int price[items];
        for (int j = 0;  j < items; j++)
        {
            fscanf(in, "%4d", &price[j]);
        }

        // check
        for (int k = 0; k < items; k++)
            for (int l = k + 1; l < items; l++)
                if (price[k] + price[l] == credit)
                {
                        fprintf(out, "Case #%d: %d %d \n", i+1, k+1, l+1);
                }
    }
}
