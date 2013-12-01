//|c| bugos 2012 CAMP pdp
#include <stdio.h>
#include <algorithm>

using namespace std;

#define MAX 1000002

int n, k, nor[MAX], existPos=0, existNeg=0, maxn=0, kj, s;

int main() {

	scanf("%d %d\n", &n, &k);

	nor[0] = 0; //only used for i=1
	for (int i=1; i<=n; i++) {
		scanf("%d", &nor[i]);
		if (nor[i]>0) existPos = 1;
		if (nor[i]<0) existNeg = 1;
		nor[i] += nor[i-1];
	}

	for (int i=1; i<=n; i++) {
		if (i!=1 && nor[i-1]>nor[i-2] && existNeg) continue;//only start after a negative
		if (nor[i]<=nor[i-1] && existPos) continue; //only start from a positive

		for (int j=1; (j<=k); j++) {
			kj = i+j-1;
			if (kj>n) break;
			s = nor[kj] - nor[i-1];
			maxn = max(maxn, s);
		}
	}
	printf("%d", maxn);

return 0;
}
