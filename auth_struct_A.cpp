#include <iostream>
using namespace std;

#define N_MOST_SUN 3
int K, minSunDays;
int	Ngood = 0;
int mostSun[N_MOST_SUN], mostSunW[N_MOST_SUN] = {}; //Priority Queue
int heapAdd(int w, int v);

int main() {
	cin >> K >> minSunDays;
	int SunDays;
	for (int i = 1; i <= K; i++) {
		try {
			cin >> SunDays;
			if (SunDays < minSunDays)
				throw i;
            ++Ngood;
            heapAdd(SunDays, i);
		}
		catch (int i) {
            cout <<"Area "<< i << " did not have enough sun days.\n";
		}
	}
	//Output
	cout << "There are " << Ngood << " areas with enough sun.\n";
	cout << "Sunny Areas:\n";
	for(int i = 0; i < N_MOST_SUN; i++) {
        cout << "Area " << mostSun[i] << " with " << mostSunW[i] << "sun days\n";
	}
	return 0;
}

// Add value v with weight w to a Priority Queue
int heapAdd(int w, int v) {
    for(int i = 0; i < N_MOST_SUN; i++) {
        if (w > mostSunW[i]) { //found a spot
            //Move previous values one position to the right
            // overwriting the last value and starting from the end.
            for(int j = N_MOST_SUN-1; j > i; j--) {
                mostSunW[j] = mostSunW[j-1];
                mostSun [j] = mostSun [j-1];
            }
            //Insert new values
            mostSunW[i] = w;
            mostSun [i] = v;
            break;
        }
    }
}
