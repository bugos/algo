// BugOS 14/10/13, Cosine Similarities http://www.ieee.org/membership_services/membership/students/awards/xtremesamples.html
#include <stdio.h> // printf
#include <math.h> // acos
#include <iostream> //cin, cout
#include <numeric> //inner_product
#include <vector>
using namespace std;


//Inputs a list of integers in the form [6,5,5,...] and stores them in v
#define ERROR 13
int cin_to_array(vector<int> &v)
{
    //Ignore '[' or ' [', return error if not found
    cin.ignore(2, '[');
    if (!cin.good()) return ERROR;


    int n;
    while (cin >> n)  //while there is an int retrieve it and ignore the ',' afterwards
    {
        v.push_back(n);

        if (cin.peek() == ',')
            cin.ignore();
    }
    if (v.empty()) return ERROR; // if no ints read return error
    cin.clear(); //clear the error that stopped the loop


    //ignore ']' that comes after no more ints, return error if not found
    cin.ignore(1, ']');
    if (!cin.good()) return ERROR;
}


int main(int argc, char *argv[])
{
    //Input and Error Checking
    vector<int> A,B;
    if ( cin_to_array(A) != ERROR &&
         cin_to_array(B) != ERROR &&
         A.size() == B.size() )
    {
        //Calculations
        double A_dot_B, A_dot_A, B_dot_B, magnA, magnB, theta;

        A_dot_B = inner_product(A.begin(), A.end(), B.begin(), 0);

        A_dot_A = inner_product(A.begin(), A.end(), A.begin(), 0);
        magnA = sqrt( A_dot_A ); //TODO: check != 0

        B_dot_B = inner_product(B.begin(), B.end(), B.begin(), 0);
        magnB = sqrt( B_dot_B );

        theta = acos( A_dot_B / (magnA * magnB) );

        //output
        printf("%.4f", theta);
        //cout << '\n' << setprecision(4) << fixed << theta; //#include <iomanip>
    }
    else
        cout << "Error";

    return 0;
}
