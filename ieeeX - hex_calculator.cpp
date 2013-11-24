//bugos, octomber 26-2013
#define DEBUG1
#define MAXITEMS 20

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <stack>
#include <iomanip>
using namespace std;

//MACROS
#ifdef DEBUG
#define DEB(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEB(str) do { } while ( false )
#endif

#define ERROR() cout << "ERROR"; return 0


int main() {
    
	stack<unsigned short> s;
	char c, op;
    unsigned short  res, a, b;
	int num, items = 0;
	
	//input and process
    while (cin.peek() != EOF && cin.peek() != '\n' ) {
        c = cin.peek();
		DEB ("READ CHAR: " << (int)c << ' ' << c);

        if ( 
		(c >= '0' && c <= '9') || 
		(c >= 'a' && c <= 'f') || 
		(c >= 'A' && c <= 'F') ) 
		{ //input number
			cin >> std::hex >> num;
			DEB ( "NUMBER " << num );
			items++;
			
			if (num > 65535 || num < 0) { //overflow
				ERROR();
			}
			s.push((short)num);
			
		}
		else if ( c == '+' || c == '-' || c == '&' || c == '|' || c == 'X' || c=='~' ) //TODO use string find?
		{ //input operator
			cin >> op; // op = c
			DEB( "OPERATOR " << op );
			items++;
			
			//check if there are enough operants in stack
			if ( ( op != '~' && s.size() < 2 ) || ( op == '~' && s.size() < 1 ) ) {
				ERROR();
			}

			//Perform the operation
			b = s.top(); s.pop();
			if (op != '~') { a = s.top(); s.pop(); }
			switch (op) {
				case '+':
					if ((int)a + (int)b > 65535) //overflow
						res = 65535; //ffff
					else 
						res = a + b;
					break;
				case '-':
					if (b > a) //overflow
						res = 0;
					else 
						res =  a - b;
					break;
				case '&':
					res = a & b;
					break;
				case '|':
					res = a | b;
					break;
				case 'X':
					res = a ^ b;
					break;
				case '~':
					res = ~b;
					break;
			}
			s.push(res);
			
		}
		else {
			ERROR();
		}
		
		if ( cin.peek() == ' ' ) cin.ignore(); //ignore one whitespace if it exists
		
		if (items > MAXITEMS) {
			DEB ("ERROR: too many items: " << items);
			ERROR();
		}
    } //end of input
	
		
	if (s.size() > 1) { // too many operants given
		DEB ("ERROR: stack has more than 1 operants: " << s.size());
		ERROR();
	}
	if (s.size() < 1) { // stack empty
		DEB ("ERROR: stack empty: " << s.size());
		ERROR();
	}
	
	cout << setfill('0') << setw(4) << uppercase << hex << s.top();
    return 0;
}
