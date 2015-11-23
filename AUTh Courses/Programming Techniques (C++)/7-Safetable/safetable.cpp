#include <iostream>
#include <exception>
using namespace std;

class ArrayOutofBoundException : public exception {};

class Safetable {
private :
	int *array;
	int size;

public:
	Safetable(int x = 10) : size(x) {
		array = new int[size];
	}
	int &operator[]( int i ) {
		if ( i >= size ) {
			throw new ArrayOutofBoundException();
		}
		return array[i];
	}
	~Safetable() {
		delete[] array;
	}
	void printValue( int i ) {
		try {
			cout << "Value of A[" << i << "]: " << (*this)[i] << endl;
		}
		catch ( ArrayOutofBoundException &e ) {
			cout << "Table out of Bounds" << endl;
		}
	}
};


int main() {
	Safetable table1;
	table1.printValue(1);
	table1.printValue(2);
	table1.printValue(14);

	Safetable table2(20);
	table2.printValue(2);
	table2.printValue(14);
	table2.printValue(32);
}
