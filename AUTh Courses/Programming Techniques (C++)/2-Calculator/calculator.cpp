// Developer Evangelos 'bugos' Mamalakis

#include <iostream>
using namespace std;

struct Complex {
	double x;
	double y;
};

double calculator(double x, double y) {
	char operationChoice;
	cout << "Choose (a)ddition, (s)ubtraction, (m)ultiplication, (d)ivision: ";
	cin >> operationChoice;

	switch (operationChoice) {
	case 'a':
		return x + y;
	case 's':
		return x - y;
	case 'm':
		return x * y;
	case 'd':
		return x / y; //dbz
	}
}

Complex calculator(Complex z, Complex w) {
	char operationChoice;
	cout << "Choose (a)ddition, (s)ubtraction, (m)ultiplication, (d)ivision: ";
	cin >> operationChoice;

	Complex res;
	switch (operationChoice) {
	case 'a':
		res.x = z.x + w.x;
		res.y = z.x + w.y;
		break;
	case 's':
		res.x = z.x - w.x;
		res.y = z.x - w.y;
		break;
	case 'm':
		res.x = z.x * w.x - z.y * w.y;
		res.y = z.x * w.y + z.y * w.x;
		break;
	case 'd':
		double wMagnitude = w.x * w.x + w.y * w.y; //dbz
		res.x = ( z.x * w.x + z.y * w.y ) / wMagnitude;
		res.y = ( z.y * w.x - z.x * w.y ) / wMagnitude;
		break;
	}
	
	return res;
}


int main()
{
	char numberTypeChoice;
	cout << "Choose (r)eal or (c)omplex numbers: ";
	cin >> numberTypeChoice;

	switch (numberTypeChoice) {
	case 'r':
	{
		double x, y;
		cout << "Insert first real number: ";
		cin >> x;
		cout << "Insert second real number: ";
		cin >> y;

		double res = calculator(x, y);
		cout << "The result is " << res;
	}
	break;
	case 'c':
	{
		Complex z, w;
		cout << "Insert first complex number: ";
		cin >> z.x >> z.y;
		cout << "Insert second complex number: ";
		cin >> w.x >> w.y;

		Complex res2 = calculator(z, w);
		cout << "The result is " << res2.x << "+" << res2.y << "i";
	}
	break;
	}

	char _pause;
	cin >> _pause; // wait for enter
    return 0;
}
