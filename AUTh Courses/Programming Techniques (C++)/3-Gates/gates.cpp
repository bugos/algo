#include <iostream>
using namespace std;

class Gate {
protected:
	int nInputs;
	bool input1, input2;
public:
	void setInput1(bool v) {
		input1 = v;
	}
	void setInput2(bool v) {
		input2 = v;
	}
	virtual bool getOutput() = 0;
	const char* name;
	
	void truthTable() {
		for ( int i = 0; i < 2 ; i++ ) {
			setInput1(i);
			if (nInputs > 1) {
				for ( int j = 0; j < 2; j++ ) {
					setInput2(j);
					cout << i << name << j << "=" << getOutput() << '\n';
				}
			}
			else {
				cout << name << i << "=" << getOutput() << '\n';
			}
		}
	}
};

class AndGate : public Gate {
public:
	AndGate() {
		name = "AND";
		nInputs = 2;
	}
	bool getOutput() {
		return input1 && input2;
	}
};
class OrGate : public Gate {
public:
	OrGate() {
		name = "OR";
		nInputs = 2;
	}
	bool getOutput() {
		return input1 || input2;
	}
};
class NotGate : public Gate {
public:
	NotGate() {
		name = "NOT";
		nInputs = 1;
	}
	bool getOutput() {
		return !input1;
	}
};

int main() {
	AndGate ag;
	OrGate og;
	NotGate ng;
	ag.truthTable();
	og.truthTable();
	ng.truthTable();
	return 0;
}