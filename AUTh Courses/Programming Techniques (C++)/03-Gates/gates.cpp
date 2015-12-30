#include <iostream>
using namespace std;

#define BASE 2
#define MAXINPUTS 10
class Gate {
protected:
	int nInputs;
	bool input[MAXINPUTS];
public:
	virtual bool getOutput() = 0;
	const char* name;
	
	Gate( int nInputs, const char* name ) : nInputs(nInputs), name(name) {
		for ( int i = 0; i < nInputs; i++ ) {
			setInput( i, 0 ); // obsolete?
		}
	}
	void setInput( int i, bool v ) {
		if ( i >= MAXINPUTS ) return;
		input[ i ] = v;
	}
	void setInput( bool v ) {
		setInput( 0, v );
	}
	void truthTable() {
		for ( int i = 0; i < BASE ; i++ ) {
			setInput( 0, i );
			
			if (nInputs > 1) {
				for ( int j = 0; j < BASE; j++ ) {
					setInput( 1, j );
					
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
	AndGate( bool input1 = 0, bool input2 = 0 ) : Gate(2, "AND") {
		setInput( 0, input1 );
		setInput( 1, input2 );
	}
	bool getOutput() {
		return input[0] && input[1];
	}
};
class OrGate : public Gate {
public:
	OrGate( bool input1 = 0, bool input2 = 0 ) : Gate(2, "OR") {
		setInput( 0, input1 );
		setInput( 1, input2 );
	}
	bool getOutput() {
		return input[0] || input[1];
	}
};
class NotGate : public Gate {
public:
	NotGate( bool input1 = 0 ) : Gate( 1, "NOT" ) {
		setInput( input1 );
	}
	bool getOutput() {
		return !input[0];
	}
};

class HalfAdder : public Gate {
public:
	HalfAdder( bool input1 = 0, bool input2 = 0 ) : Gate( 2, "HALFADDER" ) {
		setInput( 0, input1 );
		setInput( 1, input2 );
	}
	bool getSum() {
		OrGate og( input[0], input[1] );
		bool orResult = og.getOutput();

		AndGate ag( input[0], input[1] );
		NotGate ng( ag.getOutput() ); 
		bool nandResult = ng.getOutput();

		AndGate ag2( orResult, nandResult );
		bool xorResult = ag2.getOutput();
		
		return xorResult;
	}
	bool getCarry() {
		AndGate ag(input[0], input[1]);
		bool andResult = ag.getOutput();
		
		return andResult;
	}
	bool getOutput() {
		return getSum();
		//return getCarry();
	}
};

int main() {
	AndGate ag;
	OrGate og;
	NotGate ng;
	ag.truthTable();
	og.truthTable();
	ng.truthTable();
	
	HalfAdder ha;
	ha.truthTable();
	
	return 0;
}
