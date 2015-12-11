#include <iostream>
#include <fstream>
#include <vector>

class ModelValidationException : public std::exception {
public:
	virtual const char* what() const throw() {
	    return "Invalid input.\n";
	}
};

#define EMPL_FILENAME "employee.txt"
class Employee {
private:
	bool persists;
	bool synced; // synced to file

public:
	std::string name;
	double salary;

	Employee() : persists(), synced(), name(), salary() {}

	Employee& input(std::istream& is) {
		is >> name >> salary;
		if (is.fail()) {
			throw ModelValidationException();
		}
		return *this;
	}
	Employee& output(std::ostream& os) {
		os << name << ' ' << salary << std::endl;
		return *this;
	}
	Employee& persist() {
		if (!persists) {
			persistent.push_back(*this);
		}
		persists = true;
		return *this;
	}
	Employee& sync( bool alreadySynced = false ) {
		if (!synced && !alreadySynced) {
				output(savefile);
		}
		synced = true;
		return *this;
	}

	/* Static Members */
	static std::fstream savefile;
	static std::vector<Employee> persistent;
	static void inputFrom( std::istream& is ) {
		try {
			Employee employee;
			bool comesFromSavefile = &is == &(std::istream&)savefile;
			employee.input(is).sync(comesFromSavefile).persist().output(std::cout);
		}
		catch ( ModelValidationException &e ) {
			std::cout << e.what();

			is.clear();
		}
	}
	static void init() { // get employees from file
		savefile.open(EMPL_FILENAME, std::fstream::in);
		while (savefile.good()) {
			if ((savefile >> std::ws).peek(), savefile.eof()) {
				break;
			}
			inputFrom(savefile);

		}
		savefile.close();
		savefile.open(EMPL_FILENAME, std::fstream::out | std::fstream::app);
	}
	static double getMeanSalary() {
		if (!persistent.size()) return 0;
		double sum = 0;
		for ( std::vector<Employee>::iterator it = persistent.begin() ; it != persistent.end(); ++it) {
			sum += it->salary;
		}
		return sum / persistent.size();
		// use accumulate
	}

};
std::fstream Employee::savefile;
std::vector<Employee> Employee::persistent;

//todo: last line error, no return.
int main() {
	Employee::init();

	std::cout << "Please input up to 20 employees(break with !):" << std::endl;
	#define MAX_EMPL_INPUT 20
	for(int i = 0; i < MAX_EMPL_INPUT; i++ ) {
		if ((std::cin >> std::ws).peek() == '!') {
			break;
		}
		Employee::inputFrom(std::cin);
	}

	std::cout << "The mean of salaries is " << Employee::getMeanSalary();
}
