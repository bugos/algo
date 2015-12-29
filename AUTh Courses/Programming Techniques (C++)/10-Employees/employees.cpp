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
	bool saved; // synced to file

public:
	std::string name;
	double salary;

	Employee() : persists(), saved(), name(), salary() {}

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
			persists = true;
			persistent.push_back(*this);
		}
		return *this;
	}
	Employee& save( bool alreadySaved = false ) {
		if (!saved && !alreadySaved) {
				output(savefile);
		}
		saved = true;
		return *this;
	}

	/* Static Members */
	static std::fstream savefile;
	static std::vector<Employee> persistent;
	static void inputFrom( std::istream& is ) {
		try {
			Employee employee;
			bool comesFromSavefile = &is == &(std::istream&)savefile;
			employee.input(is).save(comesFromSavefile).persist().output(std::cout);
		}
		catch ( ModelValidationException &e ) {
			std::cout << e.what();
			is.clear();
		}
	}
	static void init() { // get employees from file
		savefile.open(EMPL_FILENAME, std::fstream::in); //must have newline at the end
		while (savefile.good() && !((savefile >> std::ws).peek(), savefile.eof())) {
			inputFrom(savefile);
		}
		savefile.close();
		savefile.open(EMPL_FILENAME, std::fstream::out | std::fstream::app);
	}
	static double getMeanSalary() {
		double sum = 0;
		for ( std::vector<Employee>::iterator it = persistent.begin() ; it != persistent.end(); ++it) {
			sum += it->salary;
		}
		return persistent.size() ? sum / persistent.size() : 0;
	}
};
std::fstream Employee::savefile;
std::vector<Employee> Employee::persistent;

int main() {
	Employee::init();

	std::cout << "Please input up to 20 employees(break with !):" << std::endl;
	#define MAX_EMPL_INPUT 20
	for(int i = 0; i < MAX_EMPL_INPUT && (std::cin >> std::ws).peek() != '!'; i++ ) {
		Employee::inputFrom(std::cin);
	}

	std::cout << "The mean of salaries is " << Employee::getMeanSalary() << std::endl;
}
