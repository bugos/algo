#include <iostream>
#include <set>
#include <algorithm>

class PhonebookEntry {
private:
public:
	std::string name;
	int *number; // up to 5 digits

	PhonebookEntry(std::string name = "", int number = 0) : name(name) {
		this->number = new int;
		*(this->number) = number;
	}
	~PhonebookEntry() {
		delete number;
	}
    PhonebookEntry& operator=(const PhonebookEntry& that) {
        name = that.name;
        *number = *(that.number);
        
        return *this;
    }
	bool operator<(const PhonebookEntry& that) const {
		return name < that.name;
	}
	bool operator>(const PhonebookEntry& that) const {
		return name > that.name;
	}
    friend std::ostream& operator<<(std::ostream& os, const PhonebookEntry& entry) {
        os << entry.name << ' ' << *(entry.number) << std::endl;
        return os;
    }
};
struct PhonebookEntryComparator {
	 bool operator() (const PhonebookEntry& a, const PhonebookEntry& b) const {
	 	return a.name < b.name;
	 }
};
struct PhonebookEntryComparatorReverse {
	bool operator() (const PhonebookEntry& a, const PhonebookEntry& b) const {
		return a.name > b.name;
	}
};

int main() {
	/*** Testing PhonebookEntry ***/
	PhonebookEntry a("mamalakis", 50429);
	PhonebookEntry b("tzanakakis", 41393);
	
	std::cout << a << b;
	b = a;
	a = PhonebookEntry("avgoustakis", 91205);
	std::cout << a << b;
	

	/*** Testing phonebook set ***/ 
	std::set<PhonebookEntry, std::less<PhonebookEntry>> phonebook = {
		PhonebookEntry("mamalakis", 50429),
		PhonebookEntry("avgoustakis", 91205),
		PhonebookEntry("tzanakakis", 41393)
	};
	std::for_each( phonebook.begin(), phonebook.end(), 
		[](const PhonebookEntry& entry) { //doesnt work without &!
			std::cout << entry; 
	} );

	std::set<PhonebookEntry, std::greater<PhonebookEntry>> phonebookReverse(phonebook.end(), phonebook.begin());
	std::for_each( phonebook.begin(), phonebook.end(), 
		[](const PhonebookEntry& entry) {
			std::cout << entry; 
	} );

	return 0;
}
