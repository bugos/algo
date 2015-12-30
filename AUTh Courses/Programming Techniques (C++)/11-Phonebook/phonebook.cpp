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
	PhonebookEntry(const PhonebookEntry &that) : PhonebookEntry(that.name, *(that.number)) {}
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
		PhonebookEntry("mavrokostas", 50429),
		PhonebookEntry("samolis",     91205),
		PhonebookEntry("skordalos",   41393),
		PhonebookEntry("tatakis",     49305),
		PhonebookEntry("moudakis",    95730),
		PhonebookEntry("stavrakakis", 27497),
		PhonebookEntry("giasafakis",  10953),
		PhonebookEntry("petrakis",    85486),
		PhonebookEntry("manolakis",   30572),
		PhonebookEntry("lagoudakis",  85629),
	};
	std::for_each( phonebook.begin(), phonebook.end(), 
		[](const PhonebookEntry& entry) { //doesnt work without &!
			std::cout << entry; 
	} );

	//only range constructor works with different prototypes
	std::set<PhonebookEntry, std::greater<PhonebookEntry>> phonebookReverse(phonebook.begin(), phonebook.end());

	std::for_each( phonebookReverse.begin(), phonebookReverse.end(), 
		[](const PhonebookEntry& entry) {
			std::cout << entry; 
	} );

	return 0;
}
