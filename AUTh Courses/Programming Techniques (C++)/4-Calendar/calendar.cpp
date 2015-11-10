// Developer Evangelos 'bugos' Mamalakis
#include <iostream>
using namespace std;

#define DAYS_IN_MONTH 31

class Date {
private:
    int *day, *month, *year;
    void allocateMemory() {
        day = new int;
        month = new int;
        year = new int;
    }
    // Makes sure 1 <= day <= 31, 1 <= month <= 12 etc.
    // Accepts negative values and zeros similarly to JS Date() object
    // All months have 31 days
    void normalizeNumber(int &remainder, int max, int &quotientDestination) {
        bool negativeTrick = remainder <= 0;
        if (negativeTrick) { remainder -= max + 1; } // Take a month in case of negative date

        int remainderSign = remainder > 0 ? 1 : -1;
        quotientDestination += (remainder - remainderSign) / max;
        remainder = (remainder - remainderSign) % max + remainderSign;

        if (negativeTrick) { remainder += max + 1; } // Return subtracted days.
    }
    void normalizeDate() {
        normalizeNumber(*day, DAYS_IN_MONTH, *month);
        normalizeNumber(*month, 12, *year);
        // BUG: Year 0 does exist in our calendar
    }

public:
    // Getters
    int getDay() { return *day; }
    int getMonth() { return *month; }
    int getYear() { return *year; }

    Date(int d = 0, int m = 0, int y = 0) {
        allocateMemory();

        *day = d;
        *month = m;
        *year = y;

        normalizeDate();
    }
    Date(const Date &that) : Date(*(that.day), *(that.month), *(that.year)) {} //c++11
    Date& operator=(const Date& that) {
        *day = *(that.day);
        *month = *(that.month);
        *year = *(that.year);

        return *this;
    }
    ~Date() {
        delete day;
        delete month;
        delete year;
    }
    Date operator+(const Date& that) {
        return Date(*day + *(that.day), *month + *(that.month), *year + *(that.year));
    }
    Date operator-(const Date& that) {
        return Date(*day - *(that.day), *month - *(that.month), *year - *(that.year));
    }
    friend ostream& operator<<(ostream& os, const Date& date) {
        os << *(date.day) << '/' << *(date.month) << '/' << *(date.year);
        return os;
    }
};

int main() {
    Date bday(31, 5, 1995);
    cout << bday << '\n';

    Date a(1, 7, 0);
    Date c(a);
    cout << c << '\n';

    c = bday;
    cout << a << ' ' << c << '\n';

    for (int i = -75; i <= 75; i++) { cout << Date(i, 0, 0) << '\n'; }

    int r; cin >> r;
    return 0;
}
