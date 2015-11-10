// a4-calendar.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

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
    void normalizeNumber(int &remainder, int max, int &quotientDestination, bool forcePositive = 1 ) {
        if (forcePositive) { remainder -= max + 1; } // Take a month in case of negative date

        int remainderSign = remainder > 0 ? 1 : -1;
        quotientDestination += ( remainder - remainderSign) / max;
        remainder = ( remainder - remainderSign) % max + remainderSign;

        if (forcePositive) { remainder += max + 1; } // Return subtracted days.
    }
    void normalizeDate() {
        normalizeNumber( *day, DAYS_IN_MONTH, *month );
        normalizeNumber( *month, 12, *year );
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
    Date(const Date &that) {
        Date(*(that.day), *(that.month), *(that.year));
    }
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
    friend ostream& operator<<(ostream& os, const Date& date) {
        os << *(date.day) << '/' << *(date.month) << '/' << *(date.year);
        return os;
    }

    Date operator+(const Date& that) {
        return Date(*day + *(that.day), *month + *(that.month), *year + *(that.year));
    }
    Date operator-(const Date& that) {
        return Date(*day - *(that.day), *month - *(that.month), *year - *(that.year));
    }
};

int main() {
    Date bday(31, 5, 1995);
    Date a(1, 7, 0);
    Date b(-31, 3, 0);
    //Date b(a);
    //b = bday;
    cout << bday + a << ' ' << a << ' ' << bday << ' ' << b;
    int r;
    cin >>r ;
    return 0;
}
