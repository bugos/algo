#include <iostream>
using namespace std;

class Date {
    int *day, *month, *year;

public:
    Date( int d=0, int m=0, int y=0 ) {
        day = new int;
        month = new int;
        year = new int;
        *day = d;
        *month = m;
        *year = y;
        
    }
    Date( const Date &that ) {
        Date(*(that.day), *(that.month), *(that.year));
    }
    Date& operator=(const Date& that){
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
        Date res;
        int newDay = *day + *(that.day);
        *(res.month) += newDay / 31; //alou
        *(res.day) = newDay % 31 + 1;
        
        int newMonth = *month + *(that.month);
        *(res.year) += newMonth / 12; //alou
        *(res.month) += newMonth % 12 + 1;
        
        *(res.year) += *year + *(that.year);
        
        return res;
    }
};

int main() {
    Date bday(31,5,1995);
    Date a(1,7,0);
    cout << bday+a;
    return 0;
}
