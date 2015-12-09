#include <iostream>
#include <random>
using namespace std;


template<typename T>
class Node {
#define PRNG_SEED 59915013
private:
	T data;
	Node* next;

	T getRandom() { // Generate random T
		static std::mt19937 randomGenerator(PRNG_SEED); // New one for every T
		int minChar = '0', maxChar = 'z';
		int randomNumber = minChar + (randomGenerator() % (maxChar - minChar + 1));
		return static_cast<T> ( randomNumber );
	}
public:
	Node() : data(getRandom()), next(NULL) {}
	~Node() {
		delete next;
	}
	Node* getNext() {
		return next;
	}
	T getData() {
		return data;
	}
	void add() {
		if (!next) {
			next = new Node<T>;
		}
	}
	void debug() {
		Node<T>* tail = this;
		cout << tail->getData() << ' ';
		for(int i=0; i<20; i++) {
			tail->add();
			tail = tail->getNext();
			cout << tail->getData() << ' ';
		}
		cout << endl;
	}
};

int main() {
	Node<char> a;
	a.debug();

	Node<int> b;
	b.debug();

	Node<char> c;
	c.debug();

	Node<double> d;
	d.debug();

	Node<bool> e;
	e.debug();

	Node<float> f;
	f.debug();
}
