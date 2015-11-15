#include <iostream>
using namespace std;

class Crypto {
protected:
	string Message;
	string Encrypted;
	string Decrypted;
public:
	string virtual Encryption ()=0;
	string virtual Decryption ()=0;
	Crypto(string msg);
	string getMessage();
	void setMessage();
	string getEncrypted();
	string getDecrypted();
};

class Cesar: public Crypto {

};

class XOR: public Crypto {

};


int main() {
	Cesar c_enc("hello");
	c_enc.Encryption();
	string resEncCes=c_enc.getEncrypted();
	cout << resEncCes << endl;
	c_enc.Decryption();
	cout << c_end.getDecrypted();
	XOR x_enc("hello");
	x_enc.Encryption();
	string resEncXOR = c_enc.getEncrypted();
	cout << resEncXOR << endl;
	x_enc.Decryption();
	cout << x_end.getDecrypted();

}
