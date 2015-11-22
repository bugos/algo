#include <iostream>
#include <string>
using namespace std;

class Crypto {
protected:
	string message;
	string encrypted;
	string decrypted;
public:
	void virtual Encryption() = 0;
	void virtual Decryption() = 0;
	Crypto( const string msg ) : message( msg ) {}
	string getMessage() {
		return message;
	}
	void setMessage( string message ) {
		this->message = message;
	}
	string getEncrypted() {
		return encrypted;
	}
	string getDecrypted() {
		return decrypted;
	}
};

class Cesar: public Crypto {
	static int const key = 1;
public:
	Cesar( const string msg ) : Crypto(msg) {}
	void virtual Encryption() {
		encrypted.resize( message.length() );
		for ( int i = 0; i < message.length(); i++ ) {
			encrypted[i] = message[i] + key;
		}
	}
	void virtual Decryption() {
		decrypted.resize( encrypted.length() );
		for ( int i = 0; i < encrypted.length(); i++ ) {
			decrypted[i] = encrypted[i] - key;
		}
	}
};

class XOR: public Crypto {
//	static char const key = 'k';
public:
	char key;
	XOR( const string msg ) : Crypto(msg), key('k') {}
	void virtual Encryption() {
		encrypted.resize( message.length() );
		for ( int i = 0; i < message.length(); i++ ) {
			encrypted[i] = (char)message[i] ^ key;
		}
	}
	void virtual Decryption() {
		decrypted.resize( encrypted.length() );
		for ( int i = 0; i < encrypted.length(); i++ ) {
			decrypted[i] = encrypted[i] ^ key;
		}
	}
};


int main() {
	Cesar c_enc("hello");
	c_enc.Encryption();
	string resEncCes=c_enc.getEncrypted();
	cout << resEncCes << endl;

	c_enc.Decryption();
	cout << c_enc.getDecrypted() << endl;

	XOR x_enc("hello");
	x_enc.Encryption();
	string resEncXOR = x_enc.getEncrypted();
	cout << resEncXOR << endl;

	x_enc.Decryption();
	cout << x_enc.getDecrypted() << endl;

}
