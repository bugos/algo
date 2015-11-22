#include <iostream>
#include <string>
#include <functional>
using namespace std;

class Crypto {
protected:
	string message;
	string encrypted;
	string decrypted;

	// Applies func to original char-by-char
	void charEncryption( string &result, const string &original, function<char (char)> func ) {
		result.resize( original.length() );
		for ( int i = 0; i < original.length(); i++ ) {
			result[i] = func( original[i] );
		}
	}

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
public:
	int key;
	Cesar( const string msg ) : Crypto(msg), key(1) {}
	void virtual Encryption() {
		charEncryption( encrypted, message, [this]( char c ) {
			return c + key;
		});
	}
	void virtual Decryption() {
		charEncryption( decrypted, encrypted, [this]( char c ) {
			return c - key;
		});
	}
};

class XOR: public Crypto {
//	static char const key = 'k';
public:
	char key;
	XOR( const string msg ) : Crypto(msg), key('k') {}

	void virtual Encryption() {
		charEncryption( encrypted, message, [this]( char c ) {
			return c ^ key;
		});
	}
	void virtual Decryption() {
		charEncryption( decrypted, encrypted, [this]( char c ) {
			return c ^ key;
		});
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
