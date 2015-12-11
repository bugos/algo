#Κληρονομικότητα ΙΙ: Κρυπτογράφηση

Δίνεται η ακόλουθη Abstract Class Crypto και οι κλάσεις Cesar και XOR που κληρονομούν την Crypto.

Class Crypto{

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

}

Class Cesar: public Crypto{

}

Class XOR: public Crypto{

}

Να υλοποιηθούν όλες οι μέθοδοι ώστε να λειτουργεί η ακόλουθη main

main()

{

Cesar c_enc(“hello”);

c_enc. Encryption();

string resEncCes=c_enc.getEncrypted();

coutn<<resEncCes;<<endl;

c_enc.Decryption()

cout<<c_end.getDecrypted();

XOR x_enc(“hello”);

x_enc.Encryption();

string resEncXOR=c_enc.getEncrypted();

coutn<<resEncXOR;<<endl;

x_enc.Decryption()

cout<<x_end.getDecrypted();

}

Ο αλγόριθμος του Cesar λειτουργεί ως εξής: Ας θεωρήσουμε ένα μήνυμα m ως είσοδο στον αλγόριθμο. Το κάθε σύμβολο (χαρακατήρας) του αρχικού μηνύματος αντικαθίσταται από ένα νέο σύμβολο (χαρακτήρα) ο οποίος προκύπτιε από το αρχικό σύμβολο + ν θέσεις στο αρχικό. Δηλαδή εαν το ν είναι 1 και το μήνυμα είναι ABCD τότε το κρυπτογραφημμένο μήνυμα θα είναι BCDE. Η αποκρυπτογράφηση θα γίνεται με την αντίστροφη διαδικασία.

Στην περίπτωση του XOR η κρυπτoγράφηση βασίζεται στην πράξη του XOR μεταξύ δύο bit. Συνοπτικά ως είσοδος λαμβάνεται το μήνυμα και ένα μυστικό κλειδί τα οποία τα συνδιάζουμε για να γίνει η κρυπτογράφηση. Για την πράξη της κρυπτογράφησης χρησιμοποιούμε τον τελεστή XOR (^) μεταξύ του μηνύματος και του κλειδιού. Δηλαδή εαν το μήνυμα μας είναι το “abc” και το κλειδί μας είναι το 'c' τότε θα κάνουμε κάθε χαρακτήρα του μηνύματος XOR με το χαρακτήρα 'c'. Δηλαδή το XOR(a,c)=a^c, XOR(b,c) κ.ο.κ. Θεωρείστε στη συγκεκριμένη περίπτωση ότι το κλειδί μπορεί να είναι ένα μόνος χαρακτήρας.

Oποιαδήποτε επιπλεόν μεταβλητη χρειάζεστε θα πρέπει να την ορίσετε και για την περίπτωση του Cesar και του XOR. Για παράδειγμα στην περίπτωση Cesar χρειάζεται να ορίσετε τον αριθμό n μετατώπισης ενώ στην XOR το αντίστοιχο κλειδί.