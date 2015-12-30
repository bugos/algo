#Εξαιρέσεις και κληρονομικότητα

Δίνεται ο ορισμός της ακόλουθης κλάσης `safetable`:
```cpp
class safetable {
private :
  int *array;
  int size;
public:
  safetable();
  safetable(int x)
  int operator [] (…)
  ~safetable();
}
```

Να υλοποιήσετε τις αντίστοιχες μεθόδους με τέτοιο τρόπο ώστε εαν ο χρήστης της κλάσης `safetable` πραγματοποιήσει πρόσβαση στα στοιχεία του πίνακα array εκτός ορίων, π.χ εαν το size είναι `5` και ο χρήστης πραγματοποιεί πρόσβαση στο 6ο στοιχείο του πίνακα, να δημιουργεί μια εξαίρεση. Το προκαθορισμένο μέγεθος του πίνακα θα είναι `10`, εκτός και εαν ο χρήστης θέλει να προσδιορίζει το μέγεθος του πίνακα.

Η εξαίρεση αυτή θα πρέπει να είναι τύπου `MyOutofBoundException` η οποία θα πρέπει να κληρονομεί την exception της C++. H εξαίρεση θα πρέπει να επιστρέφει ένα μήνυμα `“Table out of Bounds”`.

H main που θα πρέπει να υποστηρίζεται είναι της ακόλουθης μορφής και θα πρέπει να γίνουν οι κατάλληλες αλλαγές για να υποστηρίζει τις εξαιρέσεις.
```cpp
int main() {
  safetable table1;
  cout << “Value of A[1]:” << table1[1] << endl;
  cout << “Value of A[1]:” << table1[2] << endl;
  cout << “Value of A[2]:” << table1[14] << endl;

  safetable table2(20);
  cout << “Value of A[1]:” << table2[1] << endl;
  cout << “Value of A[1]:” << table2[2] << endl;
  cout << “Value of A[2]:” << table2[32] << endl;
}
```
