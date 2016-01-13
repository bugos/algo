public class HashTable {
	int[] array;
	int size;
	int nItems; //(3 μον) οι μεταβλητές
	
	public HashTable() { // Κάνουμε πάντα κενό constructor ακόμα και αν δε ζητείται! (1 μον)
		array = new int[0];
		size = 0;
		nItems = 0;
	}
	public HashTable(int n) { // (1 μον)
		array = new int[n];
		size = n;
		nItems = 0;
	}
	public int size() { // (1 μον)
		return size;
	}
	public int numOfItems() { // Θα μπορούσαμε να μην έχουμε μεταβητή nItems. (3 μον)
		return nItems;
	}
	public boolean isEmpty() { // παλι θα μπορουσα να μην εχω nItems (3 μον)
		return nItems == 0;
	}
	public float tableFullness() { // (3 μον)
		return 100 * (float)nItems / (float)size;
	}
	public void hash(int k, int m) { // Το m είναι τυχαίο
		if (k <= 0 || m <= 0) {
			System.out.println("Λάθος Είσοδος");
			return; // Invalid arguments
		}
		
		int index = k % m;
		while( array[index] != 0 ) {
			index = (index+1) % size; // Προσοχή στην υπερχείλιση
		}
		array[index] = k;
		nItems++;
		
		if (tableFullness() > 75) {
			int newSize = 2 * size;
			int[] newArray = new int[newSize];
			
			for (int i = 0; i < size; i++ ) {
				newArray[i] = array[i];
			}
			
			array = newArray;
			size = 2 * size;
		}
	}
}

