
class DelimiterChecker {
	public int[] delimiter;
	int index;
	
	public DelimiterChecker(int[] delimiter) {
		this.delimiter = delimiter;
		this.index = 0; // reset()
	}
	
	public DelimiterChecker(byte[] delimiter) {
		this(byteToIntArray(delimiter));
	}
	public DelimiterChecker(String delimiter) {
		this(delimiter.getBytes());
	}
	
	public static int[] byteToIntArray(byte[] delimiter) {
		int[] intDelimiter = new int[delimiter.length];
		for (int i = 0; i< delimiter.length; i++) {
			intDelimiter[i] = (int) delimiter[i];
		}
		return intDelimiter;
	}
    
	public boolean nextByte(int incoming_byte ) {
	    // Check for image end
		if ( delimiter[index] == incoming_byte ) { // Candidate delimiter 
			index++;
		}
		else { // Reset
			reset();
		}
		if ( index == delimiter.length ) { // Confirmed delimiter
			reset();
			return true;
		}
		return false;
    }
	public boolean nextByte(int incoming_byte, String buffer ) {
		boolean result = nextByte(incoming_byte );
		if (result) {
			for (int c: delimiter)
				buffer += (char)c;
		}
		return result;
	}
	
	public void reset() {
		index = 0;
	}
}
