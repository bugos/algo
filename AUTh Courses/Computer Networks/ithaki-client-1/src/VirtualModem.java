import ithakimodem.*;
import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

public class VirtualModem {
	Modem modem;
	static final String ECHO_REQUEST_CODE        = "E2718";
	static final String IMAGE_REQUEST_CODE       = "M4270";
	static final String IMAGE_ERROR_REQUEST_CODE = "G3088";
	static final String GPS_REQUEST_CODE         = "P5114";
	static final String ACK_REQUEST_CODE         = "Q1076";
	static final String NACK_RESULT_CODE         = "R0012";
	DelimiterChecker nocarrier_delimiter   = new DelimiterChecker("NO CARRIER");
	DelimiterChecker start_echo_delimiter  = new DelimiterChecker("PSTART");
	DelimiterChecker end_echo_delimiter    = new DelimiterChecker("PSTOP");
	DelimiterChecker start_image_delimiter = new DelimiterChecker(new byte[]{ (byte)0xFF, (byte)0xD8 });
	DelimiterChecker end_image_delimiter   = new DelimiterChecker(new byte[]{ (byte)0xFF, (byte)0xD9 });
	DelimiterChecker start_gps_delimiter   = new DelimiterChecker("START ITHAKI GPS TRACKING\r\n");
	DelimiterChecker end_gps_delimiter     = new DelimiterChecker("STOP ITHAKI GPS TRACKING\r\n");
	DelimiterChecker gps_line_delimiter    = new DelimiterChecker(new byte[]{ (byte)0x0D, (byte)0x0A });
	DelimiterChecker GPGGA_delimiter       = new DelimiterChecker("$GPGGA");
			
    public static void main(String[] args) {
    	try {
    		new VirtualModem().demo();
    	}
		catch (Exception e) {
	    	e.printStackTrace();
		}
    }
    public void demo() throws Exception {
    	requestAndHandleResponse(ECHO_REQUEST_CODE);
//    	requestAndHandleResponse(IMAGE_REQUEST_CODE + "CAM=PTZ");  // FIX, PTZ, 01, 02, ...
//    	requestAndHandleResponse(IMAGE_ERROR_REQUEST_CODE);
    	requestAndHandleResponse(GPS_REQUEST_CODE + "R=1004090");
//    	requestAndHandleResponse(GPS_REQUEST_CODE + GPGGATrace.getGPSImageParameters() );
//    	getPackets(ECHO_REQUEST_CODE, 4 * 60, "echolog", () -> getLine(start_echo_delimiter, end_echo_delimiter).length());
//    	getPackets(ACK_REQUEST_CODE,  4 * 60, "arqlog",  () -> getArq());
//    	
    	modem.close();
    }
    VirtualModem() {
    	modem = new Modem();
    	modem.setSpeed(10000);
    	modem.setTimeout(2000);
    	modem.open("ithaki");
    }    
    private boolean modemWrite(String s) {
    	System.out.println("#Sent: " + s); 
    	return modem.write((s + "\r").getBytes());
    }
    
    /* Read from modem and retry if -1 until timeout */
    private byte modemRead() throws Exception {
    	int incoming_byte = modem.read(); // need int byte to check for -1
    	int tries = 0;
    	while ( incoming_byte == -1) {
	    	System.out.println("#Received -1");
	    	TimeUnit.SECONDS.sleep(1);
	    	
    		if ( 5 < tries++ ) {
    	    	System.out.println("#Retry limit reached");
    			throw new Exception();
    		}
    		incoming_byte = modem.read();
		}
		return (byte)incoming_byte;
    }

    void requestAndHandleResponse(String requestContent) throws Exception {
    	modemWrite(requestContent);
    	System.out.println("#Receiving...");
    	byte incoming_byte;
    	boolean resetDelimiters = true;
    	do {
    		incoming_byte = modemRead();
    		if (resetDelimiters) {
    	    	nocarrier_delimiter.reset();
    	    	start_echo_delimiter.reset();
    	    	start_image_delimiter.reset();
    	    	start_gps_delimiter.reset();
    		}
    		resetDelimiters = true; // will reset if something is found
		    if(nocarrier_delimiter.nextByte(incoming_byte))
		    	throw new Exception();
		    else if(start_echo_delimiter.nextByte(incoming_byte))
		    	System.out.println(getLine(start_echo_delimiter, end_echo_delimiter));
		    else if(start_image_delimiter.nextByte(incoming_byte))
		    	getImage();
		    else if(start_gps_delimiter.nextByte(incoming_byte))
		    	getGPS();
		    else
		    	resetDelimiters = false; //don't reset if delimiter not found.
		} while(!resetDelimiters); //temporary accept 1 respnse
		System.out.println("#Received -1 and stoped receiving.");
    }
    
    /* Read from modem and retry if -1 until timeout */
    private void getAndProcessBytes( DelimiterChecker endDelimiter, MyConsumer<Byte> block) throws Exception {
    	endDelimiter.reset();
    	byte incoming_byte;    	
    	do {
    		incoming_byte = modemRead();
    		block.accept(incoming_byte);
    	} while (!endDelimiter.nextByte(incoming_byte));
    }
    public interface MyConsumer<T> {
        void accept(T t) throws Exception;
    }
    
    String getLine(DelimiterChecker startDelimiter, DelimiterChecker endDelimiter) throws Exception {
    	StringBuilder result = new StringBuilder( new String(startDelimiter.delimiter) ); // not always needed!!!
    	getAndProcessBytes( endDelimiter, b -> result.append((char)b.byteValue()));
    	return result.toString();
    }
 
    void getGPS() throws Exception {
    	GPGGA_delimiter.reset();
    	getAndProcessBytes( end_gps_delimiter, b -> {
		    if(GPGGA_delimiter.nextByte(b)) { //just save line and check GGA,...
	    		GPGGATrace candidateTrace = new GPGGATrace(getLine(GPGGA_delimiter, gps_line_delimiter).trim().split(","));
	    		candidateTrace.addToImage();
	    		// other commands...
		    }
    	});	
    }
    
    private File getUniqueFileName( String prefix, String suffix) {
    	File file = null;
    	for (int i = 0; file == null || file.exists(); i++) //Calculate unique name
    		file = new File(prefix + i + suffix);
    	System.out.println(file);
    	return file;
    }

    void getImage() throws Exception {
    	FileOutputStream image_file = new FileOutputStream(getUniqueFileName("img", ".jpg"));
		image_file.write(start_image_delimiter.delimiter);
		getAndProcessBytes( end_image_delimiter, b -> image_file.write(b));
		image_file.close();
	}
    
    // Get ArqPacket and return tries count
    int getArq() throws Exception {
		int resends = 0;
		boolean verificationSuccess;
		do {
			// Parse
			String line = getLine(start_echo_delimiter, end_echo_delimiter);
			int packetStart = line.indexOf('<') + 1, packetEnd = line.indexOf('>');
			String packet = line.substring(packetStart, packetEnd);
			int FCS = Integer.parseInt(line.substring(packetEnd + 2, packetEnd + 5)); // FCS length=3
			
			// Error Check
			int result = 0;
			for (byte packetByte : packet.getBytes()) {
				result ^= packetByte;
			}
			verificationSuccess = (result == FCS);
			
			if (!verificationSuccess) {
				//System.out.println("Found ARQ error");
				modemWrite(NACK_RESULT_CODE);
				resends++;
			}
		} while (!verificationSuccess);
		//System.out.println("Successfully received package.");
		return resends;
    }

	//Implements packet reception for "sessionTime" seconds and saves results.
    private void getPackets( String requestCode, long duration, String logfilename, MySupplier<Integer> supplier) throws Exception {
		System.out.println("#Receiving Packets " + logfilename);
		FileOutputStream logfile = new FileOutputStream(getUniqueFileName(logfilename, ".txt")); 
		int packets = 0;
    	long startTime = System.currentTimeMillis(), elapsedTime;
		do {
			packets++;
			modemWrite(requestCode);
			long responseTime = System.currentTimeMillis();
			int resends = supplier.get();
			responseTime = System.currentTimeMillis() - responseTime;
			logfile.write((packets + "\t" + responseTime + "\t" + resends + "\r\n").getBytes());
	    	elapsedTime = (System.currentTimeMillis() - startTime) / 1000;
		} while ( duration > elapsedTime );
		System.out.println("#Finished receiving " + logfilename);
		logfile.close();
    }
    public interface MySupplier<T> {
        T get() throws Exception;
    }
}

class GPGGATrace {
	static int TRACE_TIME_DIFFERENCE = 8;
	static int TRACE_COUNT           = 10;
	Date timestamp;
	String latitude;
	String longitude;
	SimpleDateFormat GPSTimeFormat = new SimpleDateFormat("kkmmss.SSS");
	public static Vector<GPGGATrace> GPSTraces = new Vector<GPGGATrace>(TRACE_COUNT);

	GPGGATrace(String[] data) {
		try { timestamp = GPSTimeFormat.parse(data[1]);} 
			catch (ParseException e) {};
		latitude = data[2];
		longitude = data[4];
	}
	public void addToImage() { // Check TRACE_TIME_DIFFERENCE.
		if (GPSTraces.size() >= TRACE_COUNT)
			return;
		long GPSTimeDifference = Long.MAX_VALUE; // First element always accepted
		if (!GPSTraces.isEmpty()) {
			GPSTimeDifference = this.timestamp.getTime() - GPSTraces.lastElement().timestamp.getTime();
			GPSTimeDifference /= 1000.;
		}
    	if ( TRACE_TIME_DIFFERENCE <= GPSTimeDifference ) { // convert to seconds and compare
    		System.out.println(latitude + " " + timestamp);
    		GPSTraces.add(this);
    	}
	}

	/* Convert the coordinates from DDMM.MMMM form to DDMMSS*/
	public static String getDegreesMinutesSeconds(String coordinate) {
		int minutesLength = 2;
		int degreesLength = coordinate.indexOf(".") - minutesLength; // 2 or 3
		String degrees = coordinate.substring(0, degreesLength);
		String minutes = coordinate.substring(degreesLength, degreesLength + minutesLength);
		int minutesDecimalPart = Integer.parseInt(coordinate.substring(degreesLength + minutesLength + 1));
		String seconds = String.format("%02d", Math.round(minutesDecimalPart * 60 / 10000.)); // 10000 for 4digit minutesDecimalPart
		degrees = String.format("%02d", Integer.parseInt(degrees));
		System.out.println(coordinate + " " + minutesDecimalPart + " " + seconds + " " + minutesDecimalPart * 60 / 10000.);
		return degrees + minutes + seconds; //DDMMSS
	}
	
    public static String getGPSImageParameters() {
    	String parameters = new String();
    	for (GPGGATrace trace: GPSTraces) {
    		System.out.println("ΜΟΧΟΣ " + trace.longitude + " " + trace.latitude  );
    		parameters += "T=" + getDegreesMinutesSeconds(trace.longitude) + getDegreesMinutesSeconds(trace.latitude);
    	}
    	return parameters;
    }
}

class DelimiterChecker {
	public byte[] delimiter;
	int index;
	
	public DelimiterChecker(byte[] delimiter) {
		this.delimiter = delimiter;
		this.index = 0; // reset()
	}
	public DelimiterChecker(String delimiter) {
		this(delimiter.getBytes());
	}
	public boolean nextByte(int incoming_byte ) {
		if ( delimiter[index] == incoming_byte ) // Candidate delimiter 
			index++;
		else
			reset();
		if ( index == delimiter.length ) { // Confirmed delimiter
			reset(); //remove
			return true;
		}
		return false;
    }
	public void reset() {
		index = 0;
	}
}
