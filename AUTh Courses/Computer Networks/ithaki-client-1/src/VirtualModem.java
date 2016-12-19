import ithakimodem.*;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class VirtualModem {
	Modem modem;
	static final long   BUFFER_TIMEOUT           = 4000;
	static final int    ARQ_DURATION             = 4 * 60; // seconds
	static final int    ECHO_DURATION            = 4 * 60; // seconds	
	static final int    ARQ_TIMEOUT              = 2;
	static final String REQUEST_ENDLINE          = "\r";
	static final String SERVER_NAME              = "ithaki";
	static final String ECHO_REQUEST_CODE        = "E4672";
	static final String IMAGE_REQUEST_CODE       = "M0747";
	static final String IMAGE_ERROR_REQUEST_CODE = "G8964";
	static final String GPS_REQUEST_CODE         = "P1185";
	static final String ACK_REQUEST_CODE         = "Q8820";
	static final String NACK_RESULT_CODE         = "R2996";
	DelimiterChecker nocarrier_delimiter   = new DelimiterChecker("NO CARRIER");
	DelimiterChecker start_echo_delimiter  = new DelimiterChecker("PSTART");
	DelimiterChecker end_echo_delimiter    = new DelimiterChecker("PSTOP");
	DelimiterChecker start_image_delimiter = new DelimiterChecker(new int[]{ 0xFF, 0xD8 });
	DelimiterChecker end_image_delimiter   = new DelimiterChecker(new int[]{ 0xFF, 0xD9 });
	DelimiterChecker start_gps_delimiter   = new DelimiterChecker("START ITHAKI GPS TRACKING\r\n");
	DelimiterChecker end_gps_delimiter     = new DelimiterChecker("STOP ITHAKI GPS TRACKING\r\n");
	DelimiterChecker gps_line_delimiter    = new DelimiterChecker(new int[]{ 0x0D, 0x0A });
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
    	request(ECHO_REQUEST_CODE);
    	//request(IMAGE_REQUEST_CODE + "CAM=PTZ");  // FIX, PTZ, 01, 02, ...
    	//request(IMAGE_ERROR_REQUEST_CODE);
    	//request(GPS_REQUEST_CODE + "R=1004090");
    	//request(GPS_REQUEST_CODE + GPGGATrace.getGPSImageParameters() );
    	//getEchoPackets();
//    	getArqPackets();

    	modem.close();
    }
    
    VirtualModem() {
    	modem = new Modem();
    	modem.setSpeed(10000);
    	modem.setTimeout(4000);
    	modem.open(SERVER_NAME);
    }    
    
    private boolean modemWrite(String s) {
    	//System.out.println("#Sent: " + s); 
    	return modem.write((s + REQUEST_ENDLINE).getBytes());
    }
    
    /* Read from modem and retry if -1 until timeout */
    private int modemRead() throws Exception {
    	int incoming_byte = modem.read();
    	long startTime = System.currentTimeMillis(), elapsedTime;
		while ( incoming_byte == -1) {
	    	System.out.println("#Received -1");
	    	TimeUnit.SECONDS.sleep(1);
	    	
			// Timeout timer
    		elapsedTime = System.currentTimeMillis() - startTime;
    		if ( BUFFER_TIMEOUT < elapsedTime ) {
    	    	System.out.println("#Timeout reached");
    			throw new Exception();
    		}
    		incoming_byte = modem.read();
		}
		return incoming_byte;
    }

    /* Read from modem and retry if -1 until timeout */
    private void getAndProcessBytes( DelimiterChecker endDelimiter, Consumer<Integer> block) throws Exception {
    	endDelimiter.reset();
    	int incoming_byte;    	
    	do {
    		incoming_byte = modemRead();
    		block.accept(incoming_byte);
    	} while (!endDelimiter.nextByte(incoming_byte));
    }
    
    // NOTE : Break endless loop by catching sequence "\r\n\n\n".
    void request(String requestContent) throws Exception {
    	modemWrite(requestContent);

    	System.out.println("#Receiving...");
    	int incoming_byte;
    	boolean resetDelimiters = true;
    	do {
    		incoming_byte = modem.read();

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
		} while( incoming_byte != -1 ); // fix
		System.out.println("#Received -1 and stoped receiving.");

    }
    
    String getLine(DelimiterChecker startDelimiter, DelimiterChecker endDelimiter) throws Exception {
    	StringBuilder result = new StringBuilder( startDelimiter.getString() ); // not always needed!!!
    	getAndProcessBytes( endDelimiter, b -> result.append((char)b.intValue()));
    	return result.toString();
    }
 
    void getGPS() throws Exception {
    	int incoming_byte;    	
    	do {
    		incoming_byte = modemRead();
    	 
		    if(GPGGA_delimiter.nextByte(incoming_byte)) { //just save line and check GGA,...
	    		GPGGATrace candidateTrace = new GPGGATrace(getLine(GPGGA_delimiter, gps_line_delimiter).trim().split(","));
	    		candidateTrace.addToImage();
		    }
    		// other commands...
    	} while (!end_gps_delimiter.nextByte(incoming_byte));	
    }
    
    File getUniqueFileName( String prefix, String suffix) {
    	File file = null;
    	for (int i = 0; file == null || file.exists(); i++) //Calculate unique name
    		file = new File(prefix + i + suffix);
    	System.out.println(file);
    	return file;
    }
    
    void getImage() throws Exception {
    	OutputStream image_file = null;
    	try {
    		image_file = new FileOutputStream(getUniqueFileName("img", ".jpg"));
        	
    		// Write delimiter bytes
    		image_file.write(start_image_delimiter.getBytes());
    		
    		// Get image bytes
	    	int incoming_byte;
	    	end_image_delimiter.reset();
	    	do {
	    		incoming_byte = modemRead();
    		    image_file.write((byte)incoming_byte);
	    		
	    	} while(!end_image_delimiter.nextByte(incoming_byte));
	    }
    	finally {
			if (image_file != null)
					image_file.close();
		}
	}
    
    // Get ArqPacket and return tries count
    int getArq() throws Exception {
		int resends = 0;
		boolean verificationSuccess;
    	long startTime = System.currentTimeMillis(), elapsedTime;
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

	    	elapsedTime = (System.currentTimeMillis() - startTime) / 1000;
	    	if (ARQ_TIMEOUT < elapsedTime) {
	    		//System.out.println("ARQ Timeout Reached");
	    		return -1;
	    	}
			
			if (!verificationSuccess) {
				//System.out.println("Found ARQ error");
				modemWrite(NACK_RESULT_CODE);
				resends++;
			}

		} while (!verificationSuccess);
		//System.out.println("Successfully received package.");
		return resends;
    }
    
	//Implements the ARQ packet reception for "sessionTime" seconds and saves results.
    void getArqPackets() throws Exception {
		System.out.println("#Receiving ARQ");
		FileOutputStream logfile = new FileOutputStream(getUniqueFileName("arqlog", ".txt")); 
		int packets = 0;
    	long startTime = System.currentTimeMillis(), elapsedTime;
		do {
			packets++;
			modemWrite(ACK_REQUEST_CODE);
			long errorTime = System.currentTimeMillis();
			int resends = getArq();
			errorTime = System.currentTimeMillis() - errorTime;
			logfile.write((packets + "\t" + errorTime + "\t" + resends + "\r\n").getBytes());
	    	elapsedTime = (System.currentTimeMillis() - startTime) / 1000;
		} while ( ARQ_DURATION > elapsedTime );
		System.out.println("#Finished receiving ARQ");
		logfile.close();
    }
    
	//Implements the simple packet reception for "sessionTime" seconds and saves results.
	void getEchoPackets() throws Exception {
		System.out.println("#Receiving Echo");
		FileOutputStream logfile = new FileOutputStream(getUniqueFileName("echolog", ".txt")); 
		int packets = 0;
    	long startTime = System.currentTimeMillis(), elapsedTime;
		do  {
			packets++;
			modemWrite(ECHO_REQUEST_CODE);
			long echoTime = System.currentTimeMillis();
			getLine(start_echo_delimiter, end_echo_delimiter);
			echoTime = System.currentTimeMillis() - echoTime;
			logfile.write((packets + "\t" + echoTime + "\r\n").getBytes());
	    	elapsedTime = (System.currentTimeMillis() - startTime) / 1000;
		} while ( ECHO_DURATION > elapsedTime);
		System.out.println("#Finished receiving Echo");
		logfile.close();
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

	
	GPGGATrace(String[] data) throws ParseException {
		timestamp = GPSTimeFormat.parse(data[1]);
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
	    	System.out.println("#found trace");
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