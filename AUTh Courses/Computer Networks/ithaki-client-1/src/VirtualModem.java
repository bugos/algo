import ithakimodem.*;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

public class VirtualModem {
	int speed = 10000;
	int timeout = 2000;

	Modem modem;
	long connection_timeout = 2000;
	String imageParameters = "CAM=PTZ"; // FIX, PTZ, 01, 02, ...
	int arqDuration = 4; // seconds
	
	static final String REQUEST_ENDLINE          = "\r";
	static final String SERVER_NAME              = "ithaki";
	static final String ECHO_REQUEST_CODE        = "E1170";
	static final String IMAGE_REQUEST_CODE       = "M5105"; 
	static final String IMAGE_ERROR_REQUEST_CODE = "G4980";
	static final String GPS_REQUEST_CODE         = "P9275";
	static final String ACK_REQUEST_CODE         = "G4379";
	static final String NACK_RESULT_CODE         = "G4379";
	DelimiterChecker nocarrier_delimiter = new DelimiterChecker("NO CARRIER");
	DelimiterChecker start_echo_delimiter = new DelimiterChecker("PSTART");
	DelimiterChecker end_echo_delimiter = new DelimiterChecker("PSTOP");
	DelimiterChecker start_image_delimiter = new DelimiterChecker(new int[]{ 0xFF, 0xD8 });
	DelimiterChecker end_image_delimiter = new DelimiterChecker(new int[]{ 0xFF, 0xD9 });
	DelimiterChecker start_gps_delimiter = new DelimiterChecker("START ITHAKI GPS TRACKING\r\n");
	DelimiterChecker end_gps_delimiter = new DelimiterChecker("STOP ITHAKI GPS TRACKING\r\n");
	DelimiterChecker gps_line_delimiter = new DelimiterChecker(new int[]{ 0x0D, 0x0A });
	DelimiterChecker GPGGA_delimiter = new DelimiterChecker("$GPGGA");
	
			
    public static void main(String[] args) {
    	try {
    		new VirtualModem().demo();
    	}
		catch (Exception e) { //change general exc
	    	System.out.println("#Exception while reading."); 
	    	e.printStackTrace();
		}
    }
    
    public void demo() throws Exception {
    	request(ECHO_REQUEST_CODE);
    	request(IMAGE_REQUEST_CODE + imageParameters);
    	request(GPS_REQUEST_CODE + "R=1000140");
    	request(GPS_REQUEST_CODE + GPGGATrace.getGPSImageParameters() );

    	modem.close();
    }
    
    VirtualModem() {
    	modem = new Modem();
    	modem.setSpeed(speed);
    	modem.setTimeout(timeout);
    	modem.open(SERVER_NAME);
    }    
    
    private boolean modemWrite(String s) {
    	System.out.println("#Sent: " + s); 
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
    		if ( connection_timeout < elapsedTime ) {
    	    	System.out.println("#Timeout reached");
    			throw new Exception();
    		}
		}
		return incoming_byte;
    }
    
    // NOTE : Break endless loop by catching sequence "\r\n\n\n".
    private void request(String requestContent) throws Exception {
    	modemWrite(requestContent);
    	
    	System.out.println("#Receiving...");
    	int incoming_byte;
    	nocarrier_delimiter.reset();
    	start_echo_delimiter.reset();
    	start_image_delimiter.reset();
    	start_gps_delimiter.reset();
    	do {
    		incoming_byte = modem.read();

		    if(nocarrier_delimiter.nextByte(incoming_byte))
		    	throw new Exception();
		    else if(start_echo_delimiter.nextByte(incoming_byte))
		    	System.out.println(getEcho());
		    else if(start_image_delimiter.nextByte(incoming_byte))
		    	getImage();
		    else if(start_gps_delimiter.nextByte(incoming_byte))
		    	getGPS();
		    else
		    	continue; //only reset if delimiter found.
	    	nocarrier_delimiter.reset();
	    	start_echo_delimiter.reset();
	    	start_image_delimiter.reset();
	    	start_gps_delimiter.reset();
		} while( incoming_byte != -1 );
    }
    
    private String getEcho() throws Exception {
    	String result = new String();
		for (int delimiter_byte : start_echo_delimiter.delimiter)
			result += delimiter_byte;
    	int incoming_byte;    	
    	do {
    		incoming_byte = modemRead();
    		result += (char) incoming_byte;
    	} while (!end_echo_delimiter.nextByte(incoming_byte));
    	return result;
    }
   
    private void getGPS() throws Exception {
    	int incoming_byte;    	
    	do {
    		incoming_byte = modemRead();
    	 
		    if(GPGGA_delimiter.nextByte(incoming_byte)) { //just save line and check GGA,...
	    		GPGGATrace candidateTrace = new GPGGATrace(getGPGGA().trim().split(","));
	    		candidateTrace.addToImage();
		    }
    		// other commands...
    	} while (!end_gps_delimiter.nextByte(incoming_byte));	
    }
    
    private String getGPGGA() throws Exception {
    	String lineBuffer = new String(GPGGA_delimiter.getString());
    	
	    int incoming_byte;
    	gps_line_delimiter.reset();
    	do {
    		incoming_byte = modemRead();
    		lineBuffer += (char)incoming_byte;
    	} while(!gps_line_delimiter.nextByte(incoming_byte));
    	return lineBuffer;
    }
    
    private void getImage() throws Exception {
    	OutputStream image_file = null;
    	try {
        	File img = null;
        	for (int i = 0; img == null || img.exists(); i++)//Calculate unique name
        		img = new File("img" + i + ".jpg");
    		image_file = new FileOutputStream(img);
        	
    		// Write delimiter bytes
    		for (int delimiter_byte : start_image_delimiter.delimiter)
    			image_file.write(delimiter_byte);
    		
    		// Get image bytes
	    	int incoming_byte;
	    	end_image_delimiter.reset();
	    	do {
	    		incoming_byte = modemRead();
    		    image_file.write(incoming_byte);
	    		
	    	} while(!end_image_delimiter.nextByte(incoming_byte));
	    }
    	finally {
			if (image_file != null)
					image_file.close();
		}
	}
    
    // Get ArqPacket and return tries count
    private int getArq() throws Exception {
		boolean foundError = true;
		int resends = 0;
		do {
			// Parse
			String line = getEcho();
			int packetStart = line.indexOf('<') + 1, packetEnd = line.indexOf('>');
			String packet = line.substring(packetStart, packetEnd);
			int FCS = Integer.parseInt(line.substring(packetEnd + 2, packetEnd + 5)); // FCS length=3
			
			// Error Check
			int result = 0;
			for (byte packetByte : packet.getBytes()) {
				result ^= packetByte;
			}
			foundError = (result == FCS);
			
			if (foundError) {
				request(NACK_RESULT_CODE);
				resends++;
			}
		} while (foundError);
		return resends;
    }
    
    private void requestArq() throws Exception {
		System.out.println("Receiving ARQ");
    	long startTime = System.currentTimeMillis(), elapsedTime;
		do {
			modemWrite(ACK_REQUEST_CODE);
			long errorTime = System.currentTimeMillis();
			int resends = getArq();
			errorTime = errorTime - System.currentTimeMillis();
	    	//write to file
			
	    	elapsedTime = System.currentTimeMillis() - startTime;
		} while ( arqDuration > elapsedTime );
    }
}

class GPGGATrace {
	Date timestamp;
	String latitude;
	String longitude;
	SimpleDateFormat GPSTimeFormat = new SimpleDateFormat("kkmmss.SSS");
	public static Vector<GPGGATrace> GPSTraces = new Vector<GPGGATrace>(4);
	static int TRACE_TIME_DIFFERENCE = 20;
	static int TRACE_COUNT = 7;
	
	GPGGATrace(String[] data) throws ParseException {
		timestamp = GPSTimeFormat.parse(data[1]);
		latitude = data[2];
		longitude = data[4];
	}
	
	public void addToImage() {
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