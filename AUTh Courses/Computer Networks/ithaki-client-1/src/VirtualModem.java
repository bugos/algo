import ithakimodem.*;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

class GPGGATrace {
	Date timestamp;
	String latitude;
	String longitude;
	SimpleDateFormat GPSTimeFormat = new SimpleDateFormat("kkmmss.SSS");
	
	GPGGATrace(String[] data) {
		try {
			timestamp = GPSTimeFormat.parse(data[1]);
		} catch (ParseException e) {}
		latitude = data[2].replace(".", "").substring(0, 5);
		longitude = data[4].replace(".", "").substring(0, 5);;
	}
	
	public static String getDegreesMinutesSeconds(String coordinate) {
		String degrees = coordinate.substring(0, 2);
		String minutes = coordinate.substring(2, 4);
		int minutesDecimalPart = Integer.parseInt(coordinate.substring(5)); // to seconds
		int intSeconds = minutesDecimalPart * 60 / 10000; // 10000 for 4digit minutesDecimalPart
		String seconds = String.format("%02d", intSeconds);
		System.out.println(coordinate + " " + minutesDecimalPart + " " + seconds );

		return degrees + minutes + seconds; //DDMMSS
	}
    public static String getGPSImageParameters(Vector<GPGGATrace> GPSTraces) {
    	String parameters = new String();
    	for (GPGGATrace trace: GPSTraces) {
    		parameters += "T=" + getDegreesMinutesSeconds(trace.longitude) + getDegreesMinutesSeconds(trace.latitude);
    	}
    	return parameters;
    }
}

public class VirtualModem {
	int speed = 10000;
	int timeout = 2000;

	Modem modem;
	long connection_timeout = 20000;
	String endline = "\r";
	String imageParameters = "CAM=PTZ"; // FIX, PTZ, 01, 02, ...
	String SERVER_NAME              = "ithaki";
	String ECHO_REQUEST_CODE        = "E2693";
	String IMAGE_REQUEST_CODE       = "M4806"; 
	String IMAGE_ERROR_REQUEST_CODE = "M5525";
	String GPS_REQUEST_CODE         = "G4379";
	String ACK_REQUEST_CODE         = "G4379";
	String NACK_REQUEST_CODE        = "G4379";
	DelimiterChecker start_image_delimiter = new DelimiterChecker(new int[]{ 0xFF, 0xD8 });
	DelimiterChecker end_image_delimiter = new DelimiterChecker(new int[]{ 0xFF, 0xD9 });
	DelimiterChecker start_gps_delimiter = new DelimiterChecker("START ITHAKI GPS TRACKING\r\n");
	DelimiterChecker gps_line_delimiter = new DelimiterChecker(new int[]{ 0x0D, 0x0A });
	DelimiterChecker end_gps_delimiter = new DelimiterChecker("STOP ITHAKI GPS TRACKING\r\n");
	
	Vector<GPGGATrace> GPSTraces = new Vector<GPGGATrace>(4);
	double traceTimeDifference = 4;
			
    public static void main(String[] args) {
    	new VirtualModem().demo();
    }
    
    public void demo() {
    	modemWrite(ECHO_REQUEST_CODE);
    	//modemWrite(image_request_code + imageParameters);
    	modemWrite(GPS_REQUEST_CODE + "R=1000140");
    	//modemWrite(gps_request_code + GPGGATrace.getGPSImageParameters(GPSTraces) );
    	receiveResponse();
    	
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
    	return modem.write((s + endline).getBytes());
    }
    private int modemRead() {
    	return modem.read();
//		if ( incoming_byte == -1) {
//	    	System.out.println("#Received -1"); 
//			throw ;
//		}
    }
    
    // NOTE : Break endless loop by catching sequence "\r\n\n\n".
    // NOTE : Stop program execution when "NO CARRIER" is detected.
    // NOTE : A time-out option will enhance program behavior.
    private void receiveResponse() {
    	System.out.println("#Receiving...");
    	int incoming_byte;
    	long startTime = System.currentTimeMillis(), elapsedTime, lastActiveTime = startTime;
    	start_image_delimiter.reset();
    	start_gps_delimiter.reset();
		try {	
	    	while( true ) {
				// Timeout timer
	    		elapsedTime = System.currentTimeMillis() - lastActiveTime;
	    		if ( elapsedTime > connection_timeout ) {
	    	    	System.out.println("#Timeout reached");
	    	    	break;
	    		}
	    		
	    		// Read and print
	    		incoming_byte = modem.read();
	    		if ( incoming_byte == -1) {
	    	    	System.out.println("#Received -1");
	    	    	TimeUnit.SECONDS.sleep(1);
	    			continue;
	    		}
	    		
	    		System.out.print((char) incoming_byte);
	    		
			    if(start_image_delimiter.nextByte(incoming_byte))
			    	getImage();
			    if(start_gps_delimiter.nextByte(incoming_byte))
			    	getGPS();
	    		lastActiveTime = System.currentTimeMillis();
    		}

    	}	
		catch (Exception e) {
	    	System.out.println("#Exception while reading."); 
		}
    }
    
    private void getGPS() {
		// Get gps bytes
    	int incoming_byte;
    	DelimiterChecker end_gps_delimiter = new DelimiterChecker("START ITHAKI GPS TRACKING\r\n");
    	DelimiterChecker GPGGA_delimiter = new DelimiterChecker("$GPGGA");
    	do {
    		incoming_byte = modem.read();
    		if ( incoming_byte == -1) {
    	    	System.out.println("#Received -1"); 
    			break;
    		}
    		
    	    String lineBuffer = new String();
     		
		    if(GPGGA_delimiter.nextByte(incoming_byte, lineBuffer)) { //just save line and check GGA,...
		    	getGPGGA(lineBuffer);
	    		GPGGATrace candidateTrace = new GPGGATrace(lineBuffer.trim().split(","));
		    	long GPSTimeDifference = candidateTrace.timestamp.getTime() - GPSTraces.lastElement().timestamp.getTime();
		    	if ( GPSTraces.isEmpty() || traceTimeDifference <= GPSTimeDifference / 4. ) { // convert to seconds and compare
		    		GPSTraces.add(candidateTrace);
		    	}
		    }
    		// other commands...
    		
    	} while (!end_gps_delimiter.nextByte(incoming_byte));	
    }
    
    private void getGPGGA(String lineBuffer) {
	    int incoming_byte;
    	DelimiterChecker gps_line_delimiter = new DelimiterChecker(new int[]{ 0x0D, 0x0A });
    	do {
    		incoming_byte = modem.read();
    		if ( incoming_byte == -1) {
    	    	System.out.println("#Received -1"); 
    			break;
    		}
    		
    		lineBuffer += (char)incoming_byte;
    	} while(!gps_line_delimiter.nextByte(incoming_byte));
    }
    
    private void getImage() throws IOException {
    	File img = null;
    	OutputStream image_file = null;
    	try {
        	for (int i = 0; img == null || img.exists(); i++) { //Calculate unique name
        		img = new File("img" + i + ".jpg");
        		System.out.println("img.jpg"+i); 
        	}
    		image_file = new FileOutputStream(img);
        	
    		// Write delimiter bytes
    		for (int delimiter_byte : start_image_delimiter.delimiter)
    			image_file.write(delimiter_byte);
    		
    		// Get image bytes
	    	int incoming_byte;
	    	end_image_delimiter.reset();
	    	while( true ) {
	    		incoming_byte = modem.read();
	    		if ( incoming_byte == -1) {
	    	    	System.out.println("#Received -1"); 
	    			break;
	    		}
	    		
    		    image_file.write(incoming_byte);
	    		
    		    if(end_image_delimiter.nextByte(incoming_byte))
    		    	break;  
	    	}	
	    }
    	finally {
			if (image_file != null)
					image_file.close();
		}
	}
}

