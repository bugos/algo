import ithakimodem.*;

import java.io.*;
import java.util.concurrent.TimeUnit;

public class VirtualModem {
	int speed = 10000;
	int timeout = 2000;

	String server_name        = "ithaki";
	String echo_request_code  = "E2649\r";
	String image_request_code = "M2516\r";
	String gps_request_code   = "E4541\r";
	long connection_timeout = 20000;
	int[] start_image_delimiter = { 0xFF, 0xD8 };
	int[] end_image_delimiter = { 0xFF, 0xD9 };
	
    public static void main(String[] args) {
    	new VirtualModem().demo();
    }
    
    public void demo() {
    	Modem modem = new Modem();
    	modem.setSpeed(speed);
    	modem.setTimeout(timeout);
    	modem.open(server_name);
    	
    	//echoRequest(modem);
    	//imageRequest(modem);
    	
    	modem.write(echo_request_code.getBytes());
    	modem.write(image_request_code.getBytes());
    	receiveResponse(modem);
    	
    	modem.close();
    }
    // NOTE : Break endless loop by catching sequence "\r\n\n\n".
    // NOTE : Stop program execution when "NO CARRIER" is detected.
    // NOTE : A time-out option will enhance program behavior.
    // NOTE : Continue with further Java code.
    private void receiveResponse(Modem modem) {
    	System.out.println("#Receiving...");
    	int incoming_byte;
    	long startTime = System.currentTimeMillis(), elapsedTime, lastActiveTime = startTime;
    	int image_delimeter_index = 0;
    	while( true ) {
    		try {	
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
	    		

	    		// Check for image
	    		if ( start_image_delimiter[image_delimeter_index] == incoming_byte ) { // Candidate delimiter 
	    			System.out.println("#candidate img"); 
	    			image_delimeter_index++;
	    		}
	    		else { // Reset
	    			image_delimeter_index = 0;
	    		}
	    		if ( image_delimeter_index == start_image_delimiter.length ) { //Confirmed delimiter
	    			System.out.println("#confirmed img"); 
	    			getImage(modem);
	    		}
	    		lastActiveTime = System.currentTimeMillis();
	    		
    		}
    		catch (Exception e) {
    	    	System.out.println("#Exception while reading."); 
    			break;
    		}
    	}	
    }
    
    private void getImage(Modem modem) {
    	File img = null;
    	OutputStream image_file = null;
    	try {
        	for (int i = 0; img == null || img.exists(); i++) { //Calculate unique name
        		img = new File("img" + i + ".jpg");
        		System.out.println("img.jpg"+i); 
        	}
    		image_file = new FileOutputStream(img);
        	
    		// Write delimiter bytes
    		for (int delimiter_byte : start_image_delimiter)
    			image_file.write(delimiter_byte);
    		
    		// Get image bytes
    		int image_delimeter_index = 0;
	    	int incoming_byte;
	    	while( true ) {
	    		try {
		    		incoming_byte = modem.read();
		    		if ( incoming_byte == -1) {
		    	    	System.out.println("#Received -1"); 
		    			break;
		    		}
		    		
	    		    image_file.write(incoming_byte);
		    		
	    		    // Check for image end
		    		if ( end_image_delimiter[image_delimeter_index] == incoming_byte ) { // Candidate delimiter 
		    			image_delimeter_index++;
		    		}
		    		else { // Reset
		    			image_delimeter_index = 0;
		    		}
		    		if ( image_delimeter_index == end_image_delimiter.length ) { //Confirmed delimiter
		    			System.out.println("#confirmed img end"); 
		    			break;
		    		}
	    		}
	    		catch (Exception e) {
	    	    	System.out.println("#Exception while reading."); 
	    			break;
	    		} 
	    	}	
	    }
    	catch (IOException e) {
	    	System.out.println("#IOException"); 
	    	e.printStackTrace();
    	}
    	finally {
			try {
				if (image_file != null)
					image_file.close();
			}
			catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
    
}
