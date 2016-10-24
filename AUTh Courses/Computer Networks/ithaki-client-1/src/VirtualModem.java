import ithakimodem.*;

import java.io.*;
import java.util.Arrays;

public class VirtualModem {
	int speed = 10000;
	int timeout = 2000;

	String server_name        = "ithaki";
	String echo_request_code  = "E9256\r";
	String image_request_code = "M2373\r";
	String gps_request_code   = "E4541\r";
	
    public static void main(String[] args) {
    	new VirtualModem().demo();
    }
    
    public void demo() throws IOException {
    	Modem modem = new Modem();
    	modem.setSpeed(speed);
    	modem.setTimeout(timeout);
    	modem.open(server_name);
    	echoRequest(modem);
    	imageRequest(modem);
    	modem.close();
    }
    // NOTE : Break endless loop by catching sequence "\r\n\n\n".
    // NOTE : Stop program execution when "NO CARRIER" is detected.
    // NOTE : A time-out option will enhance program behavior.
    // NOTE : Continue with further Java code.
    private void echoRequest(Modem modem) {
    	modem.write(echo_request_code.getBytes());
//    	System.out.println(Arrays.toString(echo_request_code.getBytes())); 
    	
    	int incoming_byte;
    	while( true ) {
    		try {
	    		incoming_byte = modem.read();
	    		if ( incoming_byte == -1) {
	    	    	System.out.println("#Received -1"); 
	    			break;
	    		}
	    		
	    		System.out.print((char) incoming_byte);
    		}
    		catch (Exception e) {
    	    	System.out.println("#Exception while reading."); 
    			break;
    		}
    	}	
    }

    private void imageRequest(Modem modem) {
    	modem.write(image_request_code.getBytes());
//    	System.out.println(Arrays.toString(echo_request_code.getBytes())); 
		
    	OutputStream image_file = null;
    	try {
    		image_file = new FileOutputStream("img.jpg");
        	
    		boolean receiving_image = false;
	    	int incoming_byte;
	    	while( true ) {
	    		try {
		    		incoming_byte = modem.read();
		    		if ( incoming_byte == -1) {
		    	    	System.out.println("#Received -1"); 
		    			break;
		    		}
		    		
		    		if (incoming_byte == 0xFF)
		    			receiving_image = true;
		    		
	    		    image_file.write(incoming_byte);
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
    
    private String getByteResponse() {
		return echo_request_code;
    	
    }
}
