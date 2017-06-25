import java.net.*;   //network resources
import java.io.*;    //computer resources
import javax.sound.sampled.*; //sound
import java.util.ArrayList;
import java.util.Arrays;

class MultimediaApp {
    static final int    SERVER_PORT      =  38015 ;
    static final int    CLIENT_PORT      =  48015 ;
	static final String ECHO_0_REQ       = "E0000";
	static final String ECHO_REQ         = "E5851";
	static final String IMAGE_REQ        = "M3782";
    static final String AUDIO_REQ        = "A8881";
    static final String ITHAKICOPTER_REQ = "Q8273";
    static final String VEHICLE_REQ      = "V9514";
    static final int    N_AUDIO_PACKETS  =  200   ; // 1-999
    static final int    PACKET_SIZE      =  128   ; // bytes
    static final int    LOGGING_TIME     =  4     ; // min
    static final String SERVER_IP        = "155.207.18.208";  

    byte[] rxbuffer = new byte[ PACKET_SIZE ], txbuffer;
    static InetAddress HOST_ADDRESS;
    DatagramSocket txSocket = new DatagramSocket();
	DatagramSocket rxSocket = new DatagramSocket( CLIENT_PORT );
    DatagramPacket txPacket, rxPacket = new DatagramPacket( rxbuffer, rxbuffer.length );
	
    public MultimediaApp() throws SocketException, UnknownHostException {
//        rxSocket.setSoTimeout( 10000 ); 
        HOST_ADDRESS = InetAddress.getByName(SERVER_IP);
    }
    
    void demo() throws Exception {
//    	getEcho( ECHO_REQ, LOGGING_TIME );
//    	getEcho( ECHO_0_REQ, LOGGING_TIME );
//    	getEcho( ECHO_REQ + "T00", 0 );
//    	getImage( IMAGE_REQ + "UDP=128" + "CAM=FIX" ); // 128, 256, 512, 1024
//    	getImage( IMAGE_REQ + "UDP=128" + "CAM=PTZ" );
    	playMusic(getDPCM(   AUDIO_REQ +      "T" + N_AUDIO_PACKETS ), 8 ); // Q=8,16, T=freq
//        playMusic(getDPCM(   AUDIO_REQ +   "L08F" + N_AUDIO_PACKETS ), 8 ); // L=track,F=audio
//        playMusic(getAQDPCM( AUDIO_REQ + "AQL08F" + N_AUDIO_PACKETS ), 16 );
//		ithakiCopter( ITHAKICOPTER_REQ );
//    	getCarDiagnostics(VEHICLE_REQ);
		
    }
    
    public void getEcho( String code, int duration ) throws IOException  {
    	this.setTxData( code );
        long startTime = System.currentTimeMillis(), responseTime;
        double elapsedTime;
        
        final int bitsPerPacket = 32 * 8; //32 chars in echo
        int[] packetsRxInSecond = new int[ duration * 60 + 4 ];
        int packetsRxInTimeframe[] = new int[64], currentSecond = 1;
        
        System.out.println( "Echoing in file " + code + ".txt" );
    	FileOutputStream echoLog = new FileOutputStream(getUniqueFileName(code, ".txt"));
    	FileOutputStream throughputLog = new FileOutputStream(getUniqueFileName(code + "-t", ".txt"));
        do {
        	// Receive echo
            txSocket.send( txPacket );
    		responseTime = System.currentTimeMillis();
			rxSocket.receive( rxPacket );
            responseTime = System.currentTimeMillis() - responseTime;
            elapsedTime = (System.currentTimeMillis() - startTime) / 1000.; //seconds
            echoLog.write( (elapsedTime +  "\t" + responseTime + "\n").getBytes() );

//            String message = new String( this.rxbuffer );
//            System.out.println(message);
            
            // Calculate Throughput
            packetsRxInSecond[ (int)elapsedTime + 1 ]++; // start from second 1
            if( elapsedTime > currentSecond ) { // Changed second. Calculate throughput
            	throughputLog.write(  (currentSecond + "\t").getBytes() );
            	for(int timeframeSeconds : new int[]{8, 16, 32}) {
	            	int timeframe = Math.min(currentSecond, timeframeSeconds);
	            	packetsRxInTimeframe[timeframeSeconds] -= packetsRxInSecond[ currentSecond - timeframe ];
	            	packetsRxInTimeframe[timeframeSeconds] += packetsRxInSecond[ currentSecond ]; // Add right part

	            	float throughput = bitsPerPacket * packetsRxInTimeframe[timeframeSeconds] / (float)timeframe; 
//            		System.out.println(" " + bitsPerPacket + " "+ packetsReceivedInTimeframe +" " +timeframe);
	            	throughputLog.write( (throughput + "\t").getBytes() );
            	}
            	throughputLog.write(  "\n".getBytes() );
            	currentSecond++; // Move to the next second
            }
                
        } while ( elapsedTime < duration * 60 );
        echoLog.close();
        throughputLog.close();
    }
    public void getImage( String code ) throws IOException {
    	setTxData(code);
        System.out.println( "Image will be saved on file " + code + ".jpg." );
        txSocket.send( this.txPacket );
		FileOutputStream out = new FileOutputStream( code + ".jpg" );
  		
		do {
          try {
        	  this.rxSocket.receive( this.rxPacket );
          } catch ( SocketTimeoutException e ) { 
        	  break;
          }
          out.write( this.rxPacket.getData() );
		} while(rxPacket.getLength() > 0);
        out.close();
        System.out.println("End of image creation");
    }
    public void playMusic( byte[] audio, int Q ) throws LineUnavailableException {
    	System.out.println("Playing Music");
        AudioFormat linearPCM = new AudioFormat( 8000, Q, 1, true, false );
        SourceDataLine lineOut = AudioSystem.getSourceDataLine( linearPCM );
        lineOut.open( linearPCM, 32000 );
        lineOut.start();
        lineOut.write( audio, 0, audio.length );
        lineOut.stop();
        lineOut.close();
        System.out.println("Finished Music");
    }
	private static ArrayList<Byte> decodeAQDPCM(byte[] message, int mu, int beta, BufferedWriter log) throws IOException{
		ArrayList<Byte> decoded = new ArrayList<Byte>(4 * message.length);
		int previousNibble = 0;
        for( int i = 0; i < message.length; i++ ){
        	// Differences
            int D1 = (message[ i ] >>> 4 & 0x0F);
            int D2 = (message[ i ]       & 0x0F);
            D1 = (D1 - 8) * beta;
            D2 = (D2 - 8) * beta;
            
            // Audio Values 
            int X1 = (previousNibble + D1 );
            int X2 = (X1 + D2 );
            previousNibble = X2;
           
            X1 = X1 + mu;
            X2 = X2 + mu;
            decoded.add((byte)(X1));
            if (mu != 0) // improve
            	decoded.add((byte)(X1 >> 8));
            decoded.add((byte)(X2));
            if (mu != 0)
            	decoded.add((byte)(X2 >> 8));
              
            log.write( X1 + "\t" + D1 + "\n");
            log.write( X2 + "\t" + D2 + "\n");
            
        	}
		return decoded;
	}
    public byte[] getDPCM(String code ) throws IOException {
    	System.out.println("DPCM decoding");
        this.setTxData( code );
        ArrayList<Byte> audio = new ArrayList<Byte>( N_AUDIO_PACKETS * 2 * 128 );
        BufferedWriter bw = new BufferedWriter( new FileWriter( code + ".txt" ) );
        txSocket.send( this.txPacket );
        for( int j = 0; j < N_AUDIO_PACKETS; ++j ) {
            rxSocket.receive( rxPacket );
            byte[] buff = rxPacket.getData();
    		audio.addAll(decodeAQDPCM(Arrays.copyOfRange(buff, 0, buff.length), 0, 1, bw));
        }
        bw.close();
        return  arrayFromList(audio);
    }
    public byte[] getAQDPCM(String code ) throws IOException {
    	System.out.println("AQDPCM decoding");
    	resizebuffer( 132 );
        setTxData( code );
        byte[] buff = new byte[ 132 * 2 ];
        ArrayList<Byte> audio = new ArrayList<Byte>( N_AUDIO_PACKETS * 4 * 128 );
        BufferedWriter log = new BufferedWriter( new FileWriter( code + "-muBeta.txt" ) );
        BufferedWriter bw = new BufferedWriter( new FileWriter( code + ".txt" ) );
        txSocket.send( txPacket );
        for( int j = 0; j < N_AUDIO_PACKETS; ++j ){
            rxSocket.receive( this.rxPacket );
            buff = rxPacket.getData();
            
            // Thelei sketo F?
    		int mu=   (short) (buff[1] << 8 | (buff[0] & 0x00FF)); //little-endian
    		int beta= (short) (buff[3] << 8 | (buff[2] & 0x00FF));
                     
    		log.write( mu + "\t" + beta +"\n" );
    		audio.addAll(decodeAQDPCM(Arrays.copyOfRange(buff, 4, buff.length), mu, beta, bw));
        }
        bw.close();
        log.close();
        resizebuffer( 128 );
        return arrayFromList(audio);
    }
    
    public  void ithakiCopter(String code ) throws IOException {
        BufferedWriter log = new BufferedWriter(new FileWriter(code + ".txt"));  
        DatagramSocket rxSocket1 = new DatagramSocket( 48038 );
        long startTime = System.currentTimeMillis();
        // Running for 10 seconds
        while (System.currentTimeMillis() - startTime < 20 * 1000){
        	log.write( (System.currentTimeMillis() - startTime) / 1000. + "\t" );
            rxSocket1.receive( rxPacket ); 
            
            String response = new String( this.rxbuffer );
            System.out.println(response);
            
            String[] parts = response.toLowerCase().split(" ");
            int altitude    = Integer.parseInt(parts[5].split("=")[1].trim());
            double temperature = Double.parseDouble(parts[6].split("=")[1].trim());
            double pressure    = Double.parseDouble(parts[7].split("=")[1].trim());
            System.out.println(altitude + " " + temperature + " " + pressure );
            
            log.write(altitude + "\t" + temperature + "\t" + pressure + "\n");
        }
        log.close();
        rxSocket1.close();
    }
    
    public  void getCarDiagnostics(String code) throws IOException {
    	String[] pid = { "1F", "0F", "11", "0C", "0D", "05" };
        BufferedWriter bw = new BufferedWriter( new FileWriter( code + ".txt" ) );
        
        long startTime = System.currentTimeMillis();
        while (System.currentTimeMillis() - startTime < LOGGING_TIME * 60 * 1000) {
        	bw.write( (System.currentTimeMillis() - startTime) / 1000. + "\t" );
	        for( int i = 0; i < 6; i++ ) {
		        byte[] txBuffer = (code + "OBD=01 " + pid[i] + "\r").getBytes();
		        txPacket = new DatagramPacket( txBuffer, txBuffer.length, HOST_ADDRESS, SERVER_PORT );

	        	txSocket.send( txPacket );;
	            try {
					rxSocket.receive( rxPacket );
				} catch (IOException e) {
					System.out.println(e);
					continue;
				} 
	            String response = new String( rxbuffer );
	            System.out.println(response);
	            
	            // Parse measurement
	            String[] parts = response.toLowerCase().split(" ");
	            int XX = Integer.parseInt(parts[2].trim(), 16);
	            int YY = Integer.parseInt(parts[3].trim(), 16);
	            double measurement;
	            switch(pid[i]) {
	            case "1F":
	            	measurement = 256 * XX + YY;
	            	break;
	            case "0F":
	            	measurement = XX - 40;
	            	break;
	            case "11":
	            	measurement = XX * 100./255;
	            	break;
	            case "0C":
	            	measurement = (256 * XX + YY)/4;
	            	break;
	            case "0D":
	            	measurement = XX;
	            	break;
	            case "05":
	            	measurement = XX - 40;
	            	break;
	            default:
	            	throw new IOException();
	            }
	            bw.write( measurement + "\t" );
            }
	        bw.write( "\n" ); 
	     }
        bw.close();
	 }

    public void resizebuffer(int size ){
        rxbuffer = new byte[ size ];
        try{
            rxPacket = new DatagramPacket( rxbuffer, rxbuffer.length );
        }
        catch( Exception x ){
            System.out.println( "Error while resizing buffer: " + x );
        }
    }
    private void setTxData( String data ) {
        byte[] txBuffer = data.getBytes();
        txPacket = new DatagramPacket( txBuffer, txBuffer.length, HOST_ADDRESS, SERVER_PORT );
    }
    private File getUniqueFileName( String prefix, String suffix) {
    	File file = null;
    	for (int i = 0; file == null || file.exists(); i++) //Calculate unique name
    		file = new File(prefix + i + suffix);
    	System.out.println(file);
    	return file;
    }
    private byte[] arrayFromList(ArrayList<Byte> list)  {
    	byte[] ret = new byte[list.size()];
        int i = 0;
        for (Byte e : list)  
            ret[i++] = e.byteValue();
        return ret;
    }
    
	public static void main(String[] args) {
    	try {
    		new MultimediaApp().demo();
    	}
		catch (Exception e) {
	    	e.printStackTrace();
		}
	}
}
