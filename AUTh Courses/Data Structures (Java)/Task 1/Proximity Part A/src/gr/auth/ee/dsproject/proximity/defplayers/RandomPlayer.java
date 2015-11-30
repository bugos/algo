package gr.auth.ee.dsproject.proximity.defplayers;

import gr.auth.ee.dsproject.proximity.board.Board;
import gr.auth.ee.dsproject.proximity.board.ProximityUtilities;

/**
 * Simulates a DS-Proximity player that plays randomly
 * @author Mamalakis Evangelos <mamalakis@auth.gr> <+306970489632>
 * @author Evangelou Alexandros <alexandre@auth.com> <+306980297466>
 */
public class RandomPlayer implements AbstractPlayer {
	private int id;
	private String name;
	private int score;
	private int numOfTiles;
	
	/**
	 * RandomPlayer Constructor
	 * @param id
	 */
	public RandomPlayer(Integer id) {
		this.id = id;
	}
	
	/**
	 * RandomPlayer Constructor
	 * @param id
	 * @param name
	 * @param score
	 * @param numOfTiles
	 */
	public RandomPlayer(Integer id, String name, Integer score, Integer numOfTiles) {
		this.id = id;
		this.name = name;
		this.score = score;
		this.numOfTiles = numOfTiles;
	}
	
	/**
	 * Id Setter
	 * @param id
	 */
	public void setId(int id) {
		this.id = id;
	}

	/**
	 * Id Getter
	 * @return The player id
	 */
	public int getId() {
		return id;
	}

	/**
	 * Name Setter
	 * @param name 
	 */
	public void setName(String name) {
		this.name = name;
	}
	
	/**
	 * Name Getter
	 * @return The player name
	 */
	public String getName() {
		return name;
	}
	
	/**
	 * Score Setter
	 * @param score The player's score
	 */
	public void setScore(int score) {
		this.score = score;
	}

	/**
	 * Score Getter
	 * @return The player score
	 */
	public int getScore() {
		return score;
	}

	/**
	 * numOfTilesTiles Setter
	 * @param tiles Tiles already played but the player
	 */
	public void setNumOfTiles(int tiles) {
		this.numOfTiles = tiles;
	}

	/**
	 * numOfTiles Getter
	 * @return The number of tiles already played by the player
	 */
	public int getNumOfTiles() {
		return numOfTiles;
	}
	
	/**
	 * Randomly determine the next move
	 * @param board
	 * @return The coordinates of the tile to play the next move
	 */
	public int[] getNextMove(Board board) {
		int x, y;
		do {
			x = (int)(Math.random() * ProximityUtilities.NUMBER_OF_COLUMNS);
			y = (int)(Math.random() * ProximityUtilities.NUMBER_OF_ROWS);
		} while (isTaken(board, x, y));
		
		return new int[] {x, y};
	}
	
	/**
	 * Specify if the given Tile is taken
	 * @param board
	 * @param x
	 * @param y
	 * @return Boolean true if Tile taken, else false
	 */
	private boolean isTaken(Board board, int x, int y) {
		return board.getTile(x, y).getColor() != 0;
	}
	
	/**
	 * Get the neighboring tiles starting from the eastern and moving clockwise
	 * @param board
	 * @param x
	 * @param y
	 * @return A 6x2 table with the coordinates of the neighbors
	 */
	public static int[][] getNeighborsCoordinates(Board board, int x, int y) {
		/* Check if y is even or odd */
		int yOdd = y & 1; 
		
		/* Define the neighbors table
		 * When y is even yOdd=0 and we get Even table
		 * When y is odd  yOdd=1 and we get Odd table. */
		int neighbors[][] = new int[][] {
			{x+1,      y},   // East
			{x  +yOdd, y+1}, // South-East
			{x-1+yOdd, y+1}, // South-West
			{x-1,      y},   // West
			{x-1+yOdd, y-1}, // North-West
			{x  +yOdd, y-1}	 // North-East
		};
		
		/* Determine invalid neighbors  */
		for ( int i = 0; i < neighbors.length; i++ ) {
			if ( !board.isInsideBoard( neighbors[i][0], neighbors[i][1] ) ) {
 				neighbors[i][0] = neighbors[i][1] = -1;
			}
		}
		return neighbors;
	}
}
