package gr.auth.ee.dsproject.proximity.defplayers;

import gr.auth.ee.dsproject.proximity.board.Board;
import gr.auth.ee.dsproject.proximity.board.ProximityUtilities;

public class RandomPlayer implements AbstractPlayer {
	private int id;
	private String name;
	private int score;
	private int numOfTiles;
	
	public RandomPlayer(Integer id) {
		this.id = id;
	}
	
	public RandomPlayer(Integer id, String name, Integer score, Integer numOfTiles) {
		this.id = id;
		this.name = name;
		this.score = score;
		this.numOfTiles = numOfTiles;
	}
	
	public void setId(int id) {
		this.id = id;
	}

	public int getId() {
		return id;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public void setScore(int score) {
		this.score = score;
	}

	public int getScore() {
		return score;
	}

	public void setNumOfTiles(int tiles) {
		this.numOfTiles = tiles;
	}

	public int getNumOfTiles() {
		return numOfTiles;
	}

	public int[] getNextMove(Board board) {
		int x, y;
		do {
			x = (int)(Math.random() * ProximityUtilities.NUMBER_OF_COLUMNS);
			y = (int)(Math.random() * ProximityUtilities.NUMBER_OF_ROWS);
		} while (isTaken(board, x, y));
		
		return new int[] {x, y};
	}
	
	private boolean isTaken(Board board, int x, int y) {
		return board.getTile(x, y).getColor() != 0;
	}
	
	public static int[][] getNeighborsCoordinates(Board board, int x, int y) {
		int xOdd = x & 1; // Check if x is odd
		
		// When x is even xOdd=0 and we get even table
		// When x is odd  xOdd=1 and we get Odd table.
		int res[][] = new int[][]{
			{x+1,      y},
			{x  +xOdd, y+1},
			{x-1+xOdd, y+1},
			{x-1,      y},
			{x-1+xOdd, y-1},
			{x  +xOdd, y-1}
		};
		
		for ( int i = 0; i < res.length; i++ ) {
			//if ( res[i][0] < 0 || res[i][1] < 0 || "..." ) {
			if ( !board.isInsideBoard( res[i][0], res[i][1 ] ) ) {
 				res[i][0] = res[i][1] = -1;
			}
		}
		return res;
	}

}
