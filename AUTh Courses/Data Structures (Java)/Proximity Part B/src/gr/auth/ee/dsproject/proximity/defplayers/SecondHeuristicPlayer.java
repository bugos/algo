package gr.auth.ee.dsproject.proximity.defplayers;
import java.util.PriorityQueue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Collections;


import gr.auth.ee.dsproject.proximity.board.Board;
import gr.auth.ee.dsproject.proximity.board.ProximityUtilities;
import gr.auth.ee.dsproject.proximity.board.Tile;

public class SecondHeuristicPlayer implements AbstractPlayer {

	int score;
	int id;
	String name;
	int numOfTiles;

	public SecondHeuristicPlayer (Integer pid)
	{
		id = pid;
		name = "Heuristic";
	}

	public String getName ()
	{
		return name;
	}

	public int getNumOfTiles(){
		return numOfTiles;
	}

	public void setNumOfTiles(int tiles){
		numOfTiles = tiles;
	}

	public int getId ()
	{
		return id;
	}

	public void setScore (int score)
	{
		this.score = score;
	}

	public int getScore ()
	{
		return score;
	}

	public void setId (int id)
	{
	
		this.id = id;

	}

	public void setName (String name)
	{
		this.name = name;

	}

	public int[] getNextMove (Board board , int randomNumber) {
		ArrayList<double[]> possibleMoves = new ArrayList<double[]>();
		
		// Get all empty tiles as possible moves
		for ( int x = 0; x < ProximityUtilities.NUMBER_OF_COLUMNS; x++ ) {
			for ( int y = 0; y < ProximityUtilities.NUMBER_OF_ROWS; y++ ) {
				if ( !isTaken(board, x, y) ) {
					Tile t = board.getTile(x,y);
					Double e = getEvaluation(board, randomNumber, t);
					possibleMoves.add( new double[] { x, y, e } );
				}
			}
		}
		
		// Get the move with the maximum evaluation
		double[] bestMove = Collections.max(possibleMoves, new Comparator<double[]>() {
		    public int compare(double[] move1, double[] move2) {
		    	return (int)(move1[2] - move2[2]);
		    }
		});
		
		return new int[] {(int)bestMove[0], (int)bestMove[1], randomNumber};
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
	
	double getEvaluation(Board board, int randomNumber, Tile tile) {
		double scoreGain = 0; // points gained after this move
		double coveredGain = 0; // vulnerability drop after this move
		
		coveredGain +=calcVulnerability(randomNumber, countEmptyNeighbours(board, tile)); // how covered is the tile
		
		Tile[] neighbors = ProximityUtilities.getNeighbors(tile.getX(), tile.getY(), board);
		for (Tile neighbor:neighbors) {
			if ( neighbor == null) { // out of board
			}
			else if ( neighbor.getPlayerId() ==  0 ) { // empty neighbor
			}
			else if ( neighbor.getPlayerId() == id ) { // friend neighbor
				scoreGain += 1;
				coveredGain += getVulnerabilityDiff(board, neighbor, 1, 1); // covered neighbor
				
			}
			else { // enemy neighbor
				if ( randomNumber > neighbor.getScore() ) {
					scoreGain += 2 * neighbor.getScore(); // what we get plus enemy loses
					coveredGain += calcVulnerability(neighbor.getScore(),
							countEmptyNeighbours(board, neighbor)); // (x2?)how covered is the new tile
				}
				else {
					// todo: if we don't conquer enemy, we cover him by one tile
					// coveredGain -= getVulnerabilityDiff(board, neighbor, 0, 1); // covered
				}
			}
		}
		
		if (scoreGain > 0) System.out.println(tile.getX() + " " + tile.getY() + " " + randomNumber + ": " + scoreGain + " " + coveredGain);
		return scoreGain + coveredGain;
	}
	
	/**
	 * Calculates the drop of vulnerability after we add scoreAdded score 
	 * and neighboursCovered neighbors to the tile
	 * @param board
	 * @param tile
	 * @param scoreAdded
	 * @param neighboursCovered
	 * @return the vulnerability drop
	 */
	private double getVulnerabilityDiff(Board board, Tile tile, double scoreAdded, double neighboursCovered) {
		double score = tile.getScore();
		double emptyNeighbours = countEmptyNeighbours(board, tile);
		
		double newScore = score + scoreAdded;
		double newEmptyNeighbours = emptyNeighbours - neighboursCovered;
		
		return calcVulnerability(score, emptyNeighbours) - calcVulnerability(newScore, newEmptyNeighbours);
	}
	
	/**
	 * Estimates how vulnerable a tile is to a neighbor's move next to it.
	 * @param score The score of the tile
	 * @param emptyNeighbours The number of empty neighbors of the tile.
	 * @return vulnerability values between 0 and score
	 */
	private double calcVulnerability( double score, double emptyNeighbours ) {
		// probability of enemy playing higher score tile 
		// (can use getMyPool() here)
		// values between 0 and 1
		double enemyHigherCoef = (20 - score) / 20;
		
		// empty neighbors coefficient
		// values between 0 and 1
		double emptyNeighboursCoef = Math.pow(emptyNeighbours / 6, 0.4) ;
		
		return  enemyHigherCoef * emptyNeighboursCoef * score;
	}
	
	/**
	 * Counts the empty neighbors of the tile
	 * @param board
	 * @param tile
	 * @return the number of empty neighbors
	 */
	private int countEmptyNeighbours(Board board, Tile tile) {
		int emptyNeighbours = 0;
		Tile[] neighbors = ProximityUtilities.getNeighbors(tile.getX(), tile.getY(), board);
		for (Tile neighbor:neighbors) {
			if ( neighbor == null) { // out of board
			}
			else if ( neighbor.getPlayerId() ==  0 ) { // empty neighbor
				emptyNeighbours++;
			}
		}
		return emptyNeighbours;
	}
}

