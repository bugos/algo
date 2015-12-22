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

public class HeuristicPlayer implements AbstractPlayer {

	int score;
	int id;
	String name;
	int numOfTiles;

	public HeuristicPlayer (Integer pid)
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
		System.out.println(randomNumber);
		ArrayList<double[]> possibleMoves = new ArrayList<double[]>();
		
		for ( int x = 0; x < ProximityUtilities.NUMBER_OF_COLUMNS; x++ ) {
			for ( int y = 0; y < ProximityUtilities.NUMBER_OF_ROWS; y++ ) {
				if ( !isTaken(board, x, y) ) {
					Tile t = board.getTile(x,y);
					Double e = getEvaluation(board, randomNumber, t);
					possibleMoves.add( new double[] { x, y, e } );
				}
			}
		}
		
		double[] bestMove = Collections.max(possibleMoves, new Comparator<double[]>() {
		    public int compare(double[] move1, double[] move2) {
		    	return (int)(move1[2] - move2[2]);
		    }
		});
		
		return new int[] {(int)bestMove[0], (int)bestMove[1], randomNumber};
	}
	
	private boolean isTaken(Board board, int x, int y) {
		return board.getTile(x, y).getColor() != 0;
	}
	
	double getEvaluation(Board board, int randomNumber, Tile tile) {
		Tile[] neighbors = ProximityUtilities.getNeighbors(tile.getX(), tile.getY(), board);
		int neighborsScore = 0;
		int covered = 0;
		
		for (Tile neighbor:neighbors) {
			if ( neighbor == null) { // out of board
				covered += tile.getScore();
			}
			else if ( neighbor.getPlayerId() ==  0 ) { // empty neighbor
				
			}
			else if ( neighbor.getPlayerId() == id ) { // my neighbor
				neighborsScore += 1;
				covered += neighbor.getScore();
				
				covered += tile.getScore();
			}
			else { // enemy neighbor
				if ( randomNumber > neighbor.getScore() ) {
					neighborsScore += neighbor.getScore();
					covered += neighbor.getScore();
				}
				covered += tile.getScore();
			}
		}
		
		System.out.println(covered / 6. + " " + neighborsScore);
		return neighborsScore + (covered / 6.);
		// points gained
		// safety points gained
	}

}


