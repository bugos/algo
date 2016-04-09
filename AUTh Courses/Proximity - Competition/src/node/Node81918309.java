package node;

import java.util.ArrayList;
import java.util.Comparator;

import gr.auth.ee.dsproject.proximity.board.Board;
import gr.auth.ee.dsproject.proximity.board.ProximityUtilities;
import gr.auth.ee.dsproject.proximity.board.Tile;

/**
 * A Node class used to to create the minMax Tree
 * @author Mamalakis Evangelos <mamalakis@auth.gr> <+306970489632>
 * @author Evangelou Alexandros <alexandre@auth.com> <+306980297466>
 */
public class Node81918309 {
	static int playerId;
	static int enemyId;
	
	int[] nodeMove;
	Board nodeBoard;
	Node81918309 parent;
	ArrayList<Node81918309> children;
	int nodeDepth;
	
	int nodePlayerId;
	double nodeEvaluation;
	double nodeEvaluationSum;

	/**
	 * Root node constructor. Creates the root node.
	 * @param board
	 * @param move enemy's last move. Not used.
	 * @param playerId MinMaxPlayer's id
	 */
	public Node81918309(Board board, int[] move) {
		nodeBoard = board;
		nodeMove = move;
		parent = null;
		children = new ArrayList<Node81918309>();
		nodeDepth = 0;
		nodePlayerId = enemyId;
	}
	/**
	 * Generic node constructor. Creates a node under a parent.
	 * @param x
	 * @param y
	 * @param parent
	 */
	public Node81918309(int x, int y, Node81918309 parent ) {
		this.parent = parent;
		children = new ArrayList<Node81918309>();
		nodePlayerId = parent.nodePlayerId == playerId ? enemyId : playerId;
		//int evaluationSign = parent.nodePlayerId == playerId ? 1 : -1;
		
		nodeDepth = parent.nodeDepth + 1;
		int randomNumber = Board.getNextTenNumbersToBePlayed()[nodeDepth - 1]; //should be passed.
		nodeMove = new int[] { x, y, randomNumber };
		nodeEvaluation = getEvaluation(parent.nodeBoard, randomNumber, parent.nodeBoard.getTile(x,y));
		nodeBoard = ProximityUtilities.boardAfterMove(nodePlayerId, parent.nodeBoard, nodeMove);
		
		parent.children.add(this);
	}
	
	public double getNodeEvaluation() {
		return nodeEvaluation;
	}
	
	public double getNodeEvaluationSum() {
		return nodeEvaluationSum;
	}
	public void setNodeEvaluationSum( double s) {
		nodeEvaluationSum = s;
	}
	
	public ArrayList<Node81918309> getChildren() {
		return children;
	}
	
	public int[] getNodeMove() {
		return nodeMove;
	}
	
	/**
	 * Calculate the evaluation score for a candidate move based on a heuristic algorithm
	 * @param board
	 * @param randomNumber
	 * @param tile
	 * @return the evaluation score
	 */
	protected double getEvaluation(Board board, int randomNumber, Tile tile) {
		double scoreGain = 0; // points gained after this move
		double coveredGain = 0; // vulnerability drop after this move
		
		scoreGain = randomNumber;
		// safety(= maxvuln - vuln) of the candidate move tile
		coveredGain += randomNumber - calcVulnerability(randomNumber, countEmptyNeighbours(board, tile));
		
		// Examine each neighbor separately
		Tile[] neighbors = ProximityUtilities.getNeighbors(tile.getX(), tile.getY(), board);
		for (Tile neighbor:neighbors) {
			if ( neighbor == null) { // out of board
			}
			else if ( neighbor.getPlayerId() ==  0 ) { // empty neighbor
			}
			else if ( neighbor.getPlayerId() == nodePlayerId ) { // friend neighbor
				scoreGain += 1;
				coveredGain += getVulnerabilityDiff(board, neighbor, 1, 1); // covered neighbor
				
			}
			else { // enemy neighbor
				if ( randomNumber > neighbor.getScore() ) {
					scoreGain += 2 * neighbor.getScore(); // what we get plus enemy loses
					coveredGain += neighbor.getScore() - calcVulnerability(neighbor.getScore(),
							countEmptyNeighbours(board, neighbor) + 1); // safety of the new tile
					coveredGain += neighbor.getScore() - calcVulnerability(neighbor.getScore(),
							countEmptyNeighbours(board, neighbor)); // safety lost by enemy
				}
				else {
					// todo: if we don't conquer enemy, we cover him by one tile
					coveredGain -= getVulnerabilityDiff(board, neighbor, 0, 1); // covered
				}
			}
		}
		
		//if (scoreGain > 0) System.out.println(tile.getX() + " " + tile.getY() + " " + randomNumber + ": " + scoreGain + " " + coveredGain);
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
		double enemyHigherCoef = (20 - score) / 20; // more then one tries?
		
		// empty neighbors coefficient
		// values between 0 and 1
		double emptyNeighboursCoef = Math.pow(emptyNeighbours / 6, 0.4);
		
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
	
	public static final Comparator<Node81918309> EVALUATIONSUM_ORDER_ASC = new evaluationSumComparator();
	
	/**
	 * Can be used to compare node objects according to their evaluationSum.
	 * @author bugos
	 *
	 */
	private static class evaluationSumComparator implements Comparator<Node81918309> {
	    public int compare(Node81918309 move1, Node81918309 move2) {
	    	return Double.compare(move1.getNodeEvaluationSum(), move2.getNodeEvaluationSum());
	    }
	}

}

