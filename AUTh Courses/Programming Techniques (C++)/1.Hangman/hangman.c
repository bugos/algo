#include <stdio.h>
#include <string.h>

#define FOR0( var, limit ) int var; for( var = 0; var <  limit; var++ )
#define FOR( var, start, end ) int var; for( var = start; var < end; var++ )

#define hangmanImageSize 40
char hangmanImage[] = 
"++----\n"
"|    O\n"
"|   /|\\\n"
"|   / \\\n"
;
int NHangmanParts = 6;
int hangmanParts[] = { 7+5, 7+7+4, 7+7+5, 7+7+6, 7+7+8+4, 7+7+8+6 };


void hangman( int n ) {
	char h[hangmanImageSize];
	strcpy( h, hangmanImage );
	FOR( unusedPart, n, NHangmanParts ) {
		h[ hangmanParts[ unusedPart ] ] = ' ';
	}
	printf(h);	
}

int main() {
	// your code goes here
	FOR0( i, 7 ) {
		hangman(i);
	}
	return 0;
}
