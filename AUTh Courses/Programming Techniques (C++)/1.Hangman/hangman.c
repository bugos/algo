#include <stdio.h>
#include <string.h>

#define FOR0( var, limit ) int var; for( var = 0; var <  limit; var++ )
#define FOR( var, start, end ) int var; for( var = start; var < end; var++ )

#define MAX_WORD_LENGTH 20
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
	int misses = 0;
	int found = 0;
	int wordLength;
	char word[MAX_WORD_LENGTH+1];
	char asterisks[MAX_WORD_LENGTH+1];
	char letter;
	char *match;
	int matchPos;
	
	printf( "Type a word: " );
	scanf("%20s", word);
	wordLength = strlen(word);
	
	strcpy( asterisks, word );
	memset (asterisks, '*', wordLength);
	
	FOR0( _, 24 ) printf("\n"); //clear screen 
	hangman(0);
	printf( asterisks );
	printf("\n");
	
	while( misses < NHangmanParts && found < wordLength ) {
		printf("Provide a letter: ");
		scanf(" %c", &letter);
		
		matchPos = -1; // No match
		while ( match = strchr( word, letter ) ) {
			matchPos = match - word;
			asterisks[ matchPos ] = word[ matchPos ];
			found++;
			
		}
		if ( matchPos == -1 ) { //missed 
			misses++;
			printf("Ooops...\n");
		}
		else {
			printf("Bravo!\n");
		}
		
		hangman(misses);
		printf( asterisks );
		printf( "\n" );
		
	}
	
	if ( matchPos == -1 ) { // Lost 
		printf("You are hanged!\n");
	}
	else { // Won
		printf("Congatulations, you are free!\n");
	}
	
	return 0;
}
