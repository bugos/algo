.data 0x10000000
	newLine:           .asciiz "\n"
	inputStateString1:  .asciiz "\nInsert state for switch "
	inputStateString2:  .asciiz "(0 or 1): "
	logStateString:    .asciiz "You inserted "
	menuString:        .asciiz "\nChoose logical operation:\n1.AND\n2.OR\n3.XOR\n"
	andString:        .asciiz "Result of AND operation: "
	orString:        .asciiz "Result of OR operation: "
	xorString:        .asciiz "Result of XOR operation: "

.text

main:

# Input States
move $t0, $zero # initialize loop counter

inputStatesLoop:

addi $t0, $t0, 1 # increase loop counter

li $v0, 4
la $a0, inputStateString1
syscall

li $v0, 1 # print switch number
move $a0, $t0
syscall

li $v0, 4
la $a0, inputStateString2
syscall

li $v0, 5 # input state
syscall

move $t1, $v0 # save state for immediate output

sll $s1, $s1, 1 # shift left 
or $s1, $s1, $v0 #store to lower byte of s1

li $v0, 4 # print generic string
la $a0, logStateString
syscall

li $v0, 1 # print switch state
move $a0, $t1
syscall

bne $t0, 3, inputStatesLoop # check loop counter

# Unmask States
andi $t4, $s1, 1
andi $t5, $s1, 2 # 2=10b
srl $t5, $t5, 1
andi $t6, $s1, 4 # 4=100b
srl $t6, $t6, 2

# Operations Menu
operationsMenu:

li $v0, 4
la $a0, menuString
syscall

li $v0, 5 #input operation
syscall

addi $t1, $zero, 1
addi $t2, $zero, 2
addi $t3, $zero, 3

beq $v0, $t1, doAND
beq $v0, $t2, doOR
beq $v0, $t3, doXOR
j operationsMenu #wrong input

doAND:
li $v0, 4
la $a0, andString
syscall

and $t7, $t4, $t5
and $t7, $t7, $t6 

j printResult

doOR:
li $v0, 4
la $a0, orString
syscall

or $t7, $t4, $t5
or $t7, $t7, $t6 

j printResult

doXOR:
li $v0, 4
la $a0, xorString
syscall

xor $t7, $t4, $t5
xor $t7, $t7, $t6 

j printResult

printResult:
li $v0, 1 # print result
move $a0, $t7
syscall


# Exit
li $v0, 10 
syscall

