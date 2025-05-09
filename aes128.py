import sys
from collections import deque
import numpy as np

#AES SBOX from https://en.wikipedia.org/wiki/Rijndael_S-box

sBox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
)

# Inverse AES SBOX from https://en.wikipedia.org/wiki/Rijndael_S-box

INVsBox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

rCon = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)

#converts strings to 4x4 numpy arrays (padding added if needed)
def makeBlock(hexLine):
	#pad with zeros if needed
	if (len(hexLine) < 32):
		hexLine = hexLine.ljust(32, '0')
	# Truncate if input is larger than 32
	elif (len(hexLine) > 32):
		hexLine = hexLine[:32]
		
	#convert the hex string into a list of bytes representation
	byteList = []
	for i in range(0, 32, 2):
		#get two chars at a time that represent a hex then convert to an int 
		byteList.append(int(hexLine[i:i+2], 16))
	
	#convert byteList into a 4x4 NumPy array in column major order
	# dtype=np.uint8 specifies 8 bit unsinged integers 
	# order=F specifies "fortran style" or column major order for how it is stored in memory
	arr = np.array(byteList, dtype=np.uint8).reshape(4, 4, order='F')

	return arr


def subBytes(arr):
	for i in range(4):
		for j in range(4):
			arr[i,j] = sBox[arr[i,j]]

def inverseSubBytes(arr):
	for i in range(4):
		for j in range(4):
			arr[i,j] = INVsBox[arr[i,j]]

def shiftRows(arr):
	#make a temp 4x4 for value copying purposes
	temp = np.copy(arr)

	# row1 rotate over 1 byte
	arr[1,0] = temp[1,1]
	arr[1,1] = temp[1,2]
	arr[1,2] = temp[1,3]
	arr[1,3] = temp[1,0]

	# row2 rotate over 2 bytes
	arr[2,0] = temp[2,2]
	arr[2,1] = temp[2,3]
	arr[2,2] = temp[2,0]
	arr[2,3] = temp[2,1]

	# row3 rotate over 3 bytes
	arr[3,0] = temp[3,3]
	arr[3,1] = temp[3,0]
	arr[3,2] = temp[3,1]
	arr[3,3] = temp[3,2]

def invertShiftRows(arr):
	#make a temp 4x4 for value copying purposes
	temp = np.copy(arr)

	# row1 rotate inverse(left) 1 byte
	arr[1,0] = temp[1,3]
	arr[1,1] = temp[1,0]
	arr[1,2] = temp[1,1]
	arr[1,3] = temp[1,2]

	# row2 rotate inverse(left) 2 bytes
	arr[2,0] = temp[2,2]
	arr[2,1] = temp[2,3]
	arr[2,2] = temp[2,0]
	arr[2,3] = temp[2,1]

	# row3 rotate inverse(left) 3 bytes
	arr[3,0] = temp[3,1]
	arr[3,1] = temp[3,2]
	arr[3,2] = temp[3,3]
	arr[3,3] = temp[3,0]

def galoisMultiply(a, b):
	# algo from https://en.wikipedia.org/wiki/Finite_field_arithmetic#Rijndael's_(AES)_finite_field

	p = 0  # p init to 0
	# "Run the following loop eight times (once per bit)"
	for _ in range(8):  
		# "If the rightmost bit of b is set, exclusive OR the product p by the value of a. This is polynomial addition." 
		if b & 1:
			p ^= a
        	
		# "Shift b one bit to the right, discarding the rightmost bit, and making the leftmost bit have a value of zero. This divides the polynomial by x, discarding the x0 term."
		b >>= 1

		# "Keep track of whether the leftmost bit of a is set to one and call this value carry."
		carry = a & 0x80
		
		# "Shift a one bit to the left, discarding the leftmost bit, and making the new rightmost bit zero. This multiplies the polynomial by x, but we still need to take account of carry which represented the coefficient of x7."
		a <<= 1

		# "If carry had a value of one, exclusive or a with the hexadecimal number 0x1b (00011011 in binary). 0x1b corresponds to the irreducible polynomial with the high term eliminated. Conceptually, the high term of the irreducible polynomial and carry add modulo 2 to 0." 
		# i dont really understand this math, blindly following the written algorithm though
		if carry:
			a ^= 0x1b
        
		
    	
	return p & 0xFF



def mixColumns(arr, inv):
	# see https://en.wikipedia.org/wiki/Rijndael_MixColumns Matrix representation subsection
	# modify arr in place to apply the AES MixColumns transformation."""
	# matrix used changes on if we are encrypting or decryption"
	if not inv:
		mixColumnMatrix = np.array([
			[0x02, 0x03, 0x01, 0x01],
			[0x01, 0x02, 0x03, 0x01],
			[0x01, 0x01, 0x02, 0x03],
			[0x03, 0x01, 0x01, 0x02]
		], dtype=np.uint8)
	else:
		mixColumnMatrix = np.array([
			[0x0E, 0x0B, 0x0D, 0x09],
			[0x09, 0x0E, 0x0B, 0x0D],
			[0x0D, 0x09, 0x0E, 0x0B],
			[0x0B, 0x0D, 0x09, 0x0E]
		], dtype=np.uint8)

	# make copy of the original state before modifying arr
	original = arr.copy()

	for col in range(4):  # MixColumns operates column by column
		for row in range(4):
			arr[row, col] = (
				galoisMultiply(mixColumnMatrix[row, 0], original[0, col]) ^
				galoisMultiply(mixColumnMatrix[row, 1], original[1, col]) ^
				galoisMultiply(mixColumnMatrix[row, 2], original[2, col]) ^
				galoisMultiply(mixColumnMatrix[row, 3], original[3, col])
            		)	
	
def rotWordSubBytes(key):
	# rot word functionality, last column of 4x4 block is rotated by 1
	temp = key.copy;
	key[0,3] = temp[1,3]
	key[1,3] = temp[2,3]
	key[2,3] = temp[3,3]
	key[3,3] = temp[0,3]
	# replace last column with sBox values
	for i in range(4):
		key[i,3] = sBox[key[i,3]]
	

def keySchedule(keyStr):
	#turn key into block
	key = makeBlock(keyStr)
	

	for i in range (1, 11):
		#RotWord
		newCol = np.array([key[1,(i*4)-1], key[2,(i*4)-1], key[3,(i*4)-1], key[0,(i*4)-1]], dtype=np.uint8)
		#SubBytes
		for j in range(4):
			newCol[j] = sBox[newCol[j]]
		
		
		# xor resulting column with the column four positions earliar then xor with Rcon
		#rConCol = np.array([rCon[i],0x00,0x00,0x00], dtype=np.uint8)
		# column to be insterted
		insertCol = np.array([newCol[0] ^ key[0, (i*4)-4] ^ rCon[i], newCol[1] ^ key[1, (i*4)-4], newCol[2] ^ key[2, (i*4)-4], newCol[3] ^ key[3, (i*4)-4]], dtype=np.uint8)
		# add this new column to the growing key
		key = np.column_stack((key,insertCol))
		
		# add second column for new 4x4
		secondInsertCol = np.array([insertCol[0] ^ key[0, (i*4)-3],insertCol[1] ^ key[1, (i*4)-3],insertCol[2] ^ key[2, (i*4)-3],insertCol[3] ^ key[3, (i*4)-3]], dtype=np.uint8)
		key = np.column_stack((key,secondInsertCol))

		# add third column for new 4x4
		thirdInsertCol = np.array([secondInsertCol[0] ^ key[0, (i*4)-2],secondInsertCol[1] ^ key[1, (i*4)-2],secondInsertCol[2] ^ key[2, (i*4)-2],secondInsertCol[3] ^ key[3, (i*4)-2]], dtype=np.uint8)
		key = np.column_stack((key,thirdInsertCol))

		# add fourth column for new 4x4
		fourthInsertCol = np.array([thirdInsertCol[0] ^ key[0, (i*4)-1],thirdInsertCol[1] ^ key[1, (i*4)-1],thirdInsertCol[2] ^ key[2, (i*4)-1],thirdInsertCol[3] ^ key[3, (i*4)-1]], dtype=np.uint8)
		key = np.column_stack((key,fourthInsertCol))
		
	return key
	

#def addRoundKey(arr, key):
	


def xor4x4(first, second):
	# empty array to be populated with xor'd values
	ret = np.empty((4, 4), dtype=np.uint8)
	for i in range(4):
		for j in range(4):
			ret[i,j] = first[i,j] ^ second[i,j]
	return ret	
	
# get encrypt or decrypt option
operation = sys.argv[1]

#get name of keyFile
keyFileName = sys.argv[2]
keyFile = open(keyFileName, "r")
keyStr = keyFile.readline()
#keyStr = "2b7e151628aed2a6abf7158809cf4f3c"
key = makeBlock(keyStr)

#get name of inputFile, this can either represent plaintext or ciphertext
inputFileName = sys.argv[3]
inputFile = open(inputFileName, "r")
#inputStr = keyFile.readline()
#inputStr = "3243f6a8885a308d313198a2e0370734"
#plaintext = makeBlock(inputStr)

#get from user somehow
#cipherTextHex = "3925841d02dc09fbdc118597196a0b32"

#cipherTextBlock = makeBlock(cipherTextHex)

# support for 128, 192, and 256
#mode = sys.argv[4]


# make a linked list of 4x4 array blocks to encrypt
blocks = deque()
for line in inputFile:
	
	arr = makeBlock(line)
	
	blocks.append(arr)



# this AES implementation is hilariously slow 
# Function to print a 4x4 array in hex format
def printHex(arr):
	for i in range(4):
		for j in range(4):
			print(f'{arr[i, j]:02x}', end=' ')
		print()


if (operation == "e"):
	filename = inputFileName + ".enc"
	f = open(filename, "w")
	#have to encrypt each line, i chose a deque for some reason to store each line
	numLines = len(blocks) 
	for i in range(numLines):
		#initial round 
		print("\nInitial Round")
		print("Plaintext:")
		plaintext = blocks.popleft()
		printHex(plaintext)
		currentBlock = xor4x4(key, plaintext)
		print("\nAfter Initial AddRoundKey:")
		printHex(currentBlock)
		#keySchedule
		roundKeys = keySchedule(keyStr)
		print("\nGenerated Round Keys:")
		printHex(roundKeys)
		#loop through the middle rounds (9) 
		for i in range(1, 10):
			print(f"\nRound {i}")
			#subBytes
			print("Before SubBytes:")
			printHex(currentBlock)
			subBytes(currentBlock)
			print("\nAfter SubBytes:")
			printHex(currentBlock)
			#shiftRows
			shiftRows(currentBlock)
			print("\nAfter ShiftRows:")
			printHex(currentBlock)
			#mixColumns
			mixColumns(currentBlock, False)
			print("\nAfter MixColumns:")
			printHex(currentBlock)
			#addRoundKey
			currentBlock = xor4x4(currentBlock, roundKeys[:, 4*i:4*(i+1)])
			print("\nAfter AddRoundKey:")
			printHex(currentBlock)

		#final round 
		print("\n\nFinal Round")
		print("\nBefore SubBytes:")
		printHex(currentBlock)
		subBytes(currentBlock)
		print("\nAfter SubBytes:")
		printHex(currentBlock)
		shiftRows(currentBlock)
		print("After ShiftRows:")
		printHex(currentBlock)
		ciphertext = xor4x4(currentBlock, roundKeys[:, 40:44])
		print("\nAfter Final AddRoundKey:")
		printHex(ciphertext)
		flattened = ciphertext.ravel(order='F')
		hexCiphertext = ''.join(f'{byte:02x}' for byte in flattened)
		print("\nCiphertext(hex): ", hexCiphertext)
		# write to "inputFile.enc"
		f.write(hexCiphertext)
		f.write("\n")

	f.close()
	
	
elif (operation == "d"):
	filename = inputFileName + ".dec"
	f = open(filename, "w")
	numLines = len(blocks) 
	for i in range(numLines):
		print("\nDecryption Process")
		#generate round keys
		roundKeys = keySchedule(keyStr)
		print("\nGenerated Round Keys:")
		printHex(roundKeys)
		#initial invert round
		print("\nInitial Inverse Round")
		print("\nCiphertext Block:")
		plaintext = blocks.popleft()
		printHex(plaintext)
		currentBlock = xor4x4(plaintext, roundKeys[:, 40:44])
		print("\nAfter Initial AddRoundKey:")
		printHex(currentBlock)
		# invert shift rows
		invertShiftRows(currentBlock)
		print("\nAfter Inverse ShiftRows:")
		printHex(currentBlock)
		# invert subBytes
		inverseSubBytes(currentBlock)
		print("\nAfter Inverse SubBytes:")
		printHex(currentBlock)
		for i in range(1, 10):
			print(f"\nRound {i}")
			# xor start of round with previous round key to get after mix columns for current round
			print("\nBefore AddRoundKey:")
			printHex(currentBlock) 
			currentBlock = xor4x4(currentBlock, roundKeys[:, (40-(4*i)):(40-(4*(i-1)))])
			print("\nAfter AddRoundKey:")
			printHex(currentBlock)
			#invert mixColumns
			mixColumns(currentBlock, True);
			print("\nAfter Inverse MixColumns:")
			printHex(currentBlock)
			# invert shift rows
			invertShiftRows(currentBlock)
			print("\nAfter Inverse ShiftRows:")
			printHex(currentBlock)
			# invert subBytes
			inverseSubBytes(currentBlock)
			print("\nAfter Inverse SubBytes:")
			printHex(currentBlock)
		
		
		# final invert round	
		print("\n\nFinal Inverse Round")
		print("\nBefore AddRoundKey:")
		printHex(currentBlock)
		CTDecryption = xor4x4(currentBlock, roundKeys[:, 0:4])
		print("\nAfter Final AddRoundKey:")
		printHex(CTDecryption)
		flattened = CTDecryption.ravel(order='F')
		hexPlaintext = ''.join(f'{byte:02x}' for byte in flattened)
		print("\nPlaintext(hex): ", hexPlaintext)
		f.write(hexPlaintext)
		f.write("\n")
	f.close()
	
	
else:
	raise Exception("the operation you chose doesnt exist")

