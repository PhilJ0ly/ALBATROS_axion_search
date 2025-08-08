# Compiler and flags
CC = gcc
libPath = /project/s/sievers/philj0ly/CHAIM/A-CHAIM_User_Release-6.0.2/C/lib
LDFLAGS = -L/project/s/sievers/philj0ly/CHAIM/A-CHAIM_User_Release-6.0.2/C/lib -lmathlib -lsph_cap -lcalcChar -lACHAIMDB -lACHAIM_Chars -lBACKGROUND_Chars

# Compiler and flags
CC = gcc
CFLAGS = -Wall -I/project/s/sievers/philj0ly/CHAIM/A-CHAIM_User_Release-6.0.2/C/lib

# Source files
SRCS = CHAIM_extract.c \
	   /project/s/sievers/philj0ly/CHAIM/A-CHAIM_User_Release-6.0.2/C/lib/*.c \

# Executable name
TARGET = CHAIM_extract.o

# Default rule
all: $(TARGET)

# Build rule
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) -lm -lpthread -ldl

# Clean rule
clean:
	rm -f $(TARGET)


